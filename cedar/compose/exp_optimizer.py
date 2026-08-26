"""Experimental optimizer with a conservative, separable cost model.

This module deliberately leaves ``my_optimizer`` and ``dp_optimizer`` intact.
It reuses their subset-DP search engine, but replaces the block candidate cost
model so operator compute is never discounted by fusion.  Optional stage
boundary coefficients can be supplied under ``profile["exp_cost_model"]``.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .dp_optimizer import (
    BlockCandidate,
    CacheTransitionPolicy,
    DpObjectiveCost,
    DpOptimizer,
    ExtensibleDpSearch,
)
from .optimizer import (
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
    PipeVariantType,
)


logger = logging.getLogger(__name__)


class ExpBlockCandidateProvider:
    """Build block candidates using compute + stage-boundary cost."""

    def __init__(self, optimizer: "ExpOptimizer", inner_ops: List[int]) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self._candidates_by_mask: List[List[BlockCandidate]] = [
            [] for _ in range(self.full_mask)
        ]

    def prepare(self) -> None:
        opt = self.optimizer
        if opt.logical_pipes is None or opt._dp_costs is None:
            raise RuntimeError("ExpOptimizer DP metadata is not initialized")

        candidate_variants = [PipeVariantType.INPROCESS]
        for variant, _ in opt._iter_candidate_backend_stats():
            if variant not in candidate_variants:
                candidate_variants.append(variant)

        compute_per_byte: Dict[PipeVariantType, List[float]] = {}
        for variant in candidate_variants:
            costs = [float("inf")] * self.n
            for index, p_id in enumerate(self.inner_ops):
                pipe = opt.logical_pipes.get(p_id)
                if pipe is None:
                    continue
                if (
                    variant != PipeVariantType.INPROCESS
                    and not pipe.can_mutate_to(variant)
                ):
                    continue
                if not opt._has_backend_profile(p_id, variant):
                    continue
                costs[index] = opt._exp_compute_cost_per_byte(p_id, variant)
            compute_per_byte[variant] = costs

        source_size = opt.profiled_stats["baseline"]["output_sizes"][
            opt._get_source_p_id()
        ]
        fusion_allowed = self._all_fusion_allowed()

        for mask in range(1, self.full_mask):
            is_fused = mask.bit_count() > 1
            if is_fused and (
                not opt.options.enable_fusion or not fusion_allowed[mask]
            ):
                continue

            candidates: List[BlockCandidate] = []
            for variant in candidate_variants:
                if is_fused and not opt._has_stage_boundary_profile(variant):
                    continue
                if is_fused and any(
                    not opt._pipe_can_materialize_fusion(
                        self.inner_ops[index], variant
                    )
                    for index in range(self.n)
                    if mask & (1 << index)
                ):
                    continue

                order, compute_cost = self._best_compute_order(
                    mask, compute_per_byte[variant]
                )
                if not order or not math.isfinite(compute_cost):
                    continue
                normalized_cost = compute_cost + opt._exp_boundary_cost_per_byte(
                    variant, opt._dp_r_prod[mask]
                )
                candidates.append(
                    BlockCandidate(
                        mask=mask,
                        order=order,
                        variant=variant,
                        cost=normalized_cost * source_size,
                        materializes_fusion=is_fused,
                    )
                )

            if candidates:
                # Keep backend alternatives: placement-dependent transition
                # policies may make a locally dearer implementation feasible
                # or cheaper in the complete plan.
                self._candidates_by_mask[mask] = candidates

    def candidates_for(self, mask: int) -> Sequence[BlockCandidate]:
        if mask <= 0 or mask >= self.full_mask:
            return ()
        return self._candidates_by_mask[mask]

    def _best_compute_order(
        self, mask: int, costs_per_byte: List[float]
    ) -> Tuple[Tuple[int, ...], float]:
        """Find the cheapest legal order for one fixed block and backend."""
        dp: Dict[int, float] = {0: 0.0}
        back: Dict[int, Tuple[int, int]] = {}
        submask = mask
        subsets = []
        current = submask
        while True:
            subsets.append(current)
            if current == 0:
                break
            current = (current - 1) & mask

        for placed in sorted(subsets, key=int.bit_count):
            if placed not in dp:
                continue
            remaining = mask ^ placed
            while remaining:
                bit = remaining & -remaining
                index = bit.bit_length() - 1
                remaining ^= bit
                if not math.isfinite(costs_per_byte[index]):
                    continue
                if any(
                    mask & (1 << pred) and not placed & (1 << pred)
                    for pred in self.optimizer._dp_pred_indices[index]
                ):
                    continue
                next_mask = placed | bit
                candidate = (
                    dp[placed]
                    + self.optimizer._dp_r_prod[placed]
                    * costs_per_byte[index]
                )
                if candidate < dp.get(next_mask, float("inf")):
                    dp[next_mask] = candidate
                    back[next_mask] = (placed, index)

        if mask not in dp:
            return (), float("inf")
        order_reversed: List[int] = []
        current = mask
        while current:
            previous, index = back[current]
            order_reversed.append(index)
            current = previous
        order_reversed.reverse()
        return tuple(order_reversed), dp[mask]

    def _all_fusion_allowed(self) -> List[bool]:
        allowed = [False] * self.full_mask
        allowed[0] = True
        for mask in range(1, self.full_mask):
            bit = mask & -mask
            index = bit.bit_length() - 1
            allowed[mask] = allowed[mask ^ bit] and self.optimizer._allowed_fusion(
                self.inner_ops[index]
            )
        return allowed


class ExpOptimizer(DpOptimizer):
    """Experimental DP optimizer using a separable physical cost model."""

    def __init__(self) -> None:
        super().__init__()
        self._fallback_warnings: Set[Tuple[int, PipeVariantType]] = set()

    def _dp_initial_objective_cost(self) -> DpObjectiveCost:
        # ExpOptimizer is the legacy additive-model ablation. Keep its search
        # semantics isolated from DpOptimizer's throughput objective.
        return DpObjectiveCost()

    def _dp_accumulate_objective_cost(
        self,
        previous: DpObjectiveCost,
        extra_cost: float,
        block: BlockCandidate,
        prev_mask: int,
    ) -> DpObjectiveCost:
        return DpObjectiveCost(
            local_serial=previous.local_serial + extra_cost
        )

    def _profile_entry(self, mapping: Dict, p_id: int):
        return mapping.get(p_id, mapping.get(str(p_id)))

    def _explicit_compute_per_byte(
        self, p_id: int, variant: PipeVariantType
    ) -> Optional[float]:
        model = self.profiled_stats.get("exp_cost_model", {})
        by_backend = model.get("operator_compute", {}).get(variant.name, {})
        entry = self._profile_entry(by_backend, p_id)
        if entry is None:
            return None
        if not isinstance(entry, dict) or "cost_per_byte_ms" not in entry:
            raise ValueError(
                "exp_cost_model.operator_compute entries must contain "
                "cost_per_byte_ms"
            )
        value = float(entry["cost_per_byte_ms"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "operator cost_per_byte_ms must be finite and non-negative"
            )
        return value

    def _has_backend_profile(
        self, p_id: int, variant: PipeVariantType
    ) -> bool:
        if variant == PipeVariantType.INPROCESS:
            return True
        explicit = self._explicit_compute_per_byte(p_id, variant)
        if explicit is not None:
            return True
        return p_id in self.profiled_stats.get("offloads", {}).get(
            variant.name, {}
        )

    def _warn_fallback(self, p_id: int, variant: PipeVariantType, reason: str) -> None:
        key = (p_id, variant)
        if key in self._fallback_warnings:
            return
        self._fallback_warnings.add(key)
        logger.warning(
            "[ExpOptimizer] Cannot infer %s cost for pipe %s (%s); "
            "using conservative INPROCESS compute cost instead.",
            variant.name,
            p_id,
            reason,
        )

    def _calculate_pipe_cost(
        self, p_id: int, input_size: float, desc: Optional[PipeDesc]
    ) -> float:
        baseline_input = self.profiled_stats["baseline"]["input_sizes"][p_id]
        if baseline_input <= 0:
            return self._base_cost_map[p_id]
        baseline_cost = input_size / baseline_input * self._base_cost_map[p_id]
        variant = PipeVariantType.INPROCESS if desc is None else desc.variant_type
        explicit = (
            None
            if variant is None
            else self._explicit_compute_per_byte(p_id, variant)
        )
        if explicit is not None:
            return input_size * explicit
        if variant is None or variant == PipeVariantType.INPROCESS:
            return baseline_cost

        backend_stats = self.profiled_stats.get("offloads", {}).get(
            variant.name, {}
        )
        if p_id not in backend_stats:
            raise RuntimeError(
                f"Pipe {p_id} does not have a {variant.name} profile"
            )
        baseline_tput = float(self.profiled_stats["baseline"]["throughput"])
        offload_tput = float(backend_stats[p_id]["throughput"])
        fraction = float(self._fractional_latencies[p_id])
        if baseline_tput <= 0 or offload_tput <= 0 or fraction <= 0:
            self._warn_fallback(p_id, variant, "non-positive throughput/fraction")
            return baseline_cost

        denominator = baseline_tput / offload_tput - (1.0 - fraction)
        if denominator <= 1e-12:
            self._warn_fallback(p_id, variant, "Amdahl inversion is out of range")
            return baseline_cost
        pipe_speedup = fraction / denominator
        if not math.isfinite(pipe_speedup) or pipe_speedup <= 0:
            self._warn_fallback(p_id, variant, "invalid inferred speedup")
            return baseline_cost
        inferred = baseline_cost / pipe_speedup
        if not math.isfinite(inferred) or inferred < 0:
            self._warn_fallback(p_id, variant, "invalid inferred cost")
            return baseline_cost
        return inferred

    def _exp_compute_cost_per_byte(
        self, p_id: int, variant: PipeVariantType
    ) -> float:
        baseline_input = float(
            self.profiled_stats["baseline"]["input_sizes"][p_id]
        )
        if baseline_input <= 0:
            raise ValueError(f"Pipe {p_id} has non-positive baseline input size")
        desc = PipeDesc(None, variant, None)
        return self._calculate_pipe_cost(p_id, baseline_input, desc) / baseline_input

    def _has_stage_boundary_profile(
        self, variant: PipeVariantType
    ) -> bool:
        model = self.profiled_stats.get("exp_cost_model", {})
        return variant.name in model.get("stage_boundaries", {})

    def _dp_fusion_transport_feasible(self, prev_mask: int, block) -> bool:
        # An explicit boundary model is stronger evidence than the legacy
        # Amdahl exit-pipe guard, whose very purpose is to protect profiles
        # that do not separate compute and transport.
        if self._has_stage_boundary_profile(block.variant):
            return True
        return super()._dp_fusion_transport_feasible(prev_mask, block)

    def _exp_boundary_cost_per_byte(
        self, variant: PipeVariantType, output_ratio: float
    ) -> float:
        model = self.profiled_stats.get("exp_cost_model", {})
        entry = model.get("stage_boundaries", {}).get(variant.name)
        if entry is None:
            return 0.0
        input_cost = float(entry.get("input_cost_per_byte_ms", 0.0))
        output_cost = float(entry.get("output_cost_per_byte_ms", 0.0))
        if any(
            not math.isfinite(value) or value < 0
            for value in (input_cost, output_cost)
        ):
            raise ValueError(
                "stage boundary coefficients must be finite and non-negative"
            )
        return input_cost + output_ratio * output_cost

    def _dp_reorder_offload_cache_fusion(
        self, inner_ops: List[int]
    ) -> Tuple[List[int], Optional[int]]:
        if not inner_ops:
            return [], None
        if self.options.enable_fusion and not self.profiled_stats.get(
            "exp_cost_model", {}
        ).get("stage_boundaries"):
            logger.warning(
                "[ExpOptimizer] Fusion enabled without stage-boundary profile; "
                "multi-operator fusion candidates are disabled."
            )
        provider = ExpBlockCandidateProvider(self, inner_ops)
        provider.prepare()
        cache_policy = CacheTransitionPolicy(self, inner_ops)
        result = ExtensibleDpSearch(
            optimizer=self,
            inner_ops=inner_ops,
            block_provider=provider,
            cache_policy=cache_policy,
        ).run()

        self._store_pending_fusions_from_blocks(
            result.blocks, inner_ops, result.variants_by_idx
        )
        for index, p_id in enumerate(inner_ops):
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = (
                    result.variants_by_idx.get(index, PipeVariantType.INPROCESS)
                )
        logger.info("[ExpOptimizer] DP state cost: %s", result.cost)
        order = [inner_ops[index] for index in result.order]
        cache_p_id = (
            inner_ops[result.cache_after_idx]
            if result.cache_after_idx is not None
            else None
        )
        return order, cache_p_id

    def calculate_cost(
        self,
        graph: Dict[int, Set[int]],
        physical_specs: Optional[Dict[int, PipeDesc]] = None,
        fused_pipes=None,
        caching_on: Optional[bool] = False,
        plan: PhysicalPlan = None,
    ) -> float:
        """Calculate linear-plan cost with compute and boundaries separated."""
        if caching_on:
            raise NotImplementedError(
                "ExpOptimizer cache cost is intentionally not implemented yet"
            )
        active_plan = self.physical_plan if plan is None else plan
        specs = active_plan.pipe_descs if physical_specs is None else physical_specs
        source = self._get_source_p_id()
        current = source
        path = [source]
        visited = {source}
        while graph[current]:
            if len(graph[current]) != 1:
                raise NotImplementedError(
                    "ExpOptimizer currently requires a linear plan"
                )
            current = next(iter(graph[current]))
            if current in visited:
                raise RuntimeError("Cycle in physical plan")
            visited.add(current)
            path.append(current)
        if visited != set(graph):
            raise RuntimeError("Disconnected physical plan")

        current_size = float(
            self.profiled_stats["baseline"]["output_sizes"][source]
        )
        total_cost = float(self._base_cost_map[source])
        for node in path[1:]:
            desc = specs.get(node)
            if node in self.logical_pipes:
                block = [node]
            elif desc is not None and desc.fused_pipes:
                block = list(desc.fused_pipes)
            else:
                continue
            variant = (
                PipeVariantType.INPROCESS
                if desc is None or desc.variant_type is None
                else desc.variant_type
            )
            block_input = current_size
            compute_desc = PipeDesc(None, variant, None)
            for p_id in block:
                total_cost += self._calculate_pipe_cost(
                    p_id, current_size, compute_desc
                )
                current_size *= self._data_size_ratio_map[p_id]
            boundary_coeff = self._exp_boundary_cost_per_byte(
                variant, current_size / block_input
            )
            total_cost += block_input * boundary_coeff
        return total_cost


__all__ = ["ExpBlockCandidateProvider", "ExpOptimizer"]
