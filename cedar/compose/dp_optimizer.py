import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from cedar.pipes import Pipe

from .my_optimizer import MyOptimizer
from . import constants
from .optimizer import OptimizerOptions, PhysicalPlan, PipeDesc, PipeVariantType


logger = logging.getLogger(__name__)


def _work_prod(optimizer: "DpOptimizer", mask: int) -> float:
    method = getattr(optimizer, "_dp_work_prod", None)
    if method is not None:
        return method(mask)
    return optimizer._dp_r_prod[mask]


@dataclass(frozen=True)
class DpStateSummary:
    """
    Compact state carried by the outer DP.

    The key design difference from MyOptimizer is that strategy state is a
    summary object. New strategies should extend or replace this summary only
    when future transitions need extra information.
    """

    cache_active: bool = False
    parallel_stage_cpus: int = 0


@dataclass(frozen=True)
class BlockCandidate:
    """
    One implementation choice for a block of operators.

    `mask` identifies the operators covered by the block. `order` is the
    block-local execution order in inner-op indices, and `variant` is the
    physical backend used for the whole block.
    """

    mask: int
    order: Tuple[int, ...]
    variant: PipeVariantType
    cost: float
    materializes_fusion: bool


class _BlockCostIndex:
    """Lazily compute exact subset-order costs for one backend.

    A fixed-order barrier makes most masks unreachable as prefixes of the
    complete pipeline.  Computing all ``2**n`` entries eagerly dominates plan
    generation on complex Data-Juicer recipes.  Top-down memoization visits
    only masks requested by the exact outer search while preserving the same
    recurrence and tie behavior.
    """

    def __init__(
        self,
        optimizer: "DpOptimizer",
        inner_ops: List[int],
        costs: List[float],
    ) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.source_size = optimizer.profiled_stats["baseline"][
            "output_sizes"
        ][optimizer._get_source_p_id()]
        self.per_byte = [float("inf")] * self.n
        for i, p_id in enumerate(inner_ops):
            baseline_input = optimizer.profiled_stats["baseline"][
                "input_sizes"
            ][p_id]
            if baseline_input > 0 and costs[i] != float("inf"):
                self.per_byte[i] = costs[i] / baseline_input
        self._costs: Dict[int, float] = {0: 0.0}
        self._orders: Dict[int, Tuple[int, ...]] = {0: tuple()}

    def get(self, mask: int) -> Tuple[float, Tuple[int, ...]]:
        self._compute(mask)
        return self._costs[mask] * self.source_size, self._orders[mask]

    def _compute(self, mask: int) -> None:
        if mask in self._costs:
            return

        best = float("inf")
        best_order: Tuple[int, ...] = tuple()
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            i = bit.bit_length() - 1
            if self.per_byte[i] == float("inf"):
                continue
            prev = mask ^ bit
            # ``i`` is the last operator. It cannot precede a successor that
            # is already present in ``prev``.
            if any(
                prev & (1 << successor)
                and i in self.optimizer._dp_pred_indices[successor]
                for successor in range(self.n)
            ):
                continue
            self._compute(prev)
            prev_cost = self._costs[prev]
            if prev_cost == float("inf"):
                continue
            candidate = (
                prev_cost
                + _work_prod(self.optimizer, prev) * self.per_byte[i]
            )
            if candidate < best:
                best = candidate
                best_order = self._orders[prev] + (i,)

        self._costs[mask] = best
        self._orders[mask] = best_order


@dataclass(frozen=True)
class TransitionChoice:
    state: DpStateSummary
    extra_cost: float
    cache_after_idx: Optional[int] = None
    replaces_prefix_cost: bool = False


@dataclass(frozen=True)
class BackPointer:
    prev_mask: int
    prev_state: DpStateSummary
    block: BlockCandidate
    cache_after_idx: Optional[int]


@dataclass
class SearchResult:
    order: List[int]
    blocks: List[List[int]]
    variants_by_idx: Dict[int, PipeVariantType]
    cache_after_idx: Optional[int]
    cost: float


class BlockCandidateProvider:
    """
    Generates implementation candidates for each DP block.

    The current provider reproduces MyOptimizer's block model: a block can be a
    single operator, or a multi-operator fusion block when fusion is enabled.
    The interface intentionally returns a list so future optimizations can keep
    Pareto alternatives instead of forcing each mask to one global best choice.
    """

    def __init__(self, optimizer: "DpOptimizer", inner_ops: List[int]) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self._candidates_by_mask: List[Optional[List[BlockCandidate]]] = [
            None for _ in range(self.full_mask)
        ]
        self._variant_indexes: Dict[PipeVariantType, _BlockCostIndex] = {}
        self._fused_variant_indexes: Dict[
            PipeVariantType, _BlockCostIndex
        ] = {}
        self._candidate_variants: List[PipeVariantType] = []
        self._fusion_allowed_flags: List[bool] = []

    def prepare(self) -> None:
        opt = self.optimizer
        if opt.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")
        if opt._dp_costs is None:
            raise RuntimeError("DP metadata not prepared.")

        variant_compute_costs: Dict[PipeVariantType, List[float]] = {
            PipeVariantType.INPROCESS: opt._dp_costs[:]
        }
        candidate_variants: List[PipeVariantType] = [PipeVariantType.INPROCESS]

        for vt, backend_stats in opt._iter_candidate_backend_stats():
            costs_v = [float("inf")] * self.n
            for i, p_id in enumerate(self.inner_ops):
                pipe: Optional[Pipe] = opt.logical_pipes.get(p_id)
                if pipe is None or not pipe.can_mutate_to(vt):
                    continue
                if p_id not in backend_stats:
                    continue

                base_input_size = opt.profiled_stats["baseline"]["input_sizes"][p_id]
                desc = PipeDesc(name=None, variant_type=vt, variant_ctx=None)
                try:
                    total_cost = opt._calculate_pipe_cost(
                        p_id, base_input_size, desc
                    )
                    costs_v[i] = total_cost
                except Exception:
                    continue

            if any(c != float("inf") for c in costs_v):
                variant_compute_costs[vt] = costs_v
                candidate_variants.append(vt)

        self._candidate_variants = candidate_variants
        self._fusion_allowed_flags = [
            opt._allowed_fusion(p_id) for p_id in self.inner_ops
        ]
        for vt in candidate_variants:
            self._variant_indexes[vt] = _BlockCostIndex(
                opt, self.inner_ops, variant_compute_costs[vt]
            )
            if vt != PipeVariantType.INPROCESS:
                # Fusion removes intermediate boundaries, which the outer DP
                # prices explicitly, but does not justify an unmeasured
                # discount to the profiled offload cost.
                fused_compute = variant_compute_costs[vt]
                self._fused_variant_indexes[vt] = _BlockCostIndex(
                    opt, self.inner_ops, fused_compute
                )

    def candidates_for(self, mask: int) -> Iterable[BlockCandidate]:
        if mask <= 0 or mask >= self.full_mask:
            return []
        cached = self._candidates_by_mask[mask]
        if cached is not None:
            return cached

        opt = self.optimizer
        is_multi = mask.bit_count() > 1
        if is_multi and (
            not opt.options.enable_fusion
            or not all(
                self._fusion_allowed_flags[i]
                for i in range(self.n)
                if mask & (1 << i)
            )
        ):
            self._candidates_by_mask[mask] = []
            return []

        candidates: List[BlockCandidate] = []
        for vt in self._candidate_variants:
            if is_multi and not opt._dp_has_supported_fusion_cost(vt):
                continue
            if is_multi and any(
                not opt._pipe_can_materialize_fusion(self.inner_ops[i], vt)
                for i in range(self.n)
                if mask & (1 << i)
            ):
                continue

            index = self._variant_indexes[vt]
            if is_multi and vt in self._fused_variant_indexes:
                index = self._fused_variant_indexes[vt]
            block_cost, order = index.get(mask)
            if block_cost == float("inf") or not order:
                continue
            candidates.append(
                BlockCandidate(
                    mask=mask,
                    order=order,
                    variant=vt,
                    cost=block_cost,
                    materializes_fusion=is_multi,
                )
            )

        # Placement-dependent feasibility remains in the outer DP. Keep every
        # backend alternative for this mask.
        self._candidates_by_mask[mask] = candidates
        return candidates

    def _calculate_all_fusion_allowed(self) -> List[bool]:
        opt = self.optimizer
        all_allowed = [False] * self.full_mask
        all_allowed[0] = True
        for mask in range(1, self.full_mask):
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            all_allowed[mask] = all_allowed[prev] and opt._allowed_fusion(
                self.inner_ops[idx]
            )
        return all_allowed


class CacheTransitionPolicy:
    """
    Cache-specific transition policy.

    This replaces the hard-coded `flag` dimension in MyOptimizer with an
    explicit policy. Other strategies can add similar policies without changing
    the outer DP enumeration.
    """

    def __init__(self, optimizer: "DpOptimizer", inner_ops: List[int]) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self.enabled = bool(optimizer.options.enable_caching)
        self.all_non_random = self._calculate_all_non_random()

        source_p_id = optimizer._get_source_p_id()
        read_time_per_byte = optimizer.profiled_stats["disk_info"]["read_latency"]
        source_output_size = optimizer.profiled_stats["baseline"]["output_sizes"][
            source_p_id
        ]
        self.cache_cost_per_source_sample = read_time_per_byte * 1000 * source_output_size

    def initial_state(self) -> DpStateSummary:
        return DpStateSummary(cache_active=False)

    def transitions(
        self,
        prev_mask: int,
        next_mask: int,
        prev_state: DpStateSummary,
        regular_cost: float,
        block: BlockCandidate,
        next_parallel_stage_cpus: Optional[int] = None,
    ) -> Iterable[TransitionChoice]:
        if next_parallel_stage_cpus is None:
            next_parallel_stage_cpus = prev_state.parallel_stage_cpus
        next_state = DpStateSummary(
            cache_active=prev_state.cache_active,
            parallel_stage_cpus=next_parallel_stage_cpus,
        )
        yield TransitionChoice(state=next_state, extra_cost=regular_cost)

        if not self.enabled or prev_state.cache_active:
            return
        if not self.all_non_random[next_mask]:
            return
        # With an entirely deterministic pipeline, an output cache dominates
        # recomputing any suffix on every epoch. Profiled in-memory tensor
        # sizes are not reliable serialized-cache sizes (pickle and container
        # overhead differ substantially), so do not choose an earlier cache
        # solely from that proxy. Mixed random/deterministic pipelines retain
        # the joint placement search below.
        if (
            self.all_non_random[self.full_mask - 1]
            and next_mask != self.full_mask - 1
        ):
            return

        cache_after_idx = block.order[-1]
        # The cache is materialized after the newly appended block, so its
        # entries have the size produced by every operator in ``next_mask``.
        # Using ``prev_mask`` prices the block's input instead and can
        # dramatically understate the read cost of an expanding operator.
        #
        # The DP objective excludes the source cost.  Replacing the cached
        # prefix therefore means starting from the cache read cost directly;
        # subtracting the source cost here would count a benefit that was
        # never present in the DP state.
        cache_cost = (
            self.cache_cost_per_source_sample
            * _work_prod(self.optimizer, next_mask)
        )
        yield TransitionChoice(
            state=DpStateSummary(
                cache_active=True,
                # A cache hit bypasses upstream iteration, but Cedar still
                # materializes the complete physical graph and keeps every
                # upstream Ray actor/SMP process alive for the plan lifetime
                # (also needed to fill a missing cache).  Preserve those CPU
                # reservations in the steady-state feasibility state.
                parallel_stage_cpus=next_parallel_stage_cpus,
            ),
            extra_cost=cache_cost,
            cache_after_idx=cache_after_idx,
            replaces_prefix_cost=True,
        )

    def _calculate_all_non_random(self) -> List[bool]:
        opt = self.optimizer
        if opt.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")

        all_non_random = [False] * self.full_mask
        all_non_random[0] = True
        for mask in range(1, self.full_mask):
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            pipe = opt.logical_pipes.get(self.inner_ops[idx])
            is_random = bool(pipe is not None and pipe.is_random())
            all_non_random[mask] = all_non_random[prev] and not is_random
        return all_non_random


class ExtensibleDpSearch:
    """
    Strategy-agnostic subset DP.

    The engine only knows how to append a legal block to a previous subset.
    Block construction and strategy-specific state transitions are delegated to
    provider/policy objects.
    """

    def __init__(
        self,
        optimizer: "DpOptimizer",
        inner_ops: List[int],
        block_provider: BlockCandidateProvider,
        cache_policy: CacheTransitionPolicy,
    ) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self.block_provider = block_provider
        self.cache_policy = cache_policy
        self.parallel_stage_cpu_limit = optimizer._dp_parallel_stage_cpu_limit()

    def run(self) -> SearchResult:
        initial_state = self.cache_policy.initial_state()
        dp: List[Dict[DpStateSummary, float]] = [
            {} for _ in range(self.full_mask)
        ]
        back: List[Dict[DpStateSummary, BackPointer]] = [
            {} for _ in range(self.full_mask)
        ]
        dp[0][initial_state] = 0.0

        legal_masks = self._legal_prefix_masks()
        for next_mask in legal_masks[1:]:
            # Enumerate only actual subsets of next_mask. Scanning every pair
            # of dependency-closed masks and rejecting non-subsets approaches
            # 4**n on weakly constrained filter chains; subset enumeration is
            # 3**n over the complete search while visiting the exact same
            # (previous prefix, appended block) transitions.
            prev_masks = []
            prev_mask = (next_mask - 1) & next_mask
            while True:
                prev_masks.append(prev_mask)
                if prev_mask == 0:
                    break
                prev_mask = (prev_mask - 1) & next_mask
            for prev_mask in reversed(prev_masks):
                if not dp[prev_mask]:
                    continue
                block_mask = next_mask ^ prev_mask
                if block_mask:
                    self._try_extend(
                        dp, back, prev_mask, next_mask, block_mask
                    )

        final_mask = self.full_mask - 1
        if not dp[final_mask]:
            raise RuntimeError("Extensible DP failed: no feasible final state.")

        final_state, final_cost = min(
            dp[final_mask].items(), key=lambda item: item[1]
        )
        state_counts = [len(dp[mask]) for mask in legal_masks]
        search_stats = {
            "objective": "additive",
            "legal_prefix_masks": len(legal_masks),
            "retained_states": sum(state_counts),
            "maximum_frontier": max(state_counts, default=0),
            "parallel_stage_cpu_limit": self.parallel_stage_cpu_limit,
        }
        self.optimizer._dp_last_search_stats = search_stats
        logger.info("[DpOptimizer] Exact search stats: %s", search_stats)
        return self._reconstruct(back, final_mask, final_state, final_cost)

    def _legal_prefix_masks(self) -> List[int]:
        """Return every dependency-closed subset in deterministic order."""
        pred_masks = []
        for predecessors in self.optimizer._dp_pred_indices:
            mask = 0
            for predecessor in predecessors:
                mask |= 1 << predecessor
            pred_masks.append(mask)

        legal = []
        for mask in range(self.full_mask):
            remaining = mask
            valid = True
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                idx = bit.bit_length() - 1
                if pred_masks[idx] & ~mask:
                    valid = False
                    break
            if valid:
                legal.append(mask)
        return legal

    def _try_extend(
        self,
        dp: List[Dict[DpStateSummary, float]],
        back: List[Dict[DpStateSummary, BackPointer]],
        prev_mask: int,
        next_mask: int,
        block_mask: int,
    ) -> None:
        if not dp[prev_mask]:
            return

        for block in self.block_provider.candidates_for(block_mask):
            if not self._block_can_follow(prev_mask, block):
                continue

            operator_cost = (
                _work_prod(self.optimizer, prev_mask) * block.cost
            )
            boundary_cost = self.optimizer._dp_stage_boundary_cost(
                prev_mask, block
            )
            regular_cost = operator_cost + boundary_cost
            for prev_state, prev_cost in dp[prev_mask].items():
                next_parallel_stage_cpus = (
                    prev_state.parallel_stage_cpus
                    + self.optimizer._dp_parallel_stage_cpu_cost(block.variant)
                )
                if (
                    self.parallel_stage_cpu_limit is not None
                    and next_parallel_stage_cpus
                    > self.parallel_stage_cpu_limit
                ):
                    continue
                for choice in self.cache_policy.transitions(
                    prev_mask,
                    next_mask,
                    prev_state,
                    regular_cost,
                    block,
                    next_parallel_stage_cpus,
                ):
                    if choice.replaces_prefix_cost:
                        candidate_cost = choice.extra_cost
                    else:
                        candidate_cost = prev_cost + choice.extra_cost
                    old = dp[next_mask].get(choice.state, float("inf"))
                    old_pointer = back[next_mask].get(choice.state)
                    prefer_fused_tie = (
                        math.isclose(
                            candidate_cost,
                            old,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        and block.materializes_fusion
                        and (
                            old_pointer is None
                            or block.mask.bit_count()
                            > old_pointer.block.mask.bit_count()
                        )
                    )
                    if candidate_cost < old or prefer_fused_tie:
                        dp[next_mask][choice.state] = candidate_cost
                        back[next_mask][choice.state] = BackPointer(
                            prev_mask=prev_mask,
                            prev_state=prev_state,
                            block=block,
                            cache_after_idx=choice.cache_after_idx,
                        )

    def _block_can_follow(self, prev_mask: int, block: BlockCandidate) -> bool:
        if block.mask.bit_count() == 1:
            idx = block.mask.bit_length() - 1
            return self.optimizer._dp_valid_single_last(prev_mask, idx)
        return (
            self.optimizer._dp_fusion_block_valid(prev_mask, block.mask)
            and self.optimizer._dp_fusion_transport_feasible(prev_mask, block)
        )

    def _reconstruct(
        self,
        back: List[Dict[DpStateSummary, BackPointer]],
        final_mask: int,
        final_state: DpStateSummary,
        final_cost: float,
    ) -> SearchResult:
        mask = final_mask
        state = final_state
        blocks_rev: List[List[int]] = []
        variants_by_idx: Dict[int, PipeVariantType] = {}
        cache_after_idx: Optional[int] = None

        while mask:
            pointer = back[mask].get(state)
            if pointer is None:
                raise RuntimeError("Extensible DP backtracking hit illegal state.")

            order = list(pointer.block.order)
            blocks_rev.append(order)
            for idx in order:
                variants_by_idx[idx] = pointer.block.variant
            if pointer.cache_after_idx is not None:
                cache_after_idx = pointer.cache_after_idx

            mask = pointer.prev_mask
            state = pointer.prev_state

        blocks_rev.reverse()
        flat_order: List[int] = []
        for block in blocks_rev:
            flat_order.extend(block)

        return SearchResult(
            order=flat_order,
            blocks=blocks_rev,
            variants_by_idx=variants_by_idx,
            cache_after_idx=cache_after_idx,
            cost=final_cost,
        )


class DpOptimizer(MyOptimizer):
    """
    Extensible DP optimizer.

    This class keeps MyOptimizer's current semantics but moves strategy-specific
    logic behind candidate providers and transition policies. Adding a new
    optimization should normally mean adding a provider/policy, not editing the
    subset-DP engine itself.
    """

    def run(
        self, profiled_data: Union[str, Dict[str, Any]], options: OptimizerOptions
    ) -> PhysicalPlan:
        if not self._initialized:
            raise RuntimeError("Must initialize optimizer before running.")

        from yaml import safe_load

        if isinstance(profiled_data, dict):
            self.profiled_stats = profiled_data
        else:
            import pathlib

            path = pathlib.Path(profiled_data)
            try:
                with path.open("r") as f:
                    self.profiled_stats = safe_load(f)
            except Exception as e:
                logger.error("An error occurred %s", e)
                raise RuntimeError(f"Failed to read profiled stats {profiled_data}")

        self.options = options
        self._validate_stats()
        self._init_stats()

        logger.info("[DpOptimizer] Running extensible DP optimization pass...")
        self._logical_opt()
        self._physical_opt()

        if self.options.enable_prefetch:
            logger.info("*Prefetching Pass*")
            self._insert_prefetch()

        try:
            caching_on = self._get_cache_pid(self.physical_plan) is not None
            fused_blocks = [
                list(desc.fused_pipes)
                for desc in self.physical_plan.pipe_descs.values()
                if getattr(desc, "fused_pipes", None)
                and len(getattr(desc, "fused_pipes", [])) > 1
            ]
            fused_pipes = fused_blocks if fused_blocks else None
            optimized_cost = self.calculate_cost(
                self.physical_plan.graph,
                physical_specs=self.physical_plan.pipe_descs,
                fused_pipes=fused_pipes,
                caching_on=caching_on,
                plan=self.physical_plan,
            )
            logger.info("[DpOptimizer] Optimized plan cost = %s", optimized_cost)
        except Exception as e:
            logger.info("[DpOptimizer] Failed to calculate optimized plan cost: %s", e)

        self._log_optimized_pipeline(tag="DpOptimizer")
        return self.physical_plan

    def _dp_parallel_stage_cpu_limit(self) -> Optional[int]:
        """Return the per-worker parallel-stage budget for a fixed-W run.

        Strict profile matching accounts one CPU for the local worker plus the
        profiled actor/process width of every active Ray/SMP stage.  Keeping
        this constraint in the DP state prevents selecting a cheap plan that
        cannot subsequently be materialized under the experiment's unified
        CPU budget.
        """
        if os.environ.get("CEDAR_MATCH_PROFILE_RESOURCES") != "1":
            return None
        fixed_workers_raw = os.environ.get(
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
        )
        cpu_budget_raw = os.environ.get("CEDAR_PROFILE_MATCH_CPU_BUDGET")
        if fixed_workers_raw is None or cpu_budget_raw is None:
            return None
        try:
            fixed_workers = int(fixed_workers_raw)
            cpu_budget = int(cpu_budget_raw)
        except ValueError as exc:
            raise RuntimeError(
                "Invalid fixed-worker CPU budget configuration"
            ) from exc
        if fixed_workers < 1 or cpu_budget < fixed_workers:
            raise RuntimeError(
                "Fixed local workers cannot fit under the unified CPU budget: "
                f"fixed={fixed_workers}, budget={cpu_budget}"
            )
        return cpu_budget // fixed_workers - 1

    def _dp_parallel_stage_cpu_cost(
        self, variant: PipeVariantType
    ) -> int:
        config = self.profiled_stats.get("resource_config", {})
        if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            return int(config.get("ray_actors_per_stage", 1))
        if variant == PipeVariantType.SMP:
            return int(config.get("smp_procs_per_stage", 1))
        return 0

    def _dp_reorder_offload_cache_fusion(
        self,
        inner_ops: List[int],
    ) -> Tuple[List[int], Optional[int]]:
        if not inner_ops:
            return [], None
        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")

        block_provider = BlockCandidateProvider(self, inner_ops)
        block_provider.prepare()
        cache_policy = CacheTransitionPolicy(self, inner_ops)
        search = ExtensibleDpSearch(
            optimizer=self,
            inner_ops=inner_ops,
            block_provider=block_provider,
            cache_policy=cache_policy,
        )
        result = search.run()
        self._last_dp_state_cost = result.cost

        self._store_pending_fusions_from_blocks(
            blocks_in_idx_order=result.blocks,
            inner_ops=inner_ops,
            chosen_variant_by_idx=result.variants_by_idx,
        )

        for idx, p_id in enumerate(inner_ops):
            vt = result.variants_by_idx.get(idx, PipeVariantType.INPROCESS)
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = vt

        logger.info("[DpOptimizer] DP state cost (inner ops only): %s", result.cost)
        logger.info(
            "[DpOptimizer] DP objective: %s",
            "additive_work",
        )

        best_order = [inner_ops[idx] for idx in result.order]
        cache_p_id = (
            inner_ops[result.cache_after_idx]
            if result.cache_after_idx is not None
            else None
        )
        return best_order, cache_p_id


__all__ = [
    "DpOptimizer",
    "DpStateSummary",
    "BlockCandidate",
    "BlockCandidateProvider",
    "CacheTransitionPolicy",
    "ExtensibleDpSearch",
]
