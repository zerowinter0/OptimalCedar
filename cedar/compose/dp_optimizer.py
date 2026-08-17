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


def _work_prod(
    optimizer: "DpOptimizer", mask: int, operator_idx: int
) -> float:
    method = getattr(optimizer, "_dp_compute_work_prod", None)
    if method is not None:
        return method(mask, operator_idx)
    method = getattr(optimizer, "_dp_work_prod", None)
    if method is not None:
        return method(mask)
    return optimizer._dp_r_prod[mask]


def _volume_prod(optimizer: "DpOptimizer", mask: int) -> float:
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
                denominator = optimizer._dp_compute_cost_denominator(
                    i, baseline_input, self.source_size
                )
                if denominator > 0:
                    self.per_byte[i] = costs[i] / denominator
        self._costs: Dict[Tuple[int, int], float] = {}
        self._orders: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def get(
        self, mask: int, prefix_mask: int = 0
    ) -> Tuple[float, Tuple[int, ...]]:
        if mask & prefix_mask:
            raise ValueError("A DP block cannot overlap its prefix.")
        self._compute(prefix_mask, mask)
        key = (prefix_mask, mask)
        return self._costs[key] * self.source_size, self._orders[key]

    def _compute(self, prefix_mask: int, mask: int) -> None:
        key = (prefix_mask, mask)
        if key in self._costs:
            return
        if mask == 0:
            self._costs[key] = 0.0
            self._orders[key] = tuple()
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
            self._compute(prefix_mask, prev)
            prev_key = (prefix_mask, prev)
            prev_cost = self._costs[prev_key]
            if prev_cost == float("inf"):
                continue
            candidate = (
                prev_cost
                + _work_prod(self.optimizer, prefix_mask | prev, i)
                * self.per_byte[i]
            )
            if candidate < best:
                best = candidate
                best_order = self._orders[prev_key] + (i,)

        self._costs[key] = best
        self._orders[key] = best_order


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
    prev_objective: "DpObjectiveCost"
    block: BlockCandidate
    cache_after_idx: Optional[int]


@dataclass(frozen=True)
class DpObjectiveCost:
    """Exact service-demand coordinates retained by the subset DP.

    In-process operators share a serial worker lane, while distinct Ray/SMP
    blocks form concurrently active pipeline stages. Steady-state service time
    is therefore the larger of accumulated local work and the slowest parallel
    stage, rather than their sum.
    """

    local_serial: float = 0.0
    parallel_bottleneck: float = 0.0
    parallel_total_work: float = 0.0

    @property
    def score(self) -> float:
        return max(self.local_serial, self.parallel_bottleneck)

    @property
    def total_work(self) -> float:
        return self.local_serial + self.parallel_total_work


@dataclass
class SearchResult:
    order: List[int]
    blocks: List[List[int]]
    variants_by_idx: Dict[int, PipeVariantType]
    cache_after_idx: Optional[int]
    cost: float
    objective: DpObjectiveCost


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
        self._candidates_by_prefix_and_mask: Dict[
            Tuple[int, int], List[BlockCandidate]
        ] = {}
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
                # SMP has no accelerator resource scheduling.  Treat it as an
                # infeasible placement for CUDA operators rather than hiding
                # shared-GPU contention behind an arbitrary cost multiplier.
                if (
                    vt == PipeVariantType.SMP
                    and not opt._dp_smp_supported_for_pipe(p_id)
                ):
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
        return self.candidates_for_prefix(0, mask)

    def candidates_for_prefix(
        self, prefix_mask: int, mask: int
    ) -> Iterable[BlockCandidate]:
        if mask <= 0 or mask >= self.full_mask:
            return []
        key = (prefix_mask, mask)
        cached = self._candidates_by_prefix_and_mask.get(key)
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
            self._candidates_by_prefix_and_mask[key] = []
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
            block_cost, order = index.get(mask, prefix_mask)
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
        self._candidates_by_prefix_and_mask[key] = candidates
        return candidates

    def candidate_for_order(
        self,
        order: Iterable[int],
        variant: PipeVariantType,
        prefix_mask: int = 0,
    ) -> BlockCandidate:
        """Build the exact DP block cost for one materialized block order.

        ``candidates_for`` returns only the best internal order for a mask.
        Cost-model validation must instead replay the order stored in a fixed
        physical plan.  This method uses the same per-byte backend costs and
        work-product recurrence as ``_BlockCostIndex`` without re-optimizing
        the block's order.
        """
        ordered = tuple(order)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("A DP block must contain unique operators.")
        if any(idx < 0 or idx >= self.n for idx in ordered):
            raise ValueError("A DP block references an unknown operator.")
        if variant not in self._variant_indexes:
            raise ValueError(f"Variant {variant.name} is outside the DP search space.")

        mask = 0
        for idx in ordered:
            mask |= 1 << idx
        is_multi = len(ordered) > 1
        if is_multi:
            if not self.optimizer.options.enable_fusion:
                raise ValueError("A fused plan is outside the disabled fusion space.")
            if not all(self._fusion_allowed_flags[idx] for idx in ordered):
                raise ValueError("The materialized block contains a non-fusable operator.")
            if not self.optimizer._dp_has_supported_fusion_cost(variant):
                raise ValueError(
                    f"DP has no supported fused-block cost for {variant.name}."
                )
            if any(
                not self.optimizer._pipe_can_materialize_fusion(
                    self.inner_ops[idx], variant
                )
                for idx in ordered
            ):
                raise ValueError("The materialized fusion cannot use this variant.")
        index = self._variant_indexes[variant]
        if is_multi and variant in self._fused_variant_indexes:
            index = self._fused_variant_indexes[variant]
        if mask & prefix_mask:
            raise ValueError("A replayed DP block overlaps its prefix.")
        local_mask = 0
        normalized_cost = 0.0
        for idx in ordered:
            if index.per_byte[idx] == float("inf"):
                raise ValueError(
                    f"Operator {self.inner_ops[idx]} has no {variant.name} cost."
                )
            # Enforce the exact internal topological order rather than merely
            # checking that the block mask has some legal topological order.
            if not self.optimizer._dp_valid_single_last(local_mask, idx):
                # External predecessors are checked by the outer replay, so
                # defer this check when the missing predecessor is outside
                # the current block.
                missing_internal = [
                    predecessor
                    for predecessor in self.optimizer._dp_pred_indices[idx]
                    if mask & (1 << predecessor)
                    and not local_mask & (1 << predecessor)
                ]
                if missing_internal:
                    raise ValueError("The fused block order violates dependencies.")
            normalized_cost += (
                _work_prod(self.optimizer, prefix_mask | local_mask, idx)
                * index.per_byte[idx]
            )
            local_mask |= 1 << idx

        return BlockCandidate(
            mask=mask,
            order=ordered,
            variant=variant,
            cost=normalized_cost * index.source_size,
            materializes_fusion=is_multi,
        )

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
            * _volume_prod(self.optimizer, next_mask)
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
        global_epsilon = float(
            os.environ.get("CEDAR_DP_PARETO_EPSILON", "0.10")
        )
        if not math.isfinite(global_epsilon) or global_epsilon < 0.0:
            raise ValueError("CEDAR_DP_PARETO_EPSILON must be finite and non-negative")
        # Small optimality/scalability cases remain exact. For complex formal
        # workloads, multiplicative trimming bounds the worst-case accumulated
        # coordinate error by global_epsilon across at most n prefix steps.
        self.pareto_global_epsilon = global_epsilon if self.n > 8 else 0.0
        self.pareto_step_epsilon = (
            (1.0 + self.pareto_global_epsilon) ** (1.0 / self.n) - 1.0
            if self.pareto_global_epsilon > 0.0
            else 0.0
        )

    def run(self) -> SearchResult:
        initial_state = self.cache_policy.initial_state()
        dp: List[Dict[DpStateSummary, List[DpObjectiveCost]]] = [
            {} for _ in range(self.full_mask)
        ]
        back: List[
            Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]
        ] = [
            {} for _ in range(self.full_mask)
        ]
        initial_objective = self._initial_objective()
        dp[0][initial_state] = [initial_objective]

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

        final_state, final_objective = min(
            (
                (state, objective)
                for state, objectives in dp[final_mask].items()
                for objective in objectives
            ),
            key=lambda item: (
                item[1].score,
                item[1].total_work,
                item[1].local_serial,
            ),
        )
        state_counts = [
            sum(len(frontier) for frontier in dp[mask].values())
            for mask in legal_masks
        ]
        search_stats = {
            "objective": "throughput_bottleneck_pareto",
            "legal_prefix_masks": len(legal_masks),
            "retained_states": sum(state_counts),
            "maximum_frontier": max(state_counts, default=0),
            "parallel_stage_cpu_limit": self.parallel_stage_cpu_limit,
            "pareto_global_epsilon": self.pareto_global_epsilon,
            "pareto_step_epsilon": self.pareto_step_epsilon,
            "final_local_serial": final_objective.local_serial,
            "final_parallel_bottleneck": (
                final_objective.parallel_bottleneck
            ),
            "final_parallel_total_work": final_objective.parallel_total_work,
        }
        self.optimizer._dp_last_search_stats = search_stats
        logger.info("[DpOptimizer] Exact search stats: %s", search_stats)
        return self._reconstruct(
            back, final_mask, final_state, final_objective
        )

    def _initial_objective(self) -> DpObjectiveCost:
        method = getattr(self.optimizer, "_dp_initial_objective_cost", None)
        if method is None:
            return DpObjectiveCost()
        return method()

    def _accumulate_objective(
        self,
        previous: DpObjectiveCost,
        extra_cost: float,
        block: BlockCandidate,
        replaces_prefix_cost: bool,
    ) -> DpObjectiveCost:
        if replaces_prefix_cost:
            # Cache hits run on the local lane and bypass all upstream service
            # demand. Resource reservations remain in DpStateSummary.
            return DpObjectiveCost(local_serial=extra_cost)
        method = getattr(self.optimizer, "_dp_accumulate_objective_cost", None)
        if method is None:
            return DpObjectiveCost(
                local_serial=previous.local_serial + extra_cost,
                parallel_bottleneck=previous.parallel_bottleneck,
                parallel_total_work=previous.parallel_total_work,
            )
        return method(previous, extra_cost, block)

    @staticmethod
    def _dominates(
        left: DpObjectiveCost, right: DpObjectiveCost
    ) -> bool:
        tolerance = 1e-12
        return (
            left.local_serial <= right.local_serial + tolerance
            and left.parallel_bottleneck
            <= right.parallel_bottleneck + tolerance
            and left.parallel_total_work
            <= right.parallel_total_work + tolerance
        )

    def _frontier_dominates(
        self, left: DpObjectiveCost, right: DpObjectiveCost
    ) -> bool:
        if self.pareto_step_epsilon == 0.0:
            return self._dominates(left, right)
        factor = 1.0 + self.pareto_step_epsilon
        forward = (
            left.local_serial <= factor * right.local_serial
            and left.parallel_bottleneck
            <= factor * right.parallel_bottleneck
            and left.parallel_total_work
            <= factor * right.parallel_total_work
        )
        if not forward:
            return False
        reverse = (
            right.local_serial <= factor * left.local_serial
            and right.parallel_bottleneck
            <= factor * left.parallel_bottleneck
            and right.parallel_total_work
            <= factor * left.parallel_total_work
        )
        if not reverse:
            return True
        return (
            left.score,
            left.total_work,
            left.local_serial,
        ) <= (
            right.score,
            right.total_work,
            right.local_serial,
        )

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
        dp: List[Dict[DpStateSummary, List[DpObjectiveCost]]],
        back: List[
            Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]
        ],
        prev_mask: int,
        next_mask: int,
        block_mask: int,
    ) -> None:
        if not dp[prev_mask]:
            return

        prefix_candidates = getattr(
            self.block_provider, "candidates_for_prefix", None
        )
        blocks = (
            prefix_candidates(prev_mask, block_mask)
            if prefix_candidates is not None
            else self.block_provider.candidates_for(block_mask)
        )
        for block in blocks:
            if not self._block_can_follow(prev_mask, block):
                continue

            regular_cost = self.optimizer._dp_regular_transition_cost(
                prev_mask, block
            )
            for prev_state, prev_objectives in dp[prev_mask].items():
                if self.parallel_stage_cpu_limit is None:
                    # Resource use cannot affect feasibility or any future
                    # transition when strict matching is disabled. Collapse
                    # this otherwise irrelevant state coordinate so plans
                    # with different accumulated stage widths can dominate
                    # one another by score.
                    next_parallel_stage_cpus = 0
                else:
                    next_parallel_stage_cpus = (
                        prev_state.parallel_stage_cpus
                        + self.optimizer._dp_parallel_stage_cpu_cost(
                            block.variant
                        )
                    )
                    if next_parallel_stage_cpus > self.parallel_stage_cpu_limit:
                        continue
                choices = list(
                    self.cache_policy.transitions(
                        prev_mask,
                        next_mask,
                        prev_state,
                        regular_cost,
                        block,
                        next_parallel_stage_cpus,
                    )
                )
                for prev_objective in list(prev_objectives):
                    for choice in choices:
                        candidate = self._accumulate_objective(
                            prev_objective,
                            choice.extra_cost,
                            block,
                            choice.replaces_prefix_cost,
                        )
                        # CPU usage is a monotone feasibility resource: a
                        # label with no greater service demand and fewer
                        # already-reserved stage CPUs can execute every suffix
                        # available to a more expensive label. Apply this
                        # exact dominance across resource-summary states before
                        # maintaining the within-state Pareto frontier.
                        compatible = [
                            (state, values)
                            for state, values in dp[next_mask].items()
                            if state.cache_active
                            == choice.state.cache_active
                        ]
                        if any(
                            state.parallel_stage_cpus
                            <= choice.state.parallel_stage_cpus
                            and any(
                                self._frontier_dominates(old, candidate)
                                for old in values
                            )
                            for state, values in compatible
                        ):
                            continue
                        for state, values in compatible:
                            if (
                                choice.state.parallel_stage_cpus
                                > state.parallel_stage_cpus
                            ):
                                continue
                            dominated_values = [
                                old
                                for old in values
                                if self._frontier_dominates(candidate, old)
                            ]
                            for old in dominated_values:
                                values.remove(old)
                                back[next_mask].pop((state, old), None)
                            if not values:
                                dp[next_mask].pop(state, None)
                        frontier = dp[next_mask].setdefault(
                            choice.state, []
                        )
                        equal = next(
                            (
                                old
                                for old in frontier
                                if math.isclose(
                                    old.local_serial,
                                    candidate.local_serial,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                                and math.isclose(
                                    old.parallel_bottleneck,
                                    candidate.parallel_bottleneck,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                                and math.isclose(
                                    old.parallel_total_work,
                                    candidate.parallel_total_work,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                            ),
                            None,
                        )
                        if equal is not None:
                            old_pointer = back[next_mask].get(
                                (choice.state, equal)
                            )
                            if (
                                block.materializes_fusion
                                and (
                                    old_pointer is None
                                    or block.mask.bit_count()
                                    > old_pointer.block.mask.bit_count()
                                )
                            ):
                                back[next_mask][
                                    (choice.state, equal)
                                ] = BackPointer(
                                    prev_mask=prev_mask,
                                    prev_state=prev_state,
                                    prev_objective=prev_objective,
                                    block=block,
                                    cache_after_idx=choice.cache_after_idx,
                                )
                            continue
                        if any(
                            self._frontier_dominates(old, candidate)
                            for old in frontier
                        ):
                            continue
                        dominated = [
                            old
                            for old in frontier
                            if self._frontier_dominates(candidate, old)
                        ]
                        for old in dominated:
                            frontier.remove(old)
                            back[next_mask].pop(
                                (choice.state, old), None
                            )
                        frontier.append(candidate)
                        back[next_mask][
                            (choice.state, candidate)
                        ] = BackPointer(
                            prev_mask=prev_mask,
                            prev_state=prev_state,
                            prev_objective=prev_objective,
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
        back: List[
            Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]
        ],
        final_mask: int,
        final_state: DpStateSummary,
        final_objective: DpObjectiveCost,
    ) -> SearchResult:
        mask = final_mask
        state = final_state
        objective = final_objective
        blocks_rev: List[List[int]] = []
        variants_by_idx: Dict[int, PipeVariantType] = {}
        cache_after_idx: Optional[int] = None

        while mask:
            pointer = back[mask].get((state, objective))
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
            objective = pointer.prev_objective

        blocks_rev.reverse()
        flat_order: List[int] = []
        for block in blocks_rev:
            flat_order.extend(block)

        return SearchResult(
            order=flat_order,
            blocks=blocks_rev,
            variants_by_idx=variants_by_idx,
            cache_after_idx=cache_after_idx,
            cost=final_objective.score,
            objective=final_objective,
        )


class DpOptimizer(MyOptimizer):
    """
    Extensible DP optimizer.

    This class keeps MyOptimizer's current semantics but moves strategy-specific
    logic behind candidate providers and transition policies. Adding a new
    optimization should normally mean adding a provider/policy, not editing the
    subset-DP engine itself.
    """

    def _dp_regular_transition_cost(
        self, prev_mask: int, block: BlockCandidate
    ) -> float:
        """Return the one authoritative non-cache DP transition cost."""
        return block.cost + self._dp_stage_boundary_cost(prev_mask, block)

    def _dp_initial_objective_cost(self) -> DpObjectiveCost:
        """The source runs on the same serial lane as local operators."""
        source_p_id = self._get_source_p_id()
        return DpObjectiveCost(
            local_serial=float(self._base_cost_map[source_p_id])
        )

    def _dp_stage_balance_penalty(self, stage_cost: float) -> float:
        """Return an infinitesimal convex tie-break for parallel stages.

        The throughput objective remains the maximum stage demand. When that
        maximum is identical, a squared-load term prefers two stages with
        headroom over one nearly saturated fused stage. Keeping this term in
        the existing local coordinate avoids adding a Pareto dimension and
        preserves subset-DP scalability.
        """
        raw_weight = os.environ.get("CEDAR_DP_STAGE_BALANCE_WEIGHT", "1e-3")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise RuntimeError(
                "CEDAR_DP_STAGE_BALANCE_WEIGHT must be numeric"
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise RuntimeError(
                "CEDAR_DP_STAGE_BALANCE_WEIGHT must be finite and non-negative"
            )
        scale = max(sum(self._base_cost_map.values()), 1e-12)
        return weight * stage_cost * stage_cost / scale

    def _dp_accumulate_objective_cost(
        self,
        previous: DpObjectiveCost,
        extra_cost: float,
        block: BlockCandidate,
    ) -> DpObjectiveCost:
        if block.variant in (
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
            PipeVariantType.SMP,
        ):
            # The backend compute portion can overlap across independent
            # stages, but each local worker serially submits/serializes every
            # stage boundary. Charging the complete transition to only the
            # maximum stage made extra Ray/SMP stages effectively free and
            # selected queue-heavy plans that stall at runtime.
            boundary_cost = max(0.0, extra_cost - block.cost)
            return DpObjectiveCost(
                local_serial=(
                    previous.local_serial
                    + boundary_cost
                    + self._dp_stage_balance_penalty(block.cost)
                ),
                parallel_bottleneck=max(
                    previous.parallel_bottleneck, block.cost
                ),
                parallel_total_work=(
                    previous.parallel_total_work + block.cost
                ),
            )
        return DpObjectiveCost(
            local_serial=previous.local_serial + extra_cost,
            parallel_bottleneck=previous.parallel_bottleneck,
            parallel_total_work=previous.parallel_total_work,
        )

    def _dp_blocks_from_physical_plan(
        self,
        plan: PhysicalPlan,
        inner_ops: List[int],
    ) -> List[Tuple[Tuple[int, ...], PipeVariantType, bool]]:
        """Recover DP blocks and cache placement from a linear physical plan."""
        if not plan.validate():
            raise ValueError("Cannot score an invalid physical plan.")
        if not inner_ops:
            return []

        predecessors = {child for children in plan.graph.values() for child in children}
        sources = [p_id for p_id in plan.graph if p_id not in predecessors]
        if len(sources) != 1:
            raise ValueError("DP objective scoring requires one linear source.")
        path: List[int] = []
        current = sources[0]
        visited = set()
        while True:
            if current in visited:
                raise ValueError("Physical plan contains a cycle.")
            visited.add(current)
            path.append(current)
            successors = plan.graph.get(current, set())
            if not successors:
                break
            if len(successors) != 1:
                raise ValueError("DP objective scoring requires a linear plan.")
            current = next(iter(successors))
        if len(visited) != len(plan.graph):
            raise ValueError("Physical plan contains nodes outside its main path.")

        index_by_pipe = {p_id: idx for idx, p_id in enumerate(inner_ops)}
        blocks: List[List[Any]] = []
        source_id = path[0]
        for p_id in path:
            if p_id == source_id:
                continue
            desc = plan.pipe_descs[p_id]
            if desc.name == "ObjectDiskCachePipe":
                if not blocks or blocks[-1][2]:
                    raise ValueError("Cache must follow exactly one DP block.")
                blocks[-1][2] = True
                continue
            if p_id in index_by_pipe:
                order = (index_by_pipe[p_id],)
            elif desc.fused_pipes and len(desc.fused_pipes) > 1:
                try:
                    order = tuple(index_by_pipe[item] for item in desc.fused_pipes)
                except KeyError as exc:
                    raise ValueError(
                        "Fused plan references an operator outside DP metadata."
                    ) from exc
            else:
                # Prefetch and other zero-cost optimizer nodes are inserted
                # after DP search and are intentionally outside its objective.
                if self._is_optimizer_pipe(p_id, plan):
                    continue
                raise ValueError(f"Plan pipe {p_id} is outside DP metadata.")
            variant = desc.variant_type or PipeVariantType.INPROCESS
            blocks.append([order, variant, False])

        flattened = [idx for order, _, _ in blocks for idx in order]
        if len(flattened) != len(inner_ops) or set(flattened) != set(
            range(len(inner_ops))
        ):
            raise ValueError("Physical plan does not cover every DP operator once.")
        return [
            (tuple(order), variant, bool(cache_after))
            for order, variant, cache_after in blocks
        ]

    def _replay_dp_objective(
        self,
        block_specs: Iterable[
            Tuple[Tuple[int, ...], PipeVariantType, bool]
        ],
        inner_ops: List[int],
    ) -> DpObjectiveCost:
        provider = BlockCandidateProvider(self, inner_ops)
        provider.prepare()
        cache_policy = CacheTransitionPolicy(self, inner_ops)
        search = ExtensibleDpSearch(
            optimizer=self,
            inner_ops=inner_ops,
            block_provider=provider,
            cache_policy=cache_policy,
        )
        state = cache_policy.initial_state()
        objective = search._initial_objective()
        prev_mask = 0
        for order, variant, wants_cache in block_specs:
            block = provider.candidate_for_order(
                order, variant, prefix_mask=prev_mask
            )
            if block.mask & prev_mask:
                raise ValueError("DP replay contains a duplicate operator.")
            next_mask = prev_mask | block.mask
            if not search._block_can_follow(prev_mask, block):
                raise ValueError("Materialized block is infeasible in DP search.")

            if search.parallel_stage_cpu_limit is None:
                next_parallel_stage_cpus = 0
            else:
                next_parallel_stage_cpus = (
                    state.parallel_stage_cpus
                    + self._dp_parallel_stage_cpu_cost(block.variant)
                )
                if next_parallel_stage_cpus > search.parallel_stage_cpu_limit:
                    raise ValueError("Materialized plan exceeds the DP CPU budget.")

            regular_cost = self._dp_regular_transition_cost(prev_mask, block)
            choices = list(
                cache_policy.transitions(
                    prev_mask,
                    next_mask,
                    state,
                    regular_cost,
                    block,
                    next_parallel_stage_cpus,
                )
            )
            matching = [
                choice
                for choice in choices
                if (choice.cache_after_idx is not None) == wants_cache
            ]
            if len(matching) != 1:
                raise ValueError("Materialized cache placement is infeasible in DP search.")
            choice = matching[0]
            objective = search._accumulate_objective(
                objective,
                choice.extra_cost,
                block,
                choice.replaces_prefix_cost,
            )
            state = choice.state
            prev_mask = next_mask

        if prev_mask != (1 << len(inner_ops)) - 1:
            raise ValueError("DP replay did not reach the complete operator set.")
        return objective

    def calculate_dp_objective_cost(
        self,
        plan: Optional[PhysicalPlan] = None,
        search_result: Optional[SearchResult] = None,
        inner_ops: Optional[List[int]] = None,
    ) -> float:
        """Score a plan with exactly the objective used by DP transitions.

        Exactly one of ``plan`` and ``search_result`` must be supplied.  The
        optimizer must already have loaded its profile and prepared DP
        metadata.  Unsupported or infeasible materialized plans raise instead
        of receiving a misleading cost from a different search space.
        """
        if (plan is None) == (search_result is None):
            raise ValueError("Supply exactly one of plan or search_result.")
        ops = list(inner_ops if inner_ops is not None else self._dp_inner_ops)
        if not ops or self._dp_inner_ops != ops:
            raise RuntimeError("DP metadata is not prepared for these operators.")
        if search_result is not None:
            block_specs = []
            for block in search_result.blocks:
                variant = search_result.variants_by_idx.get(
                    block[0], PipeVariantType.INPROCESS
                )
                wants_cache = (
                    search_result.cache_after_idx is not None
                    and block[-1] == search_result.cache_after_idx
                )
                block_specs.append((tuple(block), variant, wants_cache))
        else:
            block_specs = self._dp_blocks_from_physical_plan(plan, ops)
        return self._replay_dp_objective(block_specs, ops).score

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
            optimized_cost = self.calculate_dp_objective_cost(
                plan=self.physical_plan
            )
            search_cost = getattr(self, "_last_dp_state_cost", None)
            if search_cost is not None and not math.isclose(
                optimized_cost, search_cost, rel_tol=1e-10, abs_tol=1e-10
            ):
                raise RuntimeError(
                    "Materialized plan objective diverged from DP search: "
                    f"search={search_cost}, plan={optimized_cost}"
                )
            self._last_dp_state_cost = optimized_cost
            logger.info(
                "[DpOptimizer] Optimized DP objective cost = %s",
                optimized_cost,
            )
        except Exception as e:
            logger.error(
                "[DpOptimizer] Failed to replay optimized DP objective: %s", e
            )
            raise

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
        per_worker_budget = cpu_budget // fixed_workers
        reserve_raw = os.environ.get(
            "CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER", "1"
        )
        try:
            runtime_reserve = int(reserve_raw)
        except ValueError as exc:
            raise RuntimeError(
                "CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER must be an integer"
            ) from exc
        if runtime_reserve < 0:
            raise RuntimeError(
                "CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER must be non-negative"
            )
        # One slot is occupied by the local data worker. Keep one additional
        # slot by default for the Ray driver, queue threads and multiprocessing
        # coordination; plans that reserve every CPU have repeatedly stalled
        # before yielding their first record despite nominal feasibility.
        return max(0, per_worker_budget - 1 - runtime_reserve)

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
        replayed_objective = self._replay_dp_objective(
            [
                (
                    tuple(block),
                    result.variants_by_idx.get(
                        block[0], PipeVariantType.INPROCESS
                    ),
                    result.cache_after_idx is not None
                    and block[-1] == result.cache_after_idx,
                )
                for block in result.blocks
            ],
            inner_ops,
        )
        replayed_cost = replayed_objective.score
        if not math.isclose(
            replayed_cost, result.cost, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise RuntimeError(
                "DP objective replay diverged from search: "
                f"search={result.cost}, replay={replayed_cost}"
            )
        self._last_dp_state_cost = replayed_cost
        self._last_dp_search_result = result

        self._store_pending_fusions_from_blocks(
            blocks_in_idx_order=result.blocks,
            inner_ops=inner_ops,
            chosen_variant_by_idx=result.variants_by_idx,
        )

        for idx, p_id in enumerate(inner_ops):
            vt = result.variants_by_idx.get(idx, PipeVariantType.INPROCESS)
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = vt

        logger.info("[DpOptimizer] DP objective cost (inner ops only): %s", replayed_cost)
        logger.info(
            "[DpOptimizer] DP objective: throughput_bottleneck "
            "local_serial=%s parallel_bottleneck=%s parallel_total_work=%s",
            replayed_objective.local_serial,
            replayed_objective.parallel_bottleneck,
            replayed_objective.parallel_total_work,
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
    "DpObjectiveCost",
    "BlockCandidate",
    "BlockCandidateProvider",
    "CacheTransitionPolicy",
    "ExtensibleDpSearch",
]
