import logging
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from cedar.pipes import Pipe, PipeExecutionResource

from .my_optimizer import MyOptimizer
from . import constants
from .optimizer import OptimizerOptions, PhysicalPlan, PipeDesc, PipeVariantType


logger = logging.getLogger(__name__)


_FORK_SEARCH_OPTIMIZER = None
_FORK_SEARCH_INNER_OPS = None


def _initialize_conditioned_search_worker(optimizer, inner_ops) -> None:
    global _FORK_SEARCH_OPTIMIZER, _FORK_SEARCH_INNER_OPS
    _FORK_SEARCH_OPTIMIZER = optimizer
    _FORK_SEARCH_INNER_OPS = inner_ops


def _run_conditioned_search_worker(assumed_total: int):
    if _FORK_SEARCH_OPTIMIZER is None or _FORK_SEARCH_INNER_OPS is None:
        raise RuntimeError("Conditioned DP worker was not initialized.")
    return _FORK_SEARCH_OPTIMIZER._run_conditioned_dp_search(
        _FORK_SEARCH_INNER_OPS, assumed_total
    )


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
    execution_resource: PipeExecutionResource = PipeExecutionResource.CPU
    parallelism: int = 1


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
        self.successor_masks = [0] * self.n
        for successor, predecessors in enumerate(
            optimizer._dp_pred_indices
        ):
            for predecessor in predecessors:
                self.successor_masks[predecessor] |= 1 << successor
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
            if prev & self.successor_masks[i]:
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

    In-process CPU operators share a serial worker lane, while distinct CPU
    Ray/SMP blocks form concurrently active pipeline stages. CUDA blocks use a
    separate serial coordinate because all of them share the same physical GPU
    in the formal single-GPU setting. Their service demands therefore add even
    when Cedar materializes them as different Ray stages.
    """

    local_serial: float = 0.0
    parallel_bottleneck: float = 0.0
    gpu_serial: float = 0.0

    @property
    def score(self) -> float:
        return max(
            self.local_serial,
            self.parallel_bottleneck,
            self.gpu_serial,
        )

@dataclass
class SearchResult:
    order: List[int]
    blocks: List[List[int]]
    variants_by_idx: Dict[int, PipeVariantType]
    parallelism_by_idx: Dict[int, int]
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
        self._variant_indexes: Dict[
            Tuple[PipeVariantType, int], _BlockCostIndex
        ] = {}
        self._fused_variant_indexes: Dict[
            Tuple[PipeVariantType, int], _BlockCostIndex
        ] = {}
        self._candidate_variants: List[PipeVariantType] = []
        self._fusion_allowed_flags: List[bool] = []
        self._fusion_mask_allowed: Dict[int, bool] = {}
        self._fusion_variant_feasible: Dict[
            Tuple[int, PipeVariantType], bool
        ] = {}
        self._execution_resources: Dict[int, PipeExecutionResource] = {}

    def prepare(self) -> None:
        opt = self.optimizer
        if opt.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")
        if opt._dp_costs is None:
            raise RuntimeError("DP metadata not prepared.")

        # Re-evaluate INPROCESS through the same cost hook as offloaded
        # variants.  The raw _dp_costs array contains the width-one baseline;
        # layered profiles may provide an SMP worker-side width-W measurement
        # that captures contention among the formal local worker processes.
        inprocess_costs: List[float] = []
        for p_id in self.inner_ops:
            base_input_size = opt.profiled_stats["baseline"][
                "input_sizes"
            ][p_id]
            inprocess_costs.append(
                opt._calculate_pipe_cost(p_id, base_input_size, None)
            )
        variant_compute_costs: Dict[PipeVariantType, List[float]] = {
            PipeVariantType.INPROCESS: inprocess_costs
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
                    # _calculate_pipe_cost may include the formal-width
                    # contention estimate.  Candidate construction starts at
                    # one actor, so anchor it to the measured width-1 curve;
                    # wider candidates are replaced below in the same way.
                    total_cost = opt._dp_pipe_cost_at_parallelism(
                        p_id, vt, 1, total_cost
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
            max_parallelism = opt._dp_max_candidate_parallelism(vt)
            for parallelism in range(1, max_parallelism + 1):
                costs = variant_compute_costs[vt]
                if parallelism > 1:
                    costs = [
                        opt._dp_pipe_cost_at_parallelism(
                            p_id,
                            vt,
                            parallelism,
                            cost,
                        )
                        for p_id, cost in zip(self.inner_ops, costs)
                    ]
                key = (vt, parallelism)
                self._variant_indexes[key] = _BlockCostIndex(
                    opt, self.inner_ops, costs
                )
                if vt != PipeVariantType.INPROCESS:
                    # Fusion removes intermediate boundaries, while compute
                    # remains the sum of its members at the selected width.
                    self._fused_variant_indexes[key] = _BlockCostIndex(
                        opt, self.inner_ops, costs
                    )

    def candidates_for(self, mask: int) -> Iterable[BlockCandidate]:
        return self.candidates_for_prefix(0, mask)

    def candidates_for_prefix(
        self, prefix_mask: int, mask: int
    ) -> Iterable[BlockCandidate]:
        if mask <= 0 or mask >= self.full_mask:
            return []
        cache_key = (prefix_mask, mask)
        cached = self._candidates_by_prefix_and_mask.get(cache_key)
        if cached is not None:
            return cached

        opt = self.optimizer
        is_multi = mask.bit_count() > 1
        fusion_allowed = self._fusion_mask_allowed.get(mask)
        if fusion_allowed is None:
            fusion_allowed = all(
                self._fusion_allowed_flags[i]
                for i in range(self.n)
                if mask & (1 << i)
            )
            self._fusion_mask_allowed[mask] = fusion_allowed
        if is_multi and (
            not opt.options.enable_fusion
            or not fusion_allowed
        ):
            self._candidates_by_prefix_and_mask[cache_key] = []
            return []
        candidates: List[BlockCandidate] = []
        execution_resource = self._execution_resource_for_mask(mask)
        for vt in self._candidate_variants:
            if (
                execution_resource == PipeExecutionResource.CUDA
                and vt
                not in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
            ):
                # CUDA work must be isolated in a Ray actor so the physical
                # plan can declare and account its share of the single GPU.
                continue
            if is_multi and not opt._dp_has_supported_fusion_cost(vt):
                continue
            if is_multi:
                feasibility_key = (mask, vt)
                feasible = self._fusion_variant_feasible.get(
                    feasibility_key
                )
                if feasible is None:
                    feasible = all(
                        opt._pipe_can_materialize_fusion(
                            self.inner_ops[i], vt
                        )
                        for i in range(self.n)
                        if mask & (1 << i)
                    )
                    self._fusion_variant_feasible[
                        feasibility_key
                    ] = feasible
                if not feasible:
                    continue

            parallelisms = opt._dp_candidate_parallelisms(
                vt, execution_resource
            )
            for parallelism in parallelisms:
                assumed_total = getattr(
                    opt, "_dp_assumed_total_parallel_stage_cpus", None
                )
                if (
                    assumed_total is not None
                    and vt
                    in (
                        PipeVariantType.RAY,
                        PipeVariantType.TF_RAY,
                        PipeVariantType.SMP,
                    )
                    and parallelism > assumed_total
                ):
                    # This resource-conditioned search must finish at exactly
                    # ``assumed_total`` remote slots. A single block wider
                    # than that total can never participate in a feasible
                    # suffix, so do not even evaluate its subset-order cost.
                    continue
                variant_key = (vt, parallelism)
                index = self._variant_indexes[variant_key]
                if is_multi and variant_key in self._fused_variant_indexes:
                    index = self._fused_variant_indexes[variant_key]
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
                        execution_resource=execution_resource,
                        parallelism=parallelism,
                    )
                )

        # Placement-dependent feasibility remains in the outer DP. Keep every
        # backend alternative for this mask.
        self._candidates_by_prefix_and_mask[cache_key] = candidates
        return candidates

    def _execution_resource_for_mask(
        self, mask: int
    ) -> PipeExecutionResource:
        cached = self._execution_resources.get(mask)
        if cached is not None:
            return cached
        logical_pipes = self.optimizer.logical_pipes or {}
        if any(
            logical_pipes[self.inner_ops[idx]].execution_resource
            == PipeExecutionResource.CUDA
            for idx in range(self.n)
            if mask & (1 << idx)
        ):
            resource = PipeExecutionResource.CUDA
        else:
            resource = PipeExecutionResource.CPU
        self._execution_resources[mask] = resource
        return resource

    def candidate_for_order(
        self,
        order: Iterable[int],
        variant: PipeVariantType,
        prefix_mask: int = 0,
        parallelism: int = 1,
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
        key = (variant, parallelism)
        if key not in self._variant_indexes:
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
        index = self._variant_indexes[key]
        if is_multi and key in self._fused_variant_indexes:
            index = self._fused_variant_indexes[key]
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
            execution_resource=self._execution_resource_for_mask(mask),
            parallelism=parallelism,
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
        parallel_stage_cpu_limit: Optional[int] = None,
        required_final_parallel_stage_cpus: Optional[int] = None,
    ) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self.block_provider = block_provider
        self.cache_policy = cache_policy
        configured_limit = optimizer._dp_parallel_stage_cpu_limit()
        self.parallel_stage_cpu_limit = (
            configured_limit
            if parallel_stage_cpu_limit is None
            else parallel_stage_cpu_limit
        )
        self.required_final_parallel_stage_cpus = (
            required_final_parallel_stage_cpus
        )
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
        self._upper_bound_pruned = 0
        self._transition_pairs = 0
        self._frontier_cells: Dict[
            Tuple[int, DpStateSummary],
            Dict[Tuple[Optional[int], Optional[int], Optional[int]], DpObjectiveCost],
        ] = {}
        self._cell_candidates = 0
        self._cell_replacements = 0
        self.use_cell_frontier = (
            os.environ.get("CEDAR_DP_CELL_FRONTIER", "0") == "1"
        )
        frontier_cap_raw = os.environ.get("CEDAR_DP_FRONTIER_CAP", "8")
        try:
            configured_frontier_cap = int(frontier_cap_raw)
        except ValueError as exc:
            raise ValueError("CEDAR_DP_FRONTIER_CAP must be an integer") from exc
        if configured_frontier_cap < 0:
            raise ValueError("CEDAR_DP_FRONTIER_CAP must be non-negative")
        if self.n > 8:
            assumed_total = getattr(
                optimizer, "_dp_assumed_total_parallel_stage_cpus", None
            )
            if assumed_total is not None and configured_frontier_cap > 0:
                configured_frontier_cap = max(
                    1,
                    configured_frontier_cap
                    // (2 ** max(0, int(assumed_total) - 2)),
                )
            self.frontier_cap = configured_frontier_cap
        else:
            self.frontier_cap = 0
        self._frontier_cap_pruned = 0

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

        legal_masks, legal_predecessors = self._search_topology()
        legal_mask_set = set(legal_masks)
        incumbent_score = self._initial_incumbent_score()
        for next_mask in legal_masks[1:]:
            # Enumerate only actual subsets of next_mask. Scanning every pair
            # of dependency-closed masks and rejecting non-subsets approaches
            # 4**n on weakly constrained filter chains; subset enumeration is
            # 3**n over the complete search while visiting the exact same
            # (previous prefix, appended block) transitions.
            if legal_predecessors is not None:
                prev_masks = legal_predecessors[next_mask]
            else:
                prev_masks = []
                prev_mask = (next_mask - 1) & next_mask
                while True:
                    if prev_mask in legal_mask_set:
                        prev_masks.append(prev_mask)
                    if prev_mask == 0:
                        break
                    prev_mask = (prev_mask - 1) & next_mask
                prev_masks.reverse()
            for prev_mask in prev_masks:
                if not dp[prev_mask]:
                    continue
                block_mask = next_mask ^ prev_mask
                if block_mask:
                    self._transition_pairs += 1
                    self._try_extend(
                        dp,
                        back,
                        prev_mask,
                        next_mask,
                        block_mask,
                        incumbent_score,
                    )
            if self.pareto_step_epsilon > 0.0 and self.use_cell_frontier:
                self._finalize_cell_frontiers(dp, back, next_mask)

        final_mask = self.full_mask - 1
        if not dp[final_mask]:
            raise RuntimeError("Extensible DP failed: no feasible final state.")

        final_candidates = [
                (state, objective)
                for state, objectives in dp[final_mask].items()
                for objective in objectives
                if self.required_final_parallel_stage_cpus is None
                or state.parallel_stage_cpus
                == self.required_final_parallel_stage_cpus
        ]
        if not final_candidates:
            raise RuntimeError(
                "Extensible DP found no plan at the required parallel CPU "
                f"total {self.required_final_parallel_stage_cpus}."
            )
        final_state, final_objective = min(
            final_candidates,
            key=lambda item: (
                item[1].score,
                item[1].local_serial,
                item[1].parallel_bottleneck,
                item[1].gpu_serial,
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
            "required_final_parallel_stage_cpus": (
                self.required_final_parallel_stage_cpus
            ),
            "pareto_global_epsilon": self.pareto_global_epsilon,
            "pareto_step_epsilon": self.pareto_step_epsilon,
            "transition_pairs": self._transition_pairs,
            "upper_bound_score": (
                incumbent_score if math.isfinite(incumbent_score) else None
            ),
            "upper_bound_pruned": self._upper_bound_pruned,
            "cell_candidates": self._cell_candidates,
            "cell_replacements": self._cell_replacements,
            "frontier_cap": self.frontier_cap,
            "frontier_cap_pruned": self._frontier_cap_pruned,
            "final_local_serial": final_objective.local_serial,
            "final_parallel_bottleneck": (
                final_objective.parallel_bottleneck
            ),
            "final_gpu_serial": final_objective.gpu_serial,
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
        prev_mask: int,
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
                gpu_serial=previous.gpu_serial,
            )
        return method(previous, extra_cost, block, prev_mask)

    @staticmethod
    def _dominates(
        left: DpObjectiveCost, right: DpObjectiveCost
    ) -> bool:
        tolerance = 1e-12
        return (
            left.local_serial <= right.local_serial + tolerance
            and left.parallel_bottleneck
            <= right.parallel_bottleneck + tolerance
            and left.gpu_serial <= right.gpu_serial + tolerance
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
            and left.gpu_serial <= factor * right.gpu_serial
        )
        if not forward:
            return False
        reverse = (
            right.local_serial <= factor * left.local_serial
            and right.parallel_bottleneck
            <= factor * left.parallel_bottleneck
            and right.gpu_serial <= factor * left.gpu_serial
        )
        if not reverse:
            return True
        return (
            left.score,
            left.local_serial,
            left.parallel_bottleneck,
            left.gpu_serial,
        ) <= (
            right.score,
            right.local_serial,
            right.parallel_bottleneck,
            right.gpu_serial,
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

    def _search_topology(
        self,
    ) -> Tuple[List[int], Optional[Dict[int, Tuple[int, ...]]]]:
        """Build resource-independent legal transitions once per optimizer.

        Resource-conditioned searches differ only in their measured block
        costs.  Their dependency-closed masks and subset relationships are
        identical, so rebuilding and rescanning them for every resource total
        wastes most of the structural work on long pipelines.
        """
        key = (
            self.n,
            tuple(tuple(sorted(preds)) for preds in self.optimizer._dp_pred_indices),
        )
        cached = getattr(self.optimizer, "_dp_search_topology_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        legal = self._legal_prefix_masks()
        predecessors: Optional[Dict[int, Tuple[int, ...]]] = None
        pair_limit = int(os.environ.get("CEDAR_DP_TOPOLOGY_PAIR_LIMIT", "10000"))
        if len(legal) <= pair_limit:
            predecessors = {}
            for position, next_mask in enumerate(legal[1:], start=1):
                predecessors[next_mask] = tuple(
                    prev_mask
                    for prev_mask in legal[:position]
                    if prev_mask & ~next_mask == 0
                )
        self.optimizer._dp_search_topology_cache = (
            key,
            legal,
            predecessors,
        )
        return legal, predecessors

    def _initial_incumbent_score(self) -> float:
        """Return a cheap feasible upper bound for monotone non-cache search.

        A single fused remote block at the requested width is a valid member
        of the full search space whenever the pipeline can materialize it.
        Its objective is often already competitive and safely cuts any
        partial label whose monotone coordinates have exceeded that score.
        Cache-enabled search is excluded because a later cache transition can
        intentionally replace the accumulated prefix cost.
        """
        self._greedy_full_block_result = None
        cache_enabled = getattr(self.cache_policy, "enabled", None)
        if cache_enabled is None or cache_enabled:
            return float("inf")
        candidate_for_order = getattr(
            self.block_provider, "candidate_for_order", None
        )
        candidate_variants = getattr(
            self.block_provider, "_candidate_variants", None
        )
        if candidate_for_order is None or not candidate_variants:
            return float("inf")

        full_mask = self.full_mask - 1
        initial_state = self.cache_policy.initial_state()
        initial_objective = self._initial_objective()
        best = float("inf")
        blocks = []
        for variant in candidate_variants:
            is_remote = variant in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
                PipeVariantType.SMP,
            )
            if self.required_final_parallel_stage_cpus is not None:
                if is_remote:
                    parallelisms = (
                        self.required_final_parallel_stage_cpus,
                    )
                elif self.required_final_parallel_stage_cpus == 0:
                    parallelisms = (1,)
                else:
                    continue
            else:
                parallelisms = self.optimizer._dp_candidate_parallelisms(
                    variant,
                    PipeExecutionResource.CPU,
                )
            for parallelism in parallelisms:
                if parallelism < 1:
                    continue
                variant_index = getattr(
                    self.block_provider, "_variant_indexes", {}
                ).get((variant, parallelism))
                if variant_index is None:
                    continue
                prefix_mask = 0
                greedy_order = []
                while len(greedy_order) < self.n:
                    available = [
                        idx
                        for idx in range(self.n)
                        if not prefix_mask & (1 << idx)
                        and self.optimizer._dp_valid_single_last(
                            prefix_mask, idx
                        )
                        and math.isfinite(variant_index.per_byte[idx])
                    ]
                    if not available:
                        greedy_order = []
                        break
                    selected = min(
                        available,
                        key=lambda idx: (
                            _work_prod(self.optimizer, prefix_mask, idx)
                            * variant_index.per_byte[idx],
                            idx,
                        ),
                    )
                    greedy_order.append(selected)
                    prefix_mask |= 1 << selected
                if not greedy_order:
                    continue
                try:
                    blocks.append(
                        candidate_for_order(
                            tuple(greedy_order),
                            variant,
                            prefix_mask=0,
                            parallelism=parallelism,
                        )
                    )
                except ValueError:
                    continue

        for block in blocks:
            if not self._block_can_follow(0, block):
                continue
            resource_cost = self.optimizer._dp_parallel_stage_cpu_cost(block)
            if (
                self.required_final_parallel_stage_cpus is not None
                and resource_cost != self.required_final_parallel_stage_cpus
            ):
                continue
            regular_cost = self.optimizer._dp_regular_transition_cost(0, block)
            for choice in self.cache_policy.transitions(
                0,
                full_mask,
                initial_state,
                regular_cost,
                block,
                resource_cost,
            ):
                objective = self._accumulate_objective(
                    initial_objective,
                    choice.extra_cost,
                    block,
                    choice.replaces_prefix_cost,
                    0,
                )
                if objective.score < best:
                    best = objective.score
                    self._greedy_full_block_result = SearchResult(
                        order=list(block.order),
                        blocks=[list(block.order)],
                        variants_by_idx={block.order[0]: block.variant},
                        parallelism_by_idx={
                            block.order[0]: block.parallelism
                        },
                        cache_after_idx=None,
                        cost=objective.score,
                        objective=objective,
                    )
        return best

    def _try_extend(
        self,
        dp: List[Dict[DpStateSummary, List[DpObjectiveCost]]],
        back: List[
            Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]
        ],
        prev_mask: int,
        next_mask: int,
        block_mask: int,
        incumbent_score: float = float("inf"),
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
        blocks = self._prune_dominated_blocks(prev_mask, blocks)
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
                            block
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
                            prev_mask,
                        )
                        if candidate.score > incumbent_score + 1e-12:
                            self._upper_bound_pruned += 1
                            continue
                        if self.required_final_parallel_stage_cpus is not None:
                            if (
                                self.pareto_step_epsilon > 0.0
                                and self.use_cell_frontier
                            ):
                                self._insert_cell_candidate(
                                    back,
                                    next_mask,
                                    choice.state,
                                    candidate,
                                    BackPointer(
                                        prev_mask=prev_mask,
                                        prev_state=prev_state,
                                        prev_objective=prev_objective,
                                        block=block,
                                        cache_after_idx=choice.cache_after_idx,
                                    ),
                                )
                                continue
                            frontier = dp[next_mask].get(
                                choice.state, []
                            )
                            equal = None
                            dominated_values = []
                            rejected = False
                            for old in frontier:
                                if self._frontier_dominates(old, candidate):
                                    rejected = True
                                    break
                                if equal is None and (
                                    math.isclose(
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
                                        old.gpu_serial,
                                        candidate.gpu_serial,
                                        rel_tol=1e-12,
                                        abs_tol=1e-12,
                                    )
                                ):
                                    equal = old
                                if self._frontier_dominates(candidate, old):
                                    dominated_values.append(old)
                            if rejected:
                                continue
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
                            for old in dominated_values:
                                frontier.remove(old)
                                back[next_mask].pop(
                                    (choice.state, old), None
                                )
                            if choice.state not in dp[next_mask]:
                                dp[next_mask][choice.state] = frontier
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
                            self._apply_frontier_cap(
                                frontier,
                                back[next_mask],
                                choice.state,
                            )
                            continue
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
                            and (
                                self.required_final_parallel_stage_cpus
                                is None
                                or state.parallel_stage_cpus
                                == choice.state.parallel_stage_cpus
                            )
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
                                    old.gpu_serial,
                                    candidate.gpu_serial,
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

    def _apply_frontier_cap(
        self,
        frontier: List[DpObjectiveCost],
        back: Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer],
        state: DpStateSummary,
    ) -> None:
        if self.frontier_cap == 0 or len(frontier) <= self.frontier_cap:
            return
        if self.frontier_cap == 1:
            retained = [
                min(frontier, key=self._objective_order_key)
            ]
            retained_id = id(retained[0])
            for objective in frontier:
                if id(objective) != retained_id:
                    back.pop((state, objective), None)
                    self._frontier_cap_pruned += 1
            frontier[:] = retained
            return
        ordered = sorted(
            frontier,
            key=lambda objective: (
                objective.parallel_bottleneck,
                objective.local_serial,
                objective.gpu_serial,
            ),
        )
        selected = {0, len(ordered) - 1}
        best_index = min(
            range(len(ordered)),
            key=lambda index: self._objective_order_key(ordered[index]),
        )
        selected.add(best_index)
        if self.frontier_cap > 1:
            for slot in range(self.frontier_cap):
                index = round(
                    slot * (len(ordered) - 1) / (self.frontier_cap - 1)
                )
                selected.add(index)
        if len(selected) < self.frontier_cap:
            for index in sorted(
                range(len(ordered)),
                key=lambda item: self._objective_order_key(ordered[item]),
            ):
                selected.add(index)
                if len(selected) == self.frontier_cap:
                    break
        retained = [
            objective
            for index, objective in enumerate(ordered)
            if index in selected
        ]
        retained_ids = {id(objective) for objective in retained}
        for objective in frontier:
            if id(objective) not in retained_ids:
                back.pop((state, objective), None)
                self._frontier_cap_pruned += 1
        frontier[:] = retained

    def _objective_cell(
        self, objective: DpObjectiveCost
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        log_base = math.log1p(self.pareto_step_epsilon)

        def coordinate(value: float) -> Optional[int]:
            if value <= 1e-12:
                return None
            return math.floor(math.log(value) / log_base)

        return (
            (
                None
                if objective.gpu_serial <= 1e-12
                else coordinate(objective.local_serial)
            ),
            coordinate(objective.parallel_bottleneck),
            coordinate(objective.gpu_serial),
        )

    @staticmethod
    def _objective_order_key(
        objective: DpObjectiveCost,
    ) -> Tuple[float, float, float, float]:
        return (
            objective.score,
            objective.local_serial,
            objective.parallel_bottleneck,
            objective.gpu_serial,
        )

    def _insert_cell_candidate(
        self,
        back: List[Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]],
        next_mask: int,
        state: DpStateSummary,
        candidate: DpObjectiveCost,
        pointer: BackPointer,
    ) -> None:
        self._cell_candidates += 1
        cell = self._objective_cell(candidate)
        cells = self._frontier_cells.setdefault((next_mask, state), {})
        old = cells.get(cell)
        if candidate.gpu_serial <= 1e-12:
            candidate_key = (
                candidate.local_serial,
                candidate.parallel_bottleneck,
                candidate.score,
            )
            old_key = (
                old.local_serial,
                old.parallel_bottleneck,
                old.score,
            ) if old is not None else None
        else:
            candidate_key = self._objective_order_key(candidate)
            old_key = self._objective_order_key(old) if old is not None else None
        if old_key is not None and old_key <= candidate_key:
            return
        if old is not None:
            back[next_mask].pop((state, old), None)
            self._cell_replacements += 1
        cells[cell] = candidate
        back[next_mask][(state, candidate)] = pointer

    def _finalize_cell_frontiers(
        self,
        dp: List[Dict[DpStateSummary, List[DpObjectiveCost]]],
        back: List[Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]],
        next_mask: int,
    ) -> None:
        keys = [
            key for key in self._frontier_cells if key[0] == next_mask
        ]
        for key in keys:
            _, state = key
            candidates = list(self._frontier_cells.pop(key).values())
            if not candidates:
                continue
            if all(candidate.gpu_serial <= 1e-12 for candidate in candidates):
                ordered = sorted(
                    candidates,
                    key=lambda objective: (
                        objective.parallel_bottleneck,
                        objective.local_serial,
                        objective.score,
                    ),
                )
                retained = []
                best_local = float("inf")
                factor = 1.0 + self.pareto_step_epsilon
                for candidate in ordered:
                    if (
                        best_local
                        <= factor * candidate.local_serial + 1e-12
                    ):
                        back[next_mask].pop((state, candidate), None)
                        continue
                    retained.append(candidate)
                    best_local = min(best_local, candidate.local_serial)
            else:
                retained = []
                for candidate in candidates:
                    if any(
                        self._dominates(other, candidate)
                        for other in candidates
                        if other is not candidate
                    ):
                        back[next_mask].pop((state, candidate), None)
                    else:
                        retained.append(candidate)
            if retained:
                dp[next_mask][state] = retained

    def _prune_dominated_blocks(
        self,
        prev_mask: int,
        blocks: Iterable[BlockCandidate],
    ) -> List[BlockCandidate]:
        """Remove exact same-resource transition-vector dominance.

        Once a block has covered ``block.mask``, its internal backend and
        order are absent from the future DP state.  Therefore, among blocks
        that reserve the same number of remote CPUs, one whose incremental
        local/parallel/GPU service vector is component-wise no better can
        never lead to a better suffix.  Cache transitions retain all choices
        because the physical cache marker records the block's terminal pipe.
        """
        candidates = list(blocks)
        required_methods = (
            "_dp_stage_boundary_components",
            "_dp_gpu_worker_multiplier",
        )
        if (
            getattr(self.cache_policy, "enabled", False)
            or len(candidates) < 2
            or any(
                not hasattr(self.optimizer, method)
                for method in required_methods
            )
        ):
            return candidates

        vectors = []
        for block in candidates:
            boundary_local, boundary_parallel = (
                self.optimizer._dp_stage_boundary_components(
                    prev_mask, block
                )
            )
            resource_cost = self.optimizer._dp_parallel_stage_cpu_cost(block)
            if block.execution_resource == PipeExecutionResource.CUDA:
                vector = (
                    boundary_local,
                    0.0,
                    self.optimizer._dp_gpu_worker_multiplier()
                    * (block.cost + boundary_parallel),
                )
            elif block.variant in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
                PipeVariantType.SMP,
            ):
                vector = (
                    boundary_local,
                    block.cost / block.parallelism + boundary_parallel,
                    0.0,
                )
            else:
                vector = (
                    self.optimizer._dp_regular_transition_cost(
                        prev_mask, block
                    ),
                    0.0,
                    0.0,
                )
            vectors.append((block, resource_cost, vector))

        retained = []
        tolerance = 1e-12
        for index, (block, resource_cost, vector) in enumerate(vectors):
            dominated = False
            for other_index, (_, other_resource, other_vector) in enumerate(
                vectors
            ):
                if index == other_index or resource_cost != other_resource:
                    continue
                no_worse = all(
                    left <= right + tolerance
                    for left, right in zip(other_vector, vector)
                )
                strictly_better = any(
                    left < right - tolerance
                    for left, right in zip(other_vector, vector)
                )
                if no_worse and (
                    strictly_better or other_index < index
                ):
                    dominated = True
                    break
            if not dominated:
                retained.append(block)
        return retained

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
        parallelism_by_idx: Dict[int, int] = {}
        cache_after_idx: Optional[int] = None

        while mask:
            pointer = back[mask].get((state, objective))
            if pointer is None:
                raise RuntimeError("Extensible DP backtracking hit illegal state.")

            order = list(pointer.block.order)
            blocks_rev.append(order)
            for idx in order:
                variants_by_idx[idx] = pointer.block.variant
                parallelism_by_idx[idx] = pointer.block.parallelism
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
            parallelism_by_idx=parallelism_by_idx,
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

    joint_actor_allocation = True

    def _allocate_final_remote_stage_resources(self) -> None:
        """Materialize the actor/process counts selected by joint DP."""
        selected = getattr(self, "_dp_selected_stage_parallelism", {})
        for p_id in self.physical_plan.graph:
            desc = self.physical_plan.pipe_descs[p_id]
            if desc.variant_type not in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
                PipeVariantType.SMP,
            ):
                continue
            members = tuple(
                desc.fused_pipes
                if desc.fused_pipes is not None
                else [p_id]
            )
            width = int(selected.get(members, 1))
            if width < 1:
                raise RuntimeError("DP selected a non-positive stage width.")
            if desc.variant_type in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            ):
                desc.variant_ctx.n_actors = width
            else:
                desc.variant_ctx.n_procs = width

    def _dp_max_candidate_parallelism(
        self, variant: PipeVariantType
    ) -> int:
        if not self.joint_actor_allocation or variant not in (
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
            PipeVariantType.SMP,
        ):
            return 1
        return max(1, self._dp_parallel_stage_cpu_limit() or 1)

    def _dp_candidate_parallelisms(
        self,
        variant: PipeVariantType,
        execution_resource: PipeExecutionResource,
    ) -> Tuple[int, ...]:
        if (
            execution_resource == PipeExecutionResource.CUDA
            or variant
            not in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
                PipeVariantType.SMP,
            )
        ):
            return (1,)
        return tuple(range(1, self._dp_max_candidate_parallelism(variant) + 1))

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

    def _dp_gpu_worker_multiplier(self) -> int:
        """Return how many local workers share the single formal GPU.

        CPU coordinates are per-worker service times. Multiplying global GPU
        demand by W puts its capacity in the same coordinate system:
        ``W / max(cpu_ms, W * gpu_ms)`` equals the minimum of aggregate CPU
        capacity and the one-GPU capacity.
        """
        if os.environ.get("CEDAR_MATCH_PROFILE_RESOURCES") == "1":
            raw = os.environ.get("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS")
            if raw is not None:
                try:
                    workers = int(raw)
                except ValueError as exc:
                    raise RuntimeError(
                        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS must be an integer"
                    ) from exc
                if workers < 1:
                    raise RuntimeError(
                        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS must be positive"
                    )
                return workers
        return max(1, int(self.physical_plan.n_local_workers))

    def _dp_accumulate_objective_cost(
        self,
        previous: DpObjectiveCost,
        extra_cost: float,
        block: BlockCandidate,
        prev_mask: int,
    ) -> DpObjectiveCost:
        boundary_local, boundary_parallel = (
            self._dp_stage_boundary_components(prev_mask, block)
        )
        if block.execution_resource == PipeExecutionResource.CUDA:
            return DpObjectiveCost(
                local_serial=previous.local_serial + boundary_local,
                parallel_bottleneck=previous.parallel_bottleneck,
                gpu_serial=(
                    previous.gpu_serial
                    + self._dp_gpu_worker_multiplier()
                    * (block.cost + boundary_parallel)
                ),
            )
        if block.variant in (
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
            PipeVariantType.SMP,
        ):
            stage_cost = block.cost / block.parallelism + boundary_parallel
            return DpObjectiveCost(
                local_serial=previous.local_serial + boundary_local,
                parallel_bottleneck=max(
                    previous.parallel_bottleneck, stage_cost
                ),
                gpu_serial=previous.gpu_serial,
            )
        return DpObjectiveCost(
            local_serial=previous.local_serial + extra_cost,
            parallel_bottleneck=previous.parallel_bottleneck,
            gpu_serial=previous.gpu_serial,
        )

    def _dp_blocks_from_physical_plan(
        self,
        plan: PhysicalPlan,
        inner_ops: List[int],
    ) -> List[Tuple[Tuple[int, ...], PipeVariantType, bool, int]]:
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
            if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
                parallelism = int(desc.variant_ctx.n_actors)
            elif variant == PipeVariantType.SMP:
                parallelism = int(desc.variant_ctx.n_procs)
            else:
                parallelism = 1
            blocks.append([order, variant, False, parallelism])

        flattened = [idx for order, _, _, _ in blocks for idx in order]
        if len(flattened) != len(inner_ops) or set(flattened) != set(
            range(len(inner_ops))
        ):
            raise ValueError("Physical plan does not cover every DP operator once.")
        return [
            (tuple(order), variant, bool(cache_after), parallelism)
            for order, variant, cache_after, parallelism in blocks
        ]

    def _replay_dp_objective(
        self,
        block_specs: Iterable[
            Tuple[Tuple[int, ...], PipeVariantType, bool, int]
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
        for order, variant, wants_cache, parallelism in block_specs:
            block = provider.candidate_for_order(
                order,
                variant,
                prefix_mask=prev_mask,
                parallelism=parallelism,
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
                    + self._dp_parallel_stage_cpu_cost(block)
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
                prev_mask,
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
                parallelism = search_result.parallelism_by_idx.get(
                    block[0], 1
                )
                block_specs.append(
                    (tuple(block), variant, wants_cache, parallelism)
                )
        else:
            block_specs = self._dp_blocks_from_physical_plan(plan, ops)
        total_parallel_cpus = sum(
            parallelism
            for _, variant, _, parallelism in block_specs
            if variant
            in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
                PipeVariantType.SMP,
            )
        )
        previous_total = getattr(
            self, "_dp_assumed_total_parallel_stage_cpus", None
        )
        self._dp_assumed_total_parallel_stage_cpus = total_parallel_cpus
        try:
            return self._replay_dp_objective(block_specs, ops).score
        finally:
            self._dp_assumed_total_parallel_stage_cpus = previous_total

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

    def _dp_parallel_stage_cpu_cost(self, block: BlockCandidate) -> int:
        if block.variant in (
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
            PipeVariantType.SMP,
        ):
            return block.parallelism
        return 0

    def _run_conditioned_dp_search(
        self,
        inner_ops: List[int],
        assumed_total: Optional[int],
    ):
        self._dp_assumed_total_parallel_stage_cpus = assumed_total
        block_provider = BlockCandidateProvider(self, inner_ops)
        block_provider.prepare()
        cache_policy = CacheTransitionPolicy(self, inner_ops)
        search_kwargs = {}
        if assumed_total is not None:
            search_kwargs = {
                "parallel_stage_cpu_limit": assumed_total,
                "required_final_parallel_stage_cpus": assumed_total,
            }
        search = ExtensibleDpSearch(
            optimizer=self,
            inner_ops=inner_ops,
            block_provider=block_provider,
            cache_policy=cache_policy,
            **search_kwargs,
        )
        shortcut_raw = os.environ.get(
            "CEDAR_DP_GREEDY_HIGH_BUDGET_THRESHOLD", "4"
        )
        try:
            shortcut_threshold = int(shortcut_raw)
        except ValueError as exc:
            raise RuntimeError(
                "CEDAR_DP_GREEDY_HIGH_BUDGET_THRESHOLD must be an integer."
            ) from exc
        cpu_only = all(
            self.logical_pipes[p_id].execution_resource
            == PipeExecutionResource.CPU
            for p_id in inner_ops
        )
        if (
            len(inner_ops) > 8
            and assumed_total is not None
            and assumed_total >= shortcut_threshold
            and not cache_policy.enabled
            and cpu_only
        ):
            search._initial_incumbent_score()
            candidate_result = search._greedy_full_block_result
            if candidate_result is not None:
                stats = {
                    "objective": "greedy_full_fusion_high_budget",
                    "parallel_stage_cpu_limit": assumed_total,
                    "required_final_parallel_stage_cpus": assumed_total,
                    "frontier_cap": search.frontier_cap,
                    "cost": candidate_result.cost,
                }
                self._dp_last_search_stats = stats
                logger.info(
                    "[DpOptimizer] High-budget bounded search stats: %s",
                    stats,
                )
                return assumed_total, candidate_result, stats
        try:
            candidate_result = search.run()
        except RuntimeError as exc:
            if "required parallel CPU total" in str(exc):
                return None
            raise
        return (
            assumed_total,
            candidate_result,
            dict(self._dp_last_search_stats),
        )

    def _dp_reorder_offload_cache_fusion(
        self,
        inner_ops: List[int],
    ) -> Tuple[List[int], Optional[int]]:
        if not inner_ops:
            return [], None
        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")

        resource_limit = self._dp_parallel_stage_cpu_limit()
        assumed_totals = (
            [None]
            if resource_limit is None
            else list(range(resource_limit + 1))
        )
        forced_total_raw = os.environ.get("CEDAR_DP_FORCED_PARALLEL_TOTAL")
        if forced_total_raw is not None:
            if resource_limit is None:
                raise RuntimeError(
                    "CEDAR_DP_FORCED_PARALLEL_TOTAL requires strict resource matching."
                )
            try:
                forced_total = int(forced_total_raw)
            except ValueError as exc:
                raise RuntimeError(
                    "CEDAR_DP_FORCED_PARALLEL_TOTAL must be an integer."
                ) from exc
            if forced_total < 0 or forced_total > resource_limit:
                raise RuntimeError(
                    "CEDAR_DP_FORCED_PARALLEL_TOTAL is outside the feasible "
                    f"range [0, {resource_limit}]."
                )
            assumed_totals = [forced_total]
        worker_raw = os.environ.get(
            "CEDAR_DP_RESOURCE_SEARCH_WORKERS",
            str(min(len(assumed_totals), os.cpu_count() or 1)),
        )
        try:
            search_workers = int(worker_raw)
        except ValueError as exc:
            raise RuntimeError(
                "CEDAR_DP_RESOURCE_SEARCH_WORKERS must be an integer."
            ) from exc
        if search_workers < 1:
            raise RuntimeError(
                "CEDAR_DP_RESOURCE_SEARCH_WORKERS must be positive."
            )
        use_parallel_search = (
            len(inner_ops) > 8
            and len(assumed_totals) > 1
            and search_workers > 1
            and "fork" in mp.get_all_start_methods()
        )
        conditioned_results = []
        if use_parallel_search:
            # Precompute dependency topology in the parent. Forked workers
            # inherit it copy-on-write and only rebuild B-specific cost tables.
            topology_search = ExtensibleDpSearch(
                optimizer=self,
                inner_ops=inner_ops,
                block_provider=None,
                cache_policy=None,
            )
            topology_search._search_topology()
            process_count = min(search_workers, len(assumed_totals))
            logger.info(
                "[DpOptimizer] Searching %d resource totals on %d processes.",
                len(assumed_totals),
                process_count,
            )
            context = mp.get_context("fork")
            with context.Pool(
                processes=process_count,
                initializer=_initialize_conditioned_search_worker,
                initargs=(self, inner_ops),
            ) as pool:
                results = pool.map(
                    _run_conditioned_search_worker, assumed_totals
                )
            conditioned_results.extend(
                result for result in results if result is not None
            )
        else:
            for assumed_total in assumed_totals:
                candidate = self._run_conditioned_dp_search(
                    inner_ops, assumed_total
                )
                if candidate is not None:
                    conditioned_results.append(candidate)
        if not conditioned_results:
            raise RuntimeError(
                "Resource-conditioned DP found no feasible plan."
            )
        assumed_total, result, selected_stats = min(
            conditioned_results,
            key=lambda item: (
                item[1].cost,
                item[0] if item[0] is not None else 0,
            ),
        )
        self._dp_assumed_total_parallel_stage_cpus = assumed_total
        self._dp_last_search_stats = selected_stats
        replayed_objective = self._replay_dp_objective(
            [
                (
                    tuple(block),
                    result.variants_by_idx.get(
                        block[0], PipeVariantType.INPROCESS
                    ),
                    result.cache_after_idx is not None
                    and block[-1] == result.cache_after_idx,
                    result.parallelism_by_idx.get(block[0], 1),
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
        self._dp_selected_stage_parallelism = {
            tuple(inner_ops[index] for index in block):
            result.parallelism_by_idx.get(block[0], 1)
            for block in result.blocks
        }

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
            "local_serial=%s parallel_bottleneck=%s "
            "gpu_serial=%s",
            replayed_objective.local_serial,
            replayed_objective.parallel_bottleneck,
            replayed_objective.gpu_serial,
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
