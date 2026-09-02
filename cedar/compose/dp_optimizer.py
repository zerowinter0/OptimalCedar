import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from cedar.pipes import Pipe, PipeExecutionResource

from .my_optimizer import MyOptimizer
from . import constants
from .optimizer import OptimizerOptions, PhysicalPlan, PipeDesc, PipeVariantType


logger = logging.getLogger(__name__)


_FORK_LAYER_SEARCH = None
_FORK_LAYER_DP = None
_FORK_LAYER_BACK = None
_FORK_LAYER_PREDECESSORS = None
_FORK_LAYER_INCUMBENT = float("inf")
_USE_CONFIGURED_RESOURCE_LIMIT = object()


class _DpSearchDeadlineExceeded(RuntimeError):
    """Internal control flow used to return the best feasible DP incumbent."""


def _raise_if_dp_deadline_exceeded(optimizer: "DpOptimizer") -> None:
    deadline = getattr(optimizer, "_dp_search_deadline", None)
    if deadline is not None and time.monotonic() >= deadline:
        raise _DpSearchDeadlineExceeded(
            "DP optimization reached its configured time limit."
        )


def _initialize_mask_layer_worker(
    search, dp, back, predecessors, incumbent_score
) -> None:
    global _FORK_LAYER_SEARCH, _FORK_LAYER_DP, _FORK_LAYER_BACK
    global _FORK_LAYER_PREDECESSORS, _FORK_LAYER_INCUMBENT
    _FORK_LAYER_SEARCH = search
    _FORK_LAYER_DP = dp
    _FORK_LAYER_BACK = back
    _FORK_LAYER_PREDECESSORS = predecessors
    _FORK_LAYER_INCUMBENT = incumbent_score


def _run_mask_layer_worker(next_mask: int):
    if (
        _FORK_LAYER_SEARCH is None
        or _FORK_LAYER_DP is None
        or _FORK_LAYER_BACK is None
        or _FORK_LAYER_PREDECESSORS is None
    ):
        raise RuntimeError("Mask-layer DP worker was not initialized.")
    search = _FORK_LAYER_SEARCH
    before = (
        search._transition_pairs,
        search._upper_bound_pruned,
        search._suffix_lower_bound_pruned,
        search._cell_candidates,
        search._cell_replacements,
        search._frontier_cap_pruned,
    )
    if search._can_use_exact_batch_frontier():
        search._compute_exact_batch_frontier(
            _FORK_LAYER_DP,
            _FORK_LAYER_BACK,
            next_mask,
            _FORK_LAYER_PREDECESSORS[next_mask],
            _FORK_LAYER_INCUMBENT,
        )
    else:
        for prev_mask in _FORK_LAYER_PREDECESSORS[next_mask]:
            if not _FORK_LAYER_DP[prev_mask]:
                continue
            block_mask = next_mask ^ prev_mask
            if block_mask:
                search._transition_pairs += 1
                search._try_extend(
                    _FORK_LAYER_DP,
                    _FORK_LAYER_BACK,
                    prev_mask,
                    next_mask,
                    block_mask,
                    _FORK_LAYER_INCUMBENT,
                )
    if search.pareto_step_epsilon > 0.0 and search.use_cell_frontier:
        search._finalize_cell_frontiers(
            _FORK_LAYER_DP, _FORK_LAYER_BACK, next_mask
        )
    after = (
        search._transition_pairs,
        search._upper_bound_pruned,
        search._suffix_lower_bound_pruned,
        search._cell_candidates,
        search._cell_replacements,
        search._frontier_cap_pruned,
    )
    deltas = tuple(right - left for left, right in zip(before, after))
    return (
        next_mask,
        _FORK_LAYER_DP[next_mask],
        _FORK_LAYER_BACK[next_mask],
        deltas,
    )


def _is_infeasible_conditioned_search_error(exc: RuntimeError) -> bool:
    """Recognize the two failure modes of an exact resource slice."""
    message = str(exc)
    return (
        "required parallel CPU total" in message
        or message == "Extensible DP failed: no feasible final state."
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
class DpResourceUsage:
    """Per-local-worker reservations on the two physical CPU pools."""

    ray_cpus: int = 0
    smp_cpus: int = 0

    def __add__(self, other: "DpResourceUsage") -> "DpResourceUsage":
        if not isinstance(other, DpResourceUsage):
            return NotImplemented
        return DpResourceUsage(
            self.ray_cpus + other.ray_cpus,
            self.smp_cpus + other.smp_cpus,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __le__(self, other: "DpResourceUsage") -> bool:
        if not isinstance(other, DpResourceUsage):
            return NotImplemented
        return (
            self.ray_cpus <= other.ray_cpus
            and self.smp_cpus <= other.smp_cpus
        )

    def __gt__(self, other: "DpResourceUsage") -> bool:
        if not isinstance(other, DpResourceUsage):
            return NotImplemented
        return (
            self.ray_cpus > other.ray_cpus
            or self.smp_cpus > other.smp_cpus
        )

    def as_dict(self) -> Dict[str, int]:
        return {"ray_cpus": self.ray_cpus, "smp_cpus": self.smp_cpus}


@dataclass(frozen=True)
class DpStateSummary:
    """
    Compact state carried by the outer DP.

    The key design difference from MyOptimizer is that strategy state is a
    summary object. New strategies should extend or replace this summary only
    when future transitions need extra information.
    """

    cache_active: bool = False
    parallel_stage_cpus: DpResourceUsage = DpResourceUsage()


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
        track_endpoints: bool = True,
        track_first_endpoints: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.track_endpoints = track_endpoints
        self.track_first_endpoints = (
            track_first_endpoints and not track_endpoints
        )
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
        self._endpoint_costs: Dict[
            Tuple[int, int], Dict[Tuple[int, int], float]
        ] = {}
        self._endpoint_orders: Dict[
            Tuple[int, int], Dict[Tuple[int, int], Tuple[int, ...]]
        ] = {}
        self._first_costs: Dict[Tuple[int, int], Dict[int, float]] = {}
        self._first_orders: Dict[
            Tuple[int, int], Dict[int, Tuple[int, ...]]
        ] = {}

    def get(
        self, mask: int, prefix_mask: int = 0
    ) -> Tuple[float, Tuple[int, ...]]:
        if mask & prefix_mask:
            raise ValueError("A DP block cannot overlap its prefix.")
        self._compute(prefix_mask, mask)
        key = (prefix_mask, mask)
        return self._costs[key] * self.source_size, self._orders[key]

    def get_endpoint_minima(
        self, mask: int, prefix_mask: int = 0
    ) -> List[Tuple[float, Tuple[int, ...]]]:
        """Return the minimum-compute order for every legal endpoint pair.

        Boundary calibration and cache placement may depend on the first and
        last operator of a fused block. For fixed endpoints, however, every
        other transition property is identical, so a higher-compute internal
        order is losslessly dominated.
        """
        if mask & prefix_mask:
            raise ValueError("A DP block cannot overlap its prefix.")
        if not self.track_endpoints:
            return [self.get(mask, prefix_mask)]
        self._compute(prefix_mask, mask)
        key = (prefix_mask, mask)
        return [
            (self._endpoint_costs[key][endpoints] * self.source_size, order)
            for endpoints, order in self._endpoint_orders[key].items()
        ]

    def get_feasible_endpoint_minimum(
        self,
        mask: int,
        prefix_mask: int,
        endpoint_allowed,
    ) -> List[Tuple[float, Tuple[int, ...]]]:
        """Return the cheapest order whose first/last pair is feasible.

        When endpoints affect only legality (not boundary cost), retaining the
        best partial order per first endpoint is sufficient.  At the final
        recurrence level we enumerate the last operator, apply the feasibility
        predicate, and keep one globally minimum order.  This is lossless and
        avoids the quadratic endpoint map required by endpoint-priced costs.
        """
        if mask & prefix_mask:
            raise ValueError("A DP block cannot overlap its prefix.")
        if not self.track_first_endpoints:
            feasible = [
                item
                for item in self.get_endpoint_minima(mask, prefix_mask)
                if item[1]
                and endpoint_allowed(item[1][0], item[1][-1])
            ]
            if not feasible:
                return []
            return [min(feasible, key=lambda item: item[0])]

        best = float("inf")
        best_order: Tuple[int, ...] = tuple()
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            last = bit.bit_length() - 1
            if self.per_byte[last] == float("inf"):
                continue
            prev = mask ^ bit
            if prev & self.successor_masks[last]:
                continue
            operator_cost = (
                _work_prod(self.optimizer, prefix_mask | prev, last)
                * self.per_byte[last]
            )
            if prev == 0:
                predecessors = [(last, 0.0, tuple())]
            else:
                self._compute(prefix_mask, prev)
                prev_key = (prefix_mask, prev)
                predecessors = [
                    (first, cost, self._first_orders[prev_key][first])
                    for first, cost in self._first_costs[prev_key].items()
                ]
            for first, prev_cost, prev_order in predecessors:
                if not endpoint_allowed(first, last):
                    continue
                candidate = prev_cost + operator_cost
                if candidate < best:
                    best = candidate
                    best_order = prev_order + (last,)
        if not best_order:
            return []
        return [(best * self.source_size, best_order)]

    def _compute(self, prefix_mask: int, mask: int) -> None:
        _raise_if_dp_deadline_exceeded(self.optimizer)
        key = (prefix_mask, mask)
        if key in self._costs:
            return
        if mask == 0:
            self._costs[key] = 0.0
            self._orders[key] = tuple()
            self._endpoint_costs[key] = {}
            self._endpoint_orders[key] = {}
            self._first_costs[key] = {}
            self._first_orders[key] = {}
            return

        best = float("inf")
        best_order: Tuple[int, ...] = tuple()
        endpoint_costs: Dict[Tuple[int, int], float] = {}
        endpoint_orders: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        first_costs: Dict[int, float] = {}
        first_orders: Dict[int, Tuple[int, ...]] = {}
        remaining = mask
        while remaining:
            _raise_if_dp_deadline_exceeded(self.optimizer)
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
            operator_cost = (
                _work_prod(self.optimizer, prefix_mask | prev, i)
                * self.per_byte[i]
            )
            if prev == 0:
                predecessors = [(i, 0.0, tuple())]
            elif self.track_endpoints:
                self._compute(prefix_mask, prev)
                prev_key = (prefix_mask, prev)
                predecessors = [
                    (
                        endpoints[0],
                        prev_cost,
                        self._endpoint_orders[prev_key][endpoints],
                    )
                    for endpoints, prev_cost in self._endpoint_costs[
                        prev_key
                    ].items()
                ]
            elif self.track_first_endpoints:
                self._compute(prefix_mask, prev)
                prev_key = (prefix_mask, prev)
                predecessors = [
                    (
                        first,
                        prev_cost,
                        self._first_orders[prev_key][first],
                    )
                    for first, prev_cost in self._first_costs[
                        prev_key
                    ].items()
                ]
            else:
                self._compute(prefix_mask, prev)
                prev_key = (prefix_mask, prev)
                prev_order = self._orders[prev_key]
                predecessors = [
                    (
                        prev_order[0],
                        self._costs[prev_key],
                        prev_order,
                    )
                ] if prev_order else []
            for first, prev_cost, prev_order in predecessors:
                if prev_cost == float("inf"):
                    continue
                candidate = prev_cost + operator_cost
                order = prev_order + (i,)
                if self.track_endpoints:
                    endpoints = (first, i)
                    if candidate < endpoint_costs.get(
                        endpoints, float("inf")
                    ):
                        endpoint_costs[endpoints] = candidate
                        endpoint_orders[endpoints] = order
                if self.track_first_endpoints and candidate < first_costs.get(
                    first, float("inf")
                ):
                    first_costs[first] = candidate
                    first_orders[first] = order
                if candidate < best:
                    best = candidate
                    best_order = order

        if not self.track_endpoints and best_order:
            endpoints = (best_order[0], best_order[-1])
            endpoint_costs[endpoints] = best
            endpoint_orders[endpoints] = best_order
        self._costs[key] = best
        self._orders[key] = best_order
        self._endpoint_costs[key] = endpoint_costs
        self._endpoint_orders[key] = endpoint_orders
        self._first_costs[key] = first_costs
        self._first_orders[key] = first_orders


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
class _ThresholdBackPointer:
    """Backpointer for one makespan-feasibility decision problem."""

    prev_mask: int
    prev_usage: DpResourceUsage
    block: BlockCandidate


@dataclass(frozen=True)
class DpObjectiveCost:
    """Resource-family service coordinates retained by the subset DP.

    Work assigned to the same resource family is additive: local blocks form
    ``L``, Ray blocks form ``R``, SMP blocks form ``S``, and CUDA blocks form
    ``G``. Ray and SMP additionally reserve independent integer widths in the
    DP state. The predicted end-to-end bottleneck is ``max(L,R,S,G)``.
    """

    local_serial: float = 0.0
    ray_serial: float = 0.0
    smp_serial: float = 0.0
    gpu_serial: float = 0.0

    @property
    def parallel_bottleneck(self) -> float:
        """Return the slower of the additive Ray and SMP resource lanes."""
        return max(self.ray_serial, self.smp_serial)

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
        self._shared_cost_indexes: Dict[
            Tuple[Tuple[float, ...], str], _BlockCostIndex
        ] = {}
        self._endpoint_sensitive_variants: set[PipeVariantType] = set()
        self._boundary_endpoint_sensitive_variants: set[
            PipeVariantType
        ] = set()
        self._threshold_width_curves: Dict[
            Tuple[int, int], Tuple[Tuple[BlockCandidate, ...], ...]
        ] = {}

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
        logger.info(
            "[DpOptimizer] Candidate backends=%s resource_limits=%s",
            [variant.name for variant in candidate_variants],
            opt._dp_parallel_stage_cpu_limit(),
        )
        self._fusion_allowed_flags = [
            opt._allowed_fusion(p_id) for p_id in self.inner_ops
        ]
        object_boundaries = (
            opt.profiled_stats.get("physical_model", {}).get(
                "object_boundary", {}
            )
        )
        if not isinstance(object_boundaries, dict):
            object_boundaries = {}
        for variant in candidate_variants:
            key = (
                PipeVariantType.RAY.name
                if variant == PipeVariantType.TF_RAY
                else variant.name
            )
            backend = object_boundaries.get(key, {})
            operators = (
                backend.get("operators", {})
                if isinstance(backend, dict)
                else {}
            )
            if isinstance(operators, dict) and operators:
                self._endpoint_sensitive_variants.add(variant)
                self._boundary_endpoint_sensitive_variants.add(variant)
        # SMP transport feasibility additionally depends on the first and
        # last physical operators even when no endpoint boundary profile is
        # present. Ray transport has no such restriction.
        if PipeVariantType.SMP in candidate_variants:
            self._endpoint_sensitive_variants.add(PipeVariantType.SMP)
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
                # Resource-matched profiles price every candidate width at
                # the same family-wide contention level; width then changes
                # stage service through the explicit ``cost / width`` term.
                # In that common case the per-operator cost vectors are
                # identical. Reuse the exact subset-order index instead of
                # recomputing every internal fused order once per width.
                signature = tuple(costs)
                track_endpoints = (
                    vt in self._boundary_endpoint_sensitive_variants
                )
                track_first_endpoints = (
                    vt in self._endpoint_sensitive_variants
                    and not track_endpoints
                )
                endpoint_mode = (
                    "full"
                    if track_endpoints
                    else "first"
                    if track_first_endpoints
                    else "none"
                )
                shared_key = (signature, endpoint_mode)
                index = self._shared_cost_indexes.get(shared_key)
                if index is None:
                    index = _BlockCostIndex(
                        opt,
                        self.inner_ops,
                        costs,
                        track_endpoints=track_endpoints,
                        track_first_endpoints=track_first_endpoints,
                    )
                    self._shared_cost_indexes[shared_key] = index
                self._variant_indexes[key] = index
                if vt != PipeVariantType.INPROCESS:
                    # Fusion removes intermediate boundaries, while compute
                    # remains exactly the same sum as an unfused block.
                    self._fused_variant_indexes[key] = index

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
                variant_key = (vt, parallelism)
                index = self._variant_indexes[variant_key]
                if is_multi and variant_key in self._fused_variant_indexes:
                    index = self._fused_variant_indexes[variant_key]
                endpoint_orders = self._orders_for_variant(
                    index, prefix_mask, mask, vt, is_multi
                )
                for block_cost, order in endpoint_orders:
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

    def candidates_for_threshold(
        self,
        prefix_mask: int,
        mask: int,
        threshold: float,
    ) -> Iterable[BlockCandidate]:
        """Generate only the minimum feasible width per structural curve.

        Resource-matched profiles commonly map every width of a backend to
        the same ``_BlockCostIndex``.  The general provider materializes one
        candidate per width because the Pareto DP needs every trade-off.  A
        threshold decision does not: once ``cost / width + boundary <= T``,
        every larger width on that same curve only consumes more CPUs.  This
        path evaluates the shared internal order once and returns the first
        feasible width without populating the width-expanded candidate cache.
        Distinct measured width curves remain separate and therefore retain
        exact behavior for non-stationary profiles.
        """
        if mask <= 0 or mask >= self.full_mask:
            return []
        width_curves = self.threshold_width_curves_for_prefix(
            prefix_mask, mask
        )

        candidates: List[BlockCandidate] = []
        for curve in width_curves:
            for block in curve:
                extra_cost = self.optimizer._dp_regular_transition_cost(
                    prefix_mask, block
                )
                delta = self.optimizer._dp_accumulate_objective_cost(
                    DpObjectiveCost(),
                    extra_cost,
                    block,
                    prefix_mask,
                )
                if (
                    delta.local_serial <= threshold + 1e-12
                    and delta.parallel_bottleneck <= threshold + 1e-12
                    and delta.gpu_serial <= threshold + 1e-12
                ):
                    candidates.append(block)
                    break
        return candidates

    def threshold_width_curves_for_prefix(
        self,
        prefix_mask: int,
        mask: int,
    ) -> Tuple[Tuple[BlockCandidate, ...], ...]:
        """Return cached structural width curves without threshold scoring."""
        cache_key = (prefix_mask, mask)
        width_curves = self._threshold_width_curves.get(cache_key)
        if width_curves is None:
            width_curves = self._build_threshold_width_curves(
                prefix_mask, mask
            )
            self._threshold_width_curves[cache_key] = width_curves
        return width_curves

    def _build_threshold_width_curves(
        self,
        prefix_mask: int,
        mask: int,
    ) -> Tuple[Tuple[BlockCandidate, ...], ...]:
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
            not opt.options.enable_fusion or not fusion_allowed
        ):
            return ()

        execution_resource = self._execution_resource_for_mask(mask)
        curves: List[Tuple[BlockCandidate, ...]] = []
        for vt in self._candidate_variants:
            if (
                execution_resource == PipeExecutionResource.CUDA
                and vt not in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
            ):
                continue
            if is_multi and not opt._dp_has_supported_fusion_cost(vt):
                continue
            if is_multi:
                feasibility_key = (mask, vt)
                feasible = self._fusion_variant_feasible.get(feasibility_key)
                if feasible is None:
                    feasible = all(
                        opt._pipe_can_materialize_fusion(
                            self.inner_ops[i], vt
                        )
                        for i in range(self.n)
                        if mask & (1 << i)
                    )
                    self._fusion_variant_feasible[feasibility_key] = feasible
                if not feasible:
                    continue

            # Widths sharing an index have identical block compute/order.
            # Preserve separate groups when measured contention changes the
            # underlying per-operator curve.
            width_groups: Dict[int, Tuple[_BlockCostIndex, List[int]]] = {}
            for parallelism in opt._dp_candidate_parallelisms(
                vt, execution_resource
            ):
                key = (vt, parallelism)
                index = self._variant_indexes[key]
                if is_multi and key in self._fused_variant_indexes:
                    index = self._fused_variant_indexes[key]
                group = width_groups.get(id(index))
                if group is None:
                    width_groups[id(index)] = (index, [parallelism])
                else:
                    group[1].append(parallelism)

            for index, parallelisms in width_groups.values():
                endpoint_orders = self._orders_for_variant(
                    index, prefix_mask, mask, vt, is_multi
                )
                for block_cost, order in endpoint_orders:
                    if block_cost == float("inf") or not order:
                        continue
                    curve: List[BlockCandidate] = []
                    for parallelism in sorted(parallelisms):
                        block = BlockCandidate(
                            mask=mask,
                            order=order,
                            variant=vt,
                            cost=block_cost,
                            materializes_fusion=is_multi,
                            execution_resource=execution_resource,
                            parallelism=parallelism,
                        )
                        if mask.bit_count() == 1:
                            idx = mask.bit_length() - 1
                            can_follow = opt._dp_valid_single_last(
                                prefix_mask, idx
                            )
                        else:
                            can_follow = (
                                opt._dp_fusion_block_valid(prefix_mask, mask)
                                and opt._dp_fusion_transport_feasible(
                                    prefix_mask, block
                                )
                            )
                        if not can_follow:
                            break
                        curve.append(block)
                    if curve:
                        curves.append(tuple(curve))
        return tuple(curves)

    def _orders_for_variant(
        self,
        index: _BlockCostIndex,
        prefix_mask: int,
        mask: int,
        variant: PipeVariantType,
        is_multi: bool,
    ) -> List[Tuple[float, Tuple[int, ...]]]:
        if not is_multi:
            return [index.get(mask, prefix_mask)]
        if variant in self._boundary_endpoint_sensitive_variants:
            return index.get_endpoint_minima(mask, prefix_mask)
        if variant in self._endpoint_sensitive_variants:
            execution_resource = self._execution_resource_for_mask(mask)

            def endpoint_allowed(first: int, last: int) -> bool:
                probe_order = (
                    (first, last) if first != last else (first,)
                )
                probe = BlockCandidate(
                    mask=mask,
                    order=probe_order,
                    variant=variant,
                    cost=0.0,
                    materializes_fusion=True,
                    execution_resource=execution_resource,
                    parallelism=1,
                )
                return self.optimizer._dp_fusion_transport_feasible(
                    prefix_mask, probe
                )

            return index.get_feasible_endpoint_minimum(
                mask, prefix_mask, endpoint_allowed
            )
        return [index.get(mask, prefix_mask)]

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

        ``candidates_for`` retains the minimum-compute order for every endpoint
        pair. Cost-model validation must instead replay the exact order stored
        in a fixed physical plan. This method uses the same per-byte backend
        costs and work-product recurrence as ``_BlockCostIndex`` without
        re-optimizing the block's order.
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
        next_parallel_stage_cpus: Optional[DpResourceUsage] = None,
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
        parallel_stage_cpu_limit: Any = _USE_CONFIGURED_RESOURCE_LIMIT,
        required_final_parallel_stage_cpus: Optional[DpResourceUsage] = None,
        initial_incumbent_score: float = float("inf"),
    ) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self.block_provider = block_provider
        self.cache_policy = cache_policy
        self.collapse_external_service_coordinates = bool(
            getattr(
                optimizer,
                "collapse_external_service_coordinates",
                False,
            )
        )
        configured_limit = optimizer._dp_parallel_stage_cpu_limit()
        self.parallel_stage_cpu_limit = (
            configured_limit
            if parallel_stage_cpu_limit
            is _USE_CONFIGURED_RESOURCE_LIMIT
            else parallel_stage_cpu_limit
        )
        self.required_final_parallel_stage_cpus = (
            required_final_parallel_stage_cpus
        )
        self.initial_incumbent_score = initial_incumbent_score
        global_epsilon = float(
            os.environ.get("CEDAR_DP_PARETO_EPSILON", "0")
        )
        if not math.isfinite(global_epsilon) or global_epsilon < 0.0:
            raise ValueError("CEDAR_DP_PARETO_EPSILON must be finite and non-negative")
        # Exact Pareto retention is the production default. Experiments may
        # explicitly request multiplicative trimming; global_epsilon then
        # bounds its accumulated coordinate error across at most n steps.
        self.pareto_global_epsilon = global_epsilon if self.n > 8 else 0.0
        self.pareto_step_epsilon = (
            (1.0 + self.pareto_global_epsilon) ** (1.0 / self.n) - 1.0
            if self.pareto_global_epsilon > 0.0
            else 0.0
        )
        self._upper_bound_pruned = 0
        self._suffix_lower_bound_pruned = 0
        self._transition_pairs = 0
        self._mandatory_local_suffix_cost: Dict[int, float] = {}
        self._frontier_cells: Dict[
            Tuple[int, DpStateSummary],
            Dict[Tuple[Optional[int], ...], DpObjectiveCost],
        ] = {}
        self._cell_candidates = 0
        self._cell_replacements = 0
        self.use_cell_frontier = (
            os.environ.get("CEDAR_DP_CELL_FRONTIER", "0") == "1"
        )
        frontier_cap_raw = os.environ.get("CEDAR_DP_FRONTIER_CAP", "0")
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
                    // (
                        2
                        ** max(
                            0,
                            assumed_total.ray_cpus
                            + assumed_total.smp_cpus
                            - 2,
                        )
                    ),
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
        self._prepare_mandatory_suffix_lower_bounds(legal_masks)
        layer_workers_raw = os.environ.get(
            "CEDAR_DP_MASK_LAYER_WORKERS",
            str(min(32, os.cpu_count() or 1)),
        )
        try:
            layer_workers = int(layer_workers_raw)
        except ValueError as exc:
            raise ValueError(
                "CEDAR_DP_MASK_LAYER_WORKERS must be an integer"
            ) from exc
        if layer_workers < 1:
            raise ValueError("CEDAR_DP_MASK_LAYER_WORKERS must be positive")
        parallel_layers = (
            self.n > 8
            and layer_workers > 1
            and legal_predecessors is not None
            and "fork" in mp.get_all_start_methods()
        )
        masks_by_cardinality = [
            [mask for mask in legal_masks if mask.bit_count() == cardinality]
            for cardinality in range(1, self.n + 1)
        ]
        if parallel_layers:
            logger.info(
                "[DpOptimizer] Searching independent mask layers with up to "
                "%d processes.",
                layer_workers,
            )
        search_started = time.monotonic()
        for cardinality, layer_masks in enumerate(
            masks_by_cardinality, start=1
        ):
            _raise_if_dp_deadline_exceeded(self.optimizer)
            if not layer_masks:
                continue
            layer_started = time.monotonic()
            if parallel_layers and len(layer_masks) > 1:
                process_count = min(layer_workers, len(layer_masks))
                context = mp.get_context("fork")
                with context.Pool(
                    processes=process_count,
                    initializer=_initialize_mask_layer_worker,
                    initargs=(
                        self,
                        dp,
                        back,
                        legal_predecessors,
                        incumbent_score,
                    ),
                ) as pool:
                    results = pool.map(
                        _run_mask_layer_worker,
                        layer_masks,
                        chunksize=max(1, len(layer_masks) // (4 * process_count)),
                    )
                for next_mask, frontier, pointers, deltas in results:
                    dp[next_mask] = frontier
                    back[next_mask] = pointers
                    (
                        transition_pairs,
                        upper_bound_pruned,
                        suffix_lower_bound_pruned,
                        cell_candidates,
                        cell_replacements,
                        frontier_cap_pruned,
                    ) = deltas
                    self._transition_pairs += transition_pairs
                    self._upper_bound_pruned += upper_bound_pruned
                    self._suffix_lower_bound_pruned += (
                        suffix_lower_bound_pruned
                    )
                    self._cell_candidates += cell_candidates
                    self._cell_replacements += cell_replacements
                    self._frontier_cap_pruned += frontier_cap_pruned
                logger.info(
                    "[DpOptimizer] Exact layer %d/%d masks=%d states=%d "
                    "max_frontier=%d layer_sec=%.3f total_sec=%.3f",
                    cardinality,
                    self.n,
                    len(layer_masks),
                    sum(
                        sum(len(values) for values in dp[mask].values())
                        for mask in layer_masks
                    ),
                    max(
                        (
                            sum(len(values) for values in dp[mask].values())
                            for mask in layer_masks
                        ),
                        default=0,
                    ),
                    time.monotonic() - layer_started,
                    time.monotonic() - search_started,
                )
                continue

            for next_mask in layer_masks:
                _raise_if_dp_deadline_exceeded(self.optimizer)
                # Enumerate only actual subsets of next_mask. Scanning every
                # pair of dependency-closed masks and rejecting non-subsets
                # approaches 4**n; these are exactly the 3**n transitions.
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
                if self._can_use_exact_batch_frontier():
                    self._compute_exact_batch_frontier(
                        dp,
                        back,
                        next_mask,
                        prev_masks,
                        incumbent_score,
                    )
                else:
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
            logger.info(
                "[DpOptimizer] Exact layer %d/%d masks=%d states=%d "
                "max_frontier=%d layer_sec=%.3f total_sec=%.3f",
                cardinality,
                self.n,
                len(layer_masks),
                sum(
                    sum(len(values) for values in dp[mask].values())
                    for mask in layer_masks
                ),
                max(
                    (
                        sum(len(values) for values in dp[mask].values())
                        for mask in layer_masks
                    ),
                    default=0,
                ),
                time.monotonic() - layer_started,
                time.monotonic() - search_started,
            )

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
            "objective": "max_of_additive_resource_family_sums",
            "legal_prefix_masks": len(legal_masks),
            "retained_states": sum(state_counts),
            "maximum_frontier": max(state_counts, default=0),
            "parallel_stage_cpu_limit": (
                self.parallel_stage_cpu_limit.as_dict()
                if self.parallel_stage_cpu_limit is not None
                else None
            ),
            "required_final_parallel_stage_cpus": (
                self.required_final_parallel_stage_cpus.as_dict()
                if self.required_final_parallel_stage_cpus is not None
                else None
            ),
            "final_resource_states": [
                state.parallel_stage_cpus.as_dict()
                for state in dp[final_mask]
            ],
            "pareto_global_epsilon": self.pareto_global_epsilon,
            "pareto_step_epsilon": self.pareto_step_epsilon,
            "transition_pairs": self._transition_pairs,
            "upper_bound_score": (
                incumbent_score if math.isfinite(incumbent_score) else None
            ),
            "upper_bound_pruned": self._upper_bound_pruned,
            "suffix_lower_bound_pruned": self._suffix_lower_bound_pruned,
            "cell_candidates": self._cell_candidates,
            "cell_replacements": self._cell_replacements,
            "frontier_cap": self.frontier_cap,
            "frontier_cap_pruned": self._frontier_cap_pruned,
            "final_local_serial": final_objective.local_serial,
            "final_ray_serial": final_objective.ray_serial,
            "final_smp_serial": final_objective.smp_serial,
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

    def _prepare_mandatory_suffix_lower_bounds(
        self, legal_masks: Iterable[int]
    ) -> None:
        """Precompute an admissible lower bound on unfinished local work.

        Some operators have no feasible Ray/SMP/GPU implementation in the
        profiled search space and therefore must eventually execute on the
        additive local lane.  Fusion does not discount operator compute, so
        the sum of each such operator's cheapest legal-prefix local cost is a
        lower bound for every completion of a prefix.  The bound is used only
        for incumbent pruning and cannot remove an optimal plan.
        """
        self._mandatory_local_suffix_cost = {}
        if getattr(self.cache_policy, "enabled", False):
            return
        indexes = getattr(self.block_provider, "_variant_indexes", {})
        local_index = indexes.get((PipeVariantType.INPROCESS, 1))
        if local_index is None:
            return
        candidate_variants = getattr(
            self.block_provider, "_candidate_variants", ()
        )
        legal = tuple(legal_masks)
        mandatory_cost = [0.0] * self.n
        for idx in range(self.n):
            bit = 1 << idx
            execution_resource = self.block_provider._execution_resource_for_mask(
                bit
            )
            if execution_resource != PipeExecutionResource.CPU:
                continue
            has_nonlocal_implementation = False
            for variant in candidate_variants:
                if variant == PipeVariantType.INPROCESS:
                    continue
                for (indexed_variant, _), index in indexes.items():
                    if (
                        indexed_variant == variant
                        and math.isfinite(index.per_byte[idx])
                    ):
                        has_nonlocal_implementation = True
                        break
                if has_nonlocal_implementation:
                    break
            if has_nonlocal_implementation or not math.isfinite(
                local_index.per_byte[idx]
            ):
                continue
            minimum_work = min(
                (
                    _work_prod(self.optimizer, prefix, idx)
                    for prefix in legal
                    if not prefix & bit
                    and self.optimizer._dp_valid_single_last(prefix, idx)
                ),
                default=0.0,
            )
            mandatory_cost[idx] = (
                local_index.per_byte[idx]
                * local_index.source_size
                * minimum_work
            )

        full_mask = self.full_mask - 1
        for prefix in legal:
            remaining = full_mask ^ prefix
            lower_bound = 0.0
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                lower_bound += mandatory_cost[bit.bit_length() - 1]
            self._mandatory_local_suffix_cost[prefix] = lower_bound

    def _exceeds_incumbent_with_suffix_bound(
        self,
        next_mask: int,
        local_serial: float,
        ray_serial: float,
        smp_serial: float,
        gpu_serial: float,
        incumbent_score: float,
    ) -> bool:
        """Return whether a partial label cannot beat a feasible incumbent."""
        if not math.isfinite(incumbent_score):
            return False
        unavoidable_local = self._mandatory_local_suffix_cost.get(
            next_mask, 0.0
        )
        return max(
            local_serial + unavoidable_local,
            ray_serial,
            smp_serial,
            gpu_serial,
        ) > incumbent_score + 1e-12

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
                ray_serial=previous.ray_serial,
                smp_serial=previous.smp_serial,
                gpu_serial=previous.gpu_serial,
            )
        return method(previous, extra_cost, block, prev_mask)

    def _dominates(
        self, left: DpObjectiveCost, right: DpObjectiveCost
    ) -> bool:
        tolerance = 1e-12
        if not self.collapse_external_service_coordinates:
            return (
                left.local_serial <= right.local_serial + tolerance
                and left.ray_serial <= right.ray_serial + tolerance
                and left.smp_serial <= right.smp_serial + tolerance
                and left.gpu_serial <= right.gpu_serial + tolerance
            )
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
        if not self.collapse_external_service_coordinates:
            forward = (
                left.local_serial <= factor * right.local_serial
                and left.ray_serial <= factor * right.ray_serial
                and left.smp_serial <= factor * right.smp_serial
                and left.gpu_serial <= factor * right.gpu_serial
            )
            if not forward:
                return False
            reverse = (
                right.local_serial <= factor * left.local_serial
                and right.ray_serial <= factor * left.ray_serial
                and right.smp_serial <= factor * left.smp_serial
                and right.gpu_serial <= factor * left.gpu_serial
            )
            if not reverse:
                return True
            return self._objective_order_key(left) <= self._objective_order_key(
                right
            )
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
        """Generate dependency-closed subsets without scanning ``2**N``.

        Starting from the empty ideal, add only operators whose predecessor
        masks are already present. Fixed-position operators are represented by
        dependency edges, so long fixed prefixes/suffixes reduce work in
        direct proportion to the number of actually reachable prefix sets.
        """
        pred_masks = []
        for predecessors in self.optimizer._dp_pred_indices:
            mask = 0
            for predecessor in predecessors:
                mask |= 1 << predecessor
            pred_masks.append(mask)

        discovered = {0}
        frontier = [0]
        while frontier:
            mask = frontier.pop()
            for idx, predecessors in enumerate(pred_masks):
                bit = 1 << idx
                if mask & bit or predecessors & ~mask:
                    continue
                candidate = mask | bit
                if candidate not in discovered:
                    discovered.add(candidate)
                    frontier.append(candidate)
        # Numeric order preserves the deterministic predecessor ordering used
        # by the former exhaustive scan and therefore plan tie behavior.
        return sorted(discovered)

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
            return self.initial_incumbent_score
        candidate_for_order = getattr(
            self.block_provider, "candidate_for_order", None
        )
        candidate_variants = getattr(
            self.block_provider, "_candidate_variants", None
        )
        if candidate_for_order is None or not candidate_variants:
            return self.initial_incumbent_score

        restricted_start = time.monotonic()
        restricted_result = self._fixed_order_physical_incumbent()
        if os.environ.get("CEDAR_DP_LEGACY_SUBSET_INCUMBENT", "0") == "1":
            legacy_result = self._single_parallel_block_incumbent()
            if (
                legacy_result is not None
                and (
                    restricted_result is None
                    or legacy_result.cost < restricted_result.cost
                )
            ):
                restricted_result = legacy_result
        restricted_elapsed = time.monotonic() - restricted_start
        if restricted_result is not None:
            self._greedy_full_block_result = restricted_result
            logger.info(
                "[DpOptimizer] Fixed-order feasible incumbent cost=%s "
                "elapsed_sec=%.3f blocks=%s",
                restricted_result.cost,
                restricted_elapsed,
                restricted_result.blocks,
            )
        else:
            logger.info(
                "[DpOptimizer] Fixed-order feasible incumbent unavailable "
                "elapsed_sec=%.3f",
                restricted_elapsed,
            )

        full_mask = self.full_mask - 1
        initial_state = self.cache_policy.initial_state()
        initial_objective = self._initial_objective()
        best = (
            restricted_result.cost
            if restricted_result is not None
            else float("inf")
        )
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
        return min(best, self.initial_incumbent_score)

    def _fixed_order_physical_incumbent(self) -> Optional[SearchResult]:
        """Construct a strong feasible upper bound in a small plan subspace.

        The exact subset DP is still responsible for the final answer.  This
        routine first obtains several legal orders (the original topological
        order plus cheap compute-greedy orders), then performs a bounded
        fixed-order search over fusion, placement, and integer width. Every
        retained label is a fully feasible physical plan, so even though this
        preliminary search is deliberately small, its best score is a safe
        branch-and-bound incumbent and does not weaken exactness.
        """
        provider = self.block_provider
        candidate_for_order = getattr(provider, "candidate_for_order", None)
        variants = tuple(getattr(provider, "_candidate_variants", ()))
        indexes = getattr(provider, "_variant_indexes", {})
        resource_limit = self.parallel_stage_cpu_limit
        if (
            candidate_for_order is None
            or not variants
            or resource_limit is None
        ):
            return None

        full_mask = self.full_mask - 1
        orders: List[Tuple[int, ...]] = []

        def add_order(order: Iterable[int]) -> None:
            value = tuple(order)
            if (
                len(value) == self.n
                and len(set(value)) == self.n
                and value not in orders
            ):
                orders.append(value)

        add_order(self.optimizer._dp_topo_order_in_mask(full_mask))
        # A greedy order is O(N^2), dependency aware, and provides useful
        # diversity without enumerating the logical-order search space.
        width_one_indexes = [
            index
            for (variant, width), index in indexes.items()
            if width == 1 and variant in variants
        ]
        for index in width_one_indexes:
            prefix = 0
            order = []
            while len(order) < self.n:
                available = [
                    idx
                    for idx in range(self.n)
                    if not prefix & (1 << idx)
                    and self.optimizer._dp_valid_single_last(prefix, idx)
                    and math.isfinite(index.per_byte[idx])
                ]
                if not available:
                    order = []
                    break
                selected = min(
                    available,
                    key=lambda idx: (
                        _work_prod(self.optimizer, prefix, idx)
                        * index.per_byte[idx],
                        idx,
                    ),
                )
                order.append(selected)
                prefix |= 1 << selected
            add_order(order)

        try:
            beam_width = int(os.environ.get("CEDAR_DP_INCUMBENT_BEAM", "48"))
        except ValueError as exc:
            raise ValueError("CEDAR_DP_INCUMBENT_BEAM must be an integer") from exc
        if beam_width < 1:
            raise ValueError("CEDAR_DP_INCUMBENT_BEAM must be positive")

        best_result: Optional[SearchResult] = None
        initial_state = self.cache_policy.initial_state()
        initial_objective = self._initial_objective()

        def retain(labels):
            unique = {}
            for state, objective, blocks in labels:
                key = (state, objective)
                old = unique.get(key)
                if old is None or len(blocks) < len(old[2]):
                    unique[key] = (state, objective, blocks)
            return sorted(
                unique.values(),
                key=lambda item: self._objective_order_key(item[1]),
            )[:beam_width]

        for order in orders:
            prefix_masks = [0]
            for idx in order:
                prefix_masks.append(prefix_masks[-1] | (1 << idx))
            block_cache: Dict[Tuple[int, int], List[BlockCandidate]] = {}
            beams = [[] for _ in range(self.n + 1)]
            beams[0] = [(initial_state, initial_objective, tuple())]

            for start in range(self.n):
                if not beams[start]:
                    continue
                beams[start] = retain(beams[start])
                prev_mask = prefix_masks[start]
                for end in range(start + 1, self.n + 1):
                    cache_key = (start, end)
                    blocks = block_cache.get(cache_key)
                    if blocks is None:
                        block_order = order[start:end]
                        block_mask = prefix_masks[end] ^ prev_mask
                        execution_resource = (
                            provider._execution_resource_for_mask(block_mask)
                        )
                        blocks = []
                        for variant in variants:
                            if (
                                execution_resource
                                == PipeExecutionResource.CUDA
                                and variant
                                not in (
                                    PipeVariantType.RAY,
                                    PipeVariantType.TF_RAY,
                                )
                            ):
                                continue
                            for width in self.optimizer._dp_candidate_parallelisms(
                                variant, execution_resource
                            ):
                                try:
                                    block = candidate_for_order(
                                        block_order,
                                        variant,
                                        prefix_mask=prev_mask,
                                        parallelism=width,
                                    )
                                except ValueError:
                                    continue
                                if self._block_can_follow(prev_mask, block):
                                    blocks.append(block)
                        blocks = self._prune_dominated_blocks(prev_mask, blocks)
                        block_cache[cache_key] = blocks
                    if not blocks:
                        continue
                    destination = beams[end]
                    for state, objective, chosen in beams[start]:
                        for block in blocks:
                            next_usage = (
                                state.parallel_stage_cpus
                                + self.optimizer._dp_parallel_stage_cpu_cost(
                                    block
                                )
                            )
                            if next_usage > resource_limit:
                                continue
                            regular_cost = self.optimizer._dp_regular_transition_cost(
                                prev_mask, block
                            )
                            next_objective = self._accumulate_objective(
                                objective,
                                regular_cost,
                                block,
                                False,
                                prev_mask,
                            )
                            if (
                                best_result is not None
                                and next_objective.score
                                > best_result.cost + 1e-12
                            ):
                                continue
                            destination.append(
                                (
                                    DpStateSummary(
                                        cache_active=False,
                                        parallel_stage_cpus=next_usage,
                                    ),
                                    next_objective,
                                    chosen + (block,),
                                )
                            )
                    if len(destination) > 4 * beam_width:
                        beams[end] = retain(destination)

            if not beams[self.n]:
                continue
            state, objective, blocks = min(
                retain(beams[self.n]),
                key=lambda item: self._objective_order_key(item[1]),
            )
            del state
            if best_result is not None and objective.score >= best_result.cost:
                continue
            result_order: List[int] = []
            result_blocks: List[List[int]] = []
            variants_by_idx = {}
            parallelism_by_idx = {}
            for block in blocks:
                block_order = list(block.order)
                result_order.extend(block_order)
                result_blocks.append(block_order)
                for idx in block.order:
                    variants_by_idx[idx] = block.variant
                    parallelism_by_idx[idx] = block.parallelism
            best_result = SearchResult(
                order=result_order,
                blocks=result_blocks,
                variants_by_idx=variants_by_idx,
                parallelism_by_idx=parallelism_by_idx,
                cache_after_idx=None,
                cost=objective.score,
                objective=objective,
            )
        return best_result

    def _single_parallel_block_incumbent(self) -> Optional[SearchResult]:
        """Find a strong feasible bound in an O(MP2**N) subspace.

        The restricted space contains every legal plan with exactly one
        parallel block placed either before or after a sequence of local
        singleton operators.  It is used only as an upper bound for the full
        exact DP; no label below the returned score is discarded.
        """
        resource_limit = self.parallel_stage_cpu_limit
        if resource_limit is None:
            # The relaxed main DP deliberately drops resource coordinates, but
            # its pruning incumbent must still be a physically feasible plan.
            # Use the configured pool capacities solely to validate this upper
            # bound; doing so cannot exclude any plan below the bound.
            resource_limit = self.optimizer._dp_parallel_stage_cpu_limit()
        if resource_limit is None or (
            resource_limit.ray_cpus <= 0
            and resource_limit.smp_cpus <= 0
        ):
            return None
        provider = self.block_provider
        candidate_for_order = getattr(provider, "candidate_for_order", None)
        candidate_for_prefix = getattr(
            provider, "candidates_for_prefix", None
        )
        variant_indexes = getattr(provider, "_variant_indexes", {})
        local_index = variant_indexes.get((PipeVariantType.INPROCESS, 1))
        if (
            candidate_for_order is None
            or candidate_for_prefix is None
            or local_index is None
        ):
            return None

        full_mask = self.full_mask - 1
        legal_masks, _ = self._search_topology()
        parallel_variants = {
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
            PipeVariantType.SMP,
        }
        best_result: Optional[SearchResult] = None

        def local_singletons(mask: int, prefix_mask: int):
            if mask == 0:
                return []
            cost, order = local_index.get(mask, prefix_mask)
            if not math.isfinite(cost) or not order:
                return None
            blocks = []
            current_prefix = prefix_mask
            for idx in order:
                try:
                    block = candidate_for_order(
                        (idx,),
                        PipeVariantType.INPROCESS,
                        prefix_mask=current_prefix,
                        parallelism=1,
                    )
                except ValueError:
                    return None
                if not self._block_can_follow(current_prefix, block):
                    return None
                blocks.append(block)
                current_prefix |= block.mask
            return blocks

        def evaluate(blocks: List[BlockCandidate]) -> None:
            nonlocal best_result
            state = self.cache_policy.initial_state()
            objective = self._initial_objective()
            prev_mask = 0
            for block in blocks:
                if not self._block_can_follow(prev_mask, block):
                    return
                next_mask = prev_mask | block.mask
                next_parallel_cpus = (
                    state.parallel_stage_cpus
                    + self.optimizer._dp_parallel_stage_cpu_cost(block)
                )
                if next_parallel_cpus > resource_limit:
                    return
                regular_cost = self.optimizer._dp_regular_transition_cost(
                    prev_mask, block
                )
                choices = [
                    choice
                    for choice in self.cache_policy.transitions(
                        prev_mask,
                        next_mask,
                        state,
                        regular_cost,
                        block,
                        next_parallel_cpus,
                    )
                    if choice.cache_after_idx is None
                ]
                if len(choices) != 1:
                    return
                choice = choices[0]
                objective = self._accumulate_objective(
                    objective,
                    choice.extra_cost,
                    block,
                    choice.replaces_prefix_cost,
                    prev_mask,
                )
                state = choice.state
                prev_mask = next_mask
            if (
                prev_mask != full_mask
                or not math.isfinite(objective.score)
            ):
                return
            if best_result is not None and objective.score >= best_result.cost:
                return
            variants_by_idx = {}
            parallelism_by_idx = {}
            result_blocks = []
            order = []
            for block in blocks:
                block_order = list(block.order)
                result_blocks.append(block_order)
                order.extend(block_order)
                for idx in block.order:
                    variants_by_idx[idx] = block.variant
                    parallelism_by_idx[idx] = block.parallelism
            best_result = SearchResult(
                order=order,
                blocks=result_blocks,
                variants_by_idx=variants_by_idx,
                parallelism_by_idx=parallelism_by_idx,
                cache_after_idx=None,
                cost=objective.score,
                objective=objective,
            )

        def parallel_candidates(prefix_mask: int, block_mask: int):
            if block_mask == 0:
                return []
            return [
                block
                for block in candidate_for_prefix(prefix_mask, block_mask)
                if block.variant in parallel_variants
                and not (
                    self.optimizer._dp_parallel_stage_cpu_cost(block)
                    > resource_limit
                )
            ]

        # Parallel block first, followed by the best legal local order.
        for parallel_mask in legal_masks[1:]:
            suffix_mask = full_mask ^ parallel_mask
            suffix = local_singletons(suffix_mask, parallel_mask)
            if suffix is None:
                continue
            for parallel_block in parallel_candidates(0, parallel_mask):
                evaluate([parallel_block] + suffix)

        # Best legal local prefix followed by one parallel block.
        for prefix_mask in legal_masks[:-1]:
            parallel_mask = full_mask ^ prefix_mask
            prefix = local_singletons(prefix_mask, 0)
            if prefix is None:
                continue
            for parallel_block in parallel_candidates(
                prefix_mask, parallel_mask
            ):
                evaluate(prefix + [parallel_block])

        return best_result

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
            # The non-cache transition vector depends only on the prefix and
            # block, not on a retained Pareto label.  In particular, boundary
            # calibration used to be recomputed once per label here, which is
            # hundreds of millions of identical calls for a 19-op workload.
            # Compute the vector once, then add it to the corresponding
            # resource-family service lane for every retained prefix label.
            objective_delta = self._accumulate_objective(
                DpObjectiveCost(),
                regular_cost,
                block,
                False,
                prev_mask,
            )
            if objective_delta.score > incumbent_score + 1e-12:
                # Every objective coordinate is monotone. A block that alone
                # exceeds a feasible complete-plan bound cannot extend any
                # retained prefix, so reject it before the label cross product.
                self._upper_bound_pruned += 1
                continue
            for prev_state, prev_objectives in dp[prev_mask].items():
                if self.parallel_stage_cpu_limit is None:
                    # Resource use cannot affect feasibility or any future
                    # transition when strict matching is disabled. Collapse
                    # this otherwise irrelevant state coordinate so plans
                    # with different accumulated stage widths can dominate
                    # one another by score.
                    next_parallel_stage_cpus = DpResourceUsage()
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
                        if choice.replaces_prefix_cost:
                            # A cache hit discards every upstream service
                            # coordinate and resumes on the local lane.
                            local_serial = choice.extra_cost
                            ray_serial = 0.0
                            smp_serial = 0.0
                            gpu_serial = 0.0
                        else:
                            local_serial = (
                                prev_objective.local_serial
                                + objective_delta.local_serial
                            )
                            ray_serial = (
                                prev_objective.ray_serial
                                + objective_delta.ray_serial
                            )
                            smp_serial = (
                                prev_objective.smp_serial
                                + objective_delta.smp_serial
                            )
                            gpu_serial = (
                                prev_objective.gpu_serial
                                + objective_delta.gpu_serial
                            )
                        # All coordinates are non-negative.  Test the safe
                        # incumbent bound before allocating a frozen objective
                        # object; most complex-workload candidates die here.
                        if self._exceeds_incumbent_with_suffix_bound(
                            next_mask,
                            local_serial,
                            ray_serial,
                            smp_serial,
                            gpu_serial,
                            incumbent_score,
                        ):
                            self._upper_bound_pruned += 1
                            if (
                                local_serial <= incumbent_score + 1e-12
                                and ray_serial <= incumbent_score + 1e-12
                                and smp_serial <= incumbent_score + 1e-12
                                and gpu_serial <= incumbent_score + 1e-12
                            ):
                                self._suffix_lower_bound_pruned += 1
                            continue
                        candidate = DpObjectiveCost(
                            local_serial=local_serial,
                            ray_serial=ray_serial,
                            smp_serial=smp_serial,
                            gpu_serial=gpu_serial,
                        )
                        if (
                            self.required_final_parallel_stage_cpus is not None
                            and next_mask == self.full_mask - 1
                            and choice.state.parallel_stage_cpus
                            != self.required_final_parallel_stage_cpus
                        ):
                            # No suffix remains that could repair an incomplete
                            # resource reservation.  The old implementation
                            # retained these labels and discarded them only
                            # after the entire final frontier was built.
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
                                        old.ray_serial,
                                        candidate.ray_serial,
                                        rel_tol=1e-12,
                                        abs_tol=1e-12,
                                    )
                                    and math.isclose(
                                        old.smp_serial,
                                        candidate.smp_serial,
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
                                    old.ray_serial,
                                    candidate.ray_serial,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                                and math.isclose(
                                    old.smp_serial,
                                    candidate.smp_serial,
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

    def _can_use_exact_batch_frontier(self) -> bool:
        """Whether a mask can use the exact offline skyline implementation."""
        return (
            getattr(self.cache_policy, "enabled", None) is False
            and self.pareto_step_epsilon == 0.0
            and self.frontier_cap == 0
        )

    @staticmethod
    def _prefer_pointer(
        current: BackPointer, candidate: BackPointer
    ) -> BackPointer:
        """Preserve the legacy deterministic tie preference for fusion."""
        current_width = (
            current.block.mask.bit_count()
            if current.block.materializes_fusion
            else 0
        )
        candidate_width = (
            candidate.block.mask.bit_count()
            if candidate.block.materializes_fusion
            else 0
        )
        return candidate if candidate_width > current_width else current

    def _exact_skyline(
        self,
        candidates: Dict[DpObjectiveCost, BackPointer],
    ) -> List[Tuple[DpObjectiveCost, BackPointer]]:
        """Return the complete non-dominated objective set.

        Separate Ray/SMP resource use remains in :class:`DpStateSummary`.
        Service demand is suffix-equivalent under ``E=max(R,S)``, reducing a
        CPU-only frontier to the two coordinates ``(L,E)``. Sorting by local
        demand and retaining a strictly decreasing external bottleneck gives
        the exact skyline in ``O(K log K)``.
        """
        if not self.collapse_external_service_coordinates:
            items = list(candidates.items())
            retained = []
            for objective, pointer in items:
                if any(
                    self._dominates(other, objective)
                    for other, _ in items
                    if other != objective
                ):
                    continue
                retained.append((objective, pointer))
            return retained

        effective = {}
        for objective, pointer in candidates.items():
            canonical = DpObjectiveCost(
                local_serial=objective.local_serial,
                ray_serial=objective.parallel_bottleneck,
                smp_serial=objective.parallel_bottleneck,
                gpu_serial=objective.gpu_serial,
            )
            old = effective.get(canonical)
            if old is None:
                effective[canonical] = pointer
            elif isinstance(old, BackPointer) and isinstance(
                pointer, BackPointer
            ):
                effective[canonical] = self._prefer_pointer(old, pointer)
            else:
                # Prefix resource skylines store ``(state, pointer)`` values.
                # Equal effective objectives are suffix-equivalent; retaining
                # the first deterministic representative is sufficient.
                effective[canonical] = old
        items = list(effective.items())
        if not items:
            return []
        if any(objective.gpu_serial != 0.0 for objective, _ in items):
            # GPU workloads are much smaller. Keep the generic exact 3-D rule.
            retained = []
            for objective, pointer in items:
                if any(
                    self._dominates(old_objective, objective)
                    for old_objective, _ in items
                    if old_objective is not objective
                ):
                    continue
                retained.append((objective, pointer))
            return retained

        items.sort(
            key=lambda item: (
                item[0].local_serial,
                item[0].parallel_bottleneck,
            )
        )
        retained = []
        tolerance = 1e-12
        best_parallel = float("inf")
        for objective, pointer in items:
            if best_parallel <= objective.parallel_bottleneck + tolerance:
                continue
            retained.append((objective, pointer))
            best_parallel = objective.parallel_bottleneck
        return retained

    def _compute_exact_batch_frontier(
        self,
        dp: List[Dict[DpStateSummary, List[DpObjectiveCost]]],
        back: List[
            Dict[Tuple[DpStateSummary, DpObjectiveCost], BackPointer]
        ],
        next_mask: int,
        prev_masks: Iterable[int],
        incumbent_score: float,
    ) -> None:
        """Generate one mask and compute its exact Pareto set in one batch."""
        collected: Dict[
            DpStateSummary, Dict[DpObjectiveCost, BackPointer]
        ] = {}
        final_mask = self.full_mask - 1
        required_cpus = self.required_final_parallel_stage_cpus

        for prev_mask in prev_masks:
            if not dp[prev_mask]:
                continue
            block_mask = next_mask ^ prev_mask
            if not block_mask:
                continue
            self._transition_pairs += 1
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
                delta = self._accumulate_objective(
                    DpObjectiveCost(),
                    regular_cost,
                    block,
                    False,
                    prev_mask,
                )
                if delta.score > incumbent_score + 1e-12:
                    self._upper_bound_pruned += 1
                    continue
                block_cpus = self.optimizer._dp_parallel_stage_cpu_cost(block)
                for prev_state, prev_objectives in dp[prev_mask].items():
                    if self.parallel_stage_cpu_limit is None:
                        next_cpus = DpResourceUsage()
                    else:
                        next_cpus = (
                            prev_state.parallel_stage_cpus + block_cpus
                        )
                        if next_cpus > self.parallel_stage_cpu_limit:
                            continue
                    if (
                        required_cpus is not None
                        and next_mask == final_mask
                        and next_cpus != required_cpus
                    ):
                        continue
                    next_state = DpStateSummary(
                        cache_active=False,
                        parallel_stage_cpus=next_cpus,
                    )
                    state_candidates = collected.setdefault(next_state, {})
                    for previous in prev_objectives:
                        local_serial = previous.local_serial + delta.local_serial
                        ray_serial = previous.ray_serial + delta.ray_serial
                        smp_serial = previous.smp_serial + delta.smp_serial
                        gpu_serial = previous.gpu_serial + delta.gpu_serial
                        if self._exceeds_incumbent_with_suffix_bound(
                            next_mask,
                            local_serial,
                            ray_serial,
                            smp_serial,
                            gpu_serial,
                            incumbent_score,
                        ):
                            self._upper_bound_pruned += 1
                            if (
                                local_serial <= incumbent_score + 1e-12
                                and ray_serial <= incumbent_score + 1e-12
                                and smp_serial <= incumbent_score + 1e-12
                                and gpu_serial <= incumbent_score + 1e-12
                            ):
                                self._suffix_lower_bound_pruned += 1
                            continue
                        objective = DpObjectiveCost(
                            local_serial=local_serial,
                            ray_serial=ray_serial,
                            smp_serial=smp_serial,
                            gpu_serial=gpu_serial,
                        )
                        pointer = BackPointer(
                            prev_mask=prev_mask,
                            prev_state=prev_state,
                            prev_objective=previous,
                            block=block,
                            cache_after_idx=None,
                        )
                        old_pointer = state_candidates.get(objective)
                        if old_pointer is None:
                            state_candidates[objective] = pointer
                        else:
                            state_candidates[objective] = self._prefer_pointer(
                                old_pointer, pointer
                            )

        # A relaxed search deliberately drops resource counts. If its winner
        # fits the physical pools, it certifies the constrained global optimum
        # because it is optimal over a superset. Only an infeasible winner
        # requires the more expensive resource-state fallback below.
        if self.parallel_stage_cpu_limit is None:
            for state, candidates in collected.items():
                skyline = self._exact_skyline(candidates)
                if not skyline:
                    continue
                dp[next_mask][state] = [
                    objective for objective, _ in skyline
                ]
                for objective, pointer in skyline:
                    back[next_mask][(state, objective)] = pointer
            return

        # First compute the objective skyline for each exact resource state,
        # then apply exact dominance across the two resource dimensions.  A
        # label using no more Ray CPUs and no more SMP CPUs can execute every
        # suffix available to a more resource-hungry label.  Without this
        # second pass, splitting one scalar budget into two physical pools
        # retains nearly every one of the (R+1)(S+1) states at every mask.
        state_skylines = {
            state: self._exact_skyline(candidates)
            for state, candidates in collected.items()
        }
        limits = self.parallel_stage_cpu_limit
        prefix_skylines: Dict[
            Tuple[int, int],
            Dict[
                DpObjectiveCost,
                Tuple[DpStateSummary, BackPointer],
            ],
        ] = {}
        for ray_cpus in range(limits.ray_cpus + 1):
            for smp_cpus in range(limits.smp_cpus + 1):
                usage = DpResourceUsage(ray_cpus, smp_cpus)
                state = DpStateSummary(
                    cache_active=False,
                    parallel_stage_cpus=usage,
                )
                prior: Dict[
                    DpObjectiveCost,
                    Tuple[DpStateSummary, BackPointer],
                ] = {}
                if ray_cpus > 0:
                    prior.update(prefix_skylines[(ray_cpus - 1, smp_cpus)])
                if smp_cpus > 0:
                    for objective, value in prefix_skylines[
                        (ray_cpus, smp_cpus - 1)
                    ].items():
                        prior.setdefault(objective, value)

                exact = []
                for objective, pointer in state_skylines.get(state, []):
                    if any(
                        self._dominates(old, objective)
                        for old in prior
                    ):
                        continue
                    exact.append((objective, pointer))

                if exact:
                    dp[next_mask][state] = [
                        objective for objective, _ in exact
                    ]
                    for objective, pointer in exact:
                        back[next_mask][(state, objective)] = pointer

                combined = dict(prior)
                for objective, pointer in exact:
                    combined.setdefault(objective, (state, pointer))
                prefix_skylines[(ray_cpus, smp_cpus)] = dict(
                    self._exact_skyline(combined)
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
        ordered = sorted(frontier, key=self._objective_order_key)
        best_index = min(
            range(len(ordered)),
            key=lambda index: self._objective_order_key(ordered[index]),
        )
        selected = {best_index}
        coordinates = (
            lambda objective: objective.local_serial,
            lambda objective: objective.ray_serial,
            lambda objective: objective.smp_serial,
            lambda objective: objective.gpu_serial,
        )
        for coordinate in coordinates:
            for chooser in (min, max):
                if len(selected) >= self.frontier_cap:
                    break
                selected.add(
                    chooser(
                        range(len(ordered)),
                        key=lambda index: coordinate(ordered[index]),
                    )
                )
        if len(selected) < self.frontier_cap:
            for index in range(len(ordered)):
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
    ) -> Tuple[Optional[int], ...]:
        log_base = math.log1p(self.pareto_step_epsilon)

        def coordinate(value: float) -> Optional[int]:
            if value <= 1e-12:
                return None
            return math.floor(math.log(value) / log_base)

        return (
            coordinate(objective.local_serial),
            coordinate(objective.ray_serial),
            coordinate(objective.smp_serial),
            coordinate(objective.gpu_serial),
        )

    @staticmethod
    def _objective_order_key(
        objective: DpObjectiveCost,
    ) -> Tuple[float, ...]:
        return (
            objective.score,
            objective.local_serial,
            objective.ray_serial,
            objective.smp_serial,
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
        """Remove exact transition-vector and resource dominance.

        Once a block has covered ``block.mask``, its internal backend and
        order are absent from the future DP state.  Therefore, among blocks
        a block that reserves no fewer Ray/SMP CPUs and has a component-wise
        no-better local/Ray/SMP/GPU service vector can never lead to a better
        suffix.  When an API caller requests an exact final resource total we
        conservatively compare only equal-resource candidates. Cache
        transitions retain all choices because the physical cache marker
        records the block's terminal pipe.
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
                    0.0,
                    self.optimizer._dp_gpu_worker_multiplier()
                    * (block.cost + boundary_parallel),
                )
            elif block.variant in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            ):
                vector = (
                    boundary_local,
                    block.cost / block.parallelism + boundary_parallel,
                    0.0,
                    0.0,
                )
            elif block.variant == PipeVariantType.SMP:
                vector = (
                    boundary_local,
                    0.0,
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
                if index == other_index:
                    continue
                if self.required_final_parallel_stage_cpus is not None:
                    resource_dominates = other_resource == resource_cost
                else:
                    resource_dominates = other_resource <= resource_cost
                if not resource_dominates:
                    continue
                no_worse = all(
                    left <= right + tolerance
                    for left, right in zip(other_vector, vector)
                )
                strictly_better = any(
                    left < right - tolerance
                    for left, right in zip(other_vector, vector)
                )
                strictly_less_resource = other_resource != resource_cost
                if no_worse and (
                    strictly_better
                    or strictly_less_resource
                    or other_index < index
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


class ThresholdFeasibilityDpSearch:
    """Move integer Ray/SMP width choice out of the subset-DP frontier.

    For a fixed makespan threshold ``T``, every external CPU stage only has to
    satisfy ``stage_service <= T``.  Among alternatives with the same logical
    block, any wider feasible implementation that does not improve local work
    is useless: it cannot improve feasibility and only consumes more of its
    independent CPU pool.  The decision DP therefore retains only local
    additive work and exact Ray/SMP resource usage.  A monotone search over
    ``T`` recovers an epsilon-certified minimum makespan without enumerating
    width as another Pareto objective coordinate.

    The current implementation is deliberately enabled only for the formal
    CPU-only, cache-disabled problem.  Cache resets and additive GPU demand
    require extra decision-state coordinates and continue to use the general
    exact search until those cases receive an equally explicit proof.
    """

    def __init__(
        self,
        optimizer: "DpOptimizer",
        inner_ops: List[int],
        block_provider: BlockCandidateProvider,
        cache_policy: CacheTransitionPolicy,
        parallel_stage_cpu_limit: Optional[DpResourceUsage],
    ) -> None:
        self.optimizer = optimizer
        self.inner_ops = inner_ops
        self.n = len(inner_ops)
        self.full_mask = 1 << self.n
        self.block_provider = block_provider
        self.cache_policy = cache_policy
        self.parallel_stage_cpu_limit = parallel_stage_cpu_limit
        self._general_search = ExtensibleDpSearch(
            optimizer=optimizer,
            inner_ops=inner_ops,
            block_provider=block_provider,
            cache_policy=cache_policy,
            parallel_stage_cpu_limit=parallel_stage_cpu_limit,
        )
        self._decision_calls = 0
        self._transition_pairs = 0
        self._candidate_blocks = 0
        self._resource_dominated = 0
        self._best_feasible_result: Optional[SearchResult] = None
        self._current_lower_bound = 0.0
        self._current_upper_bound = math.inf
        self._incumbent_seconds = 0.0
        self._search_started: Optional[float] = None

    def supported(self) -> bool:
        if getattr(
            self.optimizer,
            "external_stage_cost_aggregation",
            "sum",
        ) != "max":
            # The compact feasibility state keeps no accumulated Ray/SMP
            # service. It is lossless only for the former maximum-stage
            # objective; additive lanes require the general Pareto DP.
            return False
        if self.parallel_stage_cpu_limit is None:
            return False
        if getattr(self.cache_policy, "enabled", False):
            return False
        logical_pipes = self.optimizer.logical_pipes or {}
        return not any(
            logical_pipes[p_id].execution_resource
            == PipeExecutionResource.CUDA
            for p_id in self.inner_ops
        )

    def run(self) -> SearchResult:
        if not self.supported():
            raise RuntimeError(
                "Threshold feasibility DP supports only constrained, "
                "CPU-only, cache-disabled searches."
            )

        incumbent_started = time.monotonic()
        incumbent = self._general_search._fixed_order_physical_incumbent()
        if incumbent is None or not math.isfinite(incumbent.cost):
            raise RuntimeError(
                "Threshold feasibility DP could not construct a feasible "
                "fixed-order upper bound."
            )
        incumbent_seconds = time.monotonic() - incumbent_started
        self._incumbent_seconds = incumbent_seconds
        logger.info(
            "[DpOptimizer] Threshold fixed-order incumbent cost=%.9f sec=%.3f",
            incumbent.cost,
            incumbent_seconds,
        )

        iterations_raw = os.environ.get(
            "CEDAR_DP_THRESHOLD_ITERATIONS", "18"
        )
        tolerance_raw = os.environ.get(
            "CEDAR_DP_THRESHOLD_REL_TOL", "1e-5"
        )
        try:
            max_iterations = int(iterations_raw)
            relative_tolerance = float(tolerance_raw)
        except ValueError as exc:
            raise RuntimeError(
                "Threshold feasibility iteration/tolerance settings are invalid."
            ) from exc
        if max_iterations < 1:
            raise RuntimeError("CEDAR_DP_THRESHOLD_ITERATIONS must be positive.")
        if (
            not math.isfinite(relative_tolerance)
            or relative_tolerance <= 0.0
        ):
            raise RuntimeError(
                "CEDAR_DP_THRESHOLD_REL_TOL must be finite and positive."
            )

        lower = 0.0
        upper = incumbent.cost
        self._current_lower_bound = lower
        self._current_upper_bound = upper
        best = incumbent
        self._best_feasible_result = incumbent
        search_started = time.monotonic()
        self._search_started = search_started
        for iteration in range(1, max_iterations + 1):
            _raise_if_dp_deadline_exceeded(self.optimizer)
            if upper - lower <= relative_tolerance * max(1.0, upper):
                break
            threshold = (lower + upper) / 2.0
            decision_started = time.monotonic()
            candidate = self._solve_decision(threshold)
            decision_seconds = time.monotonic() - decision_started
            if candidate is None:
                lower = threshold
                feasible = False
            else:
                best = candidate
                self._best_feasible_result = candidate
                # The replayed plan is itself a valid (and sometimes tighter)
                # upper bound than the probed threshold.
                upper = min(threshold, candidate.cost)
                feasible = True
            self._current_lower_bound = lower
            self._current_upper_bound = upper
            logger.info(
                "[DpOptimizer] Threshold decision %d/%d T=%.9f "
                "feasible=%s bracket=[%.9f, %.9f] sec=%.3f",
                iteration,
                max_iterations,
                threshold,
                feasible,
                lower,
                upper,
                decision_seconds,
            )

        absolute_gap = max(0.0, upper - lower)
        relative_gap = absolute_gap / max(upper, 1e-12)
        stats = {
            "objective": "threshold_feasibility_resource_allocation",
            "parallel_stage_cpu_limit": self.parallel_stage_cpu_limit.as_dict(),
            "threshold_decisions": self._decision_calls,
            "threshold_lower_bound": lower,
            "threshold_upper_bound": upper,
            "threshold_absolute_gap": absolute_gap,
            "threshold_relative_gap": relative_gap,
            "transition_pairs": self._transition_pairs,
            "candidate_blocks": self._candidate_blocks,
            "resource_dominated": self._resource_dominated,
            "incumbent_seconds": incumbent_seconds,
            "search_seconds": time.monotonic() - search_started,
            "final_local_serial": best.objective.local_serial,
            "final_ray_serial": best.objective.ray_serial,
            "final_smp_serial": best.objective.smp_serial,
            "final_parallel_bottleneck": best.objective.parallel_bottleneck,
            "final_gpu_serial": best.objective.gpu_serial,
        }
        self.optimizer._dp_last_search_stats = stats
        logger.info("[DpOptimizer] Threshold search stats: %s", stats)
        return best

    def timeout_stats(self) -> Dict[str, Any]:
        """Return the proof progress available when the deadline fires."""
        lower = self._current_lower_bound
        upper = self._current_upper_bound
        gap = (
            max(0.0, upper - lower)
            if math.isfinite(upper)
            else math.inf
        )
        return {
            "objective": "threshold_feasibility_resource_allocation",
            "parallel_stage_cpu_limit": self.parallel_stage_cpu_limit.as_dict(),
            "threshold_decisions": self._decision_calls,
            "threshold_lower_bound": lower,
            "threshold_upper_bound": upper,
            "threshold_absolute_gap": gap,
            "threshold_relative_gap": (
                gap / max(upper, 1e-12) if math.isfinite(upper) else math.inf
            ),
            "transition_pairs": self._transition_pairs,
            "candidate_blocks": self._candidate_blocks,
            "resource_dominated": self._resource_dominated,
            "incumbent_seconds": self._incumbent_seconds,
            "search_seconds": (
                time.monotonic() - self._search_started
                if self._search_started is not None
                else 0.0
            ),
        }

    def _solve_decision(self, threshold: float) -> Optional[SearchResult]:
        self._decision_calls += 1
        legal_masks, legal_predecessors = self._general_search._search_topology()
        legal_mask_set = set(legal_masks)
        initial = self.optimizer._dp_initial_objective_cost()
        if initial.local_serial > threshold + 1e-12:
            return None

        zero_usage = DpResourceUsage()
        # One scalar label per exact resource pair is sufficient: external
        # stage service has already been constrained by T, and only additive
        # local work remains visible to future transitions.
        labels: List[Dict[DpResourceUsage, float]] = [
            {} for _ in range(self.full_mask)
        ]
        back: List[Dict[DpResourceUsage, _ThresholdBackPointer]] = [
            {} for _ in range(self.full_mask)
        ]
        labels[0][zero_usage] = initial.local_serial
        decision_started = time.monotonic()
        masks_by_cardinality = [
            [mask for mask in legal_masks if mask.bit_count() == cardinality]
            for cardinality in range(1, self.n + 1)
        ]
        for cardinality, layer_masks in enumerate(
            masks_by_cardinality, start=1
        ):
            _raise_if_dp_deadline_exceeded(self.optimizer)
            if not layer_masks:
                continue
            layer_started = time.monotonic()
            for next_mask in layer_masks:
                _raise_if_dp_deadline_exceeded(self.optimizer)
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

                candidates_for_mask: Dict[
                    DpResourceUsage,
                    Tuple[float, _ThresholdBackPointer],
                ] = {}
                for prev_mask in prev_masks:
                    _raise_if_dp_deadline_exceeded(self.optimizer)
                    if not labels[prev_mask]:
                        continue
                    block_mask = next_mask ^ prev_mask
                    if not block_mask:
                        continue
                    self._transition_pairs += 1
                    options = self._threshold_block_options(
                        prev_mask, block_mask, threshold
                    )
                    for block, resource_delta, delta in options:
                        for prev_usage, prev_local in labels[prev_mask].items():
                            next_usage = prev_usage + resource_delta
                            if next_usage > self.parallel_stage_cpu_limit:
                                continue
                            next_local = prev_local + delta.local_serial
                            if next_local > threshold + 1e-12:
                                continue
                            old = candidates_for_mask.get(next_usage)
                            if (
                                old is not None
                                and old[0] <= next_local + 1e-12
                            ):
                                continue
                            candidates_for_mask[next_usage] = (
                                next_local,
                                _ThresholdBackPointer(
                                    prev_mask=prev_mask,
                                    prev_usage=prev_usage,
                                    block=block,
                                ),
                            )

                retained = self._prune_resource_dominated(
                    candidates_for_mask
                )
                for usage, (local, pointer) in retained.items():
                    labels[next_mask][usage] = local
                    back[next_mask][usage] = pointer

            logger.info(
                "[DpOptimizer] Threshold T=%.9f layer %d/%d masks=%d "
                "states=%d layer_sec=%.3f total_sec=%.3f",
                threshold,
                cardinality,
                self.n,
                len(layer_masks),
                sum(len(labels[mask]) for mask in layer_masks),
                time.monotonic() - layer_started,
                time.monotonic() - decision_started,
            )

        final_mask = self.full_mask - 1
        if not labels[final_mask]:
            return None
        final_usage = min(
            labels[final_mask],
            key=lambda usage: (
                labels[final_mask][usage],
                usage.ray_cpus + usage.smp_cpus,
                usage.ray_cpus,
                usage.smp_cpus,
            ),
        )
        return self._reconstruct(back, final_mask, final_usage)

    def _threshold_block_options(
        self,
        prev_mask: int,
        block_mask: int,
        threshold: float,
    ) -> List[
        Tuple[BlockCandidate, DpResourceUsage, DpObjectiveCost]
    ]:
        feasible: List[
            Tuple[BlockCandidate, DpResourceUsage, DpObjectiveCost]
        ] = []
        for block in self.block_provider.candidates_for_threshold(
            prev_mask, block_mask, threshold
        ):
            self._candidate_blocks += 1
            extra_cost = self.optimizer._dp_regular_transition_cost(
                prev_mask, block
            )
            delta = self.optimizer._dp_accumulate_objective_cost(
                DpObjectiveCost(), extra_cost, block, prev_mask
            )
            feasible.append(
                (
                    block,
                    self.optimizer._dp_parallel_stage_cpu_cost(block),
                    delta,
                )
            )
        return self._prune_threshold_options(feasible)

    def _prune_threshold_options(
        self,
        feasible: List[
            Tuple[BlockCandidate, DpResourceUsage, DpObjectiveCost]
        ],
    ) -> List[Tuple[BlockCandidate, DpResourceUsage, DpObjectiveCost]]:
        retained: List[
            Tuple[BlockCandidate, DpResourceUsage, DpObjectiveCost]
        ] = []
        tolerance = 1e-12
        for index, (block, usage, delta) in enumerate(feasible):
            dominated = False
            for other_index, (_, other_usage, other_delta) in enumerate(
                feasible
            ):
                if index == other_index or not other_usage <= usage:
                    continue
                no_worse = (
                    other_delta.local_serial
                    <= delta.local_serial + tolerance
                    and other_delta.gpu_serial
                    <= delta.gpu_serial + tolerance
                )
                strictly_better = (
                    other_usage != usage
                    or other_delta.local_serial
                    < delta.local_serial - tolerance
                    or other_delta.gpu_serial
                    < delta.gpu_serial - tolerance
                    or other_index < index
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if dominated:
                self._resource_dominated += 1
            else:
                retained.append((block, usage, delta))
        return retained

    def _prune_resource_dominated(
        self,
        candidates: Dict[
            DpResourceUsage,
            Tuple[float, _ThresholdBackPointer],
        ],
    ) -> Dict[DpResourceUsage, Tuple[float, _ThresholdBackPointer]]:
        if len(candidates) < 2:
            return candidates
        limits = self.parallel_stage_cpu_limit
        infinity = float("inf")
        prefix_min = [
            [infinity] * (limits.smp_cpus + 1)
            for _ in range(limits.ray_cpus + 1)
        ]
        retained = {}
        tolerance = 1e-12
        for ray_cpus in range(limits.ray_cpus + 1):
            for smp_cpus in range(limits.smp_cpus + 1):
                usage = DpResourceUsage(ray_cpus, smp_cpus)
                entry = candidates.get(usage)
                best_strict = infinity
                if ray_cpus:
                    best_strict = min(
                        best_strict, prefix_min[ray_cpus - 1][smp_cpus]
                    )
                if smp_cpus:
                    best_strict = min(
                        best_strict, prefix_min[ray_cpus][smp_cpus - 1]
                    )
                exact = entry[0] if entry is not None else infinity
                if entry is not None and best_strict > exact + tolerance:
                    retained[usage] = entry
                elif entry is not None:
                    self._resource_dominated += 1
                prefix_min[ray_cpus][smp_cpus] = min(best_strict, exact)
        return retained

    def _reconstruct(
        self,
        back: List[Dict[DpResourceUsage, _ThresholdBackPointer]],
        final_mask: int,
        final_usage: DpResourceUsage,
    ) -> SearchResult:
        mask = final_mask
        usage = final_usage
        selected_rev: List[BlockCandidate] = []
        while mask:
            pointer = back[mask].get(usage)
            if pointer is None:
                raise RuntimeError(
                    "Threshold feasibility DP backtracking hit an illegal state."
                )
            selected_rev.append(pointer.block)
            mask = pointer.prev_mask
            usage = pointer.prev_usage
        selected = list(reversed(selected_rev))

        objective = self.optimizer._dp_initial_objective_cost()
        prefix_mask = 0
        for block in selected:
            extra_cost = self.optimizer._dp_regular_transition_cost(
                prefix_mask, block
            )
            objective = self.optimizer._dp_accumulate_objective_cost(
                objective, extra_cost, block, prefix_mask
            )
            prefix_mask |= block.mask

        blocks = [list(block.order) for block in selected]
        order = [idx for block in blocks for idx in block]
        variants_by_idx = {
            idx: block.variant
            for block in selected
            for idx in block.order
        }
        parallelism_by_idx = {
            idx: block.parallelism
            for block in selected
            for idx in block.order
        }
        return SearchResult(
            order=order,
            blocks=blocks,
            variants_by_idx=variants_by_idx,
            parallelism_by_idx=parallelism_by_idx,
            cache_after_idx=None,
            cost=objective.score,
            objective=objective,
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
    external_stage_cost_aggregation = "sum"
    # R and S are additive and future transitions can affect either lane, so
    # they must remain separate Pareto coordinates.
    collapse_external_service_coordinates = False

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
        self,
        variant: PipeVariantType,
        execution_resource: PipeExecutionResource = PipeExecutionResource.CPU,
    ) -> int:
        if not self.joint_actor_allocation or variant not in (
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
            PipeVariantType.SMP,
        ):
            return 1
        if execution_resource == PipeExecutionResource.CUDA:
            return 1
        limits = self._dp_parallel_stage_cpu_limit()
        if limits is None:
            return 1
        if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            return max(1, limits.ray_cpus)
        return max(1, limits.smp_cpus)

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
        return tuple(
            range(
                1,
                self._dp_max_candidate_parallelism(
                    variant, execution_resource
                )
                + 1,
            )
        )

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
                ray_serial=previous.ray_serial,
                smp_serial=previous.smp_serial,
                gpu_serial=(
                    previous.gpu_serial
                    + self._dp_gpu_worker_multiplier()
                    * (block.cost + boundary_parallel)
                ),
            )
        if block.variant in (
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
        ):
            stage_cost = block.cost / block.parallelism + boundary_parallel
            return DpObjectiveCost(
                local_serial=previous.local_serial + boundary_local,
                ray_serial=previous.ray_serial + stage_cost,
                smp_serial=previous.smp_serial,
                gpu_serial=previous.gpu_serial,
            )
        if block.variant == PipeVariantType.SMP:
            stage_cost = block.cost / block.parallelism + boundary_parallel
            return DpObjectiveCost(
                local_serial=previous.local_serial + boundary_local,
                ray_serial=previous.ray_serial,
                smp_serial=previous.smp_serial + stage_cost,
                gpu_serial=previous.gpu_serial,
            )
        return DpObjectiveCost(
            local_serial=previous.local_serial + extra_cost,
            ray_serial=previous.ray_serial,
            smp_serial=previous.smp_serial,
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
                next_parallel_stage_cpus = DpResourceUsage()
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

    def _dp_parallel_stage_cpu_limit(
        self,
    ) -> Optional[DpResourceUsage]:
        """Return independent per-worker Ray and SMP CPU capacities."""
        if os.environ.get("CEDAR_MATCH_PROFILE_RESOURCES") != "1":
            return None
        fixed_workers_raw = os.environ.get(
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
        )
        local_budget_raw = os.environ.get("CEDAR_PROFILE_MATCH_CPU_BUDGET")
        if fixed_workers_raw is None or local_budget_raw is None:
            return None
        try:
            fixed_workers = int(fixed_workers_raw)
            local_budget = int(local_budget_raw)
            ray_budget = int(
                os.environ.get(
                    "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET",
                    local_budget_raw,
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "Invalid fixed-worker CPU budget configuration"
            ) from exc
        if (
            fixed_workers < 1
            or local_budget < fixed_workers
            or ray_budget < fixed_workers
        ):
            raise RuntimeError(
                "Fixed workers cannot fit under the local/Ray CPU budgets: "
                f"fixed={fixed_workers}, local={local_budget}, "
                f"ray={ray_budget}"
            )
        local_reserve_raw = os.environ.get(
            "CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER", "1"
        )
        ray_reserve_raw = os.environ.get(
            "CEDAR_DP_RAY_CPU_RESERVE_PER_WORKER", "1"
        )
        try:
            local_reserve = int(local_reserve_raw)
            ray_reserve = int(ray_reserve_raw)
        except ValueError as exc:
            raise RuntimeError(
                "DP local/Ray CPU reserves must be integers"
            ) from exc
        if local_reserve < 0 or ray_reserve < 0:
            raise RuntimeError(
                "DP local/Ray CPU reserves must be non-negative"
            )
        return DpResourceUsage(
            ray_cpus=max(
                0, ray_budget // fixed_workers - ray_reserve
            ),
            smp_cpus=max(
                0,
                local_budget // fixed_workers - 1 - local_reserve,
            ),
        )

    def _dp_parallel_stage_cpu_cost(
        self, block: BlockCandidate
    ) -> DpResourceUsage:
        if block.variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            return DpResourceUsage(ray_cpus=block.parallelism)
        if block.variant == PipeVariantType.SMP:
            return DpResourceUsage(smp_cpus=block.parallelism)
        return DpResourceUsage()

    def _dp_predict_general_search_complexity(
        self,
        inner_ops: List[int],
        resource_limits: Optional[DpResourceUsage],
    ) -> Dict[str, Any]:
        """Bound the structural work of the general Pareto subset DP.

        Operator count alone is misleading: a long dependency chain has only
        O(N) legal prefixes, while a shorter antichain has 2**N.  The predictor
        therefore generates dependency-closed prefixes up to a safe cap and
        counts legal prefix-to-prefix block transitions up to a second cap.
        It then accounts for the integer Ray/SMP width alternatives exposed by
        each structural transition.  Exceeding any cap selects the threshold
        feasibility search when that search supports the workload.
        """

        def positive_int(name: str, default: int) -> int:
            raw = os.environ.get(name, str(default))
            try:
                value = int(raw)
            except ValueError as exc:
                raise RuntimeError(f"{name} must be an integer") from exc
            if value < 1:
                raise RuntimeError(f"{name} must be positive")
            return value

        prefix_cap = positive_int(
            "CEDAR_DP_GENERAL_MAX_LEGAL_PREFIXES", 10000
        )
        transition_cap = positive_int(
            "CEDAR_DP_GENERAL_MAX_STRUCTURAL_TRANSITIONS", 30000
        )
        candidate_cap = positive_int(
            "CEDAR_DP_GENERAL_MAX_WIDTH_EXPANDED_TRANSITIONS", 300000
        )

        n = len(inner_ops)
        predecessor_masks: List[int] = []
        for predecessors in self._dp_pred_indices:
            mask = 0
            for predecessor in predecessors:
                mask |= 1 << predecessor
            predecessor_masks.append(mask)

        discovered = {0}
        frontier = [0]
        prefix_capped = False
        while frontier:
            placed = frontier.pop()
            for idx, predecessors in enumerate(predecessor_masks):
                bit = 1 << idx
                if placed & bit or predecessors & ~placed:
                    continue
                candidate = placed | bit
                if candidate in discovered:
                    continue
                discovered.add(candidate)
                if len(discovered) > prefix_cap:
                    prefix_capped = True
                    frontier.clear()
                    break
                frontier.append(candidate)
            if prefix_capped:
                break

        transition_count = 0
        transition_capped = False
        if not prefix_capped:
            legal_set = discovered
            for next_mask in sorted(legal_set)[1:]:
                prev_mask = (next_mask - 1) & next_mask
                while True:
                    if prev_mask in legal_set:
                        transition_count += 1
                        if transition_count > transition_cap:
                            transition_capped = True
                            break
                    if prev_mask == 0:
                        break
                    prev_mask = (prev_mask - 1) & next_mask
                if transition_capped:
                    break

        width_alternatives = 1
        if resource_limits is not None:
            width_alternatives += (
                resource_limits.ray_cpus + resource_limits.smp_cpus
            )
        projected_width_transitions = (
            transition_count * width_alternatives
        )
        high = (
            prefix_capped
            or transition_capped
            or projected_width_transitions > candidate_cap
        )
        reasons = []
        if prefix_capped:
            reasons.append(f"legal_prefixes>{prefix_cap}")
        if transition_capped:
            reasons.append(f"structural_transitions>{transition_cap}")
        if projected_width_transitions > candidate_cap:
            reasons.append(
                "width_expanded_transitions>" + str(candidate_cap)
            )
        return {
            "operator_count": n,
            "legal_prefixes": (
                f">{prefix_cap}" if prefix_capped else len(discovered)
            ),
            "structural_transitions": (
                f">{transition_cap}"
                if transition_capped
                else transition_count
            ),
            "width_alternatives": width_alternatives,
            "projected_width_expanded_transitions": (
                f">{transition_cap * width_alternatives}"
                if transition_capped
                else projected_width_transitions
            ),
            "high_complexity": high,
            "reasons": reasons,
        }

    def _run_conditioned_dp_search(
        self,
        inner_ops: List[int],
        resource_limits: Optional[DpResourceUsage],
        enforce_resource_limits: bool = True,
    ):
        limit_raw = os.environ.get(
            "CEDAR_DP_OPTIMIZATION_TIME_LIMIT_SEC", "300"
        )
        try:
            time_limit_sec = float(limit_raw)
        except ValueError as exc:
            raise RuntimeError(
                "CEDAR_DP_OPTIMIZATION_TIME_LIMIT_SEC must be numeric"
            ) from exc
        if not math.isfinite(time_limit_sec) or time_limit_sec <= 0.0:
            raise RuntimeError(
                "CEDAR_DP_OPTIMIZATION_TIME_LIMIT_SEC must be finite and positive"
            )

        search_started = time.monotonic()
        self._dp_search_deadline = search_started + time_limit_sec
        self._dp_assumed_total_parallel_stage_cpus = resource_limits
        search = None
        prediction: Dict[str, Any] = {}
        selected_mode = "general"
        timed_out = False
        try:
            block_provider = BlockCandidateProvider(self, inner_ops)
            block_provider.prepare()
            cache_policy = CacheTransitionPolicy(self, inner_ops)
            search_kwargs = {
                "parallel_stage_cpu_limit": (
                    resource_limits if enforce_resource_limits else None
                )
            }
            incumbent_raw = os.environ.get("CEDAR_DP_INITIAL_UPPER_BOUND")
            if incumbent_raw is not None:
                try:
                    incumbent_score = float(incumbent_raw)
                except ValueError as exc:
                    raise RuntimeError(
                        "CEDAR_DP_INITIAL_UPPER_BOUND must be numeric"
                    ) from exc
                if (
                    not math.isfinite(incumbent_score)
                    or incumbent_score <= 0.0
                ):
                    raise RuntimeError(
                        "CEDAR_DP_INITIAL_UPPER_BOUND must be finite and positive"
                    )
                search_kwargs["initial_incumbent_score"] = incumbent_score

            prediction = self._dp_predict_general_search_complexity(
                inner_ops, resource_limits
            )
            logger.info(
                "[DpOptimizer] General-search complexity prediction: %s",
                prediction,
            )
            requested_mode = os.environ.get(
                "CEDAR_DP_SEARCH_MODE", "auto"
            ).strip().lower()
            legacy_threshold = os.environ.get(
                "CEDAR_DP_THRESHOLD_FEASIBILITY"
            )
            if legacy_threshold == "1":
                requested_mode = "threshold"
            if requested_mode not in ("auto", "general", "threshold"):
                raise RuntimeError(
                    "CEDAR_DP_SEARCH_MODE must be auto, general, or threshold"
                )

            threshold_search = ThresholdFeasibilityDpSearch(
                optimizer=self,
                inner_ops=inner_ops,
                block_provider=block_provider,
                cache_policy=cache_policy,
                parallel_stage_cpu_limit=(
                    resource_limits if enforce_resource_limits else None
                ),
            )
            threshold_supported = threshold_search.supported()
            use_threshold_search = (
                requested_mode == "threshold"
                or (
                    requested_mode == "auto"
                    and prediction["high_complexity"]
                )
            ) and threshold_supported

            if use_threshold_search:
                selected_mode = "threshold"
                logger.info(
                    "[DpOptimizer] Using threshold-feasibility resource "
                    "search (requested=%s).",
                    requested_mode,
                )
                search = threshold_search
            else:
                if requested_mode == "threshold" and not threshold_supported:
                    logger.info(
                        "[DpOptimizer] Threshold search is not applicable; "
                        "using the general Pareto DP."
                    )
                selected_mode = "general"
                search = ExtensibleDpSearch(
                    optimizer=self,
                    inner_ops=inner_ops,
                    block_provider=block_provider,
                    cache_policy=cache_policy,
                    **search_kwargs,
                )
            try:
                candidate_result = search.run()
            except _DpSearchDeadlineExceeded:
                timed_out = True
                timeout_stats = getattr(search, "timeout_stats", None)
                if callable(timeout_stats):
                    self._dp_last_search_stats = timeout_stats()
                candidate_result = getattr(
                    search, "_best_feasible_result", None
                )
                if candidate_result is None:
                    candidate_result = getattr(
                        search, "_greedy_full_block_result", None
                    )
                if candidate_result is None:
                    raise RuntimeError(
                        "DP reached its five-minute limit before constructing "
                        "a feasible incumbent."
                    )
                logger.warning(
                    "[DpOptimizer] Search reached %.3fs; returning the best "
                    "feasible incumbent with cost %.9f.",
                    time.monotonic() - search_started,
                    candidate_result.cost,
                )
        finally:
            self._dp_search_deadline = None

        stats = dict(getattr(self, "_dp_last_search_stats", {}))
        stats.update(
            {
                "search_mode": selected_mode,
                "complexity_prediction": prediction,
                "optimization_time_limit_sec": time_limit_sec,
                "optimization_timed_out": timed_out,
                "optimization_elapsed_sec": time.monotonic() - search_started,
                "returned_feasible_cost": candidate_result.cost,
            }
        )
        self._dp_last_search_stats = stats
        return (
            resource_limits,
            candidate_result,
            dict(stats),
        )

    def _dp_reorder_offload_cache_fusion(
        self,
        inner_ops: List[int],
    ) -> Tuple[List[int], Optional[int]]:
        if not inner_ops:
            return [], None
        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")

        resource_limits = self._dp_parallel_stage_cpu_limit()
        forced_total_raw = os.environ.get("CEDAR_DP_FORCED_PARALLEL_TOTAL")
        if forced_total_raw is not None:
            raise RuntimeError(
                "CEDAR_DP_FORCED_PARALLEL_TOTAL is incompatible with separate "
                "Ray/SMP pools; use the physical pool budgets instead."
            )
        # Resource-matched formal plans go directly through the exact
        # constrained state space. The previous relaxed-first fast path was
        # lossless only when its winner happened to fit; on complex workloads
        # it completed one full exponential search merely to discover an
        # infeasible winner and then repeated all work with resource states.
        candidate = self._run_conditioned_dp_search(
            inner_ops,
            resource_limits,
            enforce_resource_limits=resource_limits is not None,
        )
        if candidate is None:
            raise RuntimeError(
                "Resource-conditioned DP found no feasible plan."
            )
        resource_limits, result, selected_stats = candidate
        self._dp_assumed_total_parallel_stage_cpus = resource_limits
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
        # Replay the chosen plan to validate and expose its four additive
        # resource-family coordinates in diagnostics and paper artifacts.
        selected_stats["final_ray_serial"] = replayed_objective.ray_serial
        selected_stats["final_smp_serial"] = replayed_objective.smp_serial
        selected_stats["final_parallel_bottleneck"] = (
            replayed_objective.parallel_bottleneck
        )
        self._dp_last_search_stats = selected_stats
        self._last_dp_state_cost = replayed_cost
        self._last_dp_search_result = result
        # Width is a first-class DP decision. Materialization must preserve it
        # exactly; a post-search allocator would change the objective that was
        # optimized and invalidate the optimality guarantee.
        self._dp_selected_stage_parallelism = {
            tuple(inner_ops[idx] for idx in block): (
                result.parallelism_by_idx.get(block[0], 1)
            )
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
            "[DpOptimizer] DP objective: max_of_additive_resource_sums "
            "local_serial=%s ray_serial=%s smp_serial=%s gpu_serial=%s",
            replayed_objective.local_serial,
            replayed_objective.ray_serial,
            replayed_objective.smp_serial,
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
    "ThresholdFeasibilityDpSearch",
]
