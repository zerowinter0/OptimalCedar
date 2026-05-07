import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from cedar.pipes import Pipe

from .my_optimizer import MyOptimizer
from .optimizer import OptimizerOptions, PhysicalPlan, PipeDesc, PipeVariantType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DpStateSummary:
    """
    Compact state carried by the outer DP.

    The key design difference from MyOptimizer is that strategy state is a
    summary object. New strategies should extend or replace this summary only
    when future transitions need extra information.
    """

    cache_active: bool = False


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
        self._candidates_by_mask: List[List[BlockCandidate]] = [
            [] for _ in range(self.full_mask)
        ]

    def prepare(self) -> None:
        opt = self.optimizer
        if opt.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")
        if opt._dp_costs is None:
            raise RuntimeError("DP metadata not prepared.")

        variant_costs: Dict[PipeVariantType, List[float]] = {
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
                    costs_v[i] = opt._calculate_pipe_cost(
                        p_id, base_input_size, desc
                    )
                except Exception:
                    continue

            if any(c != float("inf") for c in costs_v):
                variant_costs[vt] = costs_v
                candidate_variants.append(vt)

        variant_cost_by_mask = opt._dp_naive_reorder_cost_per_variant(
            self.inner_ops,
            self.n,
            candidate_variants,
            variant_costs,
        )

        source_p_id = opt._get_source_p_id()
        source_output_size = opt.profiled_stats["baseline"]["output_sizes"][
            source_p_id
        ]
        fusion_allowed = self._calculate_all_fusion_allowed()

        for mask in range(1, self.full_mask):
            is_multi = mask.bit_count() > 1
            if is_multi and not fusion_allowed[mask]:
                continue
            if is_multi and not opt.options.enable_fusion:
                continue

            candidates: List[BlockCandidate] = []
            for vt in candidate_variants:
                block_cost = variant_cost_by_mask[vt][mask]
                if block_cost == float("inf"):
                    continue
                order = tuple(opt._reconstruct_naive_reorder_order(vt, mask))
                if not order:
                    continue
                candidates.append(
                    BlockCandidate(
                        mask=mask,
                        order=order,
                        variant=vt,
                        cost=block_cost * source_output_size,
                        materializes_fusion=is_multi,
                    )
                )

            # Keep the frontier interface but prune to the cheapest candidate
            # for the current Cedar cost model. Future strategies can relax this.
            if candidates:
                best = min(candidates, key=lambda c: c.cost)
                self._candidates_by_mask[mask] = [best]

    def candidates_for(self, mask: int) -> Iterable[BlockCandidate]:
        if mask <= 0 or mask >= self.full_mask:
            return []
        return self._candidates_by_mask[mask]

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
        self.cache_benefit = -optimizer._base_cost_map[source_p_id]

    def initial_state(self) -> DpStateSummary:
        return DpStateSummary(cache_active=False)

    def transitions(
        self,
        prev_mask: int,
        next_mask: int,
        prev_state: DpStateSummary,
        regular_cost: float,
        block: BlockCandidate,
    ) -> Iterable[TransitionChoice]:
        yield TransitionChoice(state=prev_state, extra_cost=regular_cost)

        if not self.enabled or prev_state.cache_active:
            return
        if block.mask.bit_count() != 1:
            return
        if not self.all_non_random[next_mask]:
            return

        cache_after_idx = block.order[-1]
        cache_cost = self.cache_benefit + (
            self.cache_cost_per_source_sample * self.optimizer._dp_r_prod[prev_mask]
        )
        yield TransitionChoice(
            state=DpStateSummary(cache_active=True),
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

    def run(self) -> SearchResult:
        initial_state = self.cache_policy.initial_state()
        dp: List[Dict[DpStateSummary, float]] = [
            {} for _ in range(self.full_mask)
        ]
        back: List[Dict[DpStateSummary, BackPointer]] = [
            {} for _ in range(self.full_mask)
        ]
        dp[0][initial_state] = 0.0

        for mask in range(1, self.full_mask):
            t = mask
            while True:
                block_mask = mask ^ t
                if block_mask:
                    self._try_extend(dp, back, t, mask, block_mask)
                if t == 0:
                    break
                t = (t - 1) & mask

        final_mask = self.full_mask - 1
        if not dp[final_mask]:
            raise RuntimeError("Extensible DP failed: no feasible final state.")

        final_state, final_cost = min(
            dp[final_mask].items(), key=lambda item: item[1]
        )
        return self._reconstruct(back, final_mask, final_state, final_cost)

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

            regular_cost = self.optimizer._dp_r_prod[prev_mask] * block.cost
            for prev_state, prev_cost in dp[prev_mask].items():
                for choice in self.cache_policy.transitions(
                    prev_mask,
                    next_mask,
                    prev_state,
                    regular_cost,
                    block,
                ):
                    if choice.replaces_prefix_cost:
                        candidate_cost = choice.extra_cost
                    else:
                        candidate_cost = prev_cost + choice.extra_cost
                    old = dp[next_mask].get(choice.state, float("inf"))
                    if candidate_cost < old:
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
        return self.optimizer._dp_fusion_block_valid(prev_mask, block.mask)

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
