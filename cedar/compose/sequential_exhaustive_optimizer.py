"""Sequential exhaustive baselines for reorder/fusion/offload studies.

Each stage exhaustively enumerates one optimization dimension while holding
the winner of every preceding stage fixed.  This deliberately models a staged
optimizer rather than a joint search: decisions discarded by an early stage
cannot be recovered later.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

from cedar.pipes import PipeExecutionResource, PipeVariantType

from .dp_optimizer import (
    BlockCandidateProvider,
    CacheTransitionPolicy,
    DpOptimizer,
    DpResourceUsage,
    ExtensibleDpSearch,
    SearchResult,
)


logger = logging.getLogger(__name__)

STAGE_DIMENSIONS = ("r", "f", "o")


def variant_allowed_for_execution_resource(
    variant: PipeVariantType,
    execution_resource: PipeExecutionResource,
) -> bool:
    if execution_resource == PipeExecutionResource.CUDA:
        return variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
    return True


@dataclass(frozen=True)
class SequentialPlan:
    """A complete linear plan expressed as ordered physical blocks."""

    blocks: Tuple[Tuple[int, ...], ...]
    variants: Tuple[PipeVariantType, ...]

    def __post_init__(self) -> None:
        if not self.blocks or len(self.blocks) != len(self.variants):
            raise ValueError("A sequential plan needs one variant per block.")
        flattened = tuple(idx for block in self.blocks for idx in block)
        if not flattened or len(flattened) != len(set(flattened)):
            raise ValueError("Sequential plan blocks must cover unique operators.")

    @property
    def order(self) -> Tuple[int, ...]:
        return tuple(idx for block in self.blocks for idx in block)

    def sort_key(self) -> tuple:
        return (
            self.blocks,
            tuple(variant.name for variant in self.variants),
        )


def _legal_order(order: Sequence[int], predecessor_masks: Sequence[int]) -> bool:
    placed = 0
    for idx in order:
        if predecessor_masks[idx] & ~placed:
            return False
        placed |= 1 << idx
    return True


def enumerate_reorder_candidates(
    plan: SequentialPlan,
    predecessor_masks: Sequence[int],
) -> Tuple[SequentialPlan, ...]:
    """Permute existing blocks; an earlier fusion is never split."""

    candidates = []
    indexed = tuple(zip(plan.blocks, plan.variants))
    for permutation in itertools.permutations(indexed):
        blocks = tuple(item[0] for item in permutation)
        candidate = SequentialPlan(
            blocks=blocks,
            variants=tuple(item[1] for item in permutation),
        )
        if _legal_order(candidate.order, predecessor_masks):
            candidates.append(candidate)
    return tuple(sorted(set(candidates), key=SequentialPlan.sort_key))


def enumerate_fusion_candidates(plan: SequentialPlan) -> Tuple[SequentialPlan, ...]:
    """Enumerate all legal coarsenings of the current block sequence.

    Fusion may merge adjacent blocks only when they already use the same
    backend.  It cannot revise placement; that belongs to the offload stage.
    Backend-specific fusion feasibility is checked by the cost-model replay.
    """

    candidates = []
    boundary_count = len(plan.blocks) - 1
    for keep_mask in range(1 << boundary_count):
        merged_blocks = [list(plan.blocks[0])]
        merged_variants = [plan.variants[0]]
        valid = True
        for boundary, (block, variant) in enumerate(
            zip(plan.blocks[1:], plan.variants[1:])
        ):
            keep_boundary = bool(keep_mask & (1 << boundary))
            if keep_boundary:
                merged_blocks.append(list(block))
                merged_variants.append(variant)
            elif variant == merged_variants[-1]:
                merged_blocks[-1].extend(block)
            else:
                valid = False
                break
        if valid:
            candidates.append(
                SequentialPlan(
                    blocks=tuple(tuple(block) for block in merged_blocks),
                    variants=tuple(merged_variants),
                )
            )
    return tuple(sorted(set(candidates), key=SequentialPlan.sort_key))


def select_stage_winner(
    candidates: Iterable[SequentialPlan],
    score: Callable[[SequentialPlan], float],
) -> SequentialPlan:
    """Choose one deterministic winner and discard all other stage plans."""

    scored = [(float(score(candidate)), candidate) for candidate in candidates]
    finite = [item for item in scored if math.isfinite(item[0])]
    if not finite:
        raise RuntimeError("Sequential stage produced no feasible candidate plan.")
    return min(finite, key=lambda item: (item[0], item[1].sort_key()))[1]


class MinimalParallelDpOptimizer(DpOptimizer):
    """Joint DP variant whose every physical stage has width one."""

    joint_actor_allocation = False
    preserve_optimizer_widths = True


class SingleWorkerCudaDpOptimizer(MinimalParallelDpOptimizer):
    """Width-one DP that includes Cedar's legal in-process CUDA placement."""

    def _dp_variant_allowed_for_execution_resource(
        self,
        variant: PipeVariantType,
        execution_resource: PipeExecutionResource,
    ) -> bool:
        if execution_resource == PipeExecutionResource.CUDA:
            return variant in (
                PipeVariantType.INPROCESS,
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            )
        return True


class SequentialExhaustiveOptimizer(MinimalParallelDpOptimizer):
    """Exhaust one optimization at a time and commit each stage winner."""

    def _stage_order(self) -> Tuple[str, str, str]:
        raw = os.environ.get("CEDAR_STAGED_OPTIMIZATION_ORDER", "rfo").lower()
        order = tuple(raw)
        if len(order) != 3 or set(order) != set(STAGE_DIMENSIONS):
            raise ValueError(
                "CEDAR_STAGED_OPTIMIZATION_ORDER must be a permutation of rfo"
            )
        return order  # type: ignore[return-value]

    def _score_sequential_plan(
        self,
        plan: SequentialPlan,
        provider: BlockCandidateProvider,
        cache_policy: CacheTransitionPolicy,
        search: ExtensibleDpSearch,
    ):
        state = cache_policy.initial_state()
        objective = search._initial_objective()
        prev_mask = 0
        for order, variant in zip(plan.blocks, plan.variants):
            block = provider.candidate_for_order(
                order,
                variant,
                prefix_mask=prev_mask,
                parallelism=1,
            )
            if not search._block_can_follow(prev_mask, block):
                raise ValueError("Sequential candidate violates dependencies.")
            next_mask = prev_mask | block.mask
            if search.parallel_stage_cpu_limit is None:
                next_usage = DpResourceUsage()
            else:
                next_usage = (
                    state.parallel_stage_cpus
                    + self._dp_parallel_stage_cpu_cost(block)
                )
                if next_usage > search.parallel_stage_cpu_limit:
                    raise ValueError("Sequential candidate exceeds the CPU budget.")
            regular_cost = self._dp_regular_transition_cost(prev_mask, block)
            choices = list(
                cache_policy.transitions(
                    prev_mask,
                    next_mask,
                    state,
                    regular_cost,
                    block,
                    next_usage,
                )
            )
            no_cache = [choice for choice in choices if choice.cache_after_idx is None]
            if len(no_cache) != 1:
                raise ValueError("Expected exactly one cache-free transition.")
            choice = no_cache[0]
            objective = search._accumulate_objective(
                objective,
                choice.extra_cost,
                block,
                choice.replaces_prefix_cost,
                prev_mask,
            )
            state = choice.state
            prev_mask = next_mask
        if prev_mask != (1 << len(self._dp_inner_ops)) - 1:
            raise ValueError("Sequential candidate does not cover all operators.")
        return objective

    def _offload_candidates(
        self,
        plan: SequentialPlan,
        provider: BlockCandidateProvider,
    ) -> Tuple[SequentialPlan, ...]:
        candidates = []

        def visit(block_index: int, prefix_mask: int, variants) -> None:
            if block_index == len(plan.blocks):
                candidates.append(
                    SequentialPlan(plan.blocks, tuple(variants))
                )
                return
            block_order = plan.blocks[block_index]
            block_mask = sum(1 << idx for idx in block_order)
            execution_resource = provider._execution_resource_for_mask(block_mask)
            for variant in provider._candidate_variants:
                if not variant_allowed_for_execution_resource(
                    variant, execution_resource
                ):
                    continue
                try:
                    provider.candidate_for_order(
                        block_order,
                        variant,
                        prefix_mask=prefix_mask,
                        parallelism=1,
                    )
                except ValueError:
                    continue
                visit(block_index + 1, prefix_mask | block_mask, variants + [variant])

        visit(0, 0, [])
        return tuple(sorted(set(candidates), key=SequentialPlan.sort_key))

    def _initial_sequential_plan(
        self,
        provider: BlockCandidateProvider,
    ) -> SequentialPlan:
        variants = []
        prefix_mask = 0
        for idx, p_id in enumerate(self._dp_inner_ops):
            execution_resource = self.logical_pipes[p_id].execution_resource
            preferred = (
                (PipeVariantType.RAY, PipeVariantType.TF_RAY)
                if execution_resource == PipeExecutionResource.CUDA
                else (PipeVariantType.INPROCESS,)
            )
            selected = None
            for variant in preferred:
                try:
                    provider.candidate_for_order(
                        (idx,), variant, prefix_mask=prefix_mask, parallelism=1
                    )
                except ValueError:
                    continue
                selected = variant
                break
            if selected is None:
                raise RuntimeError(
                    f"No executable width-one baseline variant for operator {p_id}."
                )
            variants.append(selected)
            prefix_mask |= 1 << idx
        return SequentialPlan(
            blocks=tuple((idx,) for idx in range(len(self._dp_inner_ops))),
            variants=tuple(variants),
        )

    def _dp_reorder_offload_cache_fusion(self, inner_ops):
        if not inner_ops:
            return [], None
        provider = BlockCandidateProvider(self, inner_ops)
        provider.prepare()
        cache_policy = CacheTransitionPolicy(self, inner_ops)
        search = ExtensibleDpSearch(
            optimizer=self,
            inner_ops=inner_ops,
            block_provider=provider,
            cache_policy=cache_policy,
        )
        predecessor_masks = tuple(
            sum(1 << predecessor for predecessor in predecessors)
            for predecessors in self._dp_pred_indices
        )
        current = self._initial_sequential_plan(provider)
        trace = []

        def score(candidate: SequentialPlan) -> float:
            try:
                return self._score_sequential_plan(
                    candidate, provider, cache_policy, search
                ).score
            except ValueError:
                return float("inf")

        for dimension in self._stage_order():
            if dimension == "r":
                candidates = enumerate_reorder_candidates(current, predecessor_masks)
            elif dimension == "f":
                candidates = enumerate_fusion_candidates(current)
            else:
                candidates = self._offload_candidates(current, provider)
            current = select_stage_winner(candidates, score)
            trace.append(
                {
                    "dimension": dimension,
                    "candidate_count": len(candidates),
                    "winner_cost": score(current),
                    "winner_blocks": [list(block) for block in current.blocks],
                    "winner_variants": [variant.name for variant in current.variants],
                }
            )

        objective = self._score_sequential_plan(
            current, provider, cache_policy, search
        )
        variants_by_idx = {
            idx: variant
            for block, variant in zip(current.blocks, current.variants)
            for idx in block
        }
        result = SearchResult(
            order=list(current.order),
            blocks=[list(block) for block in current.blocks],
            variants_by_idx=variants_by_idx,
            parallelism_by_idx={idx: 1 for idx in current.order},
            cache_after_idx=None,
            cost=objective.score,
            objective=objective,
        )
        self._last_dp_search_result = result
        self._last_dp_state_cost = objective.score
        self._dp_selected_stage_parallelism = {
            tuple(inner_ops[idx] for idx in block): 1 for block in current.blocks
        }
        self._store_pending_fusions_from_blocks(
            result.blocks, inner_ops, variants_by_idx
        )
        for idx, p_id in enumerate(inner_ops):
            self.physical_plan.pipe_descs[p_id].variant_type = variants_by_idx[idx]
        self.sequential_optimizer_stats = {
            "stage_order": "".join(self._stage_order()),
            "stages": trace,
            "best_cost": objective.score,
        }
        logger.info(
            "[SequentialExhaustiveOptimizer] order=%s trace=%s",
            self.sequential_optimizer_stats["stage_order"],
            trace,
        )
        return [inner_ops[idx] for idx in current.order], None


__all__ = [
    "MinimalParallelDpOptimizer",
    "SingleWorkerCudaDpOptimizer",
    "SequentialExhaustiveOptimizer",
    "SequentialPlan",
    "enumerate_fusion_candidates",
    "enumerate_reorder_candidates",
    "select_stage_winner",
    "variant_allowed_for_execution_resource",
]
