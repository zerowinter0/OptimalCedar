import itertools
import logging
import math
from typing import Dict, List, Optional, Tuple, Type

from .dj_optimizer import DjOptimizer
from .dp_optimizer import (
    BlockCandidate,
    BlockCandidateProvider,
    CacheTransitionPolicy,
    DpObjectiveCost,
    SearchResult,
)
from .dp_two_stage_optimizer import DpTwoStageOptimizer
from .optimizer import Optimizer, PipeVariantType
from .pecan_optimizer import PecanOptimizer


logger = logging.getLogger(__name__)


class _PolicyTwoStageOptimizer(DpTwoStageOptimizer):
    """Policy reorder followed by brute-force physical enumeration.

    Stage 2 deliberately does not use the joint or fixed-order DP. It
    explicitly enumerates one backend choice per operator, one fuse/not-fuse
    bit per adjacent pair, and (when enabled) every cache position. Illegal
    materializations are rejected, while every legal combination is scored
    with the same objective used by :class:`DpOptimizer`.
    """

    reorder_policy: Type[Optimizer]

    def _dp_reorder_only(self, inner_ops: List[int]) -> List[int]:
        policy = self.reorder_policy()
        policy.init(self.logical_pipes, self.logical_graph)
        policy.profiled_stats = self.profiled_stats
        policy.options = self.options
        policy._validate_stats()
        policy._init_stats()
        reordered_graph = policy._pass_reordering()
        full_order = policy._get_single_linear_order(reordered_graph)
        if full_order is None:
            logger.warning(
                "[%s] Stage-1 policy returned a non-linear graph; preserving "
                "the original order.",
                self.__class__.__name__,
            )
            return list(range(len(inner_ops)))

        index_by_pipe = {p_id: idx for idx, p_id in enumerate(inner_ops)}
        reordered_indices = [
            index_by_pipe[p_id]
            for p_id in full_order
            if p_id in index_by_pipe
        ]
        if len(reordered_indices) != len(inner_ops) or len(
            set(reordered_indices)
        ) != len(inner_ops):
            raise RuntimeError(
                f"{self.__class__.__name__} Stage-1 reorder did not cover "
                "every physical-search operator exactly once."
            )
        logger.info(
            "[%s] Stage-1 %s order: %s",
            self.__class__.__name__,
            self.reorder_policy.__name__,
            [inner_ops[idx] for idx in reordered_indices],
        )
        return reordered_indices

    def _dp_reorder_offload_cache_fusion(self, inner_ops):
        if not inner_ops:
            return [], None

        reorder_indices = self._dp_reorder_only(inner_ops)
        reordered_inner_ops = [inner_ops[idx] for idx in reorder_indices]

        # Rebuild work products in the fixed Stage-1 order. Chain dependencies
        # ensure that validation of fused blocks cannot silently reorder them.
        self._prepare_dp_metadata(reordered_inner_ops)
        self._dp_pred_indices = [
            [] if idx == 0 else [idx - 1]
            for idx in range(len(reordered_inner_ops))
        ]
        return self._exhaustive_fixed_order_physical_search(reordered_inner_ops)

    def _exhaustive_fixed_order_physical_search(
        self, inner_ops: List[int]
    ) -> Tuple[List[int], Optional[int]]:
        """Enumerate the complete fixed-order physical-plan search space."""
        n = len(inner_ops)
        provider = BlockCandidateProvider(self, inner_ops)
        provider.prepare()
        cache_policy = CacheTransitionPolicy(self, inner_ops)

        # Use the global backend alphabet, not a per-operator pruned product:
        # the experimental baseline explicitly models M**N assignments.
        variants = tuple(provider._candidate_variants)
        fusion_pattern_count = (
            1 << max(0, n - 1) if self.options.enable_fusion else 1
        )
        cache_positions: Tuple[Optional[int], ...] = (
            (None,) + tuple(range(n))
            if self.options.enable_caching
            else (None,)
        )
        theoretical_combinations = (
            len(variants) ** n
            * fusion_pattern_count
            * len(cache_positions)
        )

        examined_combinations = 0
        legal_materializations = 0
        feasible_plans = 0
        best_key = None
        best_blocks: Optional[List[List[int]]] = None
        best_candidates: Optional[List[BlockCandidate]] = None
        best_cache_after_idx: Optional[int] = None
        best_objective: Optional[DpObjectiveCost] = None

        for assignment in itertools.product(variants, repeat=n):
            for fusion_mask in range(fusion_pattern_count):
                blocks = self._blocks_from_fusion_mask(n, fusion_mask)
                candidates = self._materialize_exhaustive_blocks(
                    provider, blocks, assignment
                )
                if candidates is None:
                    examined_combinations += len(cache_positions)
                    continue
                legal_materializations += 1

                for cache_after_idx in cache_positions:
                    examined_combinations += 1
                    objective = self._score_exhaustive_choice(
                        candidates, cache_policy, cache_after_idx
                    )
                    if objective is None or not math.isfinite(objective.score):
                        continue
                    feasible_plans += 1
                    key = (
                        objective.score,
                        objective.local_serial,
                        objective.parallel_bottleneck,
                        objective.gpu_serial,
                        len(blocks),
                        tuple(variant.value for variant in assignment),
                        fusion_mask,
                        n if cache_after_idx is None else cache_after_idx,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_blocks = blocks
                        best_candidates = candidates
                        best_cache_after_idx = cache_after_idx
                        best_objective = objective

        self._last_exhaustive_search_stats = {
            "optimizer": self.__class__.__name__,
            "operators": n,
            "backend_count": len(variants),
            "backend_assignments": len(variants) ** n,
            "fusion_patterns": fusion_pattern_count,
            "cache_positions": len(cache_positions),
            "theoretical_combinations": theoretical_combinations,
            "examined_combinations": examined_combinations,
            "legal_materializations": legal_materializations,
            "feasible_plans": feasible_plans,
            "method": "explicit_cartesian_enumeration",
        }
        logger.info(
            "[%s] Stage-2 exhaustive search stats: %s",
            self.__class__.__name__,
            self._last_exhaustive_search_stats,
        )

        if (
            best_blocks is None
            or best_candidates is None
            or best_objective is None
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} exhaustive Stage-2 search found "
                "no feasible physical plan."
            )

        variants_by_idx: Dict[int, PipeVariantType] = {}
        for block, candidate in zip(best_blocks, best_candidates):
            for idx in block:
                variants_by_idx[idx] = candidate.variant

        self._last_dp_state_cost = best_objective.score
        self._last_dp_search_result = SearchResult(
            order=list(range(n)),
            blocks=best_blocks,
            variants_by_idx=variants_by_idx,
            parallelism_by_idx={idx: 1 for idx in range(n)},
            cache_after_idx=best_cache_after_idx,
            cost=best_objective.score,
            objective=best_objective,
        )
        self._store_pending_fusions_from_blocks(
            blocks_in_idx_order=best_blocks,
            inner_ops=inner_ops,
            chosen_variant_by_idx=variants_by_idx,
        )
        for idx, p_id in enumerate(inner_ops):
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = (
                    variants_by_idx[idx]
                )

        logger.info(
            "[%s] Stage-2 exhaustive objective cost (inner ops only): %s",
            self.__class__.__name__,
            best_objective.score,
        )
        cache_p_id = (
            inner_ops[best_cache_after_idx]
            if best_cache_after_idx is not None
            else None
        )
        return list(inner_ops), cache_p_id

    @staticmethod
    def _blocks_from_fusion_mask(n: int, fusion_mask: int) -> List[List[int]]:
        """Decode bit i as 'operator i+1 fuses with its predecessor'."""
        if n <= 0:
            return []
        blocks = [[0]]
        for idx in range(1, n):
            if fusion_mask & (1 << (idx - 1)):
                blocks[-1].append(idx)
            else:
                blocks.append([idx])
        return blocks

    def _materialize_exhaustive_blocks(
        self,
        provider: BlockCandidateProvider,
        blocks: List[List[int]],
        assignment: Tuple[PipeVariantType, ...],
    ) -> Optional[List[BlockCandidate]]:
        candidates: List[BlockCandidate] = []
        prefix_mask = 0
        for block in blocks:
            variant = assignment[block[0]]
            if any(assignment[idx] != variant for idx in block):
                return None
            try:
                candidate = provider.candidate_for_order(
                    block, variant, prefix_mask=prefix_mask
                )
            except ValueError:
                return None
            if len(block) == 1:
                if not self._dp_valid_single_last(prefix_mask, block[0]):
                    return None
            elif not (
                self._dp_fusion_block_valid(prefix_mask, candidate.mask)
                and self._dp_fusion_transport_feasible(prefix_mask, candidate)
            ):
                return None
            candidates.append(candidate)
            prefix_mask |= candidate.mask
        return candidates

    def _score_exhaustive_choice(
        self,
        candidates: List[BlockCandidate],
        cache_policy: CacheTransitionPolicy,
        cache_after_idx: Optional[int],
    ) -> Optional[DpObjectiveCost]:
        state = cache_policy.initial_state()
        objective = self._dp_initial_objective_cost()
        prefix_mask = 0
        cache_materialized = False
        cpu_limit = self._dp_parallel_stage_cpu_limit()

        for candidate in candidates:
            next_mask = prefix_mask | candidate.mask
            next_cpus = (
                state.parallel_stage_cpus
                + self._dp_parallel_stage_cpu_cost(candidate)
            )
            if cpu_limit is not None and next_cpus > cpu_limit:
                return None
            regular_cost = self._dp_regular_transition_cost(
                prefix_mask, candidate
            )
            transitions = list(
                cache_policy.transitions(
                    prefix_mask,
                    next_mask,
                    state,
                    regular_cost,
                    candidate,
                    next_cpus,
                )
            )
            wants_cache = (
                cache_after_idx is not None
                and candidate.order[-1] == cache_after_idx
            )
            choice = next(
                (
                    transition
                    for transition in transitions
                    if (
                        transition.cache_after_idx == cache_after_idx
                        if wants_cache
                        else transition.cache_after_idx is None
                    )
                ),
                None,
            )
            if choice is None:
                return None
            if wants_cache:
                cache_materialized = True
            if choice.replaces_prefix_cost:
                objective = DpObjectiveCost(local_serial=choice.extra_cost)
            else:
                objective = self._dp_accumulate_objective_cost(
                    objective, choice.extra_cost, candidate, prefix_mask
                )
            state = choice.state
            prefix_mask = next_mask

        if cache_after_idx is not None and not cache_materialized:
            return None
        return objective


class PecanTwoStageOptimizer(_PolicyTwoStageOptimizer):
    reorder_policy = PecanOptimizer


class DjTwoStageOptimizer(_PolicyTwoStageOptimizer):
    reorder_policy = DjOptimizer


__all__ = ["DjTwoStageOptimizer", "PecanTwoStageOptimizer"]
