"""Joint subset DP evaluated with Cedar's original cost model.

This ablation intentionally uses only Cedar's baseline/offload/disk profile
schema.  It shares the joint search space with :class:`DpOptimizer`, but none
of the newer boundary, worker-compute, scaling, selectivity, resource, or
operator compute-semantics terms participate in its objective.
"""

import logging
import math
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from yaml import safe_load

from cedar.pipes import PipeVariantType

from .dp_optimizer import (
    BlockCandidate,
    BlockCandidateProvider,
    CacheTransitionPolicy,
    DpObjectiveCost,
    DpOptimizer,
    DpStateSummary,
    ExtensibleDpSearch,
    SearchResult,
    TransitionChoice,
)
from .my_optimizer import MyOptimizer
from .optimizer import Optimizer, OptimizerOptions, PhysicalPlan, PipeDesc


logger = logging.getLogger(__name__)


class _CedarCostBlockCandidateProvider(BlockCandidateProvider):
    """Apply Cedar's fused-I/O discount to otherwise direct DP blocks."""

    def __init__(
        self, optimizer: "SimpleDpOptimizer", inner_ops: List[int]
    ) -> None:
        super().__init__(optimizer, inner_ops)
        self._cedar_candidates: Dict[
            Tuple[int, int], List[BlockCandidate]
        ] = {}

    def candidates_for_prefix(
        self, prefix_mask: int, mask: int
    ) -> Iterable[BlockCandidate]:
        key = (prefix_mask, mask)
        cached = self._cedar_candidates.get(key)
        if cached is not None:
            return cached
        candidates = list(super().candidates_for_prefix(prefix_mask, mask))
        if mask.bit_count() > 1:
            adjusted = []
            for candidate in candidates:
                ratio = self.optimizer._cedar_fusion_io_ratio(
                    candidate.order
                )
                adjusted.append(
                    BlockCandidate(
                        mask=candidate.mask,
                        order=candidate.order,
                        variant=candidate.variant,
                        cost=candidate.cost * ratio,
                        materializes_fusion=True,
                        execution_resource=candidate.execution_resource,
                        parallelism=candidate.parallelism,
                    )
                )
            candidates = adjusted
        self._cedar_candidates[key] = candidates
        return candidates

    def candidate_for_order(
        self,
        order: Iterable[int],
        variant: PipeVariantType,
        prefix_mask: int = 0,
        parallelism: int = 1,
    ) -> BlockCandidate:
        candidate = super().candidate_for_order(
            order,
            variant,
            prefix_mask=prefix_mask,
            parallelism=parallelism,
        )
        if candidate.mask.bit_count() <= 1:
            return candidate
        ratio = self.optimizer._cedar_fusion_io_ratio(candidate.order)
        return BlockCandidate(
            mask=candidate.mask,
            order=candidate.order,
            variant=candidate.variant,
            cost=candidate.cost * ratio,
            materializes_fusion=True,
            execution_resource=candidate.execution_resource,
            parallelism=candidate.parallelism,
        )


class _CedarCacheTransitionPolicy(CacheTransitionPolicy):
    """Enumerate Cedar cache placements using serialized output size."""

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
        cache_after_idx = block.order[-1]
        cache_cost = (
            self.cache_cost_per_source_sample
            * self.optimizer._dp_r_prod[next_mask]
        )
        yield TransitionChoice(
            state=DpStateSummary(
                cache_active=True,
                parallel_stage_cpus=next_parallel_stage_cpus,
            ),
            extra_cost=cache_cost,
            cache_after_idx=cache_after_idx,
            replaces_prefix_cost=True,
        )


class SimpleDpOptimizer(DpOptimizer):
    """Joint DP whose objective is exactly Cedar's original scalar cost."""

    joint_actor_allocation = False

    def _allocate_final_remote_stage_resources(self) -> None:
        MyOptimizer._allocate_final_remote_stage_resources(self)

    def run(
        self,
        profiled_data: Union[str, Dict[str, Any]],
        options: OptimizerOptions,
    ) -> PhysicalPlan:
        if not self._initialized:
            raise RuntimeError("Must initialize optimizer before running.")
        if isinstance(profiled_data, dict):
            self.profiled_stats = profiled_data
        else:
            path = pathlib.Path(profiled_data)
            try:
                with path.open("r") as profile_file:
                    self.profiled_stats = safe_load(profile_file)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read profiled stats {profiled_data}"
                ) from exc

        self.options = options
        self._validate_stats()
        self._init_stats()
        logger.info(
            "[SimpleDpOptimizer] Running joint DP with Cedar cost model"
        )
        self._logical_opt()
        self._physical_opt()
        if self.options.enable_prefetch:
            self._insert_prefetch()

        cedar_cost = self._calculate_materialized_cedar_cost(
            self.physical_plan
        )
        search_cost = getattr(self, "_last_dp_state_cost", None)
        if search_cost is not None and not math.isclose(
            cedar_cost, search_cost, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError(
                "Materialized Cedar cost diverged from Simple DP search: "
                f"search={search_cost}, plan={cedar_cost}"
            )
        self._last_dp_state_cost = cedar_cost
        logger.info(
            "[SimpleDpOptimizer] Optimized Cedar cost = %s", cedar_cost
        )
        self._log_optimized_pipeline(tag="SimpleDpOptimizer")
        return self.physical_plan

    def _iter_candidate_backend_stats(self):
        """Read only Cedar's original whole-pipeline offload profiles."""
        if not self.options.enable_offload:
            return
        for backend_name, backend_stats in self.profiled_stats.get(
            "offloads", {}
        ).items():
            if backend_name not in PipeVariantType.__members__:
                continue
            yield PipeVariantType[backend_name], backend_stats

    def _calculate_pipe_cost(
        self, p_id: int, input_size: float, desc: Optional[PipeDesc]
    ) -> float:
        # Bypass MyOptimizer/DpOptimizer extensions. This is Cedar's original
        # baseline-size scaling plus whole-pipeline Amdahl inversion.
        return Optimizer._calculate_pipe_cost(self, p_id, input_size, desc)

    def _dp_pipe_cost_at_parallelism(
        self,
        p_id: int,
        variant_type: PipeVariantType,
        parallelism: int,
        width_one_cost: float,
    ) -> float:
        """Keep this ablation independent of PICO's scaling curves.

        Simple-DP is defined as the joint DP search evaluated strictly with
        Cedar's original whole-pipeline offload cost.  The shared candidate
        provider asks PICO optimizers for a width-specific worker cost; using
        that hook here would silently replace Cedar's estimate with the new
        layered profile and make the DP search disagree with Cedar's own
        materialized ``calculate_cost``.
        """
        return width_one_cost

    def _dp_work_prod(self, mask: int) -> float:
        return self._dp_r_prod[mask]

    def _dp_compute_work_prod(
        self, mask: int, operator_idx: Optional[int] = None
    ) -> float:
        return self._dp_r_prod[mask]

    def _dp_compute_cost_denominator(
        self,
        operator_idx: int,
        baseline_input_size: float,
        source_size: float,
    ) -> float:
        return baseline_input_size

    def _dp_has_supported_fusion_cost(
        self, variant_type: PipeVariantType
    ) -> bool:
        return True

    def _dp_fusion_transport_feasible(
        self, prev_mask: int, block: BlockCandidate
    ) -> bool:
        return True

    def _dp_stage_boundary_cost(
        self, prev_mask: int, block: BlockCandidate
    ) -> float:
        return 0.0

    def _dp_parallel_stage_cpu_limit(self) -> Optional[int]:
        return None

    def _dp_smp_supported_for_pipe(self, p_id: int) -> bool:
        return True

    def _dp_initial_objective_cost(self) -> DpObjectiveCost:
        return DpObjectiveCost(
            local_serial=float(
                self._base_cost_map[self._get_source_p_id()]
            )
        )

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

    def _dp_regular_transition_cost(
        self, prev_mask: int, block: BlockCandidate
    ) -> float:
        return block.cost

    def _cedar_fusion_io_ratio(self, order: Iterable[int]) -> float:
        ordered = tuple(order)
        if len(ordered) <= 1:
            return 1.0
        local_ratio = 1.0
        baseline_io = 1.0
        for position, idx in enumerate(ordered):
            if position > 0:
                baseline_io += 2.0 * local_ratio
            local_ratio *= self._dp_ratios[idx]
        baseline_io += local_ratio
        fused_io = 1.0 + local_ratio
        return fused_io / baseline_io

    def _dp_reorder_offload_cache_fusion(
        self, inner_ops: List[int]
    ) -> Tuple[List[int], Optional[int]]:
        if not inner_ops:
            return [], None
        provider = _CedarCostBlockCandidateProvider(self, inner_ops)
        provider.prepare()
        cache_policy = _CedarCacheTransitionPolicy(self, inner_ops)
        search = ExtensibleDpSearch(
            optimizer=self,
            inner_ops=inner_ops,
            block_provider=provider,
            cache_policy=cache_policy,
        )
        # The scalar Cedar objective has one exact label per search state.
        search.pareto_global_epsilon = 0.0
        search.pareto_step_epsilon = 0.0
        result = search.run()
        self._last_dp_state_cost = result.cost
        self._last_dp_search_result = result

        self._store_pending_fusions_from_blocks(
            blocks_in_idx_order=result.blocks,
            inner_ops=inner_ops,
            chosen_variant_by_idx=result.variants_by_idx,
        )
        for idx, p_id in enumerate(inner_ops):
            variant = result.variants_by_idx.get(
                idx, PipeVariantType.INPROCESS
            )
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = variant

        best_order = [inner_ops[idx] for idx in result.order]
        cache_p_id = (
            inner_ops[result.cache_after_idx]
            if result.cache_after_idx is not None
            else None
        )
        return best_order, cache_p_id

    def _replay_dp_objective(
        self,
        block_specs,
        inner_ops: List[int],
    ) -> DpObjectiveCost:
        # Keep external objective-validation helpers consistent with this
        # ablation's provider and cache policy.
        provider = _CedarCostBlockCandidateProvider(self, inner_ops)
        provider.prepare()
        cache_policy = _CedarCacheTransitionPolicy(self, inner_ops)
        state = cache_policy.initial_state()
        objective = self._dp_initial_objective_cost()
        prev_mask = 0
        for order, variant, wants_cache, parallelism in block_specs:
            block = provider.candidate_for_order(
                order,
                variant,
                prefix_mask=prev_mask,
                parallelism=parallelism,
            )
            next_mask = prev_mask | block.mask
            regular_cost = self._dp_regular_transition_cost(
                prev_mask, block
            )
            transitions = list(
                cache_policy.transitions(
                    prev_mask,
                    next_mask,
                    state,
                    regular_cost,
                    block,
                    0,
                )
            )
            choice = next(
                (
                    item
                    for item in transitions
                    if (item.cache_after_idx is not None) == wants_cache
                ),
                None,
            )
            if choice is None:
                raise ValueError("Plan contains an illegal Cedar cache choice")
            if choice.replaces_prefix_cost:
                objective = DpObjectiveCost(
                    local_serial=choice.extra_cost
                )
            else:
                objective = self._dp_accumulate_objective_cost(
                    objective, choice.extra_cost, block, prev_mask
                )
            state = choice.state
            prev_mask = next_mask
        return objective

    def calculate_dp_objective_cost(
        self,
        plan: Optional[PhysicalPlan] = None,
        search_result: Optional[SearchResult] = None,
        inner_ops: Optional[List[int]] = None,
    ) -> float:
        if (plan is None) == (search_result is None):
            raise ValueError("Supply exactly one of plan or search_result.")
        if plan is not None:
            return self._calculate_materialized_cedar_cost(plan)
        return float(search_result.cost)

    def _calculate_materialized_cedar_cost(
        self, plan: PhysicalPlan
    ) -> float:
        cache_on = self._get_cache_pid(plan) is not None
        fused_blocks = [
            list(desc.fused_pipes)
            for desc in plan.pipe_descs.values()
            if desc.fused_pipes and len(desc.fused_pipes) > 1
        ]
        return Optimizer.calculate_cost(
            self,
            plan.graph,
            physical_specs=plan.pipe_descs,
            fused_pipes=fused_blocks or None,
            caching_on=cache_on,
            plan=plan,
        )


__all__ = ["SimpleDpOptimizer"]
