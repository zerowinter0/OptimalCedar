import copy
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

from .optimizer import Optimizer, PhysicalPlan
from .utils import calculate_reorderings


logger = logging.getLogger(__name__)


class CedarJointOptimizer(Optimizer):
    """
    Naive joint optimizer built from Cedar's original optimization passes.

    This optimizer is intentionally exhaustive over the logical candidate
    space: every Cedar reorder candidate is paired with every legal cache
    placement, then Cedar's original physical optimizer is run on each
    candidate to choose fusion/offload decisions. It is meant as an experiment
    baseline for measuring whether a direct joint version of Cedar's own
    methods becomes too slow on simple workloads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.integrated_optimizer_stats: Dict[str, float] = {}

    def _logical_opt(self) -> None:
        baseline_cost = self.calculate_cost(self.physical_plan.graph)
        logger.info(
            "[CedarJointOptimizer] Baseline graph: %s",
            self.physical_plan.graph,
        )
        logger.info(
            "[CedarJointOptimizer] Baseline cost: %s",
            baseline_cost,
        )

    def _physical_opt(self) -> None:
        start_time = time.perf_counter()
        base_plan = copy.deepcopy(self.physical_plan)

        reorder_candidates = self._enumerate_reorder_candidates(
            base_plan.graph,
            start_time,
        )
        logger.info(
            "[CedarJointOptimizer] Generated %d reorder candidates.",
            len(reorder_candidates),
        )

        best_cost = float("inf")
        best_plan: Optional[PhysicalPlan] = None
        total_cache_candidates = 0
        total_physical_candidates = 0

        saved_forbid_local_parallelism = self.forbid_local_parallelism
        for reorder_idx, reorder_graph in enumerate(reorder_candidates):
            self._raise_if_reorder_timed_out(
                start_time,
                getattr(self.options, "reorder_timeout_sec", None),
            )
            reorder_plan = PhysicalPlan(
                graph=copy.deepcopy(reorder_graph),
                pipe_descs=copy.deepcopy(base_plan.pipe_descs),
                n_local_workers=base_plan.n_local_workers,
            )
            cache_candidates = self._enumerate_cache_candidates(reorder_plan)
            total_cache_candidates += len(cache_candidates)
            logger.info(
                "[CedarJointOptimizer] Reorder candidate %d/%d produced "
                "%d cache candidates.",
                reorder_idx + 1,
                len(reorder_candidates),
                len(cache_candidates),
            )

            for cache_plan in cache_candidates:
                total_physical_candidates += 1
                cost, plan = self._optimize_physical_candidate(cache_plan)
                if plan is not None and cost < best_cost:
                    best_cost = cost
                    best_plan = plan

        self.forbid_local_parallelism = saved_forbid_local_parallelism
        if best_plan is None:
            raise RuntimeError(
                "CedarJointOptimizer failed to produce a valid physical plan."
            )

        elapsed_sec = time.perf_counter() - start_time
        self.integrated_optimizer_stats = {
            "reorder_candidates": float(len(reorder_candidates)),
            "cache_candidates": float(total_cache_candidates),
            "physical_candidates": float(total_physical_candidates),
            "best_cost": float(best_cost),
            "elapsed_sec": float(elapsed_sec),
        }
        logger.info(
            "[CedarJointOptimizer] Evaluated %d physical candidates from "
            "%d reorder candidates and %d cache candidates in %.6fs.",
            total_physical_candidates,
            len(reorder_candidates),
            total_cache_candidates,
            elapsed_sec,
        )
        logger.info("[CedarJointOptimizer] Best cost: %s", best_cost)
        self.physical_plan = best_plan

    def _enumerate_reorder_candidates(
        self,
        base_graph: Dict[int, Set[int]],
        start_time: float,
    ) -> List[Dict[int, Set[int]]]:
        if not self.options.enable_reorder:
            return [copy.deepcopy(base_graph)]

        timeout_sec = getattr(self.options, "reorder_timeout_sec", None)
        candidates = calculate_reorderings(
            self.logical_pipes,
            base_graph,
            timeout_sec=timeout_sec,
        )
        self._raise_if_reorder_timed_out(start_time, timeout_sec)
        return candidates

    def _enumerate_cache_candidates(
        self,
        plan: PhysicalPlan,
    ) -> List[PhysicalPlan]:
        if not self.options.enable_caching:
            return [copy.deepcopy(plan)]
        if not self.options.num_samples:
            logger.info(
                "[CedarJointOptimizer] Number of samples not specified. "
                "Skipping cache enumeration for this reorder candidate."
            )
            return [copy.deepcopy(plan)]

        saved_plan = self.physical_plan
        try:
            self.physical_plan = copy.deepcopy(plan)
            return [
                copy.deepcopy(candidate)
                for candidate in self._calculate_caching_plans(
                    self.logical_pipes,
                    self.physical_plan.graph,
                )
            ]
        finally:
            self.physical_plan = saved_plan

    def _optimize_physical_candidate(
        self,
        candidate_plan: PhysicalPlan,
    ) -> Tuple[float, Optional[PhysicalPlan]]:
        saved_plan = self.physical_plan
        saved_forbid_local_parallelism = self.forbid_local_parallelism
        try:
            self.physical_plan = copy.deepcopy(candidate_plan)
            self.forbid_local_parallelism = False
            if self.options.enable_prefetch:
                self._insert_prefetch()

            Optimizer._physical_opt(self)
            optimized_plan = self.physical_plan
            cost = self._calculate_final_plan_cost(optimized_plan)
            return cost, optimized_plan
        except Exception as exc:
            logger.info(
                "[CedarJointOptimizer] Skipping invalid candidate: %s",
                exc,
            )
            return float("inf"), None
        finally:
            self.physical_plan = saved_plan
            self.forbid_local_parallelism = saved_forbid_local_parallelism

    def _calculate_final_plan_cost(self, plan: PhysicalPlan) -> float:
        fused_blocks = [
            list(desc.fused_pipes)
            for desc in plan.pipe_descs.values()
            if getattr(desc, "fused_pipes", None)
            and len(getattr(desc, "fused_pipes", [])) > 1
        ]
        cache_on = self._get_cache_pid(plan) is not None
        return self.calculate_cost(
            plan.graph,
            physical_specs=plan.pipe_descs,
            fused_pipes=fused_blocks if fused_blocks else None,
            caching_on=cache_on,
            plan=plan,
        )


__all__ = ["CedarJointOptimizer"]
