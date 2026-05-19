import logging
from typing import Any, Dict, List, Optional, Set, Union

from .dp_seperate_optimizer import DpSeperateOptimizer
from .optimizer import Optimizer, OptimizerOptions, PhysicalPlan


logger = logging.getLogger(__name__)


class DpCedarOptimizer(DpSeperateOptimizer):
    """
    Cedar optimizer with only the native reorder pass replaced by DP reorder.

    Fusion, offload, prefetch, caching, and local parallelism intentionally
    delegate to Optimizer so this remains a focused comparison against the
    original staged optimizer pipeline.
    """

    def run(
        self, profiled_data: Union[str, Dict[str, Any]], options: OptimizerOptions
    ) -> PhysicalPlan:
        return Optimizer.run(self, profiled_data, options)

    def _logical_opt(self) -> None:
        return Optimizer._logical_opt(self)

    def _physical_opt(self) -> None:
        return Optimizer._physical_opt(self)

    def _allowed_fusion(self, pipe_slice: List[int], variant_type) -> bool:
        return Optimizer._allowed_fusion(self, pipe_slice, variant_type)

    def _pass_reordering(self) -> Dict[int, Set[int]]:
        inner_ops: Optional[List[int]] = self._get_linear_inner_ops()
        if inner_ops is None:
            logger.info(
                "[DpCedarOptimizer] Graph is not linear; falling back to "
                "Optimizer._pass_reordering."
            )
            return Optimizer._pass_reordering(self)
        if not inner_ops:
            return self.physical_plan.graph

        self._prepare_dp_metadata(inner_ops)
        reorder_order = self._dp_reorder_only(inner_ops)
        reordered_inner_ops = [inner_ops[idx] for idx in reorder_order]
        logger.info("[DpCedarOptimizer] DP reorder order: %s", reordered_inner_ops)

        source_p_id = self._get_source_p_id()
        new_path = [source_p_id] + reordered_inner_ops
        new_graph: Dict[int, Set[int]] = {
            p_id: set() for p_id in self.physical_plan.graph.keys()
        }
        for u, v in zip(new_path[:-1], new_path[1:]):
            new_graph[u].add(v)
        return new_graph


__all__ = ["DpCedarOptimizer"]
