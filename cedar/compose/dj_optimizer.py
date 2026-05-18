import copy
import logging
from typing import Dict, List, Optional, Set

from cedar.pipes import FilterPipe

from .optimizer import Optimizer
from .utils import find_all_paths


logger = logging.getLogger(__name__)


class DjOptimizer(Optimizer):
    """
    Cedar optimizer variant that keeps the original staged optimizer pipeline
    but uses Data-Juicer's filter reorder policy.

    Data-Juicer only reorders filters inside a consecutive filter group. With
    profile data it sorts the group by probed speed from fast to slow; without
    profile data all speeds are zero, so the stable sort keeps the original
    order. In Cedar's baseline stats the comparable signal is per-pipe latency,
    so fast-to-slow is implemented as low-latency-to-high-latency.
    """

    def _pass_reordering(self) -> Dict[int, Set[int]]:
        graph = copy.deepcopy(self.physical_plan.graph)
        linear_order = self._get_single_linear_order(graph)
        if linear_order is None:
            logger.info(
                "[DJ Reordering] Non-linear graph detected; leaving order unchanged."
            )
            return graph

        reordered_order = self._reorder_consecutive_filters(linear_order)
        if reordered_order == linear_order:
            logger.info("[DJ Reordering] No consecutive FilterPipe group reordered.")
            return graph

        reordered_graph = self._linear_order_to_graph(reordered_order)
        logger.info("[DJ Reordering] Original order: %s", linear_order)
        logger.info("[DJ Reordering] Reordered order: %s", reordered_order)
        logger.info(
            "[DJ Reordering] Calculated reordered cost: %s",
            self.calculate_cost(reordered_graph),
        )
        return reordered_graph

    def _get_single_linear_order(
        self, graph: Dict[int, Set[int]]
    ) -> Optional[List[int]]:
        try:
            source_p_id = self._get_source_p_id()
            output_p_id = self._get_output_p_id(graph)
            paths = find_all_paths(graph, source_p_id, output_p_id)
        except Exception:
            return None

        if len(paths) != 1 or len(paths[0]) != len(graph):
            return None
        return paths[0]

    def _reorder_consecutive_filters(self, order: List[int]) -> List[int]:
        reordered: List[int] = []
        idx = 0
        while idx < len(order):
            p_id = order[idx]
            if not self._is_filter_pipe(p_id):
                reordered.append(p_id)
                idx += 1
                continue

            group: List[int] = []
            while idx < len(order) and self._is_filter_pipe(order[idx]):
                group.append(order[idx])
                idx += 1

            if len(group) > 1:
                group = sorted(group, key=self._dj_filter_sort_key, reverse=True)
            reordered.extend(group)

        return reordered

    def _is_filter_pipe(self, p_id: int) -> bool:
        pipe = self.logical_pipes.get(p_id)
        return isinstance(pipe, FilterPipe)

    def _dj_filter_sort_key(self, p_id: int) -> float:
        latency = self._base_cost_map.get(p_id)
        if latency is None or latency <= 0:
            return 0.0
        return 1.0 / latency

    @staticmethod
    def _linear_order_to_graph(order: List[int]) -> Dict[int, Set[int]]:
        graph: Dict[int, Set[int]] = {}
        for idx, p_id in enumerate(order):
            if idx + 1 < len(order):
                graph[p_id] = {order[idx + 1]}
            else:
                graph[p_id] = set()
        return graph


__all__ = ["DjOptimizer"]
