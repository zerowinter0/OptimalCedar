import copy
import heapq
import logging
from typing import Dict, List, Optional, Set

from cedar.pipes import FilterPipe

from .optimizer import Optimizer
from .utils import derive_constraint_graph, find_all_paths


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
        constraint_graph = derive_constraint_graph(self.logical_pipes)
        reordered_order = self._respect_filter_dependencies(
            reordered_order, linear_order, constraint_graph
        )
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

    def _respect_filter_dependencies(
        self,
        preferred_order: List[int],
        original_order: List[int],
        constraint_graph: Dict[int, Set[int]],
    ) -> List[int]:
        """Repair each filter group to respect explicit Cedar dependencies.

        Data-Juicer's speed ordering is retained as the priority among ready
        operators.  A dependency edge always wins over that preference, so a
        two-stage DJ plan cannot enter a physical search with an invalid
        logical order.
        """
        original_rank = {p_id: idx for idx, p_id in enumerate(original_order)}
        result: List[int] = []
        idx = 0
        while idx < len(preferred_order):
            if not self._is_filter_pipe(preferred_order[idx]):
                result.append(preferred_order[idx])
                idx += 1
                continue

            group: List[int] = []
            while idx < len(preferred_order) and self._is_filter_pipe(
                preferred_order[idx]
            ):
                group.append(preferred_order[idx])
                idx += 1
            result.extend(
                self._stable_topological_filter_order(
                    group, constraint_graph, original_rank
                )
            )
        return result

    @staticmethod
    def _stable_topological_filter_order(
        preferred: List[int],
        constraint_graph: Dict[int, Set[int]],
        original_rank: Dict[int, int],
    ) -> List[int]:
        group = set(preferred)
        preferred_rank = {p_id: idx for idx, p_id in enumerate(preferred)}
        indegree = {p_id: 0 for p_id in preferred}
        successors = {p_id: set() for p_id in preferred}
        for p_id in preferred:
            for successor in constraint_graph[p_id]:
                if successor in group:
                    successors[p_id].add(successor)
                    indegree[successor] += 1

        ready = [
            (preferred_rank[p_id], original_rank[p_id], p_id)
            for p_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        ordered: List[int] = []
        while ready:
            _, _, p_id = heapq.heappop(ready)
            ordered.append(p_id)
            for successor in successors[p_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(
                        ready,
                        (
                            preferred_rank[successor],
                            original_rank[successor],
                            successor,
                        ),
                    )
        if len(ordered) != len(preferred):
            raise ValueError("Detected cycle in DJ filter dependencies.")
        return ordered

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
