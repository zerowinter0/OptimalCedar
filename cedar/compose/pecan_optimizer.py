import copy
import heapq
import logging
import math
from typing import Dict, List, Optional, Set

from .optimizer import Optimizer
from .utils import derive_constraint_graph, find_all_paths


logger = logging.getLogger(__name__)


class PecanOptimizer(Optimizer):
    """Cedar port of Pecan's linear-time AutoOrder policy.

    Pecan divides a pipeline at fixed-position transformations. Within each
    section, deflationary transformations are prepended, neutral
    transformations are appended to the prefix, and inflationary
    transformations are appended to the suffix (ATC'24, Algorithm 2).

    Cedar ``Pipe.fix()`` annotations serve as Pecan ``keep_position`` barriers.
    Cedar additionally exposes explicit ``depends_on()`` constraints, which
    Pecan's tf.data API does not model. A stable topological repair preserves
    those constraints while retaining Pecan's preferred order whenever legal.
    """

    def _pass_reordering(self) -> Dict[int, Set[int]]:
        graph = copy.deepcopy(self.physical_plan.graph)
        linear_order = self._get_single_linear_order(graph)
        if linear_order is None:
            logger.info(
                "[Pecan AutoOrder] Non-linear graph detected; leaving order "
                "unchanged."
            )
            return graph

        constraint_graph = derive_constraint_graph(self.logical_pipes)
        reordered_order = self._autoorder(linear_order, constraint_graph)
        if not self._respects_constraints(reordered_order, constraint_graph):
            logger.warning(
                "[Pecan AutoOrder] Reordered plan violated an explicit Cedar "
                "dependency; leaving order unchanged."
            )
            return graph
        if reordered_order == linear_order:
            logger.info("[Pecan AutoOrder] No section changed order.")
            return graph

        reordered_graph = self._linear_order_to_graph(reordered_order)
        logger.info("[Pecan AutoOrder] Original order: %s", linear_order)
        logger.info("[Pecan AutoOrder] Reordered order: %s", reordered_order)
        logger.info(
            "[Pecan AutoOrder] Calculated reordered cost: %s",
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

    def _autoorder(
        self,
        order: List[int],
        constraint_graph: Dict[int, Set[int]],
    ) -> List[int]:
        reordered: List[int] = []
        section: List[int] = []

        def flush_section() -> None:
            if not section:
                return
            preferred = self._pecan_section_order(section)
            reordered.extend(
                self._stable_topological_order(
                    preferred,
                    constraint_graph,
                )
            )
            section.clear()

        for p_id in order:
            if self._is_barrier(p_id):
                flush_section()
                reordered.append(p_id)
            else:
                section.append(p_id)
        flush_section()
        return reordered

    def _pecan_section_order(self, section: List[int]) -> List[int]:
        prefix: List[int] = []
        suffix: List[int] = []
        for p_id in section:
            inflation_factor = self._data_size_ratio_map.get(p_id)
            if (
                inflation_factor is None
                or not math.isfinite(inflation_factor)
                or inflation_factor < 0
            ):
                # Profiling should normally provide a valid factor. Treat an
                # unknown factor as neutral instead of guessing a direction.
                prefix.append(p_id)
            elif inflation_factor < 1:
                prefix.insert(0, p_id)
            elif inflation_factor == 1:
                prefix.append(p_id)
            else:
                suffix.append(p_id)
        return prefix + suffix

    def _is_barrier(self, p_id: int) -> bool:
        pipe = self.logical_pipes[p_id]
        # Sources are not transformations and therefore are outside Pecan's
        # reorderable sections. Workloads mark decode/rank-changing/batching
        # transformations with fix(), matching Pecan's keep_position policy.
        return pipe.is_source() or bool(pipe._fix_order)

    @staticmethod
    def _stable_topological_order(
        preferred: List[int],
        constraint_graph: Dict[int, Set[int]],
    ) -> List[int]:
        """Return the closest legal ordering to Pecan's preferred ordering."""
        section_nodes = set(preferred)
        preferred_rank = {p_id: idx for idx, p_id in enumerate(preferred)}
        indegree = {p_id: 0 for p_id in preferred}
        successors = {p_id: set() for p_id in preferred}

        for p_id in preferred:
            for successor in constraint_graph[p_id]:
                if successor in section_nodes:
                    successors[p_id].add(successor)
                    indegree[successor] += 1

        ready = [
            (preferred_rank[p_id], p_id)
            for p_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        result: List[int] = []
        while ready:
            _, p_id = heapq.heappop(ready)
            result.append(p_id)
            for successor in successors[p_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(
                        ready,
                        (preferred_rank[successor], successor),
                    )

        if len(result) != len(preferred):
            raise ValueError("Cycle detected in Cedar ordering constraints.")
        return result

    @staticmethod
    def _respects_constraints(
        order: List[int],
        constraint_graph: Dict[int, Set[int]],
    ) -> bool:
        rank = {p_id: idx for idx, p_id in enumerate(order)}
        return all(
            rank[p_id] < rank[successor]
            for p_id, successors in constraint_graph.items()
            for successor in successors
        )

    @staticmethod
    def _linear_order_to_graph(order: List[int]) -> Dict[int, Set[int]]:
        return {
            p_id: ({order[idx + 1]} if idx + 1 < len(order) else set())
            for idx, p_id in enumerate(order)
        }


__all__ = ["PecanOptimizer"]
