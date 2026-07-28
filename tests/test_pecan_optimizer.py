from typing import List

from cedar.compose import Feature
from cedar.compose.optimizer import OptimizerOptions
from cedar.compose.pecan_optimizer import PecanOptimizer
from cedar.pipes import MapperPipe, Pipe
from cedar.sources import IterSource


def _identity(x):
    return x


class AutoOrderFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]):
        pipe = source_pipes[0]
        pipe = MapperPipe(pipe, _identity, tag="inflate")
        pipe = MapperPipe(pipe, _identity, tag="neutral")
        pipe = MapperPipe(pipe, _identity, tag="deflate")
        return pipe


class BarrierFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]):
        pipe = source_pipes[0]
        pipe = MapperPipe(pipe, _identity, tag="inflate")
        pipe = MapperPipe(pipe, _identity, tag="barrier").fix()
        pipe = MapperPipe(pipe, _identity, tag="deflate")
        return pipe


class DependencyFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]):
        pipe = source_pipes[0]
        pipe = MapperPipe(pipe, _identity, tag="inflate")
        pipe = MapperPipe(pipe, _identity, tag="deflate").depends_on(
            ["inflate"]
        )
        return pipe


def _profile_for(feature: Feature):
    ratios = {
        "inflate": 2.0,
        "neutral": 1.0,
        "deflate": 0.5,
        "barrier": 1.0,
    }
    latencies = {}
    input_sizes = {}
    output_sizes = {}
    for p_id, pipe in feature.logical_pipes.items():
        input_sizes[p_id] = 10.0
        output_sizes[p_id] = 10.0 * ratios.get(pipe.tag, 1.0)
        latencies[p_id] = 1.0
    return {
        "baseline": {
            "throughput": 100.0,
            "latencies": latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
        }
    }


def _linear_order(graph):
    predecessors = {p_id: set() for p_id in graph}
    for p_id, outputs in graph.items():
        for output in outputs:
            predecessors[output].add(p_id)
    source = next(p_id for p_id, inputs in predecessors.items() if not inputs)

    order = []
    current = source
    while True:
        order.append(current)
        if not graph[current]:
            return order
        current = next(iter(graph[current]))


def _run_pecan(feature: Feature):
    feature.apply(IterSource([1, 2, 3]))
    tags = {p_id: pipe.tag for p_id, pipe in feature.logical_pipes.items()}
    optimizer = PecanOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.run(
        _profile_for(feature),
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=False,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=False,
        ),
    )
    return [
        tags[p_id]
        for p_id in _linear_order(optimizer.physical_plan.graph)
        if tags[p_id] is not None
    ]


def test_pecan_autoorder_matches_algorithm_two():
    assert _run_pecan(AutoOrderFeature()) == [
        "deflate",
        "neutral",
        "inflate",
    ]


def test_pecan_does_not_move_transformations_across_fixed_barrier():
    assert _run_pecan(BarrierFeature()) == [
        "inflate",
        "barrier",
        "deflate",
    ]


def test_pecan_preserves_explicit_cedar_dependency():
    assert _run_pecan(DependencyFeature()) == ["inflate", "deflate"]
