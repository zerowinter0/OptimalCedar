from typing import List

from cedar.compose import Feature
from cedar.compose.dp_seperate_optimizer import DpSeperateOptimizer
from cedar.compose.optimizer import OptimizerOptions
from cedar.pipes import FilterPipe, NoopPipe, Pipe
from cedar.sources import IterSource


def _true_filter(x):
    return True


def _other_true_filter(x):
    return True


class ConsecutiveFiltersFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]):
        ft = source_pipes[0]
        ft = FilterPipe(ft, _true_filter)
        ft = FilterPipe(ft, _other_true_filter)
        ft = NoopPipe(ft)
        return ft


def _profile_for(feature: Feature, fast_filter_name: str):
    latencies = {}
    input_sizes = {}
    output_sizes = {}
    for p_id, pipe in feature.logical_pipes.items():
        input_sizes[p_id] = 1.0
        output_sizes[p_id] = 1.0
        if isinstance(pipe, FilterPipe):
            latencies[p_id] = (
                1.0 if pipe.get_logical_name() == fast_filter_name else 10.0
            )
        else:
            latencies[p_id] = 1.0

    return {
        "baseline": {
            "throughput": 100.0,
            "latencies": latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
        },
        "disk_info": {"read_latency": 0.0, "write_latency": 0.0},
    }


def _linear_order(graph):
    predecessors = {p_id: set() for p_id in graph}
    for p_id, outputs in graph.items():
        for output in outputs:
            predecessors[output].add(p_id)
    source = next(p_id for p_id, inputs in predecessors.items() if not inputs)

    order = []
    curr = source
    while True:
        order.append(curr)
        if not graph[curr]:
            return order
        curr = next(iter(graph[curr]))


def test_dp_seperate_optimizer_reorders_before_strategy_pass():
    feature = ConsecutiveFiltersFeature()
    feature.apply(IterSource([1, 2, 3]))
    name_by_pid = {
        p_id: pipe.get_logical_name() for p_id, pipe in feature.logical_pipes.items()
    }

    optimizer = DpSeperateOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    fast_filter_name = "FilterPipe__other_true_filter"
    plan = optimizer.run(
        _profile_for(feature, fast_filter_name),
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=False,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=False,
            enable_caching=False,
        ),
    )

    order = _linear_order(plan.graph)
    assert name_by_pid[order[1]] == fast_filter_name
    assert name_by_pid[order[2]] == "FilterPipe__true_filter"
    assert plan.validate()
