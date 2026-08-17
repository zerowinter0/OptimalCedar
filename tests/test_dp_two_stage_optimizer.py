from typing import List

from cedar.compose import Feature
from cedar.compose.dp_two_stage_optimizer import DpTwoStageOptimizer
from cedar.compose.policy_two_stage_optimizer import (
    DjTwoStageOptimizer,
    PecanTwoStageOptimizer,
)
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


def test_dp_two_stage_optimizer_reorders_before_strategy_pass():
    feature = ConsecutiveFiltersFeature()
    feature.apply(IterSource([1, 2, 3]))
    name_by_pid = {
        p_id: pipe.get_logical_name() for p_id, pipe in feature.logical_pipes.items()
    }

    optimizer = DpTwoStageOptimizer()
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


def test_dj_two_stage_uses_dj_reorder_before_physical_enumeration():
    feature = ConsecutiveFiltersFeature()
    feature.apply(IterSource([1, 2, 3]))
    name_by_pid = {
        p_id: pipe.get_logical_name()
        for p_id, pipe in feature.logical_pipes.items()
    }
    optimizer = DjTwoStageOptimizer()
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


def test_pecan_two_stage_uses_pecan_reorder_before_physical_enumeration():
    from tests.test_pecan_optimizer import AutoOrderFeature, _profile_for

    feature = AutoOrderFeature()
    feature.apply(IterSource([1, 2, 3]))
    tags = {p_id: pipe.tag for p_id, pipe in feature.logical_pipes.items()}
    optimizer = PecanTwoStageOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    profile = _profile_for(feature)
    profile["disk_info"] = {"read_latency": 0.0, "write_latency": 0.0}
    plan = optimizer.run(
        profile,
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=False,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=False,
            enable_caching=False,
        ),
    )

    reordered_tags = [
        tags[p_id]
        for p_id in _linear_order(plan.graph)
        if tags[p_id] is not None
    ]
    assert reordered_tags == ["deflate", "neutral", "inflate"]
    assert plan.validate()


def test_policy_two_stage_explicitly_enumerates_cartesian_physical_space(
    monkeypatch,
):
    from cedar.compose.dp_optimizer import ExtensibleDpSearch

    def reject_joint_dp(*args, **kwargs):
        raise AssertionError("policy two-stage must not invoke joint DP")

    monkeypatch.setattr(ExtensibleDpSearch, "run", reject_joint_dp)
    feature = ConsecutiveFiltersFeature()
    feature.apply(IterSource([1, 2, 3]))
    optimizer = DjTwoStageOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    profile = _profile_for(feature, "FilterPipe__other_true_filter")
    profile["offloads"] = {
        "RAY": {
            p_id: {
                "throughput": 200.0,
                "latencies": profile["baseline"]["latencies"].copy(),
                "input_sizes": profile["baseline"]["input_sizes"].copy(),
                "output_sizes": profile["baseline"]["output_sizes"].copy(),
            }
            for p_id, pipe in feature.logical_pipes.items()
            if not pipe.is_source()
        }
    }
    optimizer.run(
        profile,
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=True,
            enable_caching=True,
        ),
    )

    stats = optimizer._last_exhaustive_search_stats
    n = stats["operators"]
    assert stats["method"] == "explicit_cartesian_enumeration"
    assert stats["backend_count"] == 2
    assert stats["backend_assignments"] == stats["backend_count"] ** n
    assert stats["fusion_patterns"] == 2 ** (n - 1)
    assert stats["cache_positions"] == n + 1
    assert stats["theoretical_combinations"] == (
        stats["backend_count"] ** n * 2 ** (n - 1) * (n + 1)
    )
    assert stats["examined_combinations"] == stats[
        "theoretical_combinations"
    ]
