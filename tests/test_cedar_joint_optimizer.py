import math
from typing import List

from cedar.client import DataSet
from cedar.compose import Feature
from cedar.compose.cedar_joint_optimizer import CedarJointOptimizer
from cedar.compose.optimizer import OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import MapperPipe, Pipe
from cedar.sources import IterSource


def _add_one(x):
    return x + 1


def _double(x):
    return x * 2


def _minus_one(x):
    return x - 1


class ThreeMapFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]):
        ft = source_pipes[0]
        ft = MapperPipe(ft, _add_one)
        ft = MapperPipe(ft, _double)
        ft = MapperPipe(ft, _minus_one)
        return ft


def _profile_for(feature: Feature):
    latencies = {}
    input_sizes = {}
    output_sizes = {}
    for p_id, pipe in feature.logical_pipes.items():
        latencies[p_id] = 1.0 if pipe.is_source() else 100.0
        input_sizes[p_id] = 1.0
        output_sizes[p_id] = 1.0

    baseline_throughput = 100.0
    offloads = {"RAY": {}, "SMP": {}, "TF_RAY": {}}
    for p_id in feature.logical_pipes:
        offloads["RAY"][p_id] = {
            "throughput": baseline_throughput * 1.2,
            "latencies": dict(latencies),
            "input_sizes": dict(input_sizes),
            "output_sizes": dict(output_sizes),
        }

    return {
        "baseline": {
            "throughput": baseline_throughput,
            "latencies": latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
        },
        "disk_info": {
            "read_latency": 0.000001,
            "write_latency": 0.000001,
        },
        "offloads": offloads,
    }


def _make_feature():
    feature = ThreeMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    return feature


def test_cedar_joint_optimizer_enumerates_reorder_cache_then_physical():
    feature = _make_feature()
    optimizer = CedarJointOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)

    plan = optimizer.run(
        _profile_for(feature),
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=True,
            enable_caching=True,
            num_samples=3,
        ),
    )

    stats = optimizer.integrated_optimizer_stats
    assert plan.validate()
    assert stats["reorder_candidates"] > 1
    assert stats["cache_candidates"] >= stats["reorder_candidates"]
    assert stats["physical_candidates"] == stats["cache_candidates"]
    assert math.isfinite(stats["best_cost"])


def test_dataset_selector_uses_cedar_joint_optimizer():
    feature = _make_feature()
    dataset = DataSet(
        CedarContext(),
        {"feature": feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=False,
        optimizer_options=OptimizerOptions(use_my_optimizer=6),
    )

    assert isinstance(dataset.features["feature"].optimizer, CedarJointOptimizer)
