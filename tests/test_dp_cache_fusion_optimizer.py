import math
from typing import List, Type

import pytest

from cedar.compose import Feature
from cedar.compose.dp_optimizer import DpOptimizer
from cedar.compose.my_optimizer import MyOptimizer
from cedar.compose.optimizer import Optimizer, OptimizerOptions
from cedar.pipes import MapperPipe, Pipe
from cedar.sources import IterSource


def _add_one(x):
    return x + 1


def _double(x):
    return x * 2


class TwoMapFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]):
        ft = source_pipes[0]
        ft = MapperPipe(ft, _add_one)
        ft = MapperPipe(ft, _double)
        return ft


def _profile_for(feature: Feature):
    latencies = {}
    input_sizes = {}
    output_sizes = {}
    for p_id, pipe in feature.logical_pipes.items():
        latencies[p_id] = 1.0 if pipe.is_source() else 100.0
        if pipe.is_source():
            input_sizes[p_id] = 1.0
            output_sizes[p_id] = 1.0
        else:
            input_sizes[p_id] = 1.0
            output_sizes[p_id] = 1.0

    return {
        "baseline": {
            "throughput": 100.0,
            "latencies": latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
        },
        "disk_info": {
            "read_latency": 0.000001,
        },
        "offloads": {},
    }


def _run_optimizer(optimizer_cls: Type[Optimizer]):
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))

    optimizer = optimizer_cls()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    plan = optimizer.run(
        _profile_for(feature),
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=False,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=True,
            enable_caching=True,
        ),
    )
    return optimizer, plan, feature


@pytest.mark.parametrize("optimizer_cls", [MyOptimizer, DpOptimizer])
def test_cache_can_be_inserted_after_materialized_fusion(optimizer_cls):
    optimizer, plan, feature = _run_optimizer(optimizer_cls)

    cache_p_ids = [
        p_id
        for p_id, desc in plan.pipe_descs.items()
        if p_id in plan.graph and desc.name == "ObjectDiskCachePipe"
    ]
    assert len(cache_p_ids) == 1
    cache_p_id = cache_p_ids[0]

    fused_p_ids = [
        p_id
        for p_id, desc in plan.pipe_descs.items()
        if p_id in plan.graph and desc.name == "FusedPipe"
    ]
    assert len(fused_p_ids) == 1
    fused_p_id = fused_p_ids[0]

    assert plan.graph[fused_p_id] == {cache_p_id}

    fused_desc = plan.pipe_descs[fused_p_id]
    map_p_ids = [
        p_id
        for p_id, pipe in feature.logical_pipes.items()
        if isinstance(pipe, MapperPipe)
    ]
    assert set(fused_desc.fused_pipes) == set(map_p_ids)

    cost = optimizer.calculate_cost(
        plan.graph,
        physical_specs=plan.pipe_descs,
        fused_pipes=[fused_desc.fused_pipes],
        caching_on=True,
        plan=plan,
    )
    assert math.isfinite(cost)
