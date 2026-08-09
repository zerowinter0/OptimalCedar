import math
from typing import List, Type

import pytest

from cedar.compose import Feature
from cedar.compose import constants
from cedar.compose.dp_optimizer import BlockCandidate, DpOptimizer
from cedar.compose.dp_two_stage_optimizer import DpTwoStageOptimizer
from cedar.compose.my_optimizer import MyOptimizer
from cedar.compose.optimizer import Optimizer, OptimizerOptions, PipeDesc
from cedar.pipes import MapperPipe, Pipe, PipeVariantType
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


def _ray_profile_for(feature: Feature):
    profile = _profile_for(feature)
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
    return profile


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


@pytest.mark.parametrize(
    "optimizer_cls", [MyOptimizer, DpOptimizer, DpTwoStageOptimizer]
)
def test_dp_final_ray_stages_use_cedar_batch_tuning(optimizer_cls):
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    optimizer = optimizer_cls()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)

    plan = optimizer.run(
        _ray_profile_for(feature),
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=True,
            enable_caching=False,
        ),
    )

    ray_contexts = [
        plan.pipe_descs[p_id].variant_ctx
        for p_id in plan.graph
        if plan.pipe_descs[p_id].variant_type == PipeVariantType.RAY
    ]
    assert ray_contexts
    for ctx in ray_contexts:
        assert ctx.submit_batch_size == 500
        expected_inflight = ctx.submit_batch_size * ctx.n_actors * 3
        assert ctx.max_inflight == expected_inflight
        assert ctx.max_prefetch == expected_inflight


def test_single_smp_stage_pays_placement_dependent_boundary_cost():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    profile = _profile_for(feature)
    for p_id in feature.logical_pipes:
        profile["baseline"]["input_sizes"][p_id] = 4_000_000.0
        profile["baseline"]["output_sizes"][p_id] = 4_000_000.0

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=False)
    optimizer._validate_stats()
    optimizer._init_stats()
    inner_ops = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner_ops)

    block = BlockCandidate(
        mask=1,
        order=(0,),
        variant=PipeVariantType.SMP,
        cost=0.0,
        materializes_fusion=False,
    )
    boundary_cost = optimizer._dp_stage_boundary_cost(0, block)

    assert math.isclose(boundary_cost, 80.0)
    assert optimizer._dp_stage_boundary_cost(
        0,
        BlockCandidate(
            mask=1,
            order=(0,),
            variant=PipeVariantType.INPROCESS,
            cost=0.0,
            materializes_fusion=False,
        ),
    ) == 0.0


def test_unfixed_output_pipe_is_always_constrained_to_remain_last():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = _profile_for(feature)
    optimizer.options = OptimizerOptions(
        enable_reorder=True, enable_offload=False
    )
    optimizer._validate_stats()
    optimizer._init_stats()

    inner_ops = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner_ops)

    output_p_id = optimizer._get_output_p_id(optimizer.physical_plan.graph)
    output_idx = inner_ops.index(output_p_id)
    assert set(optimizer._dp_pred_indices[output_idx]) == (
        set(range(len(inner_ops))) - {output_idx}
    )


def test_profiled_boundary_model_overrides_compatibility_constant():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    profile = _profile_for(feature)
    profile["physical_model"] = {
        "schema_version": 1,
        "boundary": {
            "SMP": {
                "fixed_latency_ms": 3.0,
                "throughput_bytes_per_sec": 2_000_000.0,
            }
        },
    }
    for p_id in feature.logical_pipes:
        profile["baseline"]["input_sizes"][p_id] = 1000.0
        profile["baseline"]["output_sizes"][p_id] = 1000.0

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=False)
    optimizer._validate_stats()
    optimizer._init_stats()
    inner_ops = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner_ops)

    block = BlockCandidate(
        mask=1,
        order=(0,),
        variant=PipeVariantType.SMP,
        cost=0.0,
        materializes_fusion=False,
    )

    # Cedar overlaps up to 100 requests: 3/100 ms fixed latency plus
    # 2,000 bytes / 2 MB/s = 1.03 ms steady-state service time.
    assert optimizer._dp_stage_boundary_cost(0, block) == pytest.approx(1.03)


def test_profiled_ray_boundary_is_reused_for_tf_ray():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "physical_model": {
            "boundary": {
                "RAY": {
                    "fixed_latency_ms": 0.5,
                    "throughput_bytes_per_sec": 123_456_789.0,
                }
            }
        }
    }

    assert optimizer._dp_boundary_throughput(
        PipeVariantType.TF_RAY
    ) == pytest.approx(123_456_789.0)
    assert optimizer._dp_boundary_fixed_latency_ms(
        PipeVariantType.TF_RAY
    ) == pytest.approx(0.5)


def test_unidentifiable_amdahl_cost_is_finite_and_capped():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    profile = _ray_profile_for(feature)
    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._validate_stats()
    optimizer._init_stats()

    p_id = next(
        p_id
        for p_id, pipe in feature.logical_pipes.items()
        if isinstance(pipe, MapperPipe)
    )
    profile["offloads"]["RAY"][p_id]["throughput"] = 1e12
    input_size = profile["baseline"]["input_sizes"][p_id]
    cost = optimizer._calculate_pipe_cost(
        p_id,
        input_size,
        PipeDesc(None, PipeVariantType.RAY, None),
    )

    assert cost > 0
    assert math.isclose(
        cost,
        optimizer._base_cost_map[p_id]
        / constants.MAX_UNIDENTIFIABLE_OPERATOR_SPEEDUP,
    )
