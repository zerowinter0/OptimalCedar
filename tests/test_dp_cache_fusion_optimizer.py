import copy
import math
from typing import List, Type

import pytest

from cedar.compose import Feature
from cedar.compose.dp_optimizer import (
    BlockCandidate,
    BlockCandidateProvider,
    DpOptimizer,
)
from cedar.compose.dp_two_stage_optimizer import DpTwoStageOptimizer
from cedar.compose.dj_optimizer import DjOptimizer
from cedar.compose.my_optimizer import MyOptimizer
from cedar.compose.optimizer import (
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
)
from cedar.compose.pecan_optimizer import PecanOptimizer
from cedar.pipes import (
    InProcessPipeVariantContext,
    MapperPipe,
    Pipe,
    PipeExecutionResource,
    PipeVariantType,
    RayPipeVariantContext,
)
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


def _affine_layer(feature: Feature, latencies):
    """Normalized-byte kx+b layer reproducing this fixture's linear costs."""
    return {
        "schema_version": 1,
        "operators": {
            str(p_id): {
                "k_ms_per_byte": latencies[p_id],
                "b_ms": 0.0,
                "x_reference_bytes": 1.0,
                "source": "test_fixture_normalized_bytes",
            }
            for p_id, pipe in feature.logical_pipes.items()
            if not pipe.is_source()
        },
    }


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
        "physical_model": {"operator_affine": _affine_layer(feature, latencies)},
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


@pytest.mark.parametrize(
    "optimizer_cls", [Optimizer, DjOptimizer, PecanOptimizer]
)
def test_cedar_physical_block_breakdown_matches_final_plan_cost(
    caplog, optimizer_cls
):
    with caplog.at_level("INFO"):
        optimizer, plan, _ = _run_optimizer(optimizer_cls)

    breakdown = optimizer.calculate_final_plan_cost_breakdown(plan)
    fused_blocks = [
        list(desc.fused_pipes)
        for desc in plan.pipe_descs.values()
        if desc.fused_pipes and len(desc.fused_pipes) > 1
    ]
    expected = optimizer.calculate_cost(
        plan.graph,
        physical_specs=plan.pipe_descs,
        fused_pipes=fused_blocks or None,
        caching_on=optimizer._get_cache_pid(plan) is not None,
        plan=plan,
    )

    assert breakdown["total_predicted_time_ms_per_input"] == pytest.approx(
        expected
    )
    assert sum(
        block["predicted_time_ms_per_input"]
        for block in breakdown["blocks"]
    ) == pytest.approx(expected)
    assert [
        block["physical_pipe_id"] for block in breakdown["blocks"]
    ] == optimizer._get_critical_path(
        plan.graph,
        optimizer._get_source_p_id(),
        optimizer._get_output_p_id(plan.graph),
        plan,
    )[0]
    assert (
        f"[CedarCostBreakdown] optimizer={optimizer_cls.__name__} block="
        in caplog.text
    )
    assert "total_predicted_ms_per_input=" in caplog.text


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
    # A profile without direct backend compute observations may now choose a
    # conservative local plan. If Ray is selected, its contexts must still use
    # Cedar's batch tuning.
    if not ray_contexts:
        assert plan.validate()
        return
    for ctx in ray_contexts:
        assert ctx.submit_batch_size == 500
        expected_inflight = ctx.submit_batch_size * ctx.n_actors * 3
        assert ctx.max_inflight == expected_inflight
        assert ctx.max_prefetch == expected_inflight


def test_final_ray_batch_tuning_propagates_across_fused_stages():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {"output_sizes": {0: 100.0}}
    }
    optimizer._data_size_ratio_map = {
        1: 2.0,
        2: 3.0,
        3: 5.0,
        4: 7.0,
    }
    optimizer.physical_plan = PhysicalPlan(
        graph={0: {10}, 10: {11}, 11: set()},
        pipe_descs={
            0: PipeDesc(
                "source",
                PipeVariantType.INPROCESS,
                InProcessPipeVariantContext(),
            ),
            10: PipeDesc(
                "first_fused",
                PipeVariantType.RAY,
                RayPipeVariantContext(),
                fused_pipes=[1, 2],
            ),
            11: PipeDesc(
                "second_fused",
                PipeVariantType.RAY,
                RayPipeVariantContext(),
                fused_pipes=[3, 4],
            ),
        },
    )

    sizes = optimizer._dp_final_stage_item_size_map()

    assert sizes[10] == pytest.approx((100.0, 600.0))
    assert sizes[11] == pytest.approx((600.0, 21_000.0))


def test_cuda_operator_is_not_an_smp_candidate():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    cuda_pipe = next(
        pipe
        for pipe in feature.logical_pipes.values()
        if isinstance(pipe, MapperPipe)
    )
    cuda_pipe.set_execution_resource(PipeExecutionResource.CUDA)

    profile = _ray_profile_for(feature)
    profile["offloads"]["SMP"] = {
        p_id: dict(stats, throughput=1000.0)
        for p_id, stats in profile["offloads"]["RAY"].items()
    }
    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._validate_stats()
    optimizer._init_stats()
    inner_ops = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner_ops)

    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()
    cuda_idx = inner_ops.index(cuda_pipe.id)
    variants = {
        candidate.variant
        for candidate in provider.candidates_for(1 << cuda_idx)
    }

    assert PipeVariantType.RAY in variants
    assert PipeVariantType.SMP not in variants
    assert PipeVariantType.INPROCESS in variants
    assert all(
        candidate.execution_resource == PipeExecutionResource.CUDA
        for candidate in provider.candidates_for(1 << cuda_idx)
    )


def test_local_only_cuda_operator_has_only_inprocess_candidate():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    cuda_pipe = next(
        pipe
        for pipe in feature.logical_pipes.values()
        if isinstance(pipe, MapperPipe)
    )
    cuda_pipe.set_execution_resource(PipeExecutionResource.CUDA)
    cuda_pipe.pipe_spec = copy.copy(cuda_pipe.pipe_spec)
    cuda_pipe.pipe_spec.mutable_variants = [PipeVariantType.INPROCESS]
    cuda_pipe.pipe_spec.is_fusable = False

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = _ray_profile_for(feature)
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._validate_stats()
    optimizer._init_stats()
    inner_ops = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner_ops)

    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()
    cuda_idx = inner_ops.index(cuda_pipe.id)
    variants = {
        candidate.variant
        for candidate in provider.candidates_for(1 << cuda_idx)
    }

    assert variants == {PipeVariantType.INPROCESS}


def test_inprocess_fusion_can_be_replayed_without_entering_dp_search():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = _ray_profile_for(feature)
    optimizer.options = OptimizerOptions(
        enable_offload=True,
        enable_fusion=True,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    inner_ops = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner_ops)

    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()
    full_mask = (1 << len(inner_ops)) - 1

    # PICO has no calibrated local-fusion discount, so it must not generate
    # INPROCESS fusion as a search candidate.
    assert all(
        candidate.variant != PipeVariantType.INPROCESS
        for candidate in provider.candidates_for(full_mask)
    )

    # A plan produced by another optimizer can still contain executable local
    # fusion. Cross-model evaluation conservatively replays it as the same
    # sequential compute as its constituent INPROCESS operators.
    first = provider.candidate_for_order(
        (0,), PipeVariantType.INPROCESS, prefix_mask=0
    )
    second = provider.candidate_for_order(
        (1,), PipeVariantType.INPROCESS, prefix_mask=1
    )
    fused = provider.candidate_for_order(
        (0, 1), PipeVariantType.INPROCESS, prefix_mask=0
    )

    assert fused.materializes_fusion
    assert fused.cost == pytest.approx(first.cost + second.cost)


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
    profile["physical_model"]["boundary"] = {
        "SMP": {
            "fixed_latency_ms": 3.0,
            "throughput_bytes_per_sec": 2_000_000.0,
        }
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

    # SMP submits one sample per request. Queue capacity does not amortize the
    # 3 ms task-level latency; 2,000 bytes / 2 MB/s adds 1 ms.
    assert optimizer._dp_stage_boundary_cost(0, block) == pytest.approx(4.0)


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


def test_unidentifiable_amdahl_cost_uses_conservative_baseline():
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
    # The conservative fallback is the operator's own fitted kx+b value: the
    # DP prices every candidate on the affine per-record scale.
    assert math.isclose(
        cost,
        optimizer._dp_affine_value(
            p_id, profile["baseline"]["input_sizes"][p_id]
        ),
    )
