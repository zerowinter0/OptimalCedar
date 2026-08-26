import math

from cedar.compose.dp_optimizer import BlockCandidate, DpObjectiveCost, DpOptimizer
from cedar.compose.feature import apply_profile_matched_resources
from cedar.compose.optimizer import PhysicalPlan, PipeDesc
from cedar.pipes import (
    PipeExecutionResource,
    PipeVariantContextFactory,
    PipeVariantType,
)


def _ray_desc(resource: PipeExecutionResource) -> PipeDesc:
    return PipeDesc(
        name="stage",
        variant_type=PipeVariantType.RAY,
        variant_ctx=PipeVariantContextFactory.create_context(
            PipeVariantType.RAY,
            {"n_actors": 1},
        ),
        execution_resource=resource,
    )


def test_single_gpu_service_demand_accumulates_across_cuda_stages(
    monkeypatch,
):
    monkeypatch.setenv("CEDAR_MATCH_PROFILE_RESOURCES", "1")
    monkeypatch.setenv("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", "8")
    optimizer = DpOptimizer()
    monkeypatch.setattr(
        optimizer,
        "_dp_stage_boundary_components",
        lambda prev_mask, block: (0.0, 0.0),
    )
    first = BlockCandidate(
        mask=1,
        order=(0,),
        variant=PipeVariantType.RAY,
        cost=7.0,
        materializes_fusion=False,
        execution_resource=PipeExecutionResource.CUDA,
    )
    second = BlockCandidate(
        mask=2,
        order=(1,),
        variant=PipeVariantType.RAY,
        cost=11.0,
        materializes_fusion=False,
        execution_resource=PipeExecutionResource.CUDA,
    )

    objective = optimizer._dp_accumulate_objective_cost(
        DpObjectiveCost(), 7.0, first, 0
    )
    objective = optimizer._dp_accumulate_objective_cost(
        objective, 11.0, second, 1
    )

    assert objective.gpu_serial == 8 * 18.0
    assert objective.parallel_bottleneck == 0.0
    assert objective.score == 8 * 18.0


def test_profile_matched_plan_shares_exactly_one_gpu_across_all_actors():
    plan = PhysicalPlan(
        graph={0: {1}, 1: {2}, 2: set()},
        pipe_descs={
            0: _ray_desc(PipeExecutionResource.CUDA),
            1: _ray_desc(PipeExecutionResource.CPU),
            2: _ray_desc(PipeExecutionResource.CUDA),
        },
    )
    profile = {
        "resource_config": {
            "schema_version": 1,
            "profile_scope": "single_local_worker",
            "profile_local_workers": 1,
            "ray_actors_per_stage": 1,
            "smp_procs_per_stage": 1,
            "actors_per_stage": 1,
        }
    }

    signature = apply_profile_matched_resources(
        plan,
        profile,
        cpu_budget=64,
        fixed_local_workers=8,
    )

    assert signature["global_gpu_actors"] == 32
    assert math.isclose(signature["gpu_fraction_per_actor"], 1 / 32)
    assert math.isclose(signature["total_accounted_gpus"], 1.0)
    assert plan.pipe_descs[0].variant_ctx.num_gpus == 1 / 32
    assert plan.pipe_descs[1].variant_ctx.num_gpus == 0.0
    assert plan.pipe_descs[2].variant_ctx.num_gpus == 1 / 32


def test_joint_plan_preserves_widths_under_same_remote_budget():
    plan = PhysicalPlan(
        graph={0: {1}, 1: set()},
        pipe_descs={
            0: _ray_desc(PipeExecutionResource.CPU),
            1: _ray_desc(PipeExecutionResource.CPU),
        },
    )
    plan.pipe_descs[0].variant_ctx.n_actors = 4
    plan.pipe_descs[1].variant_ctx.n_actors = 2
    profile = {
        "resource_config": {
            "schema_version": 1,
            "profile_scope": "single_local_worker",
            "profile_local_workers": 1,
            "ray_actors_per_stage": 1,
            "smp_procs_per_stage": 1,
            "actors_per_stage": 1,
        }
    }

    signature = apply_profile_matched_resources(
        plan,
        profile,
        cpu_budget=64,
        fixed_local_workers=8,
        preserve_optimizer_widths=True,
    )

    assert signature["allocation_policy"] == "joint_dp"
    assert signature["remote_budget_per_worker"] == 6
    assert signature["ray_actor_widths_per_stage_per_worker"] == [4, 2]
    assert signature["global_ray_actors"] == 48
    assert plan.pipe_descs[0].variant_ctx.n_actors == 4
    assert plan.pipe_descs[1].variant_ctx.n_actors == 2


def test_physical_plan_round_trip_preserves_gpu_annotation_and_fraction():
    desc = _ray_desc(PipeExecutionResource.CUDA)
    desc.variant_ctx.num_gpus = 0.125
    plan = PhysicalPlan(graph={0: set()}, pipe_descs={0: desc})

    restored = PhysicalPlan.from_dict(plan.to_dict())

    assert (
        restored.pipe_descs[0].execution_resource
        == PipeExecutionResource.CUDA
    )
    assert restored.pipe_descs[0].variant_ctx.num_gpus == 0.125
