from cedar.compose.feature import apply_profile_matched_resources
from cedar.compose.optimizer import PhysicalPlan, PipeDesc
from cedar.pipes import PipeVariantContextFactory, PipeVariantType


def _desc(variant: PipeVariantType, fused=None) -> PipeDesc:
    return PipeDesc(
        name="FusedPipe" if fused else "pipe",
        variant_type=variant,
        variant_ctx=PipeVariantContextFactory.create_context(variant),
        fused_pipes=fused,
    )


def _profile(width: int = 1):
    return {
        "resource_config": {
            "schema_version": 1,
            "profile_scope": "single_local_worker",
            "profile_local_workers": 1,
            "actors_per_stage": width,
            "ray_actors_per_stage": width,
            "smp_procs_per_stage": width,
        }
    }


def test_mixed_plan_uses_final_active_stages_and_budget():
    plan = PhysicalPlan(
        graph={0: {7}, 7: {6}, 6: set()},
        pipe_descs={
            0: _desc(PipeVariantType.INPROCESS),
            1: _desc(PipeVariantType.RAY),  # fused-away descriptor
            7: _desc(PipeVariantType.RAY, fused=[1, 2, 3]),
            6: _desc(PipeVariantType.SMP),
        },
    )

    signature = apply_profile_matched_resources(plan, _profile(), 64)

    assert plan.n_local_workers == 21
    assert plan.pipe_descs[7].variant_ctx.n_actors == 1
    assert plan.pipe_descs[6].variant_ctx.n_procs == 1
    assert signature == {
        "cpu_budget": 64,
        "local_workers": 21,
        "local_worker_policy": "max_under_cpu_budget",
        "ray_stages": 1,
        "smp_stages": 1,
        "ray_actors_per_stage_per_worker": 1,
        "smp_procs_per_stage_per_worker": 1,
        "global_ray_actors": 21,
        "global_smp_procs": 21,
        "gpu_ray_stages": 0,
        "global_gpu_actors": 0,
        "gpu_fraction_per_actor": 0.0,
        "total_accounted_gpus": 0.0,
        "total_accounted_cpus": 63,
    }


def test_rejects_profile_with_different_width():
    plan = PhysicalPlan(
        graph={0: set()},
        pipe_descs={0: _desc(PipeVariantType.RAY)},
    )
    try:
        apply_profile_matched_resources(plan, _profile(width=8), 64)
    except RuntimeError as exc:
        assert "actors_per_stage=1" in str(exc)
    else:
        raise AssertionError("mismatched profile width was accepted")


def test_rejects_active_ray_ds_stage():
    plan = PhysicalPlan(
        graph={7: set()},
        pipe_descs={7: _desc(PipeVariantType.RAY_DS, fused=[0, 1])},
    )

    try:
        apply_profile_matched_resources(plan, _profile(), 64)
    except RuntimeError as exc:
        assert "does not support active RAY_DS stages" in str(exc)
    else:
        raise AssertionError("active RAY_DS stage was accepted")


def test_fixed_local_worker_ablation_preserves_stage_widths():
    plan = PhysicalPlan(
        graph={0: {7}, 7: {6}, 6: set()},
        pipe_descs={
            0: _desc(PipeVariantType.INPROCESS),
            7: _desc(PipeVariantType.RAY, fused=[1, 2, 3]),
            6: _desc(PipeVariantType.SMP),
        },
    )

    signature = apply_profile_matched_resources(
        plan, _profile(), 64, fixed_local_workers=1
    )

    assert plan.n_local_workers == 1
    assert plan.pipe_descs[7].variant_ctx.n_actors == 1
    assert plan.pipe_descs[6].variant_ctx.n_procs == 1
    assert signature["local_worker_policy"] == "fixed_ablation"
    assert signature["global_ray_actors"] == 1
    assert signature["global_smp_procs"] == 1
    assert signature["total_accounted_cpus"] == 3
