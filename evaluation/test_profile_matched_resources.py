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

    signature = apply_profile_matched_resources(
        plan, _profile(), 64, fixed_local_workers=8
    )

    assert plan.n_local_workers == 8
    assert plan.pipe_descs[7].variant_ctx.n_actors == 7
    assert plan.pipe_descs[6].variant_ctx.n_procs == 6
    assert signature["ray_budget_per_worker"] == 7
    assert signature["smp_budget_per_worker"] == 6
    assert signature["global_ray_actors"] == 56
    assert signature["global_smp_procs"] == 48
    assert signature["total_accounted_local_cpus"] == 56
    assert signature["total_accounted_ray_cpus"] == 56


def test_rejects_profile_with_nonpositive_width():
    plan = PhysicalPlan(
        graph={0: set()},
        pipe_descs={0: _desc(PipeVariantType.RAY)},
    )
    try:
        apply_profile_matched_resources(plan, _profile(width=0), 64)
    except RuntimeError as exc:
        assert "must be >= 1" in str(exc)
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


def test_fixed_local_worker_ablation_allocates_each_pool_independently():
    plan = PhysicalPlan(
        graph={0: {7}, 7: {6}, 6: set()},
        pipe_descs={
            0: _desc(PipeVariantType.INPROCESS),
            7: _desc(PipeVariantType.RAY, fused=[1, 2, 3]),
            6: _desc(PipeVariantType.SMP),
        },
    )

    signature = apply_profile_matched_resources(
        plan, _profile(), 64, fixed_local_workers=8
    )

    assert plan.n_local_workers == 8
    assert plan.pipe_descs[7].variant_ctx.n_actors == 7
    assert plan.pipe_descs[6].variant_ctx.n_procs == 6
    assert signature["local_worker_policy"] == "fixed_ablation"
    assert signature["global_ray_actors"] == 56
    assert signature["global_smp_procs"] == 48
    assert signature["total_accounted_cpus"] == 112
