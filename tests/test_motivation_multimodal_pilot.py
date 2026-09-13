from evaluation.motivation_multimodal.pilot import (
    iter_threshold_configs,
    select_configuration,
)
from evaluation.motivation_multimodal.calibrate import derive_threshold_payloads
from evaluation.motivation_multimodal.runner import (
    OPTIMIZER_SELECTORS,
    STAGED_OPTIMIZERS,
)
from evaluation.motivation_multimodal.four_plan_pilot import (
    optimized_plan_runtime_checks,
    plans_share_parallelism_protocol,
)


def test_runner_declares_all_six_sequential_stage_orders() -> None:
    assert STAGED_OPTIMIZERS == (
        "staged_rfo",
        "staged_rof",
        "staged_fro",
        "staged_for",
        "staged_orf",
        "staged_ofr",
    )
    assert {OPTIMIZER_SELECTORS[name] for name in STAGED_OPTIMIZERS} == {12}
    assert OPTIMIZER_SELECTORS["joint"] == 13


def test_declared_grid_has_27_unique_configs() -> None:
    configs = list(iter_threshold_configs())

    assert len(configs) == len({config.key for config in configs}) == 27
    assert {config.p_retention for config in configs} == {20, 35, 50}
    assert {config.q_retention for config in configs} == {70, 80, 90}
    assert {config.a_retention for config in configs} == {40, 60, 80}
    assert {config.c_retention for config in configs} == {80}
    assert {config.b_retention for config in configs} == {80}


def test_distinct_staged_grid_is_predeclared_and_unique() -> None:
    configs = list(iter_threshold_configs("distinct-staged"))

    assert len(configs) == len({config.key for config in configs}) == 18
    assert {config.p_retention for config in configs} == {50, 70, 90}
    assert {config.q_retention for config in configs} == {70, 90}
    assert {config.a_retention for config in configs} == {10, 20, 30}


def _qualifying_result(key: str, output_count: int = 10) -> dict:
    plan_hashes = {
        name: ("plan-a" if index % 2 == 0 else "plan-b")
        for index, name in enumerate(STAGED_OPTIMIZERS)
    }
    plans = {
        name: {"sha256": digest}
        for name, digest in plan_hashes.items()
    }
    plans["joint"] = {"sha256": "plan-j"}
    staged_times = {
        name: (9.0 if plan_hashes[name] == "plan-a" else 10.0)
        for name in STAGED_OPTIMIZERS
    }
    executions = {
        name: [{"seconds": value}] * 3
        for name, value in staged_times.items()
    }
    executions["joint"] = [{"seconds": 8.0}] * 3
    cedar_costs = {
        name: (2.0 if plan_hashes[name] == "plan-a" else 1.0)
        for name in STAGED_OPTIMIZERS
    }
    cedar_costs["joint"] = 0.5
    pico_costs = {
        name: (1.0 if plan_hashes[name] == "plan-a" else 2.0)
        for name in STAGED_OPTIMIZERS
    }
    pico_costs["joint"] = 0.5
    return {
        "key": key,
        "status": "success",
        "equivalent": True,
        "joint_backend_count": 2,
        "plans": plans,
        "executions": executions,
        "staged_median_sec": 9.0,
        "joint_median_sec": 8.0,
        "staged_medians_sec": staged_times,
        "cedar_costs": cedar_costs,
        "pico_costs": pico_costs,
        "output_count": output_count,
    }


def test_selection_uses_speedup_then_cardinality_then_key() -> None:
    results = [
        _qualifying_result("p35-q80-a60-c80-b80", 12),
        _qualifying_result("p20-q80-a60-c80-b80", 15),
        _qualifying_result("p20-q70-a60-c80-b80", 15),
        {
            **_qualifying_result("p50-q90-a80-c80-b80", 100),
            "joint_median_sec": 8.5,
        },
    ]

    chosen = select_configuration(results)

    assert chosen.key == "p20-q70-a60-c80-b80"
    assert chosen.output_count == 15


def test_selection_rejects_nonqualifying_results() -> None:
    result = _qualifying_result("p20-q70-a40-c80-b80")
    result["joint_median_sec"] = 10.0

    try:
        select_configuration([result])
    except ValueError as exc:
        assert "No pilot configuration" in str(exc)
    else:
        raise AssertionError("A slower joint plan must not qualify")


def test_selection_requires_joint_to_beat_every_staged_order() -> None:
    result = _qualifying_result("p20-q70-a40-c80-b80")
    result["staged_medians_sec"]["staged_ofr"] = 7.5
    result["executions"]["staged_ofr"] = [{"seconds": 7.5}] * 3

    try:
        select_configuration([result])
    except ValueError as exc:
        assert "No pilot configuration" in str(exc)
    else:
        raise AssertionError("Joint must beat the fastest staged order")


def test_selection_requires_a_cedar_pairwise_ranking_reversal() -> None:
    result = _qualifying_result("p20-q70-a40-c80-b80")
    result["cedar_costs"] = {
        name: result["pico_costs"][name]
        for name in (*STAGED_OPTIMIZERS, "joint")
    }

    try:
        select_configuration([result])
    except ValueError as exc:
        assert "No pilot configuration" in str(exc)
    else:
        raise AssertionError("A correct Cedar ranking cannot demonstrate misranking")


def test_selection_requires_pico_to_order_joint_and_staged_plans() -> None:
    result = _qualifying_result("p20-q70-a40-c80-b80")
    result["pico_costs"]["staged_rfo"] = 3.0
    result["pico_costs"]["staged_fro"] = 3.0
    result["pico_costs"]["staged_orf"] = 3.0

    try:
        select_configuration([result])
    except ValueError as exc:
        assert "No pilot configuration" in str(exc)
    else:
        raise AssertionError("PICO must match the full three-plan runtime order")


def test_threshold_quantiles_follow_filter_directions() -> None:
    rows = [
        {
            "record_id": str(index),
            "perplexity": float(index),
            "safety": float(index),
            "aesthetic": float(index),
            "clip": float(index),
            "blip": float(index),
        }
        for index in range(1, 101)
    ]

    payloads = derive_threshold_payloads(rows)
    low_retention = payloads["p20-q70-a40-c80-b80"]["thresholds"]

    assert low_retention["perplexity_max"] < 21.0
    assert 70.0 < low_retention["safety_max"] < 72.0
    assert 60.0 < low_retention["aesthetic_min"] < 62.0
    assert 20.0 < low_retention["clip_min"] < 22.0
    assert 20.0 < low_retention["blip_min"] < 22.0


def test_four_plan_runtime_gate_requires_clear_optimized_plan_gaps() -> None:
    checks = optimized_plan_runtime_checks(
        {
            "pico": 70.0,
            "manual": 80.0,
            "cedar": 100.0,
            "unoptimized": 120.0,
        },
        {
            "pico": 40.0,
            "manual": 80.0,
            "cedar": 60.0,
            "unoptimized": 100.0,
        },
    )

    assert checks == {
        "optimized_runtime_order_has_clear_gaps": True,
        "cedar_reverses_manual_vs_cedar": True,
    }


def test_four_plan_runtime_gate_rejects_small_runtime_gap() -> None:
    checks = optimized_plan_runtime_checks(
        {
            "pico": 91.0,
            "manual": 95.0,
            "cedar": 100.0,
            "unoptimized": 120.0,
        },
        {
            "pico": 40.0,
            "manual": 80.0,
            "cedar": 60.0,
            "unoptimized": 100.0,
        },
    )

    assert checks["optimized_runtime_order_has_clear_gaps"] is False


def test_four_plan_gate_accepts_shared_budgeted_parallelism_policy() -> None:
    plans = {
        name: {
            "parallelism_policy": "separate_pool_equal_share",
            "resource_signature": {
                "cpu_budget": 64,
                "ray_cpu_budget": 64,
                "total_accounted_local_cpus": 1,
                "total_accounted_ray_cpus": ray_cpus,
                "total_accounted_gpus": gpu_count,
            },
        }
        for name, ray_cpus, gpu_count in (
            ("unoptimized", 0, 0.0),
            ("cedar", 63, 1.0),
            ("pico", 63, 1.0),
            ("manual", 63, 1.0),
        )
    }

    assert plans_share_parallelism_protocol(plans) is True


def test_four_plan_gate_rejects_a_different_parallelism_allocator() -> None:
    plans = {
        name: {
            "parallelism_policy": (
                "joint_dp_separate_pools"
                if name == "pico"
                else "separate_pool_equal_share"
            ),
            "resource_signature": {
                "cpu_budget": 64,
                "ray_cpu_budget": 64,
                "total_accounted_local_cpus": 1,
                "total_accounted_ray_cpus": 63 if name != "unoptimized" else 0,
                "total_accounted_gpus": 1.0 if name != "unoptimized" else 0.0,
            },
        }
        for name in ("unoptimized", "cedar", "pico", "manual")
    }

    assert plans_share_parallelism_protocol(plans) is False
