from evaluation.motivation_multimodal.pilot import (
    iter_threshold_configs,
    select_configuration,
)
from evaluation.motivation_multimodal.calibrate import derive_threshold_payloads


def test_declared_grid_has_27_unique_configs() -> None:
    configs = list(iter_threshold_configs())

    assert len(configs) == len({config.key for config in configs}) == 27
    assert {config.p_retention for config in configs} == {20, 35, 50}
    assert {config.q_retention for config in configs} == {70, 80, 90}
    assert {config.a_retention for config in configs} == {40, 60, 80}
    assert {config.c_retention for config in configs} == {80}
    assert {config.b_retention for config in configs} == {80}


def test_selection_uses_speedup_then_cardinality_then_key() -> None:
    base = {
        "status": "success",
        "equivalent": True,
        "joint_backend_count": 2,
        "staged_median_sec": 10.0,
        "joint_median_sec": 8.0,
        "cedar_staged_cost": 1.0,
        "cedar_joint_cost": 2.0,
    }
    results = [
        {**base, "key": "p35-q80-a60-c80-b80", "output_count": 12},
        {**base, "key": "p20-q80-a60-c80-b80", "output_count": 15},
        {**base, "key": "p20-q70-a60-c80-b80", "output_count": 15},
        {
            **base,
            "key": "p50-q90-a80-c80-b80",
            "output_count": 100,
            "joint_median_sec": 8.5,
        },
    ]

    chosen = select_configuration(results)

    assert chosen.key == "p20-q70-a60-c80-b80"
    assert chosen.output_count == 15


def test_selection_rejects_nonqualifying_results() -> None:
    result = {
        "key": "p20-q70-a40-c80-b80",
        "status": "success",
        "equivalent": True,
        "joint_backend_count": 2,
        "staged_median_sec": 8.0,
        "joint_median_sec": 10.0,
        "cedar_staged_cost": 1.0,
        "cedar_joint_cost": 2.0,
        "output_count": 10,
    }

    try:
        select_configuration([result])
    except ValueError as exc:
        assert "No pilot configuration" in str(exc)
    else:
        raise AssertionError("A slower joint plan must not qualify")


def test_threshold_quantiles_follow_filter_directions() -> None:
    rows = [
        {
            "record_id": str(index),
            "perplexity": float(index),
            "sharpness": float(index),
            "aesthetic": float(index),
            "clip": float(index),
            "blip": float(index),
        }
        for index in range(1, 101)
    ]

    payloads = derive_threshold_payloads(rows)
    low_retention = payloads["p20-q70-a40-c80-b80"]["thresholds"]

    assert low_retention["perplexity_max"] < 21.0
    assert 30.0 < low_retention["sharpness_min"] < 32.0
    assert 60.0 < low_retention["aesthetic_min"] < 62.0
    assert 20.0 < low_retention["clip_min"] < 22.0
    assert 20.0 < low_retention["blip_min"] < 22.0
