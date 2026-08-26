import pytest

from evaluation.chapter6_experiments.cost_model_accuracy_metrics import (
    pairwise_qerrors,
    qerror_summary,
)


def test_pairwise_qerror_cancels_workload_wide_runtime_scale():
    candidates = [
        {"model_cost": 1.0, "mean_runtime_sec": 10.0},
        {"model_cost": 2.0, "mean_runtime_sec": 40.0},
        {"model_cost": 4.0, "mean_runtime_sec": 20.0},
    ]
    qerrors, correct, total = pairwise_qerrors(candidates, "model_cost")
    scaled = [
        {**candidate, "mean_runtime_sec": 100.0 * candidate["mean_runtime_sec"]}
        for candidate in candidates
    ]
    scaled_qerrors, _, _ = pairwise_qerrors(scaled, "model_cost")

    assert sorted(qerrors) == pytest.approx([2.0, 2.0, 4.0])
    assert scaled_qerrors == pytest.approx(qerrors)
    assert (correct, total) == (2, 3)
    summary = qerror_summary(qerrors)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["p90"] == pytest.approx(3.6)
