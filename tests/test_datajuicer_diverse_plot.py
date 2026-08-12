import pytest

from evaluation.chapter6_experiments.plot_datajuicer_diverse_workloads import (
    _plot,
    _rows,
)


def test_plot_rows_require_complete_six_optimizer_evidence(tmp_path):
    optimizers = (
        "optimizer",
        "dj_optimizer",
        "dp_cedar_optimizer",
        "dp_optimizer",
        "dp_two_stage_optimizer",
        "pecan_optimizer",
    )
    report = {
        "selected_workloads": ["alpaca_cot"],
        "ledger": {
            "alpaca_cot": {
                "expected_samples": 65000,
                "runs": {
                    optimizer: {
                        "valid": True,
                        "formally_unavailable": False,
                        "mean_execution_time_sec": 10.0,
                        "stddev_execution_time_sec": 0.5,
                        "execution_times_sec": [9.5, 10.0, 10.5],
                    }
                    for optimizer in optimizers
                },
            }
        },
    }

    rows = _rows(report)
    assert len(rows) == 6
    assert {row["optimizer"] for row in rows} == set(optimizers)
    assert all(row["repetitions"] == 3 for row in rows)
    figure = tmp_path / "runtime.png"
    _plot(report, rows, figure)
    assert figure.stat().st_size > 0

    report["ledger"]["alpaca_cot"]["runs"]["optimizer"] = {
        "valid": False,
        "formally_unavailable": False,
    }
    with pytest.raises(RuntimeError, match="incomplete formal evidence"):
        _rows(report)
