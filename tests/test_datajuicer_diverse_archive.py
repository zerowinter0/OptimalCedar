import json

from evaluation.chapter6_experiments.archive_datajuicer_diverse_results import (
    _copy_json_without_repo_absolute_paths,
    _validate_report,
)


def test_source_metadata_archive_removes_repo_absolute_paths(tmp_path):
    source = tmp_path / "metadata.json"
    source.write_text(
        json.dumps(
            {
                "archive": str(tmp_path / "datasets/archive.zip"),
                "upstream": "https://example.com/data",
                "revision": "abc123",
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "archive/source_metadata/data.json"

    _copy_json_without_repo_absolute_paths(source, destination, tmp_path)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "archive": "datasets/archive.zip",
        "upstream": "https://example.com/data",
        "revision": "abc123",
    }


def _formal_report():
    optimizers = (
        "optimizer",
        "dj_optimizer",
        "dp_cedar_optimizer",
        "dp_optimizer",
        "dp_two_stage_optimizer",
        "pecan_optimizer",
    )
    workloads = [f"workload_{index}" for index in range(6)]
    def run(values):
        return {
            "valid": True,
            "within_execution_limit": True,
            "formally_unavailable": False,
            "execution_times_sec": values,
            "mean_execution_time_sec": sum(values) / len(values),
        }

    def workload_item(index):
        is_win = index < 4
        dp_values = [7.5, 8.0, 8.5] if is_win else [9.5, 10.0, 10.5]
        other_values = [9.5, 10.0, 10.5]
        runs = {
            optimizer: run(
                dp_values if optimizer == "dp_optimizer" else other_values
            )
            for optimizer in optimizers
        }
        speedup = 10.0 / (8.0 if is_win else 10.0)
        return {
            "valid": True,
            "dp_at_least_20pct_faster": is_win,
            "best_other_optimizer": "optimizer",
            "best_other_execution_time_sec": 10.0,
            "dp_execution_time_sec": 8.0 if is_win else 10.0,
            "dp_speedup_over_best_other": speedup,
            "runs": runs,
        }
    return {
        "protocol": {
            "local_workers": 8,
            "cpu_budget": 64,
            "repetitions": 3,
            "optimizer_plan_limit_sec": 3600,
            "execution_limit_sec": 3600,
            "speedup_threshold": 1.20,
            "maximum_non_win_fraction": 0.40,
        },
        "selected_workloads": workloads,
        "ledger": {
            workload: workload_item(index)
            for index, workload in enumerate(workloads)
        },
        "summary": {
            "selected_count": 6,
            "wins": 4,
            "non_wins": 2,
            "failure_fraction": 2 / 6,
            "diversity": {"scenarios": [f"scenario_{index}" for index in range(6)]},
        },
    }


def test_archive_report_validation_enforces_formal_gates():
    report = _formal_report()
    _validate_report(report)

    report["ledger"]["workload_0"]["runs"].pop("pecan_optimizer")
    import pytest

    with pytest.raises(RuntimeError, match="six-optimizer"):
        _validate_report(report)


def test_archive_report_validation_rejects_more_than_forty_percent_nonwins():
    report = _formal_report()
    report["ledger"]["workload_3"]["runs"]["dp_optimizer"] = {
        "valid": True,
        "within_execution_limit": True,
        "formally_unavailable": False,
        "execution_times_sec": [9.5, 10.0, 10.5],
        "mean_execution_time_sec": 10.0,
    }
    report["ledger"]["workload_3"]["dp_at_least_20pct_faster"] = False
    report["ledger"]["workload_3"]["dp_execution_time_sec"] = 10.0
    report["ledger"]["workload_3"]["dp_speedup_over_best_other"] = 1.0
    report["summary"].update(wins=3, non_wins=3, failure_fraction=0.5)
    import pytest

    with pytest.raises(RuntimeError, match="exceeds 40%"):
        _validate_report(report)


def test_archive_report_validation_requires_successful_dp_and_comparator():
    import pytest

    report = _formal_report()
    dp = report["ledger"]["workload_0"]["runs"]["dp_optimizer"]
    dp.update(valid=False, formally_unavailable=True, execution_times_sec=[])
    with pytest.raises(RuntimeError, match="successful DP"):
        _validate_report(report)

    report = _formal_report()
    for optimizer, run in report["ledger"]["workload_0"]["runs"].items():
        if optimizer != "dp_optimizer":
            run.update(
                valid=False,
                formally_unavailable=True,
                execution_times_sec=[],
            )
    with pytest.raises(RuntimeError, match="successful comparator"):
        _validate_report(report)


def test_archive_report_validation_recomputes_means_and_speedups():
    import pytest

    report = _formal_report()
    report["ledger"]["workload_0"]["dp_speedup_over_best_other"] = 99.0
    with pytest.raises(RuntimeError, match="inconsistent derived result"):
        _validate_report(report)

    report = _formal_report()
    report["ledger"]["workload_0"]["runs"]["dp_optimizer"][
        "mean_execution_time_sec"
    ] = 99.0
    with pytest.raises(RuntimeError, match="inconsistent execution mean"):
        _validate_report(report)
