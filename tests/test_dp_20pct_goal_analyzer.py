import json

from evaluation.chapter6_experiments import analyze_dp_20pct_goal as analyzer


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_success(root, workload, optimizer, seconds):
    _write_json(
        root / workload / "plan_results" / f"{optimizer}.json",
        {
            "runs": [
                {
                    "optimizer": optimizer,
                    "setup_time_sec": 1.5,
                }
            ]
        },
    )
    for round_number, runtime in enumerate(seconds, start=1):
        _write_json(
            root
            / workload
            / "results"
            / f"round{round_number}__{optimizer}.json",
            {
                "epoch_run_times": [runtime],
                "epoch_num_samples": [20_000],
            },
        )


def _write_plan_timeout(root, workload, optimizer):
    _write_json(
        root / workload / "status" / f"plan__{optimizer}.json",
        {
            "status": "optimizer_timeout",
            "reason": "plan optimization exceeded 300s",
        },
    )
    for round_number in range(1, 4):
        _write_json(
            root
            / workload
            / "status"
            / f"round{round_number}__{optimizer}.json",
            {
                "status": "skipped",
                "reason": "optimizer did not produce a valid plan",
            },
        )


def _write_execution_timeouts(root, workload, optimizer):
    _write_json(
        root / workload / "plan_results" / f"{optimizer}.json",
        {
            "runs": [
                {
                    "optimizer": optimizer,
                    "setup_time_sec": 2.0,
                }
            ]
        },
    )
    for round_number in range(1, 4):
        _write_json(
            root
            / workload
            / "status"
            / f"round{round_number}__{optimizer}.json",
            {
                "status": "infeasible_timeout",
                "reason": "execution exceeded 3600s",
            },
        )


def _write_source_exhausted(
    root, workload, optimizer, samples=2_500, repeats=3
):
    _write_json(
        root / workload / "plan_results" / f"{optimizer}.json",
        {
            "runs": [
                {
                    "optimizer": optimizer,
                    "setup_time_sec": 2.0,
                }
            ]
        },
    )
    for round_number in range(1, repeats + 1):
        _write_json(
            root
            / workload
            / "results"
            / f"round{round_number}__{optimizer}.json",
            {
                "epoch_run_times": [10.0],
                "epoch_num_samples": [samples],
            },
        )


def test_three_complete_repetitions_are_successful(tmp_path):
    _write_success(tmp_path, "candidate", "dp_optimizer", [1.0, 1.1, 0.9])

    result = analyzer._read_candidate(
        tmp_path, "candidate", "dp_optimizer"
    )

    assert result["outcome"] == "success"
    assert result["valid"]
    assert result["processed_samples"] == [20_000, 20_000, 20_000]
    assert result["optimization_time_sec"] == 1.5


def test_one_repetition_with_reduced_output_is_configurable(tmp_path):
    _write_success(tmp_path, "candidate", "dp_optimizer", [1.0])
    result_path = (
        tmp_path
        / "candidate"
        / "results"
        / "round1__dp_optimizer.json"
    )
    payload = json.loads(result_path.read_text())
    payload["epoch_num_samples"] = [10_000]
    _write_json(result_path, payload)

    result = analyzer._read_candidate(
        tmp_path,
        "candidate",
        "dp_optimizer",
        expected_repeats=1,
        expected_samples=10_000,
    )

    assert result["outcome"] == "success"
    assert result["valid"]
    assert result["processed_samples"] == [10_000]


def test_optimizer_file_matching_does_not_consume_prefixed_optimizers(
    tmp_path,
):
    _write_success(
        tmp_path, "candidate", "optimizer", [10.0, 11.0, 12.0]
    )
    _write_success(
        tmp_path, "candidate", "dj_optimizer", [20.0, 21.0, 22.0]
    )
    _write_success(
        tmp_path, "candidate", "dp_optimizer", [30.0, 31.0, 32.0]
    )
    _write_json(
        tmp_path
        / "candidate"
        / "status"
        / "round1__dp_two_stage_optimizer.json",
        {
            "status": "infeasible_timeout",
            "reason": "execution exceeded 3600s",
        },
    )

    result = analyzer._read_candidate(
        tmp_path, "candidate", "optimizer"
    )

    assert result["outcome"] == "success"
    assert result["execution_times_sec"] == [10.0, 11.0, 12.0]
    assert result["processed_samples"] == [20_000, 20_000, 20_000]
    assert result["statuses"] == []


def test_formal_plan_timeout_is_unavailable_not_invalid(tmp_path):
    _write_plan_timeout(tmp_path, "candidate", "optimizer")

    result = analyzer._read_candidate(tmp_path, "candidate", "optimizer")

    assert result["outcome"] == "unavailable"
    assert result["formally_unavailable"]
    assert not result["valid"]


def test_unavailable_competitor_does_not_invalidate_workload(
    tmp_path, monkeypatch
):
    optimizers = ("optimizer", "dp_optimizer", "pecan_optimizer")
    monkeypatch.setattr(analyzer, "OPTIMIZERS", optimizers)
    _write_plan_timeout(tmp_path, "candidate", "optimizer")
    _write_success(
        tmp_path, "candidate", "dp_optimizer", [8.0, 8.0, 8.0]
    )
    _write_success(
        tmp_path, "candidate", "pecan_optimizer", [10.0, 10.0, 10.0]
    )

    result = analyzer._candidate_summary(tmp_path, "candidate")

    assert result["valid"]
    assert result["best_other_optimizer"] == "pecan_optimizer"
    assert result["dp_speedup_over_best_other"] == 1.25
    assert result["dp_at_least_20pct_faster"]


def test_three_execution_timeouts_provide_conservative_lower_bound(
    tmp_path, monkeypatch
):
    optimizers = ("optimizer", "dp_optimizer")
    monkeypatch.setattr(analyzer, "OPTIMIZERS", optimizers)
    _write_execution_timeouts(tmp_path, "candidate", "optimizer")
    _write_success(
        tmp_path, "candidate", "dp_optimizer", [2900.0, 3000.0, 2950.0]
    )

    result = analyzer._candidate_summary(tmp_path, "candidate")

    assert result["valid"]
    assert result["best_other_optimizer"] == "optimizer"
    assert result["best_other_execution_time_sec"] == 3600.0
    assert result["best_other_is_lower_bound"]
    assert result["dp_speedup_is_lower_bound"]
    assert result["dp_speedup_over_best_other"] > 1.20
    assert result["dp_at_least_20pct_faster"]


def test_one_execution_timeout_is_not_complete_evidence(
    tmp_path, monkeypatch
):
    optimizers = ("optimizer", "dp_optimizer")
    monkeypatch.setattr(analyzer, "OPTIMIZERS", optimizers)
    _write_json(
        tmp_path
        / "candidate"
        / "status"
        / "round1__optimizer.json",
        {
            "status": "infeasible_timeout",
            "reason": "execution exceeded 3600s",
        },
    )
    _write_success(
        tmp_path, "candidate", "dp_optimizer", [100.0, 100.0, 100.0]
    )

    result = analyzer._candidate_summary(tmp_path, "candidate")

    assert not result["valid"]
    assert result["best_other_execution_time_sec"] is None
    assert not result["dp_at_least_20pct_faster"]


def test_first_timeout_and_skipped_repeats_are_unavailable_without_bound(
    tmp_path,
):
    _write_json(
        tmp_path
        / "candidate"
        / "status"
        / "round1__optimizer.json",
        {
            "status": "infeasible_timeout",
            "reason": "execution exceeded 3600s",
        },
    )
    for round_number in (2, 3):
        _write_json(
            tmp_path
            / "candidate"
            / "status"
            / f"round{round_number}__optimizer.json",
            {
                "status": "skipped_after_timeout",
                "reason": "earlier repeat exceeded 3600s",
            },
        )

    result = analyzer._read_candidate(tmp_path, "candidate", "optimizer")

    assert result["outcome"] == "unavailable"
    assert result["formally_unavailable"] is True
    assert result["execution_time_lower_bound_sec"] is None


def test_three_consistent_short_epochs_are_source_infeasible(tmp_path):
    _write_source_exhausted(tmp_path, "candidate", "dp_optimizer")

    result = analyzer._read_candidate(
        tmp_path, "candidate", "dp_optimizer"
    )

    assert result["outcome"] == "source_infeasible"
    assert result["source_infeasible"]
    assert not result["valid"]


def test_one_short_epoch_without_workload_marker_is_not_enough(tmp_path):
    _write_source_exhausted(
        tmp_path, "candidate", "dp_optimizer", repeats=1
    )

    result = analyzer._read_candidate(
        tmp_path, "candidate", "dp_optimizer"
    )

    assert result["outcome"] == "invalid"
    assert not result["source_infeasible"]
    assert not result["valid"]


def test_workload_source_marker_is_a_denominator_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        analyzer, "OPTIMIZERS", ("optimizer", "dp_optimizer")
    )
    _write_source_exhausted(
        tmp_path, "candidate", "optimizer", repeats=1
    )
    _write_json(
        tmp_path / "candidate" / "status" / "source_infeasible.json",
        {
            "status": "source_infeasible",
            "reason": "fewer than 20000 records",
        },
    )

    result = analyzer._candidate_summary(tmp_path, "candidate")

    assert result["workload_outcome"] == "source_infeasible"
    assert not result["valid"]
    assert not result["dp_at_least_20pct_faster"]


def test_existing_report_includes_two_stage_but_ignores_no_optimizer(tmp_path):
    report_path = tmp_path / "cross_system.json"
    _write_json(
        report_path,
        {
            "workloads": {
                "workload": {
                    "entities": {
                        "cedar_dp_optimizer": {
                            "status": "success",
                            "execution_time_sec": 8.0,
                        },
                        "cedar_optimizer": {
                            "status": "success",
                            "execution_time_sec": 12.0,
                        },
                        "cedar_dp_two_stage_optimizer": {
                            "status": "success",
                            "execution_time_sec": 10.0,
                        },
                        "cedar_no_optimizer": {
                            "status": "success",
                            "execution_time_sec": 1.0,
                        },
                    }
                }
            }
        },
    )

    result = analyzer._existing_summaries(report_path)["workload"]

    assert result["best_other_optimizer"] == "cedar_dp_two_stage_optimizer"
    assert result["best_other_execution_time_sec"] == 10.0
    assert result["dp_speedup_over_best_other"] == 1.25
    assert result["dp_at_least_20pct_faster"]


def test_audit_discovers_and_combines_disjoint_candidate_roots(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        analyzer,
        "REGISTERED_CANDIDATE_WORKLOADS",
        ("pile_europarl", "pile_hackernews"),
    )
    monkeypatch.setattr(
        analyzer,
        "OPTIMIZERS",
        ("optimizer", "dp_optimizer"),
    )
    existing_report = tmp_path / "existing.json"
    _write_json(existing_report, {"workloads": {}})
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root, workload in (
        (first, "pile_europarl"),
        (second, "pile_hackernews"),
    ):
        _write_success(
            root, workload, "optimizer", [12.0, 12.0, 12.0]
        )
        _write_success(
            root, workload, "dp_optimizer", [8.0, 8.0, 8.0]
        )

    report = analyzer.audit([first, second], existing_report)

    assert set(report["candidates"]) == {
        "pile_europarl",
        "pile_hackernews",
    }
    assert report["summary"]["total_workloads"] == 2
    assert report["summary"]["dp_20pct_wins"] == 2
    assert report["summary"]["minimum_wins_required"] == 1
    assert report["summary"]["additional_wins_needed"] == 0
    assert report["summary"]["target_met"]


def test_final_audit_requires_every_pre_registered_candidate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        analyzer,
        "REGISTERED_CANDIDATE_WORKLOADS",
        ("pile_europarl", "pile_hackernews"),
    )
    monkeypatch.setattr(
        analyzer,
        "OPTIMIZERS",
        ("optimizer", "dp_optimizer"),
    )
    existing_report = tmp_path / "existing.json"
    _write_json(existing_report, {"workloads": {}})
    root = tmp_path / "candidates"
    _write_success(
        root, "pile_europarl", "optimizer", [12.0, 12.0, 12.0]
    )
    _write_success(
        root, "pile_europarl", "dp_optimizer", [8.0, 8.0, 8.0]
    )

    try:
        analyzer.audit(
            root,
            existing_report,
            require_all_registered=True,
        )
    except ValueError as exc:
        assert "pile_hackernews" in str(exc)
    else:
        raise AssertionError("missing registered candidate was accepted")


def test_six_candidates_and_ten_existing_workloads_require_five_wins(
    tmp_path, monkeypatch
):
    candidates = tuple(f"candidate_{index}" for index in range(6))
    monkeypatch.setattr(
        analyzer,
        "REGISTERED_CANDIDATE_WORKLOADS",
        candidates,
    )
    existing_workloads = {}
    for index in range(10):
        existing_workloads[f"existing_{index}"] = {
            "entities": {
                "cedar_dp_optimizer": {
                    "status": "success",
                    "execution_time_sec": 8.0 if index < 2 else 10.0,
                },
                "cedar_optimizer": {
                    "status": "success",
                    "execution_time_sec": 10.0,
                },
            },
        }
    existing_report = tmp_path / "existing.json"
    _write_json(existing_report, {"workloads": existing_workloads})
    candidate_root = tmp_path / "candidates"
    for workload in candidates:
        _write_json(
            candidate_root
            / workload
            / "status"
            / "source_infeasible.json",
            {
                "status": "source_infeasible",
                "reason": "registered denominator failure",
            },
        )

    report = analyzer.audit(
        candidate_root,
        existing_report,
        require_all_registered=True,
    )

    assert report["summary"]["total_workloads"] == 16
    assert report["summary"]["dp_20pct_wins"] == 2
    assert report["summary"]["minimum_wins_required"] == 5
    assert report["summary"]["additional_wins_needed"] == 3
    assert not report["summary"]["target_met"]
