import json

import pytest

from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
    _diversity_summary,
    _pile_summaries,
    _selection,
    _summarize_matrix,
)


def _item(valid=True, win=False, speedup=1.0):
    return {
        "valid": valid,
        "dp_at_least_20pct_faster": win,
        "dp_speedup_over_best_other": speedup,
    }


def _base_ledger():
    ledger = {
        name: _item(valid=True, win=True, speedup=1.5)
        for name in (
            "pile_europarl",
            "pile_hackernews",
            "pile_pubmed_abstracts",
            "pile_uspto_backgrounds",
        )
    }
    ledger.update(
        {
            "alpaca_cot": _item(valid=True, win=False, speedup=0.98),
            "general_video_refine": _item(
                valid=True, win=False, speedup=0.98
            ),
            "video_self_evolution": _item(valid=False),
            "llava_pretrain": _item(valid=True, win=False, speedup=0.88),
            "redpajama_code": _item(valid=False),
            "redpajama_arxiv": _item(valid=False),
        }
    )
    return ledger


def _write_result(root, round_number, optimizer, seconds, samples=100):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"round{round_number}__{optimizer}.json").write_text(
        '{"epoch_run_times": ['
        + str(seconds)
        + '], "epoch_num_samples": ['
        + str(samples)
        + "]}\n",
        encoding="utf-8",
    )


def test_summarize_matrix_uses_three_round_means_and_fastest_competitor(
    tmp_path,
):
    results = tmp_path / "results"
    competitor_times = {
        "optimizer": 20.0,
        "dj_optimizer": 15.0,
        "dp_cedar_optimizer": 18.0,
        "dp_two_stage_optimizer": 17.0,
        "pecan_optimizer": 16.0,
    }
    for round_number, dp_seconds in enumerate((10.0, 11.0, 9.0), start=1):
        _write_result(
            results, round_number, "dp_optimizer", dp_seconds
        )
        for optimizer, seconds in competitor_times.items():
            _write_result(results, round_number, optimizer, seconds)

    summary = _summarize_matrix(
        "example",
        100,
        {optimizer: [results] for optimizer in competitor_times}
        | {"dp_optimizer": [results]},
        "test evidence",
    )

    assert summary["valid"] is True
    assert summary["dp_execution_time_sec"] == pytest.approx(10.0)
    assert summary["best_other_optimizer"] == "dj_optimizer"
    assert summary["best_other_execution_time_sec"] == pytest.approx(15.0)
    assert summary["dp_speedup_over_best_other"] == pytest.approx(1.5)
    assert summary["dp_at_least_20pct_faster"] is True


def test_summarize_matrix_rejects_missing_optimizer_evidence(tmp_path):
    results = tmp_path / "results"
    for round_number in (1, 2, 3):
        _write_result(results, round_number, "dp_optimizer", 10.0)
        _write_result(results, round_number, "dj_optimizer", 15.0)

    summary = _summarize_matrix(
        "example",
        100,
        {
            "dp_optimizer": [results],
            "dj_optimizer": [results],
        },
        "incomplete evidence",
    )

    assert summary["valid"] is False


def test_pile_summary_requires_and_exposes_current_six_optimizer_evidence(
    tmp_path,
):
    workloads = {
        "pile_europarl": ("europarl_2500", 2500),
        "pile_hackernews": ("heldout_20000", 20000),
        "pile_pubmed_abstracts": ("heldout_20000", 20000),
        "pile_uspto_backgrounds": ("heldout_20000", 20000),
    }
    canonical_payload = {"candidates": {}}
    for workload, (run_id, samples) in workloads.items():
        canonical_payload["candidates"][workload] = {
            "runs": {
                "optimizer": {
                    "valid": False,
                    "formally_unavailable": True,
                    "mean_execution_time_sec": None,
                },
                "dj_optimizer": {
                    "valid": True,
                    "formally_unavailable": False,
                    "mean_execution_time_sec": 15.0,
                },
                "dp_cedar_optimizer": {
                    "valid": True,
                    "formally_unavailable": False,
                    "mean_execution_time_sec": 20.0,
                },
                "dp_optimizer": {
                    "valid": True,
                    "formally_unavailable": False,
                    "mean_execution_time_sec": 10.0,
                },
                "pecan_optimizer": {
                    "valid": True,
                    "formally_unavailable": False,
                    "mean_execution_time_sec": 18.0,
                },
            }
        }
        run_root = (
            tmp_path
            / "audit"
            / "datajuicer_candidate_runs"
            / run_id
        )
        results = run_root / workload / "results"
        for round_number in (1, 2, 3):
            _write_result(
                results, round_number, "optimizer", 16.0, samples
            )
            _write_result(
                results,
                round_number,
                "dp_two_stage_optimizer",
                12.0,
                samples,
            )
        (run_root / workload / "metadata.txt").write_text(
            "optimizer_timeout_sec=3600\n"
            "optimizers=optimizer dp_two_stage_optimizer\n",
            encoding="utf-8",
        )
        (run_root / "candidate_matrix.log").write_text(
            f"Candidate run complete: {run_root}\n",
            encoding="utf-8",
        )
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(canonical_payload), encoding="utf-8"
    )

    summaries = _pile_summaries(canonical, tmp_path / "audit")

    assert set(summaries["pile_europarl"]["runs"]) == {
        "optimizer",
        "dj_optimizer",
        "dp_cedar_optimizer",
        "dp_optimizer",
        "dp_two_stage_optimizer",
        "pecan_optimizer",
    }
    assert (
        summaries["pile_europarl"]["best_other_optimizer"]
        == "dp_two_stage_optimizer"
    )
    assert summaries["pile_europarl"][
        "dp_speedup_over_best_other"
    ] == pytest.approx(1.2)


def test_selection_fallback_has_six_workloads_and_one_third_nonwins():
    ledger = _base_ledger()
    selected = _selection(ledger)

    assert len(selected) == 6
    assert "alpaca_cot" in selected
    assert "general_video_refine" in selected
    assert sum(
        not ledger[name]["dp_at_least_20pct_faster"] for name in selected
    ) == 2


def test_selection_uses_both_positive_new_scenarios_to_reduce_pile_count():
    ledger = _base_ledger()
    ledger["redpajama_code"] = _item(valid=True, win=True, speedup=1.3)
    ledger["redpajama_arxiv"] = _item(valid=True, win=True, speedup=1.25)

    selected = _selection(ledger)

    assert len(selected) == 6
    assert "redpajama_code" in selected
    assert "redpajama_arxiv" in selected
    assert "llava_pretrain" not in selected
    assert sum(name.startswith("pile_") for name in selected) == 2
    nonwins = sum(
        not ledger[name]["dp_at_least_20pct_faster"] for name in selected
    )
    assert nonwins / len(selected) <= 0.40


def test_positive_video_replacement_admits_image_text_without_text_extra():
    ledger = _base_ledger()
    ledger["video_self_evolution"] = _item(
        valid=True, win=True, speedup=1.4
    )

    selected = _selection(ledger)

    assert "video_self_evolution" in selected
    assert "general_video_refine" not in selected
    assert "llava_pretrain" in selected
    assert len(selected) == 6


def test_selection_rejects_more_than_forty_percent_nonwins():
    ledger = _base_ledger()
    ledger["pile_europarl"] = _item(valid=True, win=False, speedup=1.0)

    with pytest.raises(RuntimeError, match="40% gate"):
        _selection(ledger)


def test_diversity_summary_records_scenario_modality_and_operator_span():
    summary = _diversity_summary(
        [
            "pile_hackernews",
            "alpaca_cot",
            "redpajama_code",
            "video_self_evolution",
            "llava_pretrain",
            "pile_pubmed_abstracts",
        ]
    )

    assert len(summary["scenarios"]) == 6
    assert summary["modalities"] == ["code", "image-text", "text", "video-text"]
    assert summary["hub_operator_count_range"] == [5, 17]
    assert summary["cedar_operator_count_range"] == [8, 19]


def test_reported_cedar_counts_match_current_logical_features():
    from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
        WORKLOAD_META,
    )

    assert WORKLOAD_META["redpajama_code"][3] == 17
    assert WORKLOAD_META["llava_pretrain"][3] == 16
