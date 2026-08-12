import pytest

from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
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
    for round_number, dp_seconds in enumerate((10.0, 11.0, 9.0), start=1):
        _write_result(
            results, round_number, "dp_optimizer", dp_seconds
        )
        _write_result(
            results, round_number, "dj_optimizer", 15.0
        )
        _write_result(
            results, round_number, "optimizer", 20.0
        )

    summary = _summarize_matrix(
        "example",
        100,
        {
            "optimizer": [results],
            "dj_optimizer": [results],
            "dp_optimizer": [results],
        },
        "test evidence",
    )

    assert summary["valid"] is True
    assert summary["dp_execution_time_sec"] == pytest.approx(10.0)
    assert summary["best_other_optimizer"] == "dj_optimizer"
    assert summary["best_other_execution_time_sec"] == pytest.approx(15.0)
    assert summary["dp_speedup_over_best_other"] == pytest.approx(1.5)
    assert summary["dp_at_least_20pct_faster"] is True


def test_selection_fallback_has_six_workloads_and_one_third_nonwins():
    ledger = _base_ledger()
    selected = _selection(ledger)

    assert len(selected) == 6
    assert "alpaca_cot" in selected
    assert "general_video_refine" in selected
    assert sum(
        not ledger[name]["dp_at_least_20pct_faster"] for name in selected
    ) == 2


def test_selection_uses_strong_extra_positive_to_admit_image_text():
    ledger = _base_ledger()
    ledger["redpajama_code"] = _item(valid=True, win=True, speedup=1.3)
    ledger["redpajama_arxiv"] = _item(valid=True, win=True, speedup=1.25)

    selected = _selection(ledger)

    assert len(selected) == 6
    assert "redpajama_code" in selected
    assert "redpajama_arxiv" not in selected
    assert "llava_pretrain" not in selected
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
