import pytest

from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
    _selection,
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
