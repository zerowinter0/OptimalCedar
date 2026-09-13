from collections import Counter

from evaluation.motivation_multimodal.scaling import (
    ScalingRow,
    image_side_grid,
    multimodal_grid,
    summarize_trials,
    text_grid,
)


def test_scaling_grid_matches_spec() -> None:
    assert text_grid() == (16, 32, 64, 128, 256, 512)
    assert image_side_grid() == (128, 256, 512, 1024)
    assert len(multimodal_grid()) == 16


def test_each_point_has_seven_trials() -> None:
    rows = [
        ScalingRow("normalize", 16, None, trial, 100 + trial)
        for trial in range(7)
    ] + [
        ScalingRow("safety", None, 128, trial, 200 + trial)
        for trial in range(7)
    ]

    counts = Counter(
        (row.operator, row.text_tokens, row.image_side) for row in rows
    )
    summaries = summarize_trials(rows)

    assert set(counts.values()) == {7}
    assert len(summaries) == 2
    assert summaries[0].trials == 7


def test_summary_uses_median_and_quartiles() -> None:
    rows = [
        ScalingRow("perplexity", 32, None, trial, value)
        for trial, value in enumerate((1, 2, 3, 4, 5, 6, 100))
    ]

    summary = summarize_trials(rows)[0]

    assert summary.median_ns == 4.0
    assert summary.q1_ns == 2.5
    assert summary.q3_ns == 5.5
