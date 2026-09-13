import json

import pytest

from evaluation.chapter6_experiments.plot_operator_scaling_figures import (
    FIGURE3_OPERATORS,
    FIGURE4_WORKLOADS,
    fixed_record_time_ms,
    load_scaling_counts,
    select_operator_rows,
)


def test_figure3_selects_three_operators_from_each_scaling_class():
    rows = []
    for tag in FIGURE3_OPERATORS:
        scaling = (
            "per_data"
            if tag in {"clean_email", "fix_unicode", "language_id"}
            else "per_record"
        )
        rows.extend(
            {"tag": tag, "scaling": scaling, "input_bytes": input_bytes}
            for input_bytes in (1024, 2048, 4096)
        )
    selected = select_operator_rows(rows)

    assert set(selected) == set(FIGURE3_OPERATORS)
    assert {row["scaling"] for row in selected["clean_email"]} == {"per_data"}
    assert {row["scaling"] for row in selected["fix_unicode"]} == {"per_data"}
    assert {row["scaling"] for row in selected["language_id"]} == {"per_data"}
    assert {row["scaling"] for row in selected["text_length"]} == {"per_record"}
    assert {row["scaling"] for row in selected["sync_text"]} == {"per_record"}
    assert {row["scaling"] for row in selected["extract_text"]} == {"per_record"}
    assert all(len(points) >= 3 for points in selected.values())


def test_fixed_record_time_uses_measured_per_record_latency():
    row = {"latency_ns_per_record": 2_500.0}

    assert fixed_record_time_ms(row, record_count=1_000) == pytest.approx(2.5)


def test_figure4_counts_exclude_the_source_default(tmp_path):
    expected = {
        "alpaca_cot": {"per_record": 3, "per_data": 5},
        "general_video_refine": {"per_record": 10, "per_data": 0},
        "redpajama_code": {"per_record": 3, "per_data": 14},
        "stackexchange": {"per_record": 3, "per_data": 16},
    }
    summary = {}
    for workload, workload_counts in expected.items():
        operators = [{"mode": "default", "scaling": "per_data"}]
        for scaling, count in workload_counts.items():
            operators.extend(
                {"mode": "explicit", "scaling": scaling}
                for _ in range(count)
            )
        summary[workload] = {"operators": operators}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    counts = load_scaling_counts(summary_path)

    assert list(counts) == list(FIGURE4_WORKLOADS)
    assert counts == expected
