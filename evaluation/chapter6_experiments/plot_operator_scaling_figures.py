#!/usr/bin/env python3
"""Reproduce the paper's operator-scaling figures from measured profiles."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE3_OPERATORS = OrderedDict(
    (
        ("clean_email", "Clean email"),
        ("fix_unicode", "Fix Unicode"),
        ("language_id", "Language ID"),
        ("text_length", "Text length"),
        ("sync_text", "Sync text"),
        ("extract_text", "Extract text"),
    )
)

FIGURE4_WORKLOADS = OrderedDict(
    (
        ("alpaca_cot", "Alpaca-CoT"),
        ("general_video_refine", "GeneralVideo"),
        ("redpajama_code", "RP-Code"),
        ("stackexchange", "StackExchange"),
    )
)


def fixed_record_time_ms(
    row: Mapping[str, Any], *, record_count: int = 1_000
) -> float:
    """Convert measured per-record latency to time for a fixed record count."""
    if record_count < 1:
        raise ValueError("record_count must be positive")
    return float(row["latency_ns_per_record"]) * record_count / 1e6


def select_operator_rows(
    rows: Sequence[Mapping[str, Any]],
) -> OrderedDict[str, list[Mapping[str, Any]]]:
    """Select and order the six representative StackExchange operators."""
    selected: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    for tag in FIGURE3_OPERATORS:
        points = sorted(
            (row for row in rows if str(row["tag"]) == tag),
            key=lambda row: float(row["input_bytes"]),
        )
        if len(points) < 3:
            raise ValueError(f"Operator {tag!r} has fewer than three points")
        selected[tag] = points
    return selected


def load_scaling_counts(
    summary_path: Path,
) -> OrderedDict[str, dict[str, int]]:
    """Load explicitly annotated non-source operator counts."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result: OrderedDict[str, dict[str, int]] = OrderedDict()
    for workload in FIGURE4_WORKLOADS:
        operators = summary[workload]["operators"]
        explicit = [op for op in operators if op["mode"] == "explicit"]
        non_explicit = [op for op in operators if op["mode"] != "explicit"]
        if len(non_explicit) != 1:
            raise ValueError(
                f"Expected exactly one unannotated source for {workload}, "
                f"found {len(non_explicit)}"
            )
        result[workload] = {
            "per_record": sum(
                op["scaling"] == "per_record" for op in explicit
            ),
            "per_data": sum(op["scaling"] == "per_data" for op in explicit),
        }
    return result


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_operator_latency(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    record_count: int = 1_000,
) -> None:
    selected = select_operator_rows(rows)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(selected)))
    markers = ("o", "s", "^", "D", "P", "X")
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    for color, marker, (tag, points) in zip(
        colors, markers, selected.items()
    ):
        x = np.array([float(row["input_bytes"]) for row in points])
        y = np.array(
            [fixed_record_time_ms(row, record_count=record_count) for row in points]
        )
        # Rate quartiles invert when converted to latency.
        lower = np.array(
            [
                record_count / float(row["rate_q3_records_per_sec"]) * 1e3
                for row in points
            ]
        )
        upper = np.array(
            [
                record_count / float(row["rate_q1_records_per_sec"]) * 1e3
                for row in points
            ]
        )
        ax.plot(
            x,
            y,
            label=FIGURE3_OPERATORS[tag],
            color=color,
            marker=marker,
            markersize=3.2,
            linewidth=1.25,
        )
        ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Input bytes per record")
    ax.set_ylabel(f"Execution time for {record_count:,} records (ms)")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"operator_input_size_scaling.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_workload_counts(summary_path: Path, output_dir: Path) -> None:
    counts = load_scaling_counts(summary_path)
    workloads = list(counts)
    labels = [FIGURE4_WORKLOADS[name] for name in workloads]
    per_record = np.array([counts[name]["per_record"] for name in workloads])
    per_byte = np.array([counts[name]["per_data"] for name in workloads])
    y = np.arange(len(workloads))

    fig, ax = plt.subplots(figsize=(3.55, 2.25))
    ax.barh(y, per_record, color="#0072B2", label="PerRecord")
    ax.barh(y, per_byte, left=per_record, color="#E69F00", label="PerByte")
    for index, (record_count, byte_count) in enumerate(
        zip(per_record, per_byte)
    ):
        if record_count:
            ax.text(
                record_count / 2,
                index,
                str(record_count),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        if byte_count:
            ax.text(
                record_count + byte_count / 2,
                index,
                str(byte_count),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of operators")
    ax.grid(axis="x", alpha=0.25, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"datajuicer_operator_scaling_counts.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--scaling-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-count", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _paper_style()
    rows = json.loads(args.raw_results.read_text(encoding="utf-8"))
    plot_operator_latency(rows, args.output_dir, record_count=args.record_count)
    plot_workload_counts(args.scaling_summary, args.output_dir)


if __name__ == "__main__":
    main()
