#!/usr/bin/env python3
"""Plot paper-ready absolute runtimes for the selected diverse workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
    OPTIMIZERS,
    WORKLOAD_META,
)


OPTIMIZER_STYLE = {
    "optimizer": ("Cedar", "#7F7F7F", ""),
    "dj_optimizer": ("Data-Juicer", "#E69F00", "//"),
    "dp_cedar_optimizer": ("DP-Cedar", "#56B4E9", "\\\\"),
    "dp_optimizer": ("PICO", "#0072B2", "..."),
    "dp_two_stage_optimizer": ("DP two-stage", "#CC79A7", "xx"),
    "pecan_optimizer": ("Pecan", "#009E73", "++"),
}


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for workload in report["selected_workloads"]:
        scenario, modality, hub_ops, cedar_ops = WORKLOAD_META[workload]
        item = report["ledger"][workload]
        for optimizer in OPTIMIZERS:
            run = item["runs"][optimizer]
            if run.get("valid"):
                outcome = "success"
            elif run.get("formally_unavailable"):
                outcome = "unavailable"
            else:
                raise RuntimeError(
                    f"incomplete formal evidence for {workload}/{optimizer}"
                )
            rows.append(
                {
                    "workload": workload,
                    "scenario": scenario,
                    "modality": modality,
                    "hub_operators": hub_ops,
                    "cedar_operators": cedar_ops,
                    "samples": item["expected_samples"],
                    "optimizer": optimizer,
                    "outcome": outcome,
                    "mean_seconds": run.get("mean_execution_time_sec"),
                    "stddev_seconds": run.get("stddev_execution_time_sec"),
                    "repetitions": len(run.get("execution_times_sec", [])),
                }
            )
    return rows


def _write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(report: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    selected = sorted(
        report["selected_workloads"], key=lambda name: WORKLOAD_META[name][3]
    )
    columns = min(3, len(selected))
    rows_count = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(3.35 * columns, 2.7 * rows_count),
        squeeze=False,
    )
    by_workload = {
        workload: [row for row in rows if row["workload"] == workload]
        for workload in selected
    }
    for index, workload in enumerate(selected):
        axis = axes[index // columns][index % columns]
        workload_rows = by_workload[workload]
        values = [
            float(row["mean_seconds"])
            for row in workload_rows
            if row["outcome"] == "success"
        ]
        if not values:
            raise RuntimeError(f"no successful optimizer for {workload}")
        top = max(values) * 1.18
        for position, row in enumerate(workload_rows):
            label, color, hatch = OPTIMIZER_STYLE[row["optimizer"]]
            if row["outcome"] == "success":
                axis.bar(
                    position,
                    row["mean_seconds"],
                    yerr=row["stddev_seconds"],
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    hatch=hatch,
                    capsize=2,
                    error_kw={"linewidth": 0.7},
                )
            else:
                axis.text(
                    position,
                    top * 0.48,
                    "Unavailable",
                    ha="center",
                    va="center",
                    rotation=90,
                    color="#B2182B",
                    fontsize=6.5,
                    fontweight="bold",
                )
                axis.scatter(
                    [position], [top * 0.08], marker="X", s=28, color="#B2182B"
                )
        scenario, _, _, cedar_ops = WORKLOAD_META[workload]
        axis.set_title(f"{scenario}\n({cedar_ops} Cedar ops)", fontsize=8)
        axis.set_ylim(0, top)
        axis.set_ylabel("Execution time (s)")
        axis.set_xticks(range(len(workload_rows)))
        axis.set_xticklabels(
            [OPTIMIZER_STYLE[row["optimizer"]][0] for row in workload_rows],
            rotation=50,
            ha="right",
            fontsize=6.3,
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.set_axisbelow(True)
    for index in range(len(selected), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(
        "Absolute execution time on selected Data-Juicer workloads "
        "(W=8, CPU budget=64)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = _rows(report)
    output_dir = args.output_dir
    _write_tsv(rows, output_dir / "datajuicer_diverse_runtime.tsv")
    for suffix in ("pdf", "png", "svg"):
        _plot(
            report,
            rows,
            output_dir / f"datajuicer_diverse_runtime.{suffix}",
        )
    print(output_dir)


if __name__ == "__main__":
    main()
