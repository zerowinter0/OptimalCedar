#!/usr/bin/env python3
"""Plot pairwise cost-model Q-error on common historical-plan coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from evaluation.chapter6_experiments.cost_model_accuracy_metrics import (
    pairwise_qerrors,
    qerror_summary,
)


MODELS = (
    ("cedar_cost", "Cedar cost", "#E69F00", "s"),
    ("reference_dp_cost", "Previous DP", "#777777", "^"),
    ("dp_objective_cost", "Revised DP", "#0072B2", "o"),
)


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = (
        Path(__file__).resolve().parent
        / "formal_results/paper_artifacts/cost_model"
    )
    parser.add_argument("--analysis", type=Path, default=default_root / "analysis.json")
    parser.add_argument("--output-root", type=Path, default=default_root)
    args = parser.parse_args()

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    required = tuple(field for field, _, _, _ in MODELS)
    common: dict[str, list[dict]] = {}
    for name, workload in payload["workloads"].items():
        candidates = [
            candidate
            for candidate in workload["candidates"]
            if all(field in candidate for field in required)
        ]
        if len(candidates) >= 2:
            common[name] = candidates

    distributions: dict[str, list[float]] = {}
    rows = []
    for field, label, _, _ in MODELS:
        values: list[float] = []
        for workload, candidates in common.items():
            workload_values, _, _ = pairwise_qerrors(candidates, field)
            values.extend(workload_values)
            rows.extend(
                {
                    "workload": workload,
                    "model": label,
                    "pairwise_qerror": value,
                }
                for value in workload_values
            )
        distributions[field] = values

    args.output_root.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "pairwise_qerror.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("workload", "model", "pairwise_qerror")
        )
        writer.writeheader()
        writer.writerows(rows)

    configure()
    fig, (cdf_ax, summary_ax) = plt.subplots(1, 2, figsize=(7.1, 2.55))
    for field, label, color, marker in MODELS:
        values = np.sort(np.asarray(distributions[field], dtype=float))
        cumulative = np.arange(1, len(values) + 1) / len(values)
        cdf_ax.step(values, cumulative, where="post", color=color, label=label)
        indices = np.linspace(0, len(values) - 1, 7, dtype=int)
        cdf_ax.scatter(
            values[indices],
            cumulative[indices],
            marker=marker,
            s=15,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    cdf_ax.set_xscale("log", base=2)
    cdf_ax.set_xlim(1, 64)
    cdf_ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    cdf_ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    cdf_ax.set_ylim(0, 1.02)
    cdf_ax.set_xlabel("Pairwise Q-error (log$_2$ scale; lower is better)")
    cdf_ax.set_ylabel("Cumulative fraction of plan pairs")
    cdf_ax.set_title("(a) Error distribution", loc="left")
    cdf_ax.grid(color="#DDDDDD", linestyle="--", linewidth=0.5)
    cdf_ax.legend(frameon=False, loc="lower right")

    statistic_names = ("geometric_mean", "p90", "maximum")
    statistic_labels = ("Geomean", "P90", "Maximum")
    x = np.arange(len(statistic_names), dtype=float)
    width = 0.23
    for model_idx, (field, label, color, _) in enumerate(MODELS):
        summary = qerror_summary(distributions[field])
        values = [summary[name] for name in statistic_names]
        positions = x + (model_idx - 1) * width
        bars = summary_ax.bar(
            positions,
            values,
            width,
            color=color,
            edgecolor="#333333",
            linewidth=0.45,
            label=label,
        )
        for bar, value in zip(bars, values):
            summary_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.1,
            )
    summary_ax.set_yscale("log", base=2)
    summary_ax.set_ylim(1, 90)
    summary_ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
    summary_ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64"])
    summary_ax.set_xticks(x, statistic_labels)
    summary_ax.set_ylabel("Pairwise Q-error (log$_2$ scale)")
    summary_ax.set_title("(b) Aggregate error", loc="left")
    summary_ax.grid(axis="y", color="#DDDDDD", linestyle="--", linewidth=0.5)
    summary_ax.text(
        0.02,
        0.98,
        f"{len(common)} workloads, {len(next(iter(distributions.values())))} plan pairs",
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
    )

    for axis in (cdf_ax, summary_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.3)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            args.output_root / f"cost_model_pairwise_qerror.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
