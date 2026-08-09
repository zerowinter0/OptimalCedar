#!/usr/bin/env python3
"""Plot broad cost-model accuracy from the canonical formal W=8 archive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


CEDAR = "#E69F00"
DP = "#0072B2"
GRAY = "#9E9E9E"
RED = "#D55E00"
ORDER = [
    "coco",
    "commonvoice",
    "commonvoice_cache",
    "simclrv2",
    "simclrv2_cache",
    "wikitext103",
    "wikitext103_cache",
    "llava_pretrain",
    "redpajama_c4",
    "stackexchange",
    "pile_europarl",
]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def style(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = (
        Path(__file__).resolve().parent
        / "formal_results"
        / "paper_artifacts"
        / "cost_model"
    )
    parser.add_argument("--analysis", type=Path, default=default_root / "analysis.json")
    parser.add_argument("--output-root", type=Path, default=default_root)
    args = parser.parse_args()

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    workloads = payload["workloads"]
    names = [name for name in ORDER if name in workloads]
    labels = [
        f"{workloads[name]['label'].replace(' [Cache]', chr(10) + '[Cache]')}\n"
        f"($n={workloads[name]['num_unique_candidates']}$)"
        for name in names
    ]
    x = np.arange(len(names), dtype=float)
    informative = [
        name
        for name in names
        if workloads[name]["models"]["cedar"].get("spearman_rho") is not None
        and workloads[name]["models"]["dp"].get("spearman_rho") is not None
    ]

    rows = []
    for name in names:
        workload = workloads[name]
        row = {
            "workload": name,
            "label": workload["label"],
            "unique_plans": workload["num_unique_candidates"],
            "execution_measurements": workload["num_execution_measurements"],
        }
        for model in ("cedar", "dp"):
            metrics = workload["models"][model]
            row[f"{model}_coverage"] = metrics["coverage"]
            row[f"{model}_spearman"] = metrics.get("spearman_rho")
            row[f"{model}_regret"] = metrics.get("selection_regret")
            row[f"{model}_top1"] = metrics.get("top1_correct")
        rows.append(row)

    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "formal_cost_model_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cedar_rhos = [
        float(workloads[name]["models"]["cedar"]["spearman_rho"])
        for name in informative
    ]
    dp_rhos = [
        float(workloads[name]["models"]["dp"]["spearman_rho"])
        for name in informative
    ]
    cedar_regrets = [
        100.0 * float(workloads[name]["models"]["cedar"]["selection_regret"])
        for name in informative
    ]
    dp_regrets = [
        100.0 * float(workloads[name]["models"]["dp"]["selection_regret"])
        for name in informative
    ]
    summary = {
        "informative_workloads": len(informative),
        "unique_plans_in_main_comparison": sum(
            workloads[name]["num_unique_candidates"] for name in informative
        ),
        "execution_measurements_in_main_comparison": sum(
            workloads[name]["num_execution_measurements"] for name in informative
        ),
        "total_unique_plans_audited": sum(
            workloads[name]["num_unique_candidates"] for name in names
        ),
        "total_execution_measurements_audited": sum(
            workloads[name]["num_execution_measurements"] for name in names
        ),
        "cedar_macro_spearman": float(np.mean(cedar_rhos)),
        "dp_macro_spearman": float(np.mean(dp_rhos)),
        "cedar_mean_regret_percent": float(np.mean(cedar_regrets)),
        "dp_mean_regret_percent": float(np.mean(dp_regrets)),
        "cedar_top1": int(
            sum(bool(workloads[name]["models"]["cedar"]["top1_correct"]) for name in informative)
        ),
        "dp_top1": int(
            sum(bool(workloads[name]["models"]["dp"]["top1_correct"]) for name in informative)
        ),
    }
    (data_dir / "formal_cost_model_accuracy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    configure()
    fig, (rank_ax, regret_ax) = plt.subplots(2, 1, figsize=(7.2, 4.65), sharex=True)

    for index, name in enumerate(names):
        cedar_rho = workloads[name]["models"]["cedar"].get("spearman_rho")
        dp_rho = workloads[name]["models"]["dp"].get("spearman_rho")
        if cedar_rho is None or dp_rho is None:
            rank_ax.axvspan(index - 0.43, index + 0.43, color="#F0F0F0", zorder=0)
            note = "tied costs" if name == "commonvoice_cache" else "insufficient\ncoverage"
            rank_ax.text(index, 0, note, color="#777777", ha="center", va="center", fontsize=6.1)
            continue
        rank_ax.plot(
            [index, index],
            [cedar_rho, dp_rho],
            color="#BBBBBB",
            linewidth=1.0,
            zorder=1,
        )
        rank_ax.scatter(index, cedar_rho, marker="s", s=24, color=CEDAR, edgecolor="white", linewidth=0.45, zorder=3)
        rank_ax.scatter(index, dp_rho, marker="o", s=27, color=DP, edgecolor="white", linewidth=0.45, zorder=3)
    rank_ax.axhline(0, color="#777777", linewidth=0.7)
    rank_ax.set_ylim(-1.12, 1.12)
    rank_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    rank_ax.set_ylabel("Spearman rank correlation")
    rank_ax.set_title("(a) Plan-ranking accuracy (higher is better)", loc="left")
    rank_ax.text(
        0.01,
        0.04,
        f"Macro mean: {summary['cedar_macro_spearman']:.2f} → {summary['dp_macro_spearman']:.2f}",
        transform=rank_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#BBBBBB"},
    )
    style(rank_ax)

    width = 0.34
    cap = 12.0
    for offset, model, label, color, hatch in (
        (-0.5, "cedar", "Cedar cost", CEDAR, "\\\\"),
        (0.5, "dp", "DP objective", DP, "///"),
    ):
        for index, name in enumerate(names):
            metrics = workloads[name]["models"][model]
            rho = metrics.get("spearman_rho")
            regret = metrics.get("selection_regret")
            if rho is None or regret is None:
                if model == "cedar":
                    regret_ax.axvspan(index - 0.43, index + 0.43, color="#F0F0F0", zorder=0)
                    regret_ax.text(index, 5.2, "N/A", color="#777777", ha="center", va="center", fontsize=6.3)
                continue
            value = 100.0 * float(regret)
            shown = min(value, cap)
            regret_ax.bar(
                index + offset * width,
                shown,
                width,
                color=color,
                edgecolor="#333333",
                linewidth=0.5,
                hatch=hatch,
                label=label if index == 0 else None,
            )
            if value > cap:
                regret_ax.text(
                    index + offset * width,
                    cap - 0.15,
                    f"↑{value:.0f}%",
                    color=RED,
                    ha="center",
                    va="top",
                    rotation=90,
                    fontsize=6.5,
                    fontweight="bold",
                )
            elif value >= 0.45:
                regret_ax.text(
                    index + offset * width,
                    shown + 0.28,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.0,
                )
    regret_ax.set_ylim(0, 13.4)
    regret_ax.set_ylabel("Selected-plan runtime regret (%)")
    regret_ax.set_title("(b) Cost-based plan selection (lower is better; axis capped at 12%)", loc="left")
    regret_ax.set_xticks(x, labels)
    regret_ax.text(
        0.01,
        0.96,
        (
            f"Mean regret: {summary['cedar_mean_regret_percent']:.1f}% → "
            f"{summary['dp_mean_regret_percent']:.1f}%   |   "
            f"Top-1: {summary['cedar_top1']}/{len(informative)} → "
            f"{summary['dp_top1']}/{len(informative)}"
        ),
        transform=regret_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#BBBBBB"},
    )
    style(regret_ax)

    cedar_handle = plt.Line2D([], [], marker="s", linestyle="none", color=CEDAR, markersize=5, label="Cedar cost")
    dp_handle = plt.Line2D([], [], marker="o", linestyle="none", color=DP, markersize=5, label="DP objective")
    fig.legend(
        handles=[cedar_handle, dp_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.925, bottom=0.14, hspace=0.34)

    figure_dir = args.output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "formal_cost_model_accuracy_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
