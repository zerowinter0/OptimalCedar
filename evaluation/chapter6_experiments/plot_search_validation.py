#!/usr/bin/env python3
"""Generate the paper figure for DP scalability and exactness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"


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
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def style(ax: plt.Axes, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    root = (
        Path(__file__).resolve().parent
        / "formal_results"
        / "paper_artifacts"
        / "search"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()

    dp = read_csv(args.data_dir / "dp_scalability.csv")
    cedar = read_csv(args.data_dir / "cedar_scalability.csv")
    optimality = json.loads(
        (args.data_dir / "optimality_summary.json").read_text(encoding="utf-8")
    )
    planning = json.loads(
        (args.data_dir / "real_pipeline_planning.json").read_text(encoding="utf-8")
    )

    configure()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))

    ax = axes[0]
    for rows, label, color, marker in (
        (dp, "PICO DP", BLUE, "o"),
        (cedar, "Cedar enum.", ORANGE, "s"),
    ):
        x = np.asarray([int(row["num_independent_ops"]) for row in rows])
        y = np.asarray([float(row["mean_seconds"]) for row in rows])
        ax.plot(x, y, color=color, marker=marker, markersize=3.4, label=label)
        if rows[-1]["status"] == "timeout":
            ax.scatter(x[-1], y[-1], marker="X", s=35, color=RED, zorder=5)
            ax.annotate(
                ">1 h", (x[-1], y[-1]), xytext=(4, -2),
                textcoords="offset points", color=RED, fontsize=6.7,
            )
    cedar10 = next(float(row["mean_seconds"]) for row in cedar if row["num_independent_ops"] == "10")
    dp10 = next(float(row["mean_seconds"]) for row in dp if row["num_independent_ops"] == "10")
    ax.annotate(
        f"{cedar10 / dp10:,.0f}× at 10 ops",
        xy=(10, dp10),
        xytext=(8.5, 0.0022),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": "#555555"},
        fontsize=6.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Reorderable operators")
    ax.set_ylabel("Optimization time (s)")
    ax.set_title("(a) Search latency")
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.legend(frameon=False, loc="upper left")
    style(ax, "both")

    ax = axes[1]
    x = np.asarray([int(row["num_independent_ops"]) for row in dp])
    orders = np.asarray([int(row["candidate_orders"]) for row in dp], dtype=float)
    states = np.asarray([int(row["dp_states"]) for row in dp], dtype=float)
    ax.plot(x, orders, color=ORANGE, marker="s", markersize=3.1, label="Enumeration ($n!$)")
    ax.plot(x, states, color=BLUE, marker="o", markersize=3.1, label="DP states ($2^n$)")
    ax.set_yscale("log")
    ax.set_xlabel("Reorderable operators")
    ax.set_ylabel("Search-space size")
    ax.set_title("(b) State-space reduction")
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.04,
        0.06,
        f"Exact optimum: {optimality['num_cases']}/{optimality['num_cases']} cases\n"
        f"{optimality['total_enumerated_plans'] / 1e6:.1f}M plans checked",
        transform=ax.transAxes,
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB", "alpha": 0.9},
    )
    style(ax, "both")

    ax = axes[2]
    centers = np.arange(len(planning["workloads"]))
    width = 0.32
    for offset, (method, color, hatch) in enumerate(
        (("cedar", ORANGE, "\\\\"), ("pico", BLUE, "///"))
    ):
        values = [float(item[method]["seconds"]) for item in planning["workloads"]]
        bars = ax.bar(
            centers + (offset - 0.5) * width,
            values,
            width,
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
            hatch=hatch,
            label="Cedar" if method == "cedar" else "PICO",
        )
        for bar, value, item in zip(bars, values, planning["workloads"]):
            timeout = bool(item[method].get("timeout"))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.12,
                f">{value:.0f}" if timeout else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color=RED if timeout else "#333333",
            )
    ax.set_yscale("log")
    ax.set_ylim(1.5, 650)
    ax.set_xticks(
        centers,
        [f"{item['label']}\n({item['operators']} ops)" for item in planning["workloads"]],
    )
    ax.set_ylabel("Plan time (s)")
    ax.set_title("(c) Real pipelines")
    ax.legend(frameon=False, loc="upper left")
    style(ax)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22, wspace=0.38)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "dp_search_evaluation"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
