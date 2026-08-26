#!/usr/bin/env python3
"""Plot reorder, staged-enumeration, and joint-enumeration time growth.

The projection does not invent a per-plan runtime.  For every operator count
``n <= 10``, it takes Cedar's measured complete-order enumeration time and
divides by ``n!`` to obtain the measured mean time per visited candidate.
For 11 and 12 operators, reorder time is extrapolated by fitting log(runtime)
against log(n!) over the five largest measured pipelines.  The same candidate
rate gives deliberately optimistic estimates for a staged enumerator that
fixes reorder heuristically and exhaustively visits ``k**n * 2**n * n``
physical choices, and for a joint enumerator that additionally visits all
``n!`` orders.  Physical-plan construction and heuristic-reorder time are
ignored, so neither estimate overstates enumeration overhead.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


def _read_measurements(path: Path) -> Dict[int, float]:
    measurements: Dict[int, float] = {}
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            measurements[int(raw["num_independent_ops"])] = float(
                raw["mean_seconds"]
            )
    return measurements


def _fit_reorder_trend(
    measurements: Dict[int, float], fit_tail: int
) -> Tuple[float, float, int]:
    positive = sorted(n for n, value in measurements.items() if value > 0)
    if len(positive) < 2:
        raise ValueError("At least two positive reorder measurements required")
    selected = positive[-min(fit_tail, len(positive)) :]
    x = np.asarray([math.lgamma(n + 1) for n in selected], dtype=float)
    y = np.log([measurements[n] for n in selected])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), selected[0]


def load_rows(
    path: Path, backends: int, max_ops: int, fit_tail: int
) -> List[Dict[str, Union[float, int, str]]]:
    measurements = _read_measurements(path)
    slope, intercept, fit_start = _fit_reorder_trend(
        measurements, fit_tail
    )
    rows: List[Dict[str, Union[float, int, str]]] = []
    for n in range(min(measurements), max_ops + 1):
        if n in measurements:
            reorder_seconds = measurements[n]
            source = "measured"
        else:
            reorder_seconds = math.exp(
                intercept + slope * math.lgamma(n + 1)
            )
            source = "trend_extrapolation"
        reorder_orders = math.factorial(n)
        placement_choices = backends**n
        fusion_choices = 2**n
        cache_choices = n
        staged_plans = placement_choices * fusion_choices * cache_choices
        joint_plans = reorder_orders * staged_plans
        seconds_per_order = reorder_seconds / reorder_orders
        rows.append(
            {
                "num_operators": n,
                "backends_k": backends,
                "reorder_source": source,
                "reorder_orders": reorder_orders,
                "placement_choices": placement_choices,
                "fusion_choices_upper_bound": fusion_choices,
                "cache_positions": cache_choices,
                "staged_plan_upper_bound": staged_plans,
                "joint_plan_upper_bound": joint_plans,
                "reorder_seconds": reorder_seconds,
                "candidate_visit_seconds": seconds_per_order,
                "optimistic_staged_seconds": (
                    seconds_per_order * staged_plans
                ),
                "optimistic_joint_seconds": (
                    seconds_per_order * joint_plans
                ),
                "extrapolation_fit_start_n": fit_start,
                "extrapolation_log_factorial_slope": slope,
                "extrapolation_intercept": intercept,
            }
        )
    return rows


def write_csv(
    rows: List[Dict[str, Union[float, int, str]]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: List[Dict[str, Union[float, int, str]]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["num_operators"]) for row in rows]
    reorder = [row["reorder_seconds"] for row in rows]
    staged = [row["optimistic_staged_seconds"] for row in rows]
    joint = [row["optimistic_joint_seconds"] for row in rows]
    k = int(rows[0]["backends_k"])

    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    ax.plot(
        x,
        reorder,
        color="#2878B5",
        marker="o",
        linewidth=1.35,
        markersize=3.2,
        label="Reorder only",
    )
    ax.plot(
        x,
        staged,
        color="#F28E2B",
        marker="^",
        markerfacecolor="white",
        markeredgewidth=1.3,
        linestyle="--",
        linewidth=1.3,
        markersize=3.2,
        label="Staged exhaustive physical",
    )
    ax.plot(
        x,
        joint,
        color="#C82423",
        marker="D",
        markerfacecolor="white",
        markeredgewidth=1.4,
        linestyle="--",
        linewidth=1.35,
        markersize=3.2,
        label=f"Exhaustive joint ($K={k}$)",
    )
    for seconds, label in (
        (3600, "1 hour"),
        (86400, "1 day"),
        (365 * 86400, "1 year"),
    ):
        ax.axhline(seconds, color="#888888", linewidth=0.75,
                   linestyle=":", zorder=0)
        ax.text(
            max(x) + 0.08,
            seconds,
            label,
            fontsize=6.2,
            color="#666666",
            va="center",
            clip_on=False,
        )

    measured_max = max(
        int(row["num_operators"])
        for row in rows
        if row["reorder_source"] == "measured"
    )
    if measured_max < max(x):
        ax.axvspan(
            measured_max + 0.5,
            max(x) + 0.5,
            color="#999999",
            alpha=0.10,
            linewidth=0,
        )
        ax.text(
            measured_max + 0.62,
            max(joint) / 20,
            "extrapolated",
            fontsize=6.2,
            color="#666666",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of operators")
    ax.set_ylabel("Optimization time (s, log scale)")
    ax.set_xticks(x)
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.24)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/plots/cedar_reorder_time.csv"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "evaluation/chapter6_experiments/formal_results/"
            "paper_artifacts/figures/joint_search_explosion.csv"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "evaluation/chapter6_experiments/formal_results/"
            "paper_artifacts/figures/joint_search_explosion.pdf"
        ),
    )
    parser.add_argument("--backends", type=int, default=3)
    parser.add_argument("--max-ops", type=int, default=12)
    parser.add_argument("--fit-tail", type=int, default=5)
    args = parser.parse_args()
    if args.backends < 1:
        raise ValueError("--backends must be positive")
    if args.max_ops < 1 or args.fit_tail < 2:
        raise ValueError("Require --max-ops >= 1 and --fit-tail >= 2")
    rows = load_rows(
        args.input, args.backends, args.max_ops, args.fit_tail
    )
    write_csv(rows, args.csv)
    plot(rows, args.figure)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
