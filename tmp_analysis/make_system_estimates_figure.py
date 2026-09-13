"""What every system's cost model says each of the three SimCLRv2 plans costs.

Single panel: for each plan (ordered by measured time) we show the per-record
cost predicted by every cost model we implemented, against the measured value.

Data comes from ``outputs/plumber_bench_20260912/system_cost_matrix.json``,
which ``tmp_analysis/score_plans_all_models.py`` writes.  Two profiling bases
are available there; ``--basis`` selects which one to plot (default: the
pipeline-trace basis each system would collect by itself).

Usage:
  python tmp_analysis/make_system_estimates_figure.py [--basis baseline|affine]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# The paper is typeset in Times; plot with a Times-metric font so the figure
# matches the body text.  ``Liberation Serif`` and ``Nimbus Roman`` are both
# metric-compatible with Times New Roman and are what this image provides.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Liberation Serif",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)

OUT = Path("/workspace/OptimalCedar/outputs/plumber_bench_20260912/figures")
PAPER = Path("/workspace/OptimalCedar/my_paper/69e75a0100d7b4afeb1cfc20/figures")
MATRIX = Path("/workspace/OptimalCedar/outputs/plumber_bench_20260912/system_cost_matrix.json")

# Plan order is the measured order (fastest first).
PLAN_ORDER = [
    ("pico", "PICO's plan"),
    ("plumber", "Plumber-style plan"),
    ("cedar", "Cedar's plan"),
    ("unoptimized", "Unoptimized plan"),
    ("raydata", "Ray Data plan"),
]
# The five rate-based models all predict the same bottleneck-stage cost, so
# they are drawn as one bar; the legend names them.
RATE_MAX_MODELS = ["plumber", "pecan", "tfdata", "fastflow", "aero"]
MODEL_ORDER = [
    ("pico", "PICO (ours)", "#54A24B"),
    ("cedar", "Cedar", "#4C78A8"),
    ("raydata", "Ray Data", "#BAB0AC"),
    (
        "rate_max",
        "rate-max family (Plumber, Pecan,\ntf.data, FastFlow, Aero)",
        "#F58518",
    ),
]
# Highest prediction in either basis is ~38.8 ms; Ray Data's measured 110 ms is
# drawn as a clipped line with an explicit note.
Y_CLIP = 45.0
# Bottom of the log axis used by the ratio plot (0.04 is the largest error a
# baseline model makes on the plans in this figure).
RATIO_FLOOR = 0.03


def estimate(row, model_key, basis):
    if model_key == "pico":
        return float(row["pico"])
    if model_key == "rate_max":
        values = [rate_max_estimate(row, key, basis) for key in RATE_MAX_MODELS]
        return sum(values) / len(values)
    if basis == "affine" and isinstance(row["affine"].get(model_key), (int, float)):
        return float(row["affine"][model_key])
    return float(row["baseline"][model_key])


def rate_max_estimate(row, model_key, basis):
    if basis == "affine" and isinstance(row["affine"].get(model_key), (int, float)):
        return float(row["affine"][model_key])
    return float(row["baseline"][model_key])


def _draw(
    container,
    plans,
    values,
    measured,
    width,
    low,
    high,
    log_scale=False,
    label_fmt="{:.1f}",
    measured_fmt="{:.2f}",
):
    """Draw bars and measured lines whose value falls inside [low, high]."""
    for index, (plan_key, _) in enumerate(plans):
        for offset, (model_key, _, color) in enumerate(MODEL_ORDER):
            value = values[plan_key][model_key]
            inside = low < value < high
            if not inside and (low > 0.0 or high < float("inf")):
                continue
            x = index + (offset - (len(MODEL_ORDER) - 1) / 2) * (width + 0.05)
            container.bar(
                x,
                value - (low if log_scale else 0.0),
                bottom=low if log_scale else 0.0,
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.4,
            )
            container.text(
                x,
                value * 1.12 if log_scale else value + (high - low) * 0.015,
                label_fmt.format(value),
                ha="center",
                va="bottom" if log_scale else "baseline",
                fontsize=7,
                rotation=90,
            )
        line = measured[plan_key]
        if not (low < line < high) and (low > 0.0 or high < float("inf")):
            continue
        container.plot(
            [index - 0.46, index + 0.46],
            [line, line],
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        # The dashed line carries the measurement; print its value so the
        # reader does not have to interpolate against the axis.
        container.text(
            index + 0.44,
            line * (1.06 if log_scale else 1.0) + (0.0 if log_scale else (high - low) * 0.02),
            measured_fmt.format(line),
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75),
        )


def _legend(container, measured_label="measured (dashed)"):
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
        for _, _, color in MODEL_ORDER
    ] + [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black")]
    container.legend(
        handles,
        [label for _, label, _ in MODEL_ORDER] + [measured_label],
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        ncol=2,
    )


def _xpos(index, offset):
    return index + (offset - (len(MODEL_ORDER) - 1) / 2) * (0.17 + 0.05)


def _draw_markers(container, plans, values, measured, measured_fmt="{:.2f}"):
    """Marker variant of the bar chart: positions carry the value."""
    for index, (plan_key, _) in enumerate(plans):
        for offset, (model_key, _, color) in enumerate(MODEL_ORDER):
            value = values[plan_key][model_key]
            container.plot(
                [_xpos(index, offset) - 0.085, _xpos(index, offset) + 0.085],
                [value, value],
                color=color,
                linewidth=3.0,
                solid_capstyle="butt",
                zorder=3,
            )
        line = measured[plan_key]
        container.plot(
            [index - 0.46, index + 0.46],
            [line, line],
            color="black",
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        container.text(
            index + 0.44,
            line * 1.06,
            measured_fmt.format(line),
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75),
        )


def _draw_ratio(container, plans, values, measured, floor=0.03):
    """Prediction error: 1.0 means the model matches the measurement."""
    for index, (plan_key, _) in enumerate(plans):
        for offset, (model_key, _, color) in enumerate(MODEL_ORDER):
            ratio = values[plan_key][model_key] / measured[plan_key]
            x = _xpos(index, offset)
            container.bar(
                x,
                ratio - floor,
                bottom=floor,
                width=0.17,
                color=color,
                edgecolor="black",
                linewidth=0.4,
            )
            container.text(
                x,
                ratio * 1.10,
                f"{ratio:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    container.axhline(1.0, color="black", linewidth=1.1, zorder=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="baseline", choices=["baseline", "affine"])
    parser.add_argument(
        "--matrix",
        default=str(MATRIX),
        help="system cost matrix JSON (use the trace- or affine-scored file).",
    )
    parser.add_argument(
        "--out",
        default="simclr_model_estimates",
        help="output stem in the figures directories.",
    )
    parser.add_argument(
        "--yscale",
        default="log",
        choices=["linear", "log", "broken"],
        help=(
            "linear: one absolute axis (the Ray Data plan compresses the rest); "
            "log: logarithmic absolute axis; broken: split absolute axis that "
            "keeps the four fast plans readable and still shows the Ray Data "
            "plan."
        ),
    )
    parser.add_argument(
        "--metric",
        default="time",
        choices=["time", "throughput"],
        help=(
            "time: per-record milliseconds (lower is better); throughput: "
            "records per second (higher is better)."
        ),
    )
    parser.add_argument(
        "--plot",
        default="bar",
        choices=["bar", "marker", "ratio"],
        help=(
            "bar: grouped bars per plan; marker: horizontal markers on a log "
            "axis (position, not length, carries the value); ratio: predicted "
            "over measured, with 1.0 the perfect prediction."
        ),
    )
    args = parser.parse_args()

    table = json.loads(Path(args.matrix).read_text())
    width = 0.17

    def to_metric(milliseconds):
        """Convert a per-record time into the plotted metric."""
        if args.metric == "time":
            return milliseconds
        return 1000.0 / milliseconds

    label_fmt = "{:.0f}" if args.metric == "throughput" else "{:.1f}"
    measured_fmt = "{:.0f}" if args.metric == "throughput" else "{:.2f}"
    values, measured = {}, {}
    for plan_key, _ in PLAN_ORDER:
        row = table[plan_key]
        workers = max(1, int(row.get("workers", 1) or 1))
        values[plan_key] = {
            model_key: to_metric(estimate(row, model_key, args.basis) / workers)
            for model_key, _, _ in MODEL_ORDER
        }
        measured[plan_key] = to_metric(float(row["measured_aggregate_ms"]))
    highest = max(
        list(measured.values())
        + [value for per_plan in values.values() for value in per_plan.values()]
    )
    # The log axis must start below the smallest datum, otherwise that bar is
    # drawn with zero height and disappears (the Cedar model's own estimate of
    # the Cedar plan and several rate-max estimates sit at the old floor).
    lowest = min(
        list(measured.values())
        + [value for per_plan in values.values() for value in per_plan.values()]
    )
    # Bars are centred on x, so the axis must cover the outermost bar's edge.
    half_group = (len(MODEL_ORDER) - 1) / 2 * (width + 0.05)
    x_left = -half_group - width / 2 - 0.12
    x_right = (len(PLAN_ORDER) - 1) + half_group + width / 2 + 0.12

    if args.plot in ("marker", "ratio"):
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        if args.plot == "marker":
            _draw_markers(
                ax, PLAN_ORDER, values, measured, measured_fmt=measured_fmt
            )
            low, high = lowest * 0.35, highest * 1.9
            ax.set_ylabel(
                "throughput (records/s, log scale)"
                if args.metric == "throughput"
                else "per-record time (ms, log scale)",
                fontsize=11,
            )
        else:
            ratios = [
                values[plan][model] / measured[plan]
                for plan, _ in PLAN_ORDER
                for model, _, _ in MODEL_ORDER
            ]
            ratio_floor = min(ratios) * 0.35
            _draw_ratio(
                ax, PLAN_ORDER, values, measured, floor=ratio_floor
            )
            low, high = ratio_floor, max(ratios) * 2.0
            ax.set_ylabel("predicted / measured (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.set_xlim(x_left, x_right)
        ax.set_xticks(range(len(PLAN_ORDER)))
        ax.set_xticklabels([label for _, label in PLAN_ORDER], fontsize=10)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        ax.tick_params(axis="y", labelsize=10)
        _legend(
            ax,
            measured_label=(
                "measured (=1)" if args.plot == "ratio" else "measured (dashed)"
            ),
        )
        fig.subplots_adjust(left=0.11, right=0.985, top=0.97, bottom=0.38)
    elif args.yscale == "broken":
        # Two stacked axes over one shared x: the four short plans stay
        # readable on a linear 0-5 axis, and the Ray Data plan keeps its true
        # height instead of flattening everybody else.
        fig = plt.figure(figsize=(7.4, 3.7))
        grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.08)
        top = fig.add_subplot(grid[0])
        bottom = fig.add_subplot(grid[1], sharex=top)
        panels = [(bottom, 0.0, 5.0), (top, 14.0, highest * 1.25)]
        for container, low, high in panels:
            _draw(
                container, PLAN_ORDER, values, measured, width, low, high,
                label_fmt=label_fmt, measured_fmt=measured_fmt,
            )
            container.set_ylim(low, high)
            container.grid(True, axis="y", linewidth=0.3, alpha=0.4)
            container.tick_params(axis="y", labelsize=9)
        for axis, y in ((top, 0.0), (bottom, 1.0)):
            axis.plot(
                (-0.014, 0.014),
                (y - 0.014, y + 0.014),
                transform=axis.transAxes,
                color="black",
                clip_on=False,
                linewidth=0.8,
            )
            axis.plot(
                (0.986, 1.014),
                (y - 0.014, y + 0.014),
                transform=axis.transAxes,
                color="black",
                clip_on=False,
                linewidth=0.8,
            )
        bottom.spines["top"].set_visible(False)
        top.spines["bottom"].set_visible(False)
        top.tick_params(labelbottom=False, bottom=False)
        bottom.set_xlim(x_left, x_right)
        bottom.set_xticks(range(len(PLAN_ORDER)))
        bottom.set_xticklabels([label for _, label in PLAN_ORDER], fontsize=10)
        # One shared label: the two panels are one axis with a cut in it.
        fig.text(
            0.028,
            0.62,
            "throughput (records/s)"
            if args.metric == "throughput"
            else "per-record time (ms)",
            rotation=90,
            va="center",
            fontsize=11,
        )
        _legend(bottom)
        fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.30)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        if args.yscale == "log":
            low, high = lowest * 0.35, highest * 1.9
        else:
            low, high = 0.0, highest * 1.18
        _draw(
            ax,
            PLAN_ORDER,
            values,
            measured,
            width,
            low,
            high,
            log_scale=args.yscale == "log",
            label_fmt=label_fmt,
            measured_fmt=measured_fmt,
        )
        if args.yscale == "log":
            ax.set_yscale("log")
        ax.set_ylim(low, high)
        ax.set_xlim(x_left, x_right)
        ax.set_xticks(range(len(PLAN_ORDER)))
        ax.set_xticklabels([label for _, label in PLAN_ORDER], fontsize=10)
        if args.metric == "throughput":
            ax.set_ylabel(
                "throughput (records/s)"
                + (", log scale" if args.yscale == "log" else ""),
                fontsize=11,
            )
        else:
            ax.set_ylabel(
                "per-record time (ms)"
                + (", log scale" if args.yscale == "log" else ""),
                fontsize=11,
            )
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        ax.tick_params(axis="y", labelsize=10)
        _legend(ax)
        fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.38)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{args.out}.pdf")
    fig.savefig(OUT / f"{args.out}.png", dpi=220)
    if PAPER.is_dir() and args.out == "simclr_model_estimates":
        fig.savefig(PAPER / f"{args.out}.pdf")
        fig.savefig(PAPER / f"{args.out}.png", dpi=220)
    print("wrote", OUT / f"{args.out}.pdf", "basis =", args.basis)


if __name__ == "__main__":
    main()
