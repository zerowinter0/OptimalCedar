"""Draw the formal-campaign result figures from the collected figure data.

Reads outputs/figure_data_20260921/figure_data.json (built by
scripts/collect_figure_data_20260921.py) and writes

  fig1_throughput        steady-state throughput per workload x optimizer
  fig2_optimization_time optimization time, missing cells marked
  fig3_cost_rank         model cost rank vs measured throughput rank
  fig4_rank_agreement    Spearman rho summary per workload

to outputs/figures_20260921/ as PNG (300 dpi), PDF and SVG.

Usage (inside the container):
  python -u scripts/make_experiment_figures_20260921.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path("/workspace/OptimalCedar")
DATA = json.loads(
    (ROOT / "outputs/figure_data_20260921/figure_data.json").read_text()
)
OUT = ROOT / "outputs/figures_20260921"

WORKLOADS = [
    "simclrv2",
    "simclrv2_cache",
    "commonvoice",
    "coco",
    "llava_pretrain",
    "stackexchange",
]
WLABEL = {
    "simclrv2": "simclrv2",
    "simclrv2_cache": "simclrv2\n-cache",
    "commonvoice": "commonvoice",
    "coco": "coco",
    "llava_pretrain": "llava\n-pretrain",
    "stackexchange": "stackexchange",
}
LABELS = [
    "unopt",
    "plumber",
    "raydata",
    "cedar",
    "cedar-dp",
    "PICO-Resource",
    "PICO-Resource-Op",
    "PICO",
]
COLORS = {
    "unopt": "#9e9e9e",
    "plumber": "#009e73",
    "raydata": "#56b4e9",
    "cedar": "#d55e00",
    "cedar-dp": "#e69f00",
    "PICO-Resource": "#cc79a7",
    "PICO-Resource-Op": "#0072b2",
    "PICO": "#003049",
}
MODELS = [("cedar", "cedar", "#d55e00"), ("plumber", "plumber", "#009e73"),
          ("pico", "PICO", "#003049")]

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)


def measured(workload, label):
    return DATA["workloads"][workload]["measured"][label]


def score(workload, label):
    return DATA["workloads"][workload]["scores"].get(label)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)
    print(f"wrote {OUT / (name + '.png')}")


def fig_throughput():
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    positions = np.arange(len(WORKLOADS))
    width = 0.10
    for index, label in enumerate(LABELS):
        values, missing = [], []
        for workload in WORKLOADS:
            cell = measured(workload, label)
            rate = cell.get("throughput_records_per_sec")
            values.append(rate if rate else np.nan)
            missing.append(rate is None)
        offset = (index - (len(LABELS) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            values,
            width * 0.92,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, value, gap in zip(bars, values, missing):
            if gap:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    2.2,
                    "×",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#b00020",
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=4.6,
                color="#333333",
            )
    ax.set_yscale("log")
    ax.set_ylim(1.2, 2.6e4)
    ax.set_ylabel("steady throughput (records/s, log)")
    ax.set_xticks(positions)
    ax.set_xticklabels([WLABEL[w] for w in WORKLOADS])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", which="both", color="#e6e6e6", linewidth=0.5)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#b00020", marker="$×$", linestyle="none"))
    labels.append("cell missing (timeout / skipped)")
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=False,
    )
    fig.suptitle(
        "Steady-state throughput on the formal campaign (log scale)", y=1.02
    )
    save(fig, "fig1_throughput")


def fig_optimization_time():
    fig, ax = plt.subplots(figsize=(10.0, 3.4))
    positions = np.arange(len(WORKLOADS))
    width = 0.10
    for index, label in enumerate(LABELS):
        values = []
        for workload in WORKLOADS:
            cell = measured(workload, label)
            values.append(cell.get("setup_time_sec") or np.nan)
        offset = (index - (len(LABELS) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            values,
            width * 0.92,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.4,
            hatch="///" if label == "PICO" else None,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                ax.bar(
                    bar.get_x(),
                    0.16,
                    bar.get_width(),
                    bottom=0.5,
                    color="#f2f2f2",
                    edgecolor="#b00020",
                    linewidth=0.5,
                    hatch="//",
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.63,
                    "×",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#b00020",
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=4.6,
                color="#333333",
            )
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1.2e4)
    ax.set_ylabel("optimization / setup time (s, log)")
    ax.set_xticks(positions)
    ax.set_xticklabels([WLABEL[w] for w in WORKLOADS])
    ax.grid(axis="y", which="both", color="#e6e6e6", linewidth=0.5)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#b00020", marker="$×$", linestyle="none"))
    labels.append("no cell (timeout / skipped)")
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=False,
    )
    ax.text(
        0.5,
        1.02,
        "cedar-opt has no cell on llava-pretrain / stackexchange "
        "(known Cedar optimization timeout); PICO has none on stackexchange",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#b00020",
    )
    fig.suptitle("Optimization (setup) time per optimizer and workload", y=1.02)
    save(fig, "fig2_optimization_time")


def fig_cost_rank():
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    handles = {}
    for ax, workload in zip(axes.ravel(), WORKLOADS):
        agreement = DATA["workloads"][workload]["rank_agreement"]
        labels = agreement.get("labels", [])
        if not labels:
            ax.text(0.5, 0.5, "no priceable plan", ha="center", va="center")
            continue
        measured_rank = agreement["throughput_rank"]
        limit = len(labels) + 0.6
        ax.plot([0.4, limit], [0.4, limit], color="#cccccc", linewidth=0.8,
                linestyle="--", zorder=1)
        rho_text = []
        for index, (model, nice, color) in enumerate(MODELS):
            ranks = agreement.get(f"{model}_rank")
            if not ranks:
                continue
            # A small deterministic dodge keeps coincident ranks visible.
            dodge = (index - 1) * 0.085
            handle = ax.scatter(
                [r + dodge for r in measured_rank],
                [r + dodge for r in ranks],
                s=26,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                marker=("o", "s", "^")[index],
                label=nice,
                zorder=3,
            )
            handles.setdefault(model, handle)
            rho_text.append(f"{nice} {agreement.get(f'{model}_spearman')}")
        ax.set_xlim(0.4, limit)
        ax.set_ylim(limit, 0.4)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_title(workload)
        ax.grid(color="#f0f0f0", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(
            0.03,
            0.03,
            "ρ: " + " · ".join(rho_text),
            transform=ax.transAxes,
            fontsize=6,
            color="#555555",
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("model cost rank (1 = cheapest)")
    for ax in axes[1, :]:
        ax.set_xlabel("measured rank (1 = fastest)")
    fig.legend(
        [handles[model] for model, _nice, _color in MODELS],
        [nice for _m, nice, _c in MODELS],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        "Does the cost model rank the plans the way the machine does?", y=0.99
    )
    fig.tight_layout()
    save(fig, "fig3_cost_rank")


def fig_rank_agreement():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    positions = np.arange(len(WORKLOADS))
    width = 0.26
    for index, (model, nice, color) in enumerate(MODELS):
        values = [
            DATA["workloads"][w]["rank_agreement"].get(f"{model}_spearman", np.nan)
            for w in WORKLOADS
        ]
        bars = ax.bar(
            positions + (index - 1) * width,
            values,
            width * 0.92,
            label=nice,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.03 if value >= 0 else -0.07),
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=5.6,
            )
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_ylim(-0.35, 1.15)
    ax.set_ylabel("Spearman ρ (cost rank vs speed rank)")
    ax.set_xticks(positions)
    ax.set_xticklabels([WLABEL[w] for w in WORKLOADS])
    ax.legend(ncol=3, frameon=False, loc="lower right")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Rank agreement with the measured throughput (1 = perfect)")
    save(fig, "fig4_rank_agreement")


def main() -> int:
    fig_throughput()
    fig_optimization_time()
    fig_cost_rank()
    fig_rank_agreement()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
