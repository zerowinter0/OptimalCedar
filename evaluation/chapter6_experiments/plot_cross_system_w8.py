#!/usr/bin/env python3
"""Render publication-ready figures from a cross-system W=8 report."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter


WORKLOADS = [
    ("coco", "COCO"),
    ("commonvoice", "CV"),
    ("commonvoice_cache", "CV-C"),
    ("llava_pretrain", "LLaVA"),
    ("redpajama_c4", "RP-C4"),
    ("stackexchange", "StackEx"),
    ("simclrv2", "SimCLR"),
    ("simclrv2_cache", "SimCLR-C"),
    ("wikitext103", "Wiki"),
    ("wikitext103_cache", "Wiki-C"),
]

OPTIMIZERS = [
    ("cedar_optimizer", "Cedar", "#7F7F7F", ""),
    ("cedar_dp_optimizer", "DP (ours)", "#0072B2", "///"),
    ("cedar_dp_cedar_optimizer", "DP-Cedar", "#56B4E9", "\\\\\\"),
    ("cedar_dj_optimizer", "Data-Juicer", "#E69F00", "---"),
    ("cedar_pecan_optimizer", "Pecan", "#009E73", "xxx"),
]

SYSTEMS = [
    ("cedar_dp_optimizer", "DP (ours)", "#0072B2", "///"),
    ("cedar_optimizer", "Cedar", "#7F7F7F", ""),
    ("pytorch", "PyTorch", "#D55E00", "\\\\\\"),
    ("tensorflow", "tf.data", "#CC79A7", "---"),
    ("ray", "Ray Data", "#009E73", "xxx"),
    ("plumber", "Plumber", "#E69F00", "..."),
]

AVAILABILITY_SYSTEMS = SYSTEMS + [
    ("fastflow", "FastFlow", "#000000", "+++"),
]

STATUS_STYLE = {
    "success": ("S", "#009E73", "success"),
    "optimizer_timeout": ("TO", "#D55E00", "timeout"),
    "infeasible_timeout": ("TO", "#D55E00", "timeout"),
    "infeasible": ("IF", "#E69F00", "infeasible"),
    "unsupported": ("U", "#BDBDBD", "unsupported"),
    "environment_unavailable": ("ENV", "#8C8C8C", "environment unavailable"),
    "invalid": ("INV", "#CC0000", "invalid"),
    "failed": ("F", "#CC0000", "failed"),
    "not_run": ("NR", "#FFFFFF", "not run"),
}


def configure_matplotlib() -> None:
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


def load_report(path: Path) -> Dict:
    with path.open() as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1:
        raise ValueError(f"Unsupported report schema: {report.get('schema_version')}")
    missing = [name for name, _ in WORKLOADS if name not in report["workloads"]]
    if missing:
        raise ValueError(f"Report is missing workloads: {missing}")
    invalid = []
    for workload, _ in WORKLOADS:
        for entity, record in report["workloads"][workload]["entities"].items():
            if record["status"] in {"invalid", "failed", "not_run"}:
                invalid.append((workload, entity, record["status"]))
    if invalid:
        raise ValueError(f"Refusing to plot invalid/incomplete records: {invalid}")
    return report


def record(report: Dict, workload: str, entity: str) -> Dict:
    return report["workloads"][workload]["entities"][entity]


def execution_baseline(report: Dict, workload: str) -> Tuple[str, Dict]:
    """Use Cedar when available, otherwise fall back to DP-Cedar."""
    cedar = record(report, workload, "cedar_optimizer")
    if cedar["status"] == "success" and positive_number(
        cedar.get("execution_time_sec")
    ):
        return "cedar_optimizer", cedar

    dp_cedar = record(report, workload, "cedar_dp_cedar_optimizer")
    if dp_cedar["status"] == "success" and positive_number(
        dp_cedar.get("execution_time_sec")
    ):
        return "cedar_dp_cedar_optimizer", dp_cedar
    raise ValueError(
        f"Neither Cedar nor DP-Cedar has a valid execution time for {workload}"
    )


def speedup_vs_execution_baseline(
    report: Dict, workload: str, item: Dict
) -> float:
    if item["status"] != "success" or not positive_number(
        item.get("execution_time_sec")
    ):
        return None
    _, baseline = execution_baseline(report, workload)
    return baseline["execution_time_sec"] / item["execution_time_sec"]


def positive_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def log_tick(value, _position) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.4g}"


def grouped_bars(
    ax,
    report: Dict,
    series: List[Tuple[str, str, str, str]],
    value_getter,
    missing_y: float,
    annotate_timeout: bool = False,
    mark_fallback_baseline: bool = False,
) -> None:
    width = 0.82 / len(series)
    centers = list(range(len(WORKLOADS)))
    for series_idx, (entity, label, color, hatch) in enumerate(series):
        offset = (series_idx - (len(series) - 1) / 2) * width
        for workload_idx, (workload, _) in enumerate(WORKLOADS):
            item = record(report, workload, entity)
            value = value_getter(workload, item)
            x = centers[workload_idx] + offset
            if positive_number(value):
                edge = "#333333"
                linewidth = 0.45
                if item["status"] == "optimizer_timeout":
                    edge = "#D55E00"
                    linewidth = 1.0
                ax.bar(
                    x,
                    value,
                    width=width * 0.92,
                    color=color,
                    edgecolor=edge,
                    linewidth=linewidth,
                    hatch=hatch,
                    zorder=3,
                )
                if annotate_timeout and item["status"] == "optimizer_timeout":
                    ax.text(
                        x,
                        value * 1.07,
                        "TO",
                        ha="center",
                        va="bottom",
                        color="#D55E00",
                        fontsize=5.5,
                        fontweight="bold",
                    )
            elif item["status"] != "success":
                marker_color = (
                    "#D55E00"
                    if "timeout" in item["status"] or item["status"] == "infeasible"
                    else "#777777"
                )
                ax.plot(
                    x,
                    missing_y,
                    marker="x",
                    markersize=4.3,
                    markeredgewidth=0.9,
                    color=marker_color,
                    zorder=4,
                )

    ax.set_xlim(-0.55, len(WORKLOADS) - 0.45)
    tick_labels = []
    for workload, label in WORKLOADS:
        if (
            mark_fallback_baseline
            and execution_baseline(report, workload)[0]
            == "cedar_dp_cedar_optimizer"
        ):
            label += "*"
        tick_labels.append(label)
    ax.set_xticks(centers, tick_labels)


def save_all(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def optimizer_figure(report: Dict, output_dir: Path) -> None:
    fig, (perf_ax, setup_ax) = plt.subplots(
        2,
        1,
        figsize=(7.15, 4.15),
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.18},
    )

    grouped_bars(
        perf_ax,
        report,
        OPTIMIZERS,
        lambda workload, item: speedup_vs_execution_baseline(
            report, workload, item
        ),
        missing_y=0.28,
        mark_fallback_baseline=True,
    )
    perf_ax.set_yscale("log", base=2)
    perf_ax.set_ylim(0.25, 4)
    perf_ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    perf_ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
    perf_ax.axhline(1, color="#333333", linewidth=0.8)
    perf_ax.set_ylabel("Execution speedup\nvs. Cedar baseline")
    perf_ax.set_xticklabels([])
    perf_ax.set_title("(a) Optimized-plan execution performance", loc="left", pad=2)
    style_axis(perf_ax)

    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in OPTIMIZERS
    ]
    handles.append(
        mpl.lines.Line2D(
            [], [], marker="x", linestyle="None", color="#D55E00", label="timeout"
        )
    )
    perf_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.29),
        ncol=6,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.6,
    )

    grouped_bars(
        setup_ax,
        report,
        OPTIMIZERS,
        lambda _workload, item: item.get("optimization_or_setup_time_sec"),
        missing_y=1.0,
        annotate_timeout=True,
        mark_fallback_baseline=True,
    )
    setup_ax.set_yscale("log")
    setup_ax.set_ylim(1, 520)
    setup_ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    setup_ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
    setup_ax.set_ylabel("Optimization/setup\ntime (s)")
    setup_ax.set_title(
        "(b) Optimization/setup overhead (TO is capped at 300 s)",
        loc="left",
        pad=2,
    )
    setup_ax.tick_params(axis="x", rotation=25, pad=1)
    for label in setup_ax.get_xticklabels():
        label.set_ha("right")
    style_axis(setup_ax)
    fig.align_ylabels([perf_ax, setup_ax])
    fig.text(
        0.995,
        0.003,
        "* DP-Cedar baseline where Cedar optimizer times out",
        ha="right",
        va="bottom",
        fontsize=6,
    )
    save_all(fig, output_dir, "optimizer_effectiveness")


def system_figure(report: Dict, output_dir: Path) -> None:
    fig, (perf_ax, status_ax) = plt.subplots(
        2,
        1,
        figsize=(7.15, 4.3),
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.16},
    )

    grouped_bars(
        perf_ax,
        report,
        SYSTEMS,
        lambda workload, item: speedup_vs_execution_baseline(
            report, workload, item
        ),
        missing_y=1.35e-4,
        mark_fallback_baseline=True,
    )
    perf_ax.set_yscale("log")
    perf_ax.set_ylim(1e-4, 500)
    perf_ax.yaxis.set_major_locator(
        FixedLocator([1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100])
    )
    perf_ax.yaxis.set_major_formatter(FuncFormatter(log_tick))
    perf_ax.axhline(1, color="#333333", linewidth=0.8)
    perf_ax.set_ylabel("Execution speedup vs. Cedar baseline")
    perf_ax.set_xticklabels([])
    perf_ax.set_title(
        "(a) Cross-system execution performance (optimization and warmup excluded)",
        loc="left",
        pad=2,
    )
    style_axis(perf_ax)
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in SYSTEMS
    ]
    perf_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=6,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.7,
    )

    status_ax.set_xlim(-0.5, len(WORKLOADS) - 0.5)
    status_ax.set_ylim(-0.5, len(AVAILABILITY_SYSTEMS) - 0.5)
    for row, (entity, label, _, _) in enumerate(AVAILABILITY_SYSTEMS):
        for col, (workload, _) in enumerate(WORKLOADS):
            state = record(report, workload, entity)["status"]
            code, color, _ = STATUS_STYLE.get(
                state, (state[:3].upper(), "#CC0000", state)
            )
            status_ax.add_patch(
                mpl.patches.Rectangle(
                    (col - 0.48, row - 0.43),
                    0.96,
                    0.86,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            text_color = "white" if color not in {"#BDBDBD", "#FFFFFF"} else "#333333"
            status_ax.text(
                col,
                row,
                code,
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.5,
                fontweight="bold",
            )
    status_tick_labels = []
    for workload, label in WORKLOADS:
        if execution_baseline(report, workload)[0] == "cedar_dp_cedar_optimizer":
            label += "*"
        status_tick_labels.append(label)
    status_ax.set_xticks(
        range(len(WORKLOADS)), status_tick_labels, rotation=25
    )
    for label in status_ax.get_xticklabels():
        label.set_ha("right")
    status_ax.set_yticks(
        range(len(AVAILABILITY_SYSTEMS)),
        [label for _, label, _, _ in AVAILABILITY_SYSTEMS],
    )
    status_ax.invert_yaxis()
    status_ax.tick_params(length=0, pad=1.5)
    status_ax.set_title("(b) Experimental availability", loc="left", pad=2)
    for spine in status_ax.spines.values():
        spine.set_visible(False)

    present_states = {
        record(report, workload, entity)["status"]
        for workload, _ in WORKLOADS
        for entity, _, _, _ in AVAILABILITY_SYSTEMS
    }
    status_handles = []
    used_labels = set()
    for state, (code, color, description) in STATUS_STYLE.items():
        if state not in present_states or description in used_labels:
            continue
        used_labels.add(description)
        status_handles.append(
            Patch(facecolor=color, edgecolor="white", label=f"{code}: {description}")
        )
    status_ax.legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.48),
        ncol=min(5, len(status_handles)),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.2,
    )
    fig.text(
        0.995,
        0.003,
        "* DP-Cedar baseline where Cedar optimizer times out",
        ha="right",
        va="bottom",
        fontsize=6,
    )
    save_all(fig, output_dir, "cross_system_speedup")


def optimizer_total_figure(report: Dict, output_dir: Path) -> None:
    totals = report["optimizer_totals"]
    keys = [entity.removeprefix("cedar_") for entity, _, _, _ in OPTIMIZERS]
    labels = [label for _, label, _, _ in OPTIMIZERS]
    colors = [color for _, _, color, _ in OPTIMIZERS]
    hatches = [hatch for _, _, _, hatch in OPTIMIZERS]
    values = [totals[key]["total_optimization_or_setup_time_sec"] for key in keys]

    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    y = list(range(len(labels)))
    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        height=0.68,
    )
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        ax.text(
            value * 1.06,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=6.5,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(20, 1200)
    ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(log_tick))
    ax.set_xlabel("Aggregate optimization/setup time (s)")
    ax.set_title("All ten workloads; timeouts capped at 300 s", loc="left", pad=3)
    style_axis(ax)
    save_all(fig, output_dir, "optimizer_total_setup_time")


def export_tsv(report: Dict, output_dir: Path) -> None:
    fields = [
        "workload",
        "workload_label",
        "entity",
        "status",
        "execution_time_sec",
        "throughput_samples_per_sec",
        "optimization_or_setup_time_sec",
        "profile_or_cache_warmup_time_sec",
        "speedup_vs_cedar_no_optimizer",
        "speedup_vs_cedar_optimizer",
        "execution_baseline",
        "speedup_vs_execution_baseline",
        "reason",
    ]
    with (output_dir / "figure_data.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for workload, workload_label in WORKLOADS:
            for entity, item in sorted(
                report["workloads"][workload]["entities"].items()
            ):
                baseline_entity, _ = execution_baseline(report, workload)
                row = {field: item.get(field) for field in fields}
                row.update(
                    {
                        "workload": workload,
                        "workload_label": workload_label,
                        "entity": entity,
                        "execution_baseline": baseline_entity,
                        "speedup_vs_execution_baseline": (
                            speedup_vs_execution_baseline(
                                report, workload, item
                            )
                        ),
                    }
                )
                writer.writerow(row)


def geometric_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if positive_number(value)]
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_notes(report: Dict, report_path: Path, output_dir: Path) -> None:
    dp_vs_selected_baseline = []
    dp_vs_raw = []
    fallback_workloads = []
    for workload, _ in WORKLOADS:
        dp = record(report, workload, "cedar_dp_optimizer")
        baseline_entity, baseline = execution_baseline(report, workload)
        if baseline_entity == "cedar_dp_cedar_optimizer":
            fallback_workloads.append(workload)
        if dp["status"] == "success":
            dp_vs_raw.append(dp["speedup_vs_cedar_no_optimizer"])
            dp_vs_selected_baseline.append(
                baseline["execution_time_sec"] / dp["execution_time_sec"]
            )

    totals = report["optimizer_totals"]
    original_total = totals["optimizer"]["total_optimization_or_setup_time_sec"]
    dp_total = totals["dp_optimizer"]["total_optimization_or_setup_time_sec"]
    notes = f"""# Cross-system W=8 paper figures

Source report: `{report_path.resolve()}`

Generated figures:

- `optimizer_effectiveness`: optimizer execution speedup against Cedar (falling back to DP-Cedar) and per-workload setup overhead.
- `cross_system_speedup`: proposed DP versus Cedar and external systems, plus an explicit availability matrix.
- `optimizer_total_setup_time`: compact aggregate optimizer-overhead figure.

Reproduce inside `optimalcedar-torch201-dev` after activating `env`:

```bash
python evaluation/chapter6_experiments/plot_cross_system_w8.py \\
  {report_path.resolve()}
```

Suggested captions:

1. **Optimizer effectiveness.** Execution speedup over the original Cedar optimizer and optimization/setup overhead at W=8 and CPU budget 64. Where Cedar optimization exceeds the 300 s limit (LLaVA and StackExchange, marked `*`), DP-Cedar is the execution baseline. Execution excludes optimization, profiling, and cache materialization. TO denotes the 300 s optimizer timeout. Each bar is one formal measured run.
2. **Cross-system comparison.** Execution-only speedup over the original Cedar optimizer under the same bounded inputs; DP-Cedar is used as the fallback baseline for the two `*` workloads where Cedar times out. Cache warmup and system setup are excluded and reported separately. Missing bars are explained by the availability matrix. Plumber values are normalized from its measured batch-1 steady-state throughput; PyTorch has no native dataset-cache policy in cache workloads.
3. **Aggregate optimization overhead.** Total measured optimization/setup time over all ten workloads. Cedar's two optimizer timeouts are conservatively counted at the 300 s cap.

Headline values (computed from successful comparable cells):

- DP geometric-mean execution speedup over the selected Cedar/DP-Cedar baseline: **{geometric_mean(dp_vs_selected_baseline):.2f}x** across 10 workloads.
- The fallback baseline is DP-Cedar for: **{", ".join(fallback_workloads)}**; all other workloads use original Cedar.
- For context, DP geometric-mean execution speedup over the unoptimized raw Cedar plan is **{geometric_mean(dp_vs_raw):.2f}x**.
- Aggregate setup reduction versus original Cedar: **{original_total / dp_total:.2f}x** ({original_total:.1f} s versus {dp_total:.1f} s; Cedar timeout values are capped).

Interpretation cautions:

- The current protocol contains one formal measured execution per cell, so the figures intentionally contain no error bars.
- Do not claim statistical significance until repeated formal runs are available.
- Cache policies are not identical across all external systems; retain the cache-policy qualification in the caption.
- `unsupported`, `infeasible`, and timeout cells are categorical outcomes and are never converted into numeric bars.
"""
    (output_dir / "README.md").write_text(notes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="Path to report.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <report directory>/paper_figures",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.report.parent / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    report = load_report(args.report)
    optimizer_figure(report, output_dir)
    system_figure(report, output_dir)
    optimizer_total_figure(report, output_dir)
    export_tsv(report, output_dir)
    write_notes(report, args.report, output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
