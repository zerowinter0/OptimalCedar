#!/usr/bin/env python3
"""Plot optimizer-only W=8 results using DP-Cedar as the fixed baseline."""

import argparse
import csv
import json
import math
from pathlib import Path

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

# Put the fixed baseline first and the proposed joint DP optimizer last among
# the non-Cedar alternatives so it remains visually prominent.
OPTIMIZERS = [
    ("cedar_dp_cedar_optimizer", "DP-Cedar", "#9E9E9E", ""),
    ("cedar_dj_optimizer", "Data-Juicer", "#E69F00", "---"),
    ("cedar_pecan_optimizer", "Pecan", "#009E73", "xxx"),
    ("cedar_dp_optimizer", "DP (ours)", "#0072B2", "///"),
    ("cedar_optimizer", "Cedar", "#CC79A7", "\\\\\\"),
]
BASELINE = "cedar_dp_cedar_optimizer"


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


def positive(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def load_report(path: Path) -> dict:
    with path.open() as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1:
        raise ValueError("Expected report schema_version=1")
    for workload, _ in WORKLOADS:
        entities = report["workloads"][workload]["entities"]
        baseline = entities[BASELINE]
        if baseline["status"] != "success" or not positive(
            baseline.get("execution_time_sec")
        ):
            raise ValueError(f"Missing DP-Cedar baseline for {workload}")
        for entity, _, _, _ in OPTIMIZERS:
            if entity not in entities:
                raise ValueError(f"Missing {entity} for {workload}")
    return report


def record(report: dict, workload: str, entity: str) -> dict:
    return report["workloads"][workload]["entities"][entity]


def speedup(report: dict, workload: str, entity: str):
    item = record(report, workload, entity)
    baseline = record(report, workload, BASELINE)
    if item["status"] != "success" or not positive(
        item.get("execution_time_sec")
    ):
        return None
    return baseline["execution_time_sec"] / item["execution_time_sec"]


def geometric_mean(values) -> float:
    values = [float(value) for value in values if positive(value)]
    if not values:
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_all(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def optimizer_effectiveness(report: dict, output_dir: Path) -> None:
    fig, (execution_ax, overhead_ax) = plt.subplots(
        2,
        1,
        figsize=(7.15, 4.15),
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.18},
    )
    width = 0.82 / len(OPTIMIZERS)
    centers = list(range(len(WORKLOADS)))

    for series_idx, (entity, label, color, hatch) in enumerate(OPTIMIZERS):
        offset = (series_idx - (len(OPTIMIZERS) - 1) / 2) * width
        for workload_idx, (workload, _) in enumerate(WORKLOADS):
            x = centers[workload_idx] + offset
            item = record(report, workload, entity)
            value = speedup(report, workload, entity)
            if positive(value):
                execution_ax.bar(
                    x,
                    value,
                    width=width * 0.92,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.45,
                    hatch=hatch,
                    zorder=3,
                )
            elif "timeout" in item["status"]:
                execution_ax.plot(
                    x,
                    0.08,
                    marker="x",
                    color="#D55E00",
                    markersize=4.5,
                    markeredgewidth=1.0,
                    zorder=4,
                )
                execution_ax.text(
                    x,
                    0.14,
                    "TO",
                    ha="center",
                    va="bottom",
                    color="#D55E00",
                    fontsize=5.5,
                    fontweight="bold",
                )

            setup = item.get("optimization_or_setup_time_sec")
            if positive(setup):
                edge = "#D55E00" if "timeout" in item["status"] else "#333333"
                linewidth = 1.0 if "timeout" in item["status"] else 0.45
                overhead_ax.bar(
                    x,
                    setup,
                    width=width * 0.92,
                    color=color,
                    edgecolor=edge,
                    linewidth=linewidth,
                    hatch=hatch,
                    zorder=3,
                )
                if "timeout" in item["status"]:
                    overhead_ax.text(
                        x,
                        setup * 1.08,
                        "TO",
                        ha="center",
                        va="bottom",
                        color="#D55E00",
                        fontsize=5.5,
                        fontweight="bold",
                    )

    execution_ax.axhline(1, color="#333333", linewidth=0.8)
    execution_ax.set_xlim(-0.55, len(WORKLOADS) - 0.45)
    execution_ax.set_ylim(0, 2.45)
    execution_ax.set_ylabel("Execution speedup\nvs. DP-Cedar")
    execution_ax.set_xticks(centers, [])
    execution_ax.set_title(
        "(a) Optimized-plan execution (higher is better)", loc="left", pad=2
    )
    style_axis(execution_ax)

    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in OPTIMIZERS
    ]
    handles.append(
        mpl.lines.Line2D(
            [], [], marker="x", linestyle="None", color="#D55E00", label="timeout"
        )
    )
    execution_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=6,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.6,
    )

    overhead_ax.set_xlim(-0.55, len(WORKLOADS) - 0.45)
    overhead_ax.set_yscale("log")
    overhead_ax.set_ylim(1, 520)
    overhead_ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    overhead_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g}"))
    overhead_ax.set_ylabel("Optimization/setup\ntime (s)")
    overhead_ax.set_xticks(centers, [label for _, label in WORKLOADS], rotation=25)
    for tick in overhead_ax.get_xticklabels():
        tick.set_ha("right")
    overhead_ax.set_title(
        "(b) End-to-end optimization/setup overhead (lower is better)",
        loc="left",
        pad=2,
    )
    style_axis(overhead_ax)
    fig.align_ylabels([execution_ax, overhead_ax])
    fig.text(
        0.995,
        0.003,
        "TO: Cedar optimization exceeded 300 s; execution was not run",
        ha="right",
        va="bottom",
        fontsize=6,
    )
    save_all(fig, output_dir, "optimizer_effectiveness_dp_cedar_baseline")


def aggregate_summary(report: dict, output_dir: Path) -> dict:
    common_workloads = [
        workload
        for workload, _ in WORKLOADS
        if record(report, workload, "cedar_optimizer")["status"] == "success"
    ]
    aggregate = {}
    for entity, label, color, hatch in OPTIMIZERS:
        all_valid = [
            speedup(report, workload, entity)
            for workload, _ in WORKLOADS
            if positive(speedup(report, workload, entity))
        ]
        common = [speedup(report, workload, entity) for workload in common_workloads]
        aggregate[entity] = {
            "label": label,
            "valid_workloads": len(all_valid),
            "geomean_all_valid": geometric_mean(all_valid),
            "geomean_common_8": geometric_mean(common),
        }

    totals = report["optimizer_totals"]
    labels = [label for _, label, _, _ in OPTIMIZERS]
    colors = [color for _, _, color, _ in OPTIMIZERS]
    hatches = [hatch for _, _, _, hatch in OPTIMIZERS]
    entities = [entity for entity, _, _, _ in OPTIMIZERS]
    common_values = [aggregate[entity]["geomean_common_8"] for entity in entities]
    setup_values = [
        totals[entity.removeprefix("cedar_")][
            "total_optimization_or_setup_time_sec"
        ]
        for entity in entities
    ]

    fig, (speed_ax, setup_ax) = plt.subplots(1, 2, figsize=(7.15, 2.25))
    x = list(range(len(OPTIMIZERS)))
    bars = speed_ax.bar(
        x,
        common_values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        width=0.7,
    )
    for bar, hatch, value in zip(bars, hatches, common_values):
        bar.set_hatch(hatch)
        speed_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}×",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    speed_ax.axhline(1, color="#333333", linewidth=0.8)
    speed_ax.set_ylim(0, max(common_values) * 1.23)
    speed_ax.set_xticks(x, labels, rotation=25)
    for tick in speed_ax.get_xticklabels():
        tick.set_ha("right")
    speed_ax.set_ylabel("Geomean speedup\nvs. DP-Cedar")
    speed_ax.set_title(
        "(a) Common 8 workloads with valid Cedar runs", loc="left", pad=3
    )
    style_axis(speed_ax)

    bars = setup_ax.bar(
        x,
        setup_values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        width=0.7,
    )
    for bar, hatch, value in zip(bars, hatches, setup_values):
        bar.set_hatch(hatch)
        setup_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    setup_ax.set_yscale("log")
    setup_ax.set_ylim(20, 1200)
    setup_ax.yaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
    setup_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _p: f"{y:g}"))
    setup_ax.set_xticks(x, labels, rotation=25)
    for tick in setup_ax.get_xticklabels():
        tick.set_ha("right")
    setup_ax.set_ylabel("Total optimization/setup time (s)")
    setup_ax.set_title(
        "(b) All 10 workloads; Cedar timeouts capped at 300 s",
        loc="left",
        pad=3,
    )
    style_axis(setup_ax)
    fig.subplots_adjust(wspace=0.32)
    save_all(fig, output_dir, "optimizer_aggregate_dp_cedar_baseline")
    return aggregate


def export_data(report: dict, aggregate: dict, output_dir: Path) -> None:
    with (output_dir / "optimizer_dp_cedar_baseline_data.tsv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "workload",
            "entity",
            "status",
            "execution_time_sec",
            "speedup_vs_dp_cedar",
            "optimization_or_setup_time_sec",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for workload, _ in WORKLOADS:
            for entity, _, _, _ in OPTIMIZERS:
                item = record(report, workload, entity)
                writer.writerow(
                    {
                        "workload": workload,
                        "entity": entity,
                        "status": item["status"],
                        "execution_time_sec": item.get("execution_time_sec"),
                        "speedup_vs_dp_cedar": speedup(report, workload, entity),
                        "optimization_or_setup_time_sec": item.get(
                            "optimization_or_setup_time_sec"
                        ),
                    }
                )

    with (output_dir / "optimizer_dp_cedar_aggregate.tsv").open(
        "w", newline=""
    ) as handle:
        fields = [
            "entity",
            "label",
            "valid_workloads",
            "geomean_all_valid",
            "geomean_common_8",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for entity, _, _, _ in OPTIMIZERS:
            writer.writerow({"entity": entity, **aggregate[entity]})


def write_readme(report_path: Path, aggregate: dict, output_dir: Path) -> None:
    dp_all = aggregate["cedar_dp_optimizer"]["geomean_all_valid"]
    dp_common = aggregate["cedar_dp_optimizer"]["geomean_common_8"]
    cedar_common = aggregate["cedar_optimizer"]["geomean_common_8"]
    text = f"""# Optimizer figures with DP-Cedar baseline

Source: `{report_path.resolve()}`.

The source contains ten W=8, CPU-budget-64 workloads under the reduced
single-run protocol. Execution time excludes profiling, optimization/setup,
and cache warmup. Every execution value is normalized to DP-Cedar for the
same workload. Cedar timed out during optimization on LLaVA and StackExchange;
those execution cells are categorical `TO` values and are not imputed.

Generated figures:

- `optimizer_effectiveness_dp_cedar_baseline`: per-workload execution speedup
  and optimization/setup overhead.
- `optimizer_aggregate_dp_cedar_baseline`: common-eight execution geomean and
  all-ten aggregate optimization/setup time.

Headline values:

- DP (ours) geomean speedup over DP-Cedar across all ten workloads: **{dp_all:.3f}x**.
- DP (ours) geomean speedup over DP-Cedar on the common eight workloads where
  Cedar also completed: **{dp_common:.3f}x**.
- Cedar geomean speedup over DP-Cedar on that same common-eight subset:
  **{cedar_common:.3f}x**; Cedar has only 8/10 executable plans.

Interpretation cautions:

- The source protocol has one measured execution per cell, so these figures
  intentionally contain no error bars and do not establish significance.
- The aggregate execution panel uses the common eight workloads to avoid
  comparing Cedar's 8-workload geomean against other optimizers' 10-workload
  geomeans. The per-workload panel retains LLaVA and StackExchange for the
  four optimizers that completed.
- Optimization/setup is an end-to-end measure. Cedar's two timeouts are
  conservatively counted at the 300-second cap in the aggregate panel.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.report.parent / "paper_figures_dp_cedar"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    report = load_report(args.report)
    optimizer_effectiveness(report, output_dir)
    aggregate = aggregate_summary(report, output_dir)
    export_data(report, aggregate, output_dir)
    write_readme(args.report, aggregate, output_dir)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
