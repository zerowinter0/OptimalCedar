#!/usr/bin/env python3
"""Plot all Cedar optimizers and external systems using absolute run time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from evaluation.chapter6_experiments.plot_paper_cross_system_speedup import (
    EXPECTED_REPEATS,
    aggregate_system,
    positive,
)


# Workloads are ordered by logical Cedar operator count (excluding the source).
# Within a tie, non-cache and cache variants of the same pipeline stay adjacent.
WORKLOADS = [
    ("commonvoice", "CV", 7),
    ("commonvoice_cache", "CV [Cache]", 7),
    ("simclrv2", "SimCLR", 9),
    ("simclrv2_cache", "SimCLR [Cache]", 9),
    ("wikitext103", "Wiki", 9),
    ("wikitext103_cache", "Wiki [Cache]", 9),
    ("redpajama_c4", "RP-C4", 17),
    ("pile_hackernews", "HN", 18),
    ("stackexchange", "StackEx", 19),
    ("pile_pubmed_abstracts", "PubMed", 19),
    ("pile_uspto_backgrounds", "USPTO", 19),
    ("pile_europarl", "EuroParl", 19),
]


OPTIMIZERS = [
    ("optimizer", "Cedar", "#7F7F7F", ""),
    ("dj_optimizer", "DJ Opt.", "#F0A202", "---"),
    ("dp_cedar_optimizer", "DP-Cedar", "#56B4E9", "\\\\\\"),
    ("pecan_optimizer", "Pecan", "#009E73", "xxx"),
    ("dp_optimizer", "PICO", "#0072B2", "///"),
]

EXTERNAL_SYSTEMS = [
    ("pytorch", "PyTorch", "#D55E00", "\\\\\\"),
    ("tensorflow", "tf.data", "#CC79A7", "---"),
    ("ray", "Ray Data", "#44AA99", "xxx"),
    ("plumber", "Plumber", "#8C6BB1", "+++"),
    ("fastflow", "FastFlow", "#000000", "ooo"),
]

SERIES = [
    (f"optimizer:{name}", label, color, hatch)
    for name, label, color, hatch in OPTIMIZERS
] + [
    (f"system:{name}", label, color, hatch)
    for name, label, color, hatch in EXTERNAL_SYSTEMS
]


def read_optimizer_results(path: Path) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = {
        name: {} for name, _, _ in WORKLOADS
    }
    allowed = {name for name, *_ in OPTIMIZERS}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["workload"] not in output or row["optimizer"] not in allowed:
                continue
            successful = row["status"] == "success"
            output[row["workload"]][row["optimizer"]] = {
                "status": "success" if successful else row["status"],
                "mean_execution_sec": (
                    float(row["mean_execution_sec"]) if successful else None
                ),
                "sd_execution_sec": (
                    float(row["sd_execution_sec"]) if successful else None
                ),
                "repeats": int(row["repeats"]),
                "samples": int(row["samples"]) if row["samples"] else None,
                "measurement_kind": "exact measured execution",
                "protocol": row.get("protocol", ""),
                "sources": [row.get("source", str(path))],
                "reasons": [],
            }
    missing = [
        (workload, optimizer)
        for workload, rows in output.items()
        for optimizer in allowed
        if optimizer not in rows
    ]
    if missing:
        raise ValueError(f"Optimizer TSV is incomplete: {missing}")
    for workload, rows in output.items():
        samples = rows["dp_optimizer"]["samples"]
        if samples is None:
            raise ValueError(f"PICO sample count is unavailable for {workload}")
        for item in rows.values():
            if item["samples"] is None:
                item["samples"] = samples
    return output


def build_report(run_root: Path, optimizer_tsv: Path) -> dict:
    optimizers = read_optimizer_results(optimizer_tsv)
    workloads = {}
    for workload, _, _ in WORKLOADS:
        entities = {
            f"optimizer:{name}": optimizers[workload][name]
            for name, *_ in OPTIMIZERS
        }
        samples = optimizers[workload]["dp_optimizer"]["samples"]
        for system, *_ in EXTERNAL_SYSTEMS:
            entities[f"system:{system}"] = aggregate_system(
                run_root, workload, system, samples
            )
        workloads[workload] = {"samples": samples, "entities": entities}
    invalid = [
        (workload, entity, item["status"])
        for workload, payload in workloads.items()
        for entity, item in payload["entities"].items()
        if item["status"] in {"failed", "invalid", "not_run"}
    ]
    if invalid:
        raise ValueError(f"Refusing to plot incomplete results: {invalid}")
    return {
        "schema_version": 1,
        "protocol": {
            "workers": 8,
            "cpu_budget": 64,
            "repeats": EXPECTED_REPEATS,
            "cell_timeout_sec": 3600,
            "metric": "absolute execution time in seconds",
            "excluded_workload": "pile_freelaw",
            "operator_count_excludes_source": True,
            "displayed_workloads": [
                {
                    "workload": workload,
                    "label": label,
                    "operator_count": operator_count,
                }
                for workload, label, operator_count in WORKLOADS
            ],
        },
        "workloads": workloads,
    }


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def tick_label(value, _position) -> str:
    if value >= 10000:
        return f"{value / 1000:.0f}k"
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}" if value >= 1 else f"{value:.2g}"


def plot_workload(
    ax,
    report: dict,
    workload: str,
    title: str,
    *,
    show_series_labels: bool,
) -> None:
    items = report["workloads"][workload]["entities"]
    upper_candidates = []
    for entity, *_ in SERIES:
        item = items[entity]
        value = item.get("mean_execution_sec")
        error = item.get("sd_execution_sec") or 0.0
        if item["status"] == "success" and positive(value):
            upper_candidates.append(float(value) + float(error))
    upper = max(upper_candidates) * 1.15
    marker_y = upper * 0.025

    for series_index, (entity, _, color, hatch) in enumerate(SERIES):
        item = items[entity]
        value = item.get("mean_execution_sec")
        if item["status"] == "success" and positive(value):
            error = item.get("sd_execution_sec")
            ax.bar(
                series_index,
                value,
                yerr=error if positive(error) else None,
                error_kw={"elinewidth": 0.45, "capsize": 1.0},
                width=0.76,
                color=color,
                edgecolor="#333333",
                linewidth=0.42,
                hatch=hatch,
                zorder=3,
            )
        elif item["status"] == "infeasible_timeout":
            # A timeout is categorical rather than a completed 3600-second
            # measurement. Give it a full, conspicuous bar slot without using
            # its timeout threshold to flatten the workload's linear scale.
            ax.bar(
                series_index,
                upper * 0.94,
                width=0.76,
                facecolor="#FDE0DD",
                edgecolor="#C62828",
                linewidth=1.0,
                hatch="////",
                zorder=4,
            )
            ax.text(
                series_index,
                upper * 0.90,
                "TO",
                ha="center",
                va="top",
                fontsize=6.2,
                fontweight="bold",
                color="#8E0000",
                zorder=5,
            )
        else:
            ax.plot(
                series_index,
                marker_y,
                marker="x",
                color="#777777",
                markersize=3.2,
                markeredgewidth=0.7,
                zorder=4,
            )
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.65, len(SERIES) - 0.35)
    if show_series_labels:
        ax.set_xticks(
            range(len(SERIES)),
            [label for _, label, *_ in SERIES],
            rotation=48,
            ha="right",
        )
    else:
        ax.set_xticks([])
    ax.set_title(title, loc="left", pad=2)


def render(report: dict, output_dir: Path) -> None:
    configure()
    fig, axes = plt.subplots(
        3, 4, figsize=(10.0, 6.7), gridspec_kw={"hspace": 0.43, "wspace": 0.34}
    )
    for index, ((workload, label, operator_count), ax) in enumerate(
        zip(WORKLOADS, axes.flat)
    ):
        plot_workload(
            ax,
            report,
            workload,
            f"({chr(ord('a') + index)}) {label} ({operator_count} ops)",
            show_series_labels=False,
        )
        if index % 4 == 0:
            ax.set_ylabel("Execution time\n(s)")
    for ax in axes.flat[len(WORKLOADS):]:
        ax.set_visible(False)
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in SERIES
    ]
    handles.extend(
        [
            Patch(
                facecolor="#FDE0DD",
                edgecolor="#C62828",
                hatch="////",
                label="timeout (> 1 h)",
            ),
            mpl.lines.Line2D(
                [], [], marker="x", linestyle="None", color="#777777",
                markersize=4, label="unsupported/unavailable",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        handlelength=1.55,
        columnspacing=0.75,
    )
    fig.subplots_adjust(top=0.86, bottom=0.075)
    fig.text(
        0.995,
        0.006,
        "Each workload uses an independent linear y-axis. Bars show three-run means; error bars show one SD.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"optimizer_and_system_execution_time.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def write_tsv(report: dict, path: Path) -> None:
    fields = [
        "workload", "entity", "status", "mean_execution_sec",
        "sd_execution_sec", "repeats", "samples", "measurement_kind",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for workload, _, _ in WORKLOADS:
            for entity, *_ in SERIES:
                item = report["workloads"][workload]["entities"][entity]
                writer.writerow(
                    {
                        "workload": workload,
                        "entity": entity,
                        "status": item["status"],
                        "mean_execution_sec": item.get("mean_execution_sec"),
                        "sd_execution_sec": item.get("sd_execution_sec"),
                        "repeats": item["repeats"],
                        "samples": item["samples"],
                        "measurement_kind": item.get("measurement_kind"),
                        "reasons": "; ".join(item.get("reasons", [])),
                    }
                )




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--optimizer-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.run_root, args.optimizer_tsv)
    (args.output_dir / "optimizer_and_system_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_tsv(report, args.output_dir / "optimizer_and_system_data.tsv")
    render(report, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
