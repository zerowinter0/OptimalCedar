#!/usr/bin/env python3
"""Plot the canonical seven-optimizer matrix with absolute execution time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


WORKLOADS = [
    ("alpaca_cot", "Alpaca-CoT", 8),
    ("general_video_refine", "GenerateVideo", 10),
    ("redpajama_code", "RP-Code", 17),
    ("pile_hackernews", "HN", 18),
    ("pile_pubmed_abstracts", "PubMed", 19),
    ("pile_uspto_backgrounds", "USPTO", 19),
    ("pile_europarl", "EuroParl", 19),
    ("stackexchange", "StackEx", 19),
]

OPTIMIZERS = [
    ("optimizer", "Cedar", "#8C8C8C", ""),
    ("dj_optimizer", "DJ", "#E69F00", "///"),
    ("pecan_optimizer", "Pecan", "#009E73", "\\\\"),
    ("dj_two_stage_optimizer", "DJ-TS", "#F0E442", "xx"),
    ("pecan_two_stage_optimizer", "Pecan-TS", "#D55E00", "++"),
    ("simple_dp_optimizer", "Simple-DP", "#56B4E9", ".."),
    ("dp_optimizer", "PICO", "#0072B2", "oo"),
]

TASK_TIMEOUT_SEC = 3600


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metadata(path: Path) -> dict[str, str]:
    metadata = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    return metadata


def load_matrix(matrix_root: Path) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for workload, label, operator_count in WORKLOADS:
        root = matrix_root / workload
        metadata = read_metadata(root / "metadata.txt")
        workload_data: dict[str, Any] = {
            "label": label,
            "operator_count": operator_count,
            "metadata": metadata,
            "optimizers": {},
        }
        expected_samples = int(metadata["samples"])
        for optimizer, optimizer_label, _color, _hatch in OPTIMIZERS:
            times = []
            observed_samples = []
            for repeat in (1, 2, 3):
                result_path = root / "results" / f"round{repeat}__{optimizer}.json"
                if not result_path.exists():
                    continue
                result = read_json(result_path)
                times.extend(float(value) for value in result["epoch_run_times"])
                observed_samples.extend(
                    int(value) for value in result["epoch_num_samples"]
                )

            unavailable_path = root / "plans" / f"{optimizer}.unavailable.json"
            timeout_path = root / "results" / f"round1__{optimizer}.timeout.json"
            setup_path = root / "warmup_results" / f"plan_only__{optimizer}.json"
            setup_sec = None
            if setup_path.exists():
                setup_result = read_json(setup_path)
                setup_sec = float(setup_result["runs"][0]["setup_time_sec"])

            if len(times) == 3:
                if len(set(observed_samples)) != 1:
                    raise ValueError(
                        f"Inconsistent samples for {workload}/{optimizer}: "
                        f"{observed_samples}"
                    )
                if observed_samples[0] != expected_samples:
                    raise ValueError(
                        f"Expected {expected_samples} samples for "
                        f"{workload}/{optimizer}, got {observed_samples[0]}"
                    )
                status = "success"
            elif unavailable_path.exists():
                status = "plan_timeout"
                setup_sec = float(read_json(unavailable_path)["timeout_sec"])
            elif timeout_path.exists():
                timeout = read_json(timeout_path)
                status = (
                    "plan_timeout"
                    if "optimization" in timeout.get("reason", "")
                    else "execution_timeout"
                )
                if status == "plan_timeout":
                    setup_sec = float(timeout.get("task_timeout_sec", TASK_TIMEOUT_SEC))
            else:
                raise ValueError(
                    f"Missing terminal result for {workload}/{optimizer}"
                )

            workload_data["optimizers"][optimizer] = {
                "label": optimizer_label,
                "status": status,
                "execution_times_sec": times,
                "mean_execution_sec": statistics.mean(times) if times else None,
                "sd_execution_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
                "repeats": len(times),
                "samples": expected_samples,
                "setup_sec": setup_sec,
            }
        matrix[workload] = workload_data
    return matrix


def style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles() -> list[Patch]:
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _optimizer, label, color, hatch in OPTIMIZERS
    ]
    handles.append(
        Patch(
            facecolor="#FFF2F0",
            edgecolor="#B2182B",
            hatch="////",
            linewidth=1.0,
            label="TO (unified task ≥ 1 h)",
        )
    )
    return handles


def draw_execution(
    matrix: dict[str, Any],
    output_dir: Path,
    *,
    ncols: int = 4,
    stem: str = "formal_seven_optimizer_execution",
) -> None:
    nrows = math.ceil(len(WORKLOADS) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.7 * ncols, 2.625 * nrows),
        squeeze=False,
    )
    for ax, (workload, label, operator_count) in zip(axes.flat, WORKLOADS):
        items = matrix[workload]["optimizers"]
        successes = [
            item["mean_execution_sec"]
            for item in items.values()
            if item["status"] == "success"
        ]
        maximum = max(successes)
        timeout_height = maximum * 1.10
        for index, (optimizer, _optimizer_label, color, hatch) in enumerate(
            OPTIMIZERS
        ):
            item = items[optimizer]
            if item["status"] == "success":
                ax.bar(
                    index,
                    item["mean_execution_sec"],
                    yerr=item["sd_execution_sec"],
                    width=0.78,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.5,
                    hatch=hatch,
                    error_kw={"elinewidth": 0.65, "capsize": 1.8},
                    zorder=3,
                )
            else:
                ax.bar(
                    index,
                    timeout_height,
                    width=0.78,
                    color="#FFF2F0",
                    edgecolor="#B2182B",
                    linewidth=1.0,
                    hatch="////",
                    zorder=3,
                )
                ax.text(
                    index,
                    timeout_height * 0.52,
                    "TO\n≥1h",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#8B0000",
                    fontweight="bold",
                )
        ax.set_ylim(0, timeout_height * 1.08)
        ax.set_xlim(-0.65, len(OPTIMIZERS) - 0.35)
        ax.set_xticks([])
        ax.set_title(f"{label} ({operator_count} ops)", loc="left", pad=2)
        style(ax)
    for ax in axes.flat[len(WORKLOADS) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Execution time (s)")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(len(legend_handles()), 9),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.6,
    )
    fig.suptitle(
        "End-to-end execution time (lower is better; mean ± SD, three rounds)",
        y=0.955,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.065, right=0.995, hspace=0.34, wspace=0.28)
    save(fig, output_dir, stem)


def draw_overhead(
    matrix: dict[str, Any],
    output_dir: Path,
    *,
    ncols: int = 4,
    stem: str = "formal_seven_optimizer_overhead",
) -> None:
    nrows = math.ceil(len(WORKLOADS) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.7 * ncols, 2.625 * nrows),
        squeeze=False,
    )
    for ax, (workload, label, operator_count) in zip(axes.flat, WORKLOADS):
        items = matrix[workload]["optimizers"]
        for index, (optimizer, _optimizer_label, color, hatch) in enumerate(
            OPTIMIZERS
        ):
            item = items[optimizer]
            value = item["setup_sec"]
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing setup time for {workload}/{optimizer}")
            plan_timeout = item["status"] == "plan_timeout"
            ax.bar(
                index,
                value,
                width=0.78,
                color="#FFF2F0" if plan_timeout else color,
                edgecolor="#B2182B" if plan_timeout else "#333333",
                linewidth=1.0 if plan_timeout else 0.5,
                hatch="////" if plan_timeout else hatch,
                zorder=3,
            )
            if plan_timeout:
                ax.text(
                    index,
                    value / 2.2,
                    "TO",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#8B0000",
                    fontweight="bold",
                )
        ax.set_yscale("log", base=2)
        ax.set_ylim(1, 4096)
        ax.set_xlim(-0.65, len(OPTIMIZERS) - 0.35)
        ax.set_xticks([])
        ax.set_title(f"{label} ({operator_count} ops)", loc="left", pad=2)
        style(ax)
    for ax in axes.flat[len(WORKLOADS) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Optimization time (s)")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(len(legend_handles()), 9),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.6,
    )
    fig.suptitle(
        "Optimization time (lower is better; log₂ scale)",
        y=0.955,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.065, right=0.995, hspace=0.34, wspace=0.28)
    save(fig, output_dir, stem)


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def export(matrix: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "formal_seven_optimizer_data.json"
    json_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "workload",
        "operator_count",
        "samples",
        "optimizer",
        "status",
        "mean_execution_sec",
        "sd_execution_sec",
        "repeats",
        "execution_times_sec",
        "setup_sec",
    ]
    with (output_dir / "formal_seven_optimizer_data.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for workload, _label, operator_count in WORKLOADS:
            for optimizer, _optimizer_label, _color, _hatch in OPTIMIZERS:
                item = matrix[workload]["optimizers"][optimizer]
                writer.writerow(
                    {
                        "workload": workload,
                        "operator_count": operator_count,
                        "samples": item["samples"],
                        "optimizer": optimizer,
                        "status": item["status"],
                        "mean_execution_sec": item["mean_execution_sec"],
                        "sd_execution_sec": item["sd_execution_sec"],
                        "repeats": item["repeats"],
                        "setup_sec": item["setup_sec"],
                        "execution_times_sec": ",".join(
                            str(value) for value in item["execution_times_sec"]
                        ),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure()
    matrix = load_matrix(args.matrix_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_execution(matrix, args.output_dir)
    draw_overhead(matrix, args.output_dir)
    export(matrix, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
