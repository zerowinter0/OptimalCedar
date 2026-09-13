#!/usr/bin/env python3
"""Plot the paper subset where a DP optimizer beats both reorder heuristics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Sorted by logical operator count, excluding the source.
WORKLOADS = [
    ("coco", "COCO", 6),
    ("commonvoice", "CommonVoice", 7),
    ("alpaca_cot", "Alpaca-CoT", 8),
    ("simclrv2", "SimCLR-v2", 9),
    ("simclrv2_cache", "SimCLR-v2 [Cache]", 9),
    ("wikitext103_cache", "WikiText-103 [Cache]", 9),
    ("redpajama_code", "RP-Code", 17),
    ("pile_hackernews", "HackerNews", 18),
    ("stackexchange", "StackExchange", 19),
    ("pile_pubmed_abstracts", "PubMed", 19),
    ("pile_uspto_backgrounds", "USPTO", 19),
]

OPTIMIZERS = [
    ("dj_optimizer", "DJ", "#E69F00", "///"),
    ("pecan_optimizer", "Pecan", "#009E73", "\\\\"),
    ("simple_dp_optimizer", "Simple-DP", "#56B4E9", ".."),
    ("dp_optimizer", "PICO", "#0072B2", "oo"),
]


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
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def load_matrix(
    matrix_root: Path,
    *,
    workloads: Sequence[tuple[str, str, int]] = WORKLOADS,
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for workload, label, operator_count in workloads:
        workload_root = matrix_root / workload
        metadata = read_metadata(workload_root / "metadata.txt")
        expected_samples = int(metadata["samples"])
        workload_data: dict[str, Any] = {
            "label": label,
            "operator_count": operator_count,
            "samples": expected_samples,
            "metadata_source": _relative(
                workload_root / "metadata.txt", matrix_root
            ),
            "optimizers": {},
        }
        for optimizer, optimizer_label, _color, _hatch in OPTIMIZERS:
            result_path = (
                workload_root / "results" / f"round1__{optimizer}.json"
            )
            timeout_path = (
                workload_root
                / "results"
                / f"round1__{optimizer}.timeout.json"
            )
            unavailable_path = (
                workload_root / "plans" / f"{optimizer}.unavailable.json"
            )
            setup_path = (
                workload_root
                / "warmup_results"
                / f"plan_only__{optimizer}.json"
            )
            execution_sec = None
            setup_sec = None
            result_source = None
            setup_source = None

            if result_path.exists():
                result = read_json(result_path)
                times = [float(value) for value in result["epoch_run_times"]]
                samples = [int(value) for value in result["epoch_num_samples"]]
                if len(times) != 1 or samples != [expected_samples]:
                    raise ValueError(
                        f"Invalid single-run result for {workload}/{optimizer}: "
                        f"times={times}, samples={samples}, expected={expected_samples}"
                    )
                status = "success"
                execution_sec = times[0]
                result_source = _relative(result_path, matrix_root)
            elif unavailable_path.exists():
                unavailable = read_json(unavailable_path)
                status = "plan_timeout"
                setup_sec = float(unavailable["timeout_sec"])
                result_source = _relative(unavailable_path, matrix_root)
            elif timeout_path.exists():
                timeout = read_json(timeout_path)
                reason = str(timeout.get("reason", ""))
                status = (
                    "plan_timeout"
                    if "optimization" in reason
                    else "execution_timeout"
                )
                if status == "plan_timeout":
                    setup_sec = float(timeout["task_timeout_sec"])
                result_source = _relative(timeout_path, matrix_root)
            else:
                raise ValueError(
                    f"Missing terminal result for {workload}/{optimizer}"
                )

            if setup_path.exists():
                setup = read_json(setup_path)
                setup_sec = float(setup["runs"][0]["setup_time_sec"])
                setup_source = _relative(setup_path, matrix_root)
            if setup_sec is None:
                raise ValueError(f"Missing setup time for {workload}/{optimizer}")

            workload_data["optimizers"][optimizer] = {
                "label": optimizer_label,
                "status": status,
                "execution_sec": execution_sec,
                "setup_sec": setup_sec,
                "result_source": result_source,
                "setup_source": setup_source,
            }
        matrix[workload] = workload_data
    return matrix


def validate_selection(matrix: dict[str, Any]) -> None:
    """Require a completed DP method to beat both heuristic outcomes."""
    for workload, workload_data in matrix.items():
        items = workload_data["optimizers"]
        dp_times = [
            items[name]["execution_sec"]
            for name in ("simple_dp_optimizer", "dp_optimizer")
            if items[name]["status"] == "success"
        ]
        if not dp_times:
            raise ValueError(
                f"{workload} violates selection rule: no DP execution"
            )
        best_dp = min(dp_times)
        qualifies = True
        for name in ("dj_optimizer", "pecan_optimizer"):
            item = items[name]
            if item["status"] == "success":
                qualifies = qualifies and best_dp < item["execution_sec"]
            else:
                qualifies = qualifies and item["status"] in {
                    "execution_timeout",
                    "plan_timeout",
                }
        if not qualifies:
            raise ValueError(
                f"{workload} violates selection rule: best DP={best_dp}"
            )


def _style(ax, *, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles(*, overhead: bool) -> list[Patch]:
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _optimizer, label, color, hatch in OPTIMIZERS
    ]
    if not overhead:
        handles.extend(
            [
                Patch(
                    facecolor="#FFF2F0",
                    edgecolor="#B2182B",
                    hatch="////",
                    label="Execution TO (≥1 h)",
                ),
                Patch(
                    facecolor="#F3E5F5",
                    edgecolor="#6A1B9A",
                    hatch="xx",
                    label="Plan TO",
                ),
            ]
        )
    return handles


def draw_execution(matrix: dict[str, Any], output_dir: Path) -> None:
    ncols = 4
    nrows = math.ceil(len(WORKLOADS) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.7 * ncols, 2.3 * nrows),
        squeeze=False,
    )
    for ax, (workload, label, operator_count) in zip(axes.flat, WORKLOADS):
        items = matrix[workload]["optimizers"]
        completed = [
            item["execution_sec"]
            for item in items.values()
            if item["status"] == "success"
        ]
        panel_max = max(completed)
        has_execution_timeout = any(
            item["status"] == "execution_timeout" for item in items.values()
        )
        y_max = max(panel_max * 1.16, 3600.0 * 1.04) if has_execution_timeout else panel_max * 1.18
        for index, (optimizer, _name, color, hatch) in enumerate(OPTIMIZERS):
            item = items[optimizer]
            if item["status"] == "success":
                value = item["execution_sec"]
                facecolor, edgecolor, bar_hatch = color, "#333333", hatch
                text = None
            elif item["status"] == "execution_timeout":
                value = 3600.0
                facecolor, edgecolor, bar_hatch = "#FFF2F0", "#B2182B", "////"
                text = "TO\n≥1h"
            else:
                value = panel_max * 1.08
                facecolor, edgecolor, bar_hatch = "#F3E5F5", "#6A1B9A", "xx"
                text = "Plan\nTO"
            ax.bar(
                index,
                value,
                width=0.78,
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=0.8,
                hatch=bar_hatch,
                zorder=3,
            )
            if text:
                ax.text(
                    index,
                    value * 0.52,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=edgecolor,
                    fontweight="bold",
                )
        ax.set_ylim(0, y_max)
        ax.set_xlim(-0.6, len(OPTIMIZERS) - 0.4)
        ax.set_xticks([])
        ax.set_title(f"{label} ({operator_count} ops)", loc="left", pad=2)
        _style(ax)
    for ax in axes.flat[len(WORKLOADS) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Execution time (s)")
    fig.legend(
        handles=_legend_handles(overhead=False),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=6,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.6,
    )
    fig.suptitle(
        "End-to-end execution time (lower is better; one formal run)",
        y=0.955,
        fontsize=9,
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.055,
        left=0.065,
        right=0.995,
        hspace=0.38,
        wspace=0.28,
    )
    save(fig, output_dir, "optimizer_execution")


def draw_overhead(matrix: dict[str, Any], output_dir: Path) -> None:
    ncols = 4
    nrows = math.ceil(len(WORKLOADS) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.7 * ncols, 2.3 * nrows),
        squeeze=False,
    )
    for ax, (workload, label, operator_count) in zip(axes.flat, WORKLOADS):
        items = matrix[workload]["optimizers"]
        for index, (optimizer, _name, color, hatch) in enumerate(OPTIMIZERS):
            item = items[optimizer]
            plan_timeout = item["status"] == "plan_timeout"
            value = item["setup_sec"]
            ax.bar(
                index,
                value,
                width=0.78,
                color="#F3E5F5" if plan_timeout else color,
                edgecolor="#6A1B9A" if plan_timeout else "#333333",
                linewidth=0.8,
                hatch="xx" if plan_timeout else hatch,
                zorder=3,
            )
            if plan_timeout:
                ax.text(
                    index,
                    value / 2.4,
                    "TO",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#6A1B9A",
                    fontweight="bold",
                )
        ax.set_yscale("log", base=2)
        ax.set_ylim(2, 512)
        ax.set_xlim(-0.6, len(OPTIMIZERS) - 0.4)
        ax.set_xticks([])
        ax.set_title(f"{label} ({operator_count} ops)", loc="left", pad=2)
        _style(ax)
    for ax in axes.flat[len(WORKLOADS) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Planning/setup time (s)")
    fig.legend(
        handles=_legend_handles(overhead=True),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.suptitle("Planning and setup time (lower is better; log₂ scale)", y=0.955, fontsize=9)
    fig.subplots_adjust(
        top=0.88,
        bottom=0.055,
        left=0.065,
        right=0.995,
        hspace=0.38,
        wspace=0.28,
    )
    save(fig, output_dir, "optimizer_overhead")


def save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def export_matrix(matrix: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_dp_optimizer_data.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "workload",
        "operator_count",
        "samples",
        "optimizer",
        "status",
        "execution_sec",
        "setup_sec",
        "result_source",
        "setup_source",
    ]
    with (output_dir / "selected_dp_optimizer_data.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for workload, _label, operator_count in WORKLOADS:
            if workload not in matrix:
                continue
            workload_data = matrix[workload]
            for optimizer, _name, _color, _hatch in OPTIMIZERS:
                item = workload_data["optimizers"][optimizer]
                writer.writerow(
                    {
                        "workload": workload,
                        "operator_count": operator_count,
                        "samples": workload_data["samples"],
                        "optimizer": optimizer,
                        **{
                            key: item[key]
                            for key in fields
                            if key
                            not in {
                                "workload",
                                "operator_count",
                                "samples",
                                "optimizer",
                            }
                        },
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure()
    matrix = load_matrix(args.matrix_root)
    validate_selection(matrix)
    draw_execution(matrix, args.output_dir)
    draw_overhead(matrix, args.output_dir)
    export_matrix(matrix, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
