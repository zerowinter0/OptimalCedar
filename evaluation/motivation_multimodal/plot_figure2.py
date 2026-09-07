"""Render the data-driven multimodal running-example figure."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import yaml

from evaluation.motivation_multimodal.artifacts import atomic_write_json, sha256_file


TAG_LABELS = {
    "normalize": "N\nNormalize",
    "perplexity": "P\nPerplexity",
    "sharpness": "Q\nSharpness",
    "aesthetic": "A\nAesthetic",
    "clip": "C\nCLIP",
    "blip": "B\nBLIP",
}
BACKEND_COLORS = {
    "local": "#D9D9D9",
    "smp": "#56B4E9",
    "ray": "#E69F00",
    "cuda-ray": "#CC79A7",
}


@dataclass(frozen=True)
class RenderedFigure:
    pdf: Path
    png: Path
    manifest: Path


def _artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_figure2_data(formal_root: str | Path) -> dict[str, Any]:
    root = Path(formal_root)
    summary_path = root / "formal_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "success"
        or payload.get("formal_protocol") is not True
        or payload.get("records") != 3000
        or payload.get("equivalent") is not True
    ):
        raise ValueError("Figure 2 requires a valid 3,000-record formal artifact")
    for name in ("staged", "joint"):
        runs = payload.get("executions", {}).get(name, [])
        if len(runs) != 3:
            raise ValueError("Figure 2 requires three formal repetitions per optimizer")
        plan_meta = payload.get("plans", {}).get(name, {})
        plan_path = _artifact_path(root, plan_meta.get("path", ""))
        if not plan_path.is_file() or sha256_file(plan_path) != plan_meta.get("sha256"):
            raise ValueError(f"Figure 2 {name} plan is missing or has a stale hash")
    cedar_costs = payload.get("cedar_costs", {})
    if not all(
        name in cedar_costs and float(cedar_costs[name]) > 0
        for name in ("staged", "joint")
    ):
        raise ValueError("Figure 2 requires Cedar costs for both plans")
    payload["_summary_path"] = summary_path
    return payload


def _topological_order(graph: dict[str, Any]) -> list[int]:
    outgoing = {
        int(node): {int(value) for value in str(children).split(",") if value.strip()}
        for node, children in graph.items()
    }
    incoming = {node: 0 for node in outgoing}
    for children in outgoing.values():
        for child in children:
            incoming[child] = incoming.get(child, 0) + 1
    ready = sorted(node for node, count in incoming.items() if count == 0)
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(outgoing.get(node, ())):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(outgoing):
        raise ValueError("Physical plan is not acyclic")
    return order


def _backend(desc: dict[str, Any]) -> str:
    variant = desc["variant"]
    if variant in {"RAY", "TF_RAY"}:
        return "cuda-ray" if desc.get("execution_resource") == "cuda" else "ray"
    if variant == "SMP":
        return "smp"
    return "local"


def _stages(plan_path: Path, operator_ids: dict[str, int]) -> list[dict[str, Any]]:
    physical = yaml.safe_load(plan_path.read_text(encoding="utf-8"))["physical_plan"]
    pipes = {int(key): value for key, value in physical["pipes"].items()}
    id_to_tag = {int(value): key for key, value in operator_ids.items()}
    stages = []
    for pipe_id in _topological_order(physical["graph"]):
        desc = pipes[pipe_id]
        members = [int(value) for value in desc.get("fused_pipes", [pipe_id])]
        tags = [id_to_tag[value] for value in members if value in id_to_tag]
        if not tags:
            continue
        context = desc.get("variant_ctx", {}) or {}
        width = context.get("n_actors", context.get("n_procs", 1))
        stages.append(
            {
                "tags": tags,
                "backend": _backend(desc),
                "width": int(width),
            }
        )
    if set(tag for stage in stages for tag in stage["tags"]) != set(TAG_LABELS):
        raise ValueError("Physical plan does not cover all six running-example operators")
    return stages


def _operator_panel(axis: plt.Axes) -> None:
    axis.set_xlim(-1.0, 5.6)
    axis.set_ylim(-0.9, 1.35)
    axis.axis("off")
    positions = {
        "normalize": (0, 0.75),
        "perplexity": (1.25, 0.75),
        "sharpness": (0, -0.25),
        "aesthetic": (1.25, -0.25),
        "clip": (3.15, 0.25),
        "blip": (4.75, 0.25),
    }
    dependencies = (
        ("normalize", "perplexity"),
        ("sharpness", "aesthetic"),
        ("perplexity", "clip"),
        ("aesthetic", "clip"),
        ("clip", "blip"),
    )
    for source, target in dependencies:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        axis.add_patch(
            FancyArrowPatch(
                (x1 + 0.42, y1),
                (x2 - 0.42, y2),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color="#555555",
            )
        )
    modality = {
        "normalize": "#F0E442",
        "perplexity": "#F0E442",
        "sharpness": "#009E73",
        "aesthetic": "#009E73",
        "clip": "#0072B2",
        "blip": "#0072B2",
    }
    for tag, (x, y) in positions.items():
        axis.add_patch(
            FancyBboxPatch(
                (x - 0.42, y - 0.28),
                0.84,
                0.56,
                boxstyle="round,pad=0.04",
                facecolor=modality[tag],
                edgecolor="#333333",
                linewidth=0.8,
            )
        )
        axis.text(x, y, TAG_LABELS[tag], ha="center", va="center", fontsize=7.5)
    axis.text(-0.55, 1.23, "(a) Logical pipeline", fontsize=9, fontweight="bold")
    axis.text(-0.78, 0.75, "Text", rotation=90, va="center", fontsize=7, color="#666666")
    axis.text(-0.78, -0.25, "Image", rotation=90, va="center", fontsize=7, color="#666666")


def _plan_row(axis: plt.Axes, y: float, name: str, stages: list[dict[str, Any]]) -> None:
    axis.text(-0.04, y, name, transform=axis.transAxes, ha="right", va="center", fontsize=8)
    left = 0.02
    usable = 0.94
    gap = 0.025
    width = (usable - gap * (len(stages) - 1)) / len(stages)
    for index, stage in enumerate(stages):
        x = left + index * (width + gap)
        axis.add_patch(
            FancyBboxPatch(
                (x, y - 0.14),
                width,
                0.28,
                transform=axis.transAxes,
                boxstyle="round,pad=0.012",
                facecolor=BACKEND_COLORS[stage["backend"]],
                edgecolor="#333333",
                linewidth=0.8,
            )
        )
        labels = "+".join(TAG_LABELS[tag].split("\n")[0] for tag in stage["tags"])
        axis.text(
            x + width / 2,
            y + 0.018,
            labels,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        axis.text(
            x + width / 2,
            y - 0.075,
            f"{stage['backend']} ×{stage['width']}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=5.8,
        )
        if index + 1 < len(stages):
            axis.annotate(
                "",
                xy=(x + width + gap * 0.88, y),
                xytext=(x + width + gap * 0.12, y),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555555"},
            )


def render_figure2(
    formal_root: str | Path,
    output_root: str | Path,
) -> RenderedFigure:
    root = Path(formal_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    data = load_figure2_data(root)
    plan_paths = {
        name: _artifact_path(root, data["plans"][name]["path"])
        for name in ("staged", "joint")
    }
    stage_rows = {
        name: _stages(plan_paths[name], data["plans"][name]["operator_ids"])
        for name in ("staged", "joint")
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(13.2, 3.15))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.5, 1.55, 1.05))
    logical_axis = figure.add_subplot(grid[0, 0])
    _operator_panel(logical_axis)

    plan_axis = figure.add_subplot(grid[0, 1])
    plan_axis.axis("off")
    plan_axis.text(0, 0.97, "(b) Generated physical plans", transform=plan_axis.transAxes, fontsize=9, fontweight="bold")
    _plan_row(plan_axis, 0.66, "Staged", stage_rows["staged"])
    _plan_row(plan_axis, 0.30, "Joint", stage_rows["joint"])

    result_grid = grid[0, 2].subgridspec(1, 2, wspace=0.42)
    runtime_axis = figure.add_subplot(result_grid[0, 0])
    cost_axis = figure.add_subplot(result_grid[0, 1])
    names = ("staged", "joint")
    colors = ("#999999", "#0072B2")
    runtimes = [
        [float(run["seconds"]) for run in data["executions"][name]]
        for name in names
    ]
    medians = [statistics.median(values) for values in runtimes]
    errors = np.asarray(
        [
            [median - np.quantile(values, 0.25) for median, values in zip(medians, runtimes)],
            [np.quantile(values, 0.75) - median for median, values in zip(medians, runtimes)],
        ]
    )
    bars = runtime_axis.bar(names, medians, yerr=errors, color=colors, capsize=2, width=0.62)
    runtime_axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=2)
    runtime_axis.set_ylabel("Execution time (s)")
    runtime_axis.set_title("(c) Measured", fontsize=8, fontweight="bold")

    raw_costs = [float(data["cedar_costs"][name]) for name in names]
    normalized_costs = [value / raw_costs[0] for value in raw_costs]
    cost_bars = cost_axis.bar(names, normalized_costs, color=colors, width=0.62)
    cost_axis.bar_label(cost_bars, fmt="%.2f", fontsize=6, padding=2)
    cost_axis.set_ylabel("Normalized cost\n(staged = 1)")
    cost_axis.set_title("Cedar estimate", fontsize=8, fontweight="bold")

    for axis in (runtime_axis, cost_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=18, labelsize=6.5)
        axis.tick_params(axis="y", labelsize=6.5)

    figure.subplots_adjust(left=0.025, right=0.99, bottom=0.16, top=0.91, wspace=0.28)

    pdf = output / "pipeline_cooptimization_equivalent.pdf"
    png = output / "pipeline_cooptimization_equivalent.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(figure)
    inputs = [data["_summary_path"], *plan_paths.values()]
    manifest = output / "figure_manifest.json"
    atomic_write_json(
        manifest,
        {
            "figure": 2,
            "inputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in inputs
            ],
            "outputs": [
                {"path": str(pdf), "sha256": sha256_file(pdf)},
                {"path": str(png), "sha256": sha256_file(png)},
            ],
        },
    )
    return RenderedFigure(pdf, png, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    render_figure2(args.formal_root, args.output_root)


if __name__ == "__main__":
    main()
