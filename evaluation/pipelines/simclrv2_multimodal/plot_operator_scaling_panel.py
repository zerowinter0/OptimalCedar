"""Render and embed the Figure 1 operator-scaling panel."""

from __future__ import annotations

import argparse
import base64
import csv
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.motivation_multimodal.artifacts import atomic_write_json, sha256_file
from evaluation.pipelines.simclrv2_multimodal.modality_scaling import (
    IMAGE_SWEEP_SIDES,
    TEXT_SWEEP_WORDS,
)
from evaluation.pipelines.simclrv2_multimodal.operator_scaling import OPERATOR_NAMES


DISPLAY = {
    "normalize": "L: Language",
    "clip": "P: Perplexity",
    "random_crop": "M: Motion",
    "random_flip": "A: Aesthetic",
    "color_jitter": "S: V--T Sim.",
    "gaussian_blur": "N: NSFW",
}
COLORS = {
    "normalize": "#4F8A3C",
    "clip": "#BF9000",
    "random_crop": "#C65911",
    "random_flip": "#7E57C2",
    "color_jitter": "#3973AC",
    "gaussian_blur": "#A61C00",
}
MARKERS = {
    "normalize": "s",
    "clip": "h",
    "random_crop": "*",
    "random_flip": "o",
    "color_jitter": "D",
    "gaussian_blur": "p",
}
PANEL_CELL_ID = "5T0ilBr9OxNLVZLFwcjT-13"
PLACEHOLDER_EDGE_IDS = (
    "5T0ilBr9OxNLVZLFwcjT-14",
    "5T0ilBr9OxNLVZLFwcjT-15",
    "5T0ilBr9OxNLVZLFwcjT-16",
    "5T0ilBr9OxNLVZLFwcjT-17",
)


def load_normalized_rates(
    summary_csv: str | Path,
    required_operators: Iterable[str] = OPERATOR_NAMES,
    baseline_scale: float = 1.0,
) -> dict[str, dict[float, tuple[float, float, float]]]:
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = tuple(required_operators)
    curves: dict[str, dict[float, tuple[float, float, float]]] = {}
    for operator in required:
        selected = [row for row in rows if row["operator"] == operator]
        baseline = [
            row
            for row in selected
            if float(row["work_scale"]) == baseline_scale
        ]
        if len(baseline) != 1:
            raise ValueError(f"Expected one baseline row for {operator}")
        denominator = float(baseline[0]["median_records_per_sec"])
        if denominator <= 0:
            raise ValueError(f"Invalid baseline rate for {operator}")
        curves[operator] = {
            float(row["work_scale"]): (
                float(row["median_records_per_sec"]) / denominator,
                float(row["q1_records_per_sec"]) / denominator,
                float(row["q3_records_per_sec"]) / denominator,
            )
            for row in selected
        }
        if set(curves[operator]) != {0.25, 1.0, 4.0, 16.0}:
            raise ValueError(f"Incomplete scale grid for {operator}")
    return curves


def load_modality_durations(
    summary_csv: str | Path,
    required_operators: Iterable[str] = OPERATOR_NAMES,
) -> dict[str, dict[str, dict[int, tuple[float, float, float]]]]:
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    curves: dict[str, dict[str, dict[int, tuple[float, float, float]]]] = {
        "text": {},
        "image": {},
    }
    expected_grids = {
        "text": set(TEXT_SWEEP_WORDS),
        "image": set(IMAGE_SWEEP_SIDES),
    }
    for dimension in ("text", "image"):
        for operator in tuple(required_operators):
            selected = [
                row
                for row in rows
                if row["operator"] == operator
                and row["varied_dimension"] == dimension
            ]
            curve = {
                int(row["input_value"]): (
                    float(row["median_ms_per_record"]),
                    float(row["q1_ms_per_record"]),
                    float(row["q3_ms_per_record"]),
                )
                for row in selected
            }
            if set(curve) != expected_grids[dimension]:
                raise ValueError(
                    f"Incomplete {dimension} grid for operator {operator}"
                )
            if any(value <= 0 for triple in curve.values() for value in triple):
                raise ValueError(f"Non-positive duration for operator {operator}")
            curves[dimension][operator] = curve
    return curves


def normalize_modality_rates(
    durations: dict[str, dict[str, dict[int, tuple[float, float, float]]]],
) -> dict[str, dict[str, dict[int, tuple[float, float, float]]]]:
    rates: dict[str, dict[str, dict[int, tuple[float, float, float]]]] = {}
    for dimension, operator_curves in durations.items():
        rates[dimension] = {}
        for operator, curve in operator_curves.items():
            if 128 not in curve:
                raise ValueError(
                    f"Missing 128-input baseline for {dimension}/{operator}"
                )
            baseline_median = curve[128][0]
            if baseline_median <= 0:
                raise ValueError(
                    f"Invalid 128-input baseline for {dimension}/{operator}"
                )
            rates[dimension][operator] = {
                input_value: (
                    baseline_median / median,
                    baseline_median / q3,
                    baseline_median / q1,
                )
                for input_value, (median, q1, q3) in curve.items()
            }
    return rates


def image_pixel_positions(image_sides: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(side) ** 2 for side in image_sides)


def render_modality_panel(
    curves: dict[str, dict[str, dict[int, tuple[float, float, float]]]],
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )
    figure, axes = plt.subplots(figsize=(5.0, 4.0), nrows=2, sharey=True)
    panel_specs = (
        (
            axes[0],
            "text",
            np.asarray(TEXT_SWEEP_WORDS),
            np.asarray(TEXT_SWEEP_WORDS),
            tuple(str(value) for value in TEXT_SWEEP_WORDS),
            {"normalize", "clip"},
            "(a) Vary caption length; image = 224×224",
            "Caption length (words)",
        ),
        (
            axes[1],
            "image",
            np.asarray(IMAGE_SWEEP_SIDES),
            np.asarray(image_pixel_positions(IMAGE_SWEEP_SIDES)),
            tuple(
                rf"${side}^2$" if side in {128, 384, 640, 896, 1024} else ""
                for side in IMAGE_SWEEP_SIDES
            ),
            {"clip", "random_crop", "random_flip", "color_jitter", "gaussian_blur"},
            "(b) Vary image size; caption = 128 words",
            "Image size (pixels)",
        ),
    )
    legend_handles = []
    for (
        axis,
        dimension,
        input_grid,
        x_grid,
        tick_labels,
        consumers,
        title,
        xlabel,
    ) in panel_specs:
        for operator in OPERATOR_NAMES:
            values = curves[dimension][operator]
            median = np.asarray([values[int(value)][0] for value in input_grid])
            low = np.asarray([values[int(value)][1] for value in input_grid])
            high = np.asarray([values[int(value)][2] for value in input_grid])
            consumes_dimension = operator in consumers
            (line,) = axis.plot(
                x_grid,
                median,
                color=COLORS[operator],
                marker=MARKERS[operator],
                markersize=3.6,
                linewidth=1.45 if consumes_dimension else 1.05,
                linestyle="-" if consumes_dimension else "--",
                alpha=1.0 if consumes_dimension else 0.65,
                label=DISPLAY[operator],
            )
            axis.fill_between(
                x_grid,
                low,
                high,
                color=COLORS[operator],
                alpha=0.10 if consumes_dimension else 0.05,
                linewidth=0,
            )
            if dimension == "text":
                legend_handles.append(line)
        x_span = float(x_grid[-1] - x_grid[0])
        axis.set_xlim(
            float(x_grid[0]) - 0.025 * x_span,
            float(x_grid[-1]) + 0.025 * x_span,
        )
        axis.set_ylim(0.0, 1.25)
        axis.set_xticks(x_grid, tick_labels)
        axis.set_yticks(
            (0.0, 0.25, 0.5, 0.75, 1.0, 1.25),
            ("0×", "0.25×", "0.5×", "0.75×", "1×", "1.25×"),
        )
        axis.tick_params(axis="x", labelsize=5.7, pad=1.5)
        axis.tick_params(axis="y", labelsize=6.2, pad=1.5)
        axis.set_xlabel(xlabel, fontsize=6.8, labelpad=1.5)
        axis.set_ylabel("Relative rate (128 = 1×)", fontsize=6.8, labelpad=2)
        axis.set_title(title, fontsize=7.6, fontweight="bold", pad=2.5)
        axis.grid(True, which="major", color="#DDDDDD", linewidth=0.5)
        axis.axhline(1.0, color="#888888", linestyle=":", linewidth=0.65, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        legend_handles,
        [DISPLAY[operator] for operator in OPERATOR_NAMES],
        loc="upper center",
        ncol=3,
        fontsize=6.2,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.65,
        bbox_to_anchor=(0.54, 0.995),
    )
    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.12, top=0.855, hspace=0.59)
    svg = output / "operator_modality_scaling.svg"
    pdf = output / "operator_modality_scaling.pdf"
    png = output / "operator_modality_scaling.png"
    figure.savefig(svg, bbox_inches="tight", transparent=True)
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(figure)
    return svg, pdf, png


def render_total_input_panel(
    curves: dict[str, dict[float, tuple[float, float, float]]],
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )
    figure, axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    scales = np.asarray((0.25, 1.0, 4.0, 16.0))
    for operator in OPERATOR_NAMES:
        values = curves[operator]
        median = np.asarray([values[scale][0] for scale in scales])
        axis.plot(
            scales,
            median,
            color=COLORS[operator],
            marker=MARKERS[operator],
            markersize=5,
            linewidth=1.8,
            label=DISPLAY[operator],
        )
    axis.set_xscale("log", base=4)
    axis.set_xlim(0.19, 21.0)
    axis.set_ylim(0.0, 1.2)
    axis.set_xticks(scales, ("0.25×", "1×", "4×", "16×"))
    axis.set_yticks(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        ("0×", "0.25×", "0.5×", "0.75×", "1×"),
    )
    axis.set_xlabel("Record size")
    axis.set_ylabel("Relative throughput")
    axis.set_title("Operator Scaling Behavior", fontsize=10, fontweight="bold")
    axis.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8, zorder=0)
    axis.grid(True, which="major", color="#DDDDDD", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper right",
        ncol=2,
        fontsize=7.1,
        frameon=True,
        framealpha=0.92,
        columnspacing=0.8,
        handlelength=1.8,
    )
    svg = output / "figure1_total_input_scaling.svg"
    pdf = output / "figure1_total_input_scaling.pdf"
    png = output / "figure1_total_input_scaling.png"
    figure.savefig(svg, bbox_inches="tight", transparent=True)
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(figure)
    return svg, pdf, png


def embed_panel(drawio: str | Path, image: str | Path) -> None:
    drawio_path = Path(drawio)
    image_path = Path(image)
    mime_types = {".png": "image/png", ".svg": "image/svg+xml"}
    try:
        mime_type = mime_types[image_path.suffix.lower()]
    except KeyError as exc:
        raise ValueError("Figure 1 panel must be PNG or SVG") from exc
    tree = ET.parse(drawio_path)
    cells = {cell.get("id"): cell for cell in tree.getroot().iter("mxCell")}
    if PANEL_CELL_ID not in cells:
        raise ValueError("Figure 1 panel cell is missing")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    panel = cells[PANEL_CELL_ID]
    panel.set("value", "")
    panel.set(
        "style",
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        f"image=data:{mime_type}%3Bbase64,{encoded};",
    )
    for cell_id in PLACEHOLDER_EDGE_IDS:
        if cell_id in cells:
            cells[cell_id].set("style", "visible=0;strokeColor=none;opacity=0;")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{drawio_path.name}.", dir=drawio_path.parent
    )
    os.close(fd)
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=False)
        os.replace(temporary, drawio_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--total-summary-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--drawio", type=Path, required=True)
    args = parser.parse_args()
    durations = load_modality_durations(args.summary_csv)
    curves = normalize_modality_rates(durations)
    modality_outputs = render_modality_panel(curves, args.output_root)
    total_curves = load_normalized_rates(
        args.total_summary_csv, baseline_scale=0.25
    )
    total_outputs = render_total_input_panel(total_curves, args.output_root)
    # diagrams.net's headless Electron renderer is not reliable for embedded
    # SVG data URIs. Keep the standalone vector artifacts and embed the
    # high-resolution PNG in the combined Draw.io canvas.
    embed_panel(args.drawio, total_outputs[2])
    atomic_write_json(
        args.output_root / "figure_manifest.json",
        {
            "modality_summary_csv": str(args.summary_csv),
            "modality_summary_sha256": sha256_file(args.summary_csv),
            "total_summary_csv": str(args.total_summary_csv),
            "total_summary_sha256": sha256_file(args.total_summary_csv),
            "drawio": str(args.drawio),
            "modality_outputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in modality_outputs
            ],
            "total_input_outputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in total_outputs
            ],
        },
    )


if __name__ == "__main__":
    main()
