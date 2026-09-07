"""Render six operator input-sensitivity panels from archived measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.motivation_multimodal.artifacts import atomic_write_json, sha256_file
from evaluation.motivation_multimodal.plot_figure2 import RenderedFigure


DISPLAY = {
    "normalize": "N: Normalize",
    "perplexity": "P: Perplexity",
    "sharpness": "Q: Sharpness",
    "aesthetic": "A: Aesthetic",
    "clip": "C: CLIP similarity",
    "blip": "B: BLIP matching",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "operator",
        "text_tokens",
        "image_side",
        "trials",
        "median_ns",
        "q1_ns",
        "q3_ns",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Figure 5 summary CSV is empty or incomplete")
    parsed = []
    for row in rows:
        if int(row["trials"]) != 7:
            raise ValueError("Figure 5 requires seven trials at every point")
        parsed.append(
            {
                "operator": row["operator"],
                "text_tokens": int(row["text_tokens"]) if row["text_tokens"] else None,
                "image_side": int(row["image_side"]) if row["image_side"] else None,
                "median_ms": float(row["median_ns"]) / 1e6,
                "q1_ms": float(row["q1_ns"]) / 1e6,
                "q3_ms": float(row["q3_ns"]) / 1e6,
            }
        )
    if set(row["operator"] for row in parsed) != set(DISPLAY):
        raise ValueError("Figure 5 requires all six running-example operators")
    return parsed


def _line_panel(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    operator: str,
    x_key: str,
    xlabel: str,
) -> None:
    selected = sorted(
        (row for row in rows if row["operator"] == operator),
        key=lambda row: row[x_key],
    )
    x = np.asarray([row[x_key] for row in selected], dtype=float)
    if x_key == "image_side":
        x = x * x / 1e6
    median = np.asarray([row["median_ms"] for row in selected])
    q1 = np.asarray([row["q1_ms"] for row in selected])
    q3 = np.asarray([row["q3_ms"] for row in selected])
    axis.plot(x, median, color="#0072B2", marker="o", markersize=3, linewidth=1.4)
    axis.fill_between(x, q1, q3, color="#56B4E9", alpha=0.28, linewidth=0)
    axis.set_title(DISPLAY[operator], fontsize=8.5, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Time / record (ms)")
    axis.spines[["top", "right"]].set_visible(False)


def _heatmap(axis: plt.Axes, rows: list[dict[str, Any]], operator: str) -> None:
    selected = [row for row in rows if row["operator"] == operator]
    tokens = sorted({row["text_tokens"] for row in selected})
    sides = sorted({row["image_side"] for row in selected})
    lookup = {(row["text_tokens"], row["image_side"]): row["median_ms"] for row in selected}
    values = np.asarray([[lookup[(token, side)] for token in tokens] for side in sides])
    image = axis.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(range(len(tokens)), tokens)
    axis.set_yticks(range(len(sides)), [f"{side * side / 1e6:.2f}" for side in sides])
    axis.set_xlabel("Caption tokens")
    axis.set_ylabel("Decoded image (MP)")
    axis.set_title(DISPLAY[operator], fontsize=8.5, fontweight="bold")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("ms / record", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)


def render_figure5(
    summary_csv: str | Path,
    output_root: str | Path,
) -> RenderedFigure:
    source = Path(summary_csv)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(source)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.25), constrained_layout=True)
    _line_panel(axes[0, 0], rows, "normalize", "text_tokens", "Caption tokens")
    _line_panel(axes[0, 1], rows, "perplexity", "text_tokens", "Caption tokens")
    _line_panel(axes[0, 2], rows, "sharpness", "image_side", "Decoded image (MP)")
    _line_panel(axes[1, 0], rows, "aesthetic", "image_side", "Decoded image (MP)")
    _heatmap(axes[1, 1], rows, "clip")
    _heatmap(axes[1, 2], rows, "blip")
    for axis in axes.flat:
        axis.tick_params(labelsize=6.5)
        axis.grid(False)
    pdf = output / "operator_input_size_scaling.pdf"
    png = output / "operator_input_size_scaling.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(figure)
    manifest = output / "figure_manifest.json"
    atomic_write_json(
        manifest,
        {
            "figure": 5,
            "inputs": [{"path": str(source), "sha256": sha256_file(source)}],
            "outputs": [
                {"path": str(pdf), "sha256": sha256_file(pdf)},
                {"path": str(png), "sha256": sha256_file(png)},
            ],
        },
    )
    return RenderedFigure(pdf, png, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    render_figure5(args.summary_csv, args.output_root)


if __name__ == "__main__":
    main()
