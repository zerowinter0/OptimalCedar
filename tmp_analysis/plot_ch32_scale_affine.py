"""Sketch figures for the §3.2 scale/affine panel data.

Reads ``outputs/ch32_scale_affine_20260927/{figure_data.json,fits.csv}`` and
writes quick-look PNGs (no paper layout) so the delivery can be checked
without re-running anything.

Usage (inside the container):
  python -m tmp_analysis.plot_ch32_scale_affine [--out <dir>]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/workspace/OptimalCedar")
RESULT = ROOT / "outputs/ch32_scale_affine_20260927"


def load():
    figure = json.loads((RESULT / "figure_data.json").read_text())
    with (RESULT / "fits.csv").open() as handle:
        fits = list(csv.DictReader(handle))
    return figure, fits


def cell_map(figure):
    return {cell["cell_id"]: cell for cell in figure["cells"]}


def scatter_by_class(ax, cells, x_key, y_key="mean_ms", label_suffix=""):
    grouped = defaultdict(list)
    for cell in cells:
        if cell["role"] == "diagnostic":
            continue
        grouped[cell["representation_class"]].append(cell)
    for klass, items in sorted(grouped.items()):
        items.sort(key=lambda item: item[x_key])
        xs = [item[x_key] for item in items]
        ys = [item[y_key] for item in items]
        errs = [item["block_std_ms"] for item in items]
        marker = "o" if klass.startswith("float") else "s"
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            marker=marker,
            linestyle="none",
            capsize=2,
            label=f"{klass}{label_suffix}",
        )
        for item in items:
            if item["role"] == "validation":
                ax.plot(item[x_key], item[y_key], marker=marker, mfc="none",
                        mec="black", markersize=9, linestyle="none")


def panel_blur(figure, fits, out):
    cells = [cell for cell in figure["cells"] if cell["operator"] == "blur"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    scatter_by_class(axes[0], cells, "serialized_bytes")
    axes[0].set_xlabel("serialized bytes")
    axes[0].set_ylabel("operator self time (ms/call)")
    axes[0].set_title("Blur: bytes")
    scatter_by_class(axes[1], cells, "elements")
    axes[1].set_xlabel("elements")
    axes[1].set_title("Blur: elements")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "panel_blur_bytes_vs_elements.png", dpi=140)
    plt.close(fig)


def panel_elements(figure, fits, operator, out, title=None):
    cells = [cell for cell in figure["cells"] if cell["operator"] == operator]
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_by_class(ax, cells, "elements")
    grouped = defaultdict(list)
    for cell in cells:
        if cell["role"] == "diagnostic":
            grouped[cell["pair_id"]].append(cell)
    for pair_id, items in grouped.items():
        if not pair_id:
            continue
        for item in items:
            ax.annotate(
                pair_id,
                (item["elements"], item["mean_ms"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    ax.set_xlabel("elements")
    ax.set_ylabel("operator self time (ms/call)")
    ax.set_title(title or f"{operator}: elements by representation class")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"panel_{operator}_elements.png", dpi=140)
    plt.close(fig)


def panel_crop(figure, fits, out):
    cells = [cell for cell in figure["cells"] if cell["operator"] == "crop"]
    by_id = {cell["cell_id"]: cell for cell in cells}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, group in zip(axes, ("1ch", "3ch")):
        relevant = [
            cell
            for cell in cells
            if cell["representation_class"].endswith(group)
        ]
        scatter_by_class(ax, relevant, "elements")
        # Per-class affine fits (M5) drawn over each class's own range.
        for klass in sorted({cell["representation_class"] for cell in relevant}):
            row = next(
                (
                    fit
                    for fit in fits
                    if fit["operator"] == "crop"
                    and fit["group"] == group
                    and fit["model"] == "M5_repr_element_affine"
                    and fit["class_scope"] == klass
                ),
                None,
            )
            if row is None:
                continue
            coefficients = json.loads(row["coefficients"])
            xs = sorted(
                cell["elements"]
                for cell in relevant
                if cell["representation_class"] == klass
            )
            ys = [coefficients["k"] * x + coefficients["b"] for x in xs]
            ax.plot(xs, ys, "-", linewidth=1.5, label=f"M5 affine {klass}")
        # Anchored byte-proportional prediction for the group (a single point
        # anchored on the group's real-class reference cell).
        anchor_row = next(
            (
                fit
                for fit in fits
                if fit["operator"] == "crop"
                and fit["group"] == group
                and fit["model"] == "M1_byte_prop_anchored"
            ),
            None,
        )
        anchor_cell = by_id.get(anchor_row["anchor_cell"]) if anchor_row else None
        if anchor_cell:
            xs = sorted(cell["elements"] for cell in relevant)
            ys = [
                anchor_cell["mean_ms"] * x / anchor_cell["elements"] for x in xs
            ]
            ax.plot(xs, ys, "--", linewidth=1.5, label="M1 byte-proportional")
        ax.set_xlabel("elements")
        ax.set_ylabel("operator self time (ms/call)")
        ax.set_title(f"Crop: {group}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "panel_crop.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULT / "figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure, fits = load()
    panel_blur(figure, fits, out)
    panel_elements(figure, fits, "jitter", out)
    panel_elements(figure, fits, "grayscale", out)
    panel_elements(figure, fits, "flip", out)
    panel_crop(figure, fits, out)
    print(f"wrote sketches to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
