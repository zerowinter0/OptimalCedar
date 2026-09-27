"""Sketch panels for the extended §3.2 data (byte-normalised axes).

x = native bytes / reference native bytes, y = mean time / reference mean time,
with the reference frozen from round 1.  Points are coloured by representation
class; solid lines are the full-range per-class affine fits and dashed lines
the pre-declared local x<=2 fits.

Usage (inside the container):
  python -m tmp_analysis.plot_ch32_scale_affine_ext [--out DIR]
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
import numpy as np  # noqa: E402

ROOT = Path("/workspace/OptimalCedar")
RESULT = ROOT / "outputs/ch32_scale_affine_ext_20260927"
LOCAL_MAX = 2.0


def load():
    figure = json.loads((RESULT / "figure_data.json").read_text())
    with (RESULT / "fits.csv").open() as handle:
        fits = list(csv.DictReader(handle))
    return figure, fits


def panel(figure, fits, operator, out, x_max=None):
    cells = [cell for cell in figure["cells"] if cell["operator"] == operator]
    reference = figure["panels"][operator]["reference"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    grouped = defaultdict(list)
    for cell in cells:
        grouped[cell["representation_class"]].append(cell)
    markers = {"float32": "o", "uint8": "s"}
    for klass, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["x_bytes_norm"])
        xs = [item["x_bytes_norm"] for item in items]
        ys = [item["y_time_norm"] for item in items]
        marker = markers[klass.split(":")[0]]
        ax.plot(xs, ys, marker=marker, linestyle="none", markersize=5,
                label=f"{klass} (measured)")
        for item in items:
            if item["role"] == "validation":
                ax.plot(item["x_bytes_norm"], item["y_time_norm"], marker=marker,
                        mfc="none", mec="black", markersize=9, linestyle="none")
            if item["origin"] == "extension" or "round2" in ",".join(item["batches"]):
                ax.plot(item["x_bytes_norm"], item["y_time_norm"], marker="x",
                        color="black", markersize=4, linestyle="none")
        fit = next(
            (
                row
                for row in fits
                if row["operator"] == operator
                and row["model"] == "M5_repr_element_affine"
                and row["class_scope"] == klass
                and row["fit_scope"] == "full"
            ),
            None,
        )
        if fit:
            coefficients = json.loads(fit["coefficients"])
            elements = np.array([item["elements"] for item in items])
            order = np.argsort(elements)
            predicted = (
                coefficients["k"] * elements + coefficients["b"]
            ) / reference["mean_ms"]
            x_curve = np.array(
                [
                    reference["native_bytes"]
                    * (item["native_bytes"] / reference["native_bytes"])
                    for item in items
                ]
            ) / reference["native_bytes"]
            ax.plot(
                x_curve[order],
                predicted[order],
                "-",
                linewidth=1.4,
                label=f"{klass} M5 full",
            )
        local = next(
            (
                row
                for row in fits
                if row["operator"] == operator
                and row["model"] == "M4_element_affine"
                and row["class_scope"] == klass
                and row["fit_scope"] == f"local_x_le_{LOCAL_MAX:g}"
            ),
            None,
        )
        if local:
            coefficients = json.loads(local["coefficients"])
            in_window = [item for item in items if item["x_bytes_norm"] <= LOCAL_MAX]
            if in_window:
                elements = np.array([item["elements"] for item in in_window])
                predicted = (
                    coefficients["k"] * elements + coefficients["b"]
                ) / reference["mean_ms"]
                xs = [item["x_bytes_norm"] for item in in_window]
                order = np.argsort(xs)
                ax.plot(
                    np.array(xs)[order],
                    predicted[order],
                    "--",
                    linewidth=1.4,
                    label=f"{klass} M4 local x<={LOCAL_MAX:g}",
                )
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    if x_max:
        ax.set_xlim(0, x_max)
    ax.set_xlabel(f"native bytes / reference native bytes "
                  f"({reference['cell_id']} = 1)")
    ax.set_ylabel("mean time / reference mean time")
    ax.set_title(f"{operator}: frozen-reference normalisation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"ext_{operator}_normalized.png", dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULT / "figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure, fits = load()
    for operator in ("jitter", "crop", "blur", "grayscale", "flip"):
        panel(figure, fits, operator, out)
    print(f"wrote sketches to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
