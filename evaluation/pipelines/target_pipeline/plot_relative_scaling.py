"""Backend/modality-separated multi-operator relative compute-rate plots."""
import argparse
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def classify(op):
    if op["field"] == "caption":
        return "text"
    if op["workload"] in ("clip", "blip") and (op["batch"] or op["tag"] == "collect"):
        return "image_text"
    return "image"


def page_group(op):
    view = re.match(r"view(\d+)_", op["tag"])
    if view:
        return op["workload"] + " / view" + view.group(1)
    return op["workload"]


def render(source, output):
    operators = json.loads((source / "operators.json").read_text())
    raw = pd.read_csv(source / "raw.csv")
    keys = ["operator_index", "backend", "round", "point"]
    if raw.duplicated(keys).any():
        raise ValueError("Duplicate measurement rows")
    if len(raw) != len(operators) * 12 * 3 * 3:
        raise ValueError("Incomplete measurements")
    if not np.isfinite(raw["mib_per_sec"]).all() or not (raw["mib_per_sec"] > 0).all():
        raise ValueError("Invalid execution rates")
    # Pair numerator and denominator within each independent round.
    baseline = raw.loc[raw["point"] == 0,
                       ["operator_index", "backend", "round", "mib_per_sec"]].rename(
                           columns={"mib_per_sec": "baseline_mib_per_sec"})
    data = raw.merge(baseline, on=["operator_index", "backend", "round"],
                     how="left", validate="many_to_one")
    data["relative_rate"] = data["mib_per_sec"] / data["baseline_mib_per_sec"]
    if data["relative_rate"].isna().any():
        raise ValueError("Missing baselines")
    summary = data.groupby(["operator_index", "backend", "point"]).agg(
        input_bytes=("input_bytes", "mean"),
        relative_mean=("relative_rate", "mean"),
        relative_std=("relative_rate", "std"),
        rounds=("round", "count")).reset_index()
    if not (summary["rounds"] == 3).all():
        raise ValueError("Each size requires all three rounds")
    summary["modality"] = summary.operator_index.map(
        {i: classify(op) for i, op in enumerate(operators)})
    summary["workload"] = summary.operator_index.map(
        {i: op["workload"] for i, op in enumerate(operators)})
    summary["operator"] = summary.operator_index.map(
        {i: op["tag"] for i, op in enumerate(operators)})
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "relative_raw.csv", index=False)
    summary.to_csv(output / "relative_summary.csv", index=False)
    names = {"image": "Image", "text": "Text", "image_text": "Image + text"}
    catalog = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    for backend in ("local", "smp", "ray"):
        for modality in names:
            folder = output / f"{backend}_{modality}"
            folder.mkdir(exist_ok=True)
            groups = {}
            for index, op in enumerate(operators):
                if classify(op) == modality:
                    groups.setdefault(page_group(op), []).append(index)
            entries = []
            with PdfPages(folder / "relative_rate.pdf") as pdf:
                for group, indices in groups.items():
                    # At most nine curves; preserve individual view occurrences.
                    for start in range(0, len(indices), 9):
                        chunk = indices[start:start+9]
                        fig, ax = plt.subplots(figsize=(9.5, 5.5))
                        for color, index in zip(plt.get_cmap("tab10").colors, chunk):
                            values = summary[(summary.operator_index == index) &
                                             (summary.backend == backend)].sort_values("point")
                            x = values.input_bytes.to_numpy() / (1024 if modality == "text" else 1024**2)
                            y = values.relative_mean.to_numpy()
                            sd = values.relative_std.to_numpy()
                            tag = operators[index]["tag"]
                            ax.plot(x, y, marker="o", markersize=3, linewidth=1.5,
                                    label=tag, color=color)
                        ax.axhline(1, color="#777777", linestyle="--", linewidth=.8)
                        unit = "KiB" if modality == "text" else "MiB"
                        ax.set_xlabel(f"Actual active input size ({unit}; linear)")
                        ax.set_ylabel("Relative execution rate (MiB/s / baseline MiB/s)")
                        ax.set_ylim(bottom=0)
                        ax.set_title(f"{backend.upper()} | {names[modality]} | {group}")
                        ax.grid(alpha=.2)
                        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                                  fontsize=8, frameon=False)
                        fig.text(.1, .015,
                                 "Each operator: smallest input = 1; mean of 3 paired rounds.",
                                 fontsize=8, color="#555555")
                        fig.tight_layout(rect=(0, .045, 1, 1))
                        name = re.sub(r"[^a-zA-Z0-9]+", "_", group) + f"_{start//9+1}"
                        fig.savefig(folder / f"{name}.png", dpi=150)
                        pdf.savefig(fig)
                        plt.close(fig)
                        entries.append({"title": group, "image": name + ".png",
                                        "operator_indices": chunk})
            title = f"{backend.upper()} / {names[modality]}"
            body = "".join(
                f'<h2>{html.escape(e["title"])}</h2><img loading="lazy" src="{e["image"]}" style="max-width:100%">'
                for e in entries)
            (folder / "index.html").write_text(
                '<meta charset="utf-8"><title>' + title + '</title>'
                '<style>body{max-width:1200px;margin:30px auto;font-family:system-ui}</style>'
                '<p><a href="../index.html">All groups</a> · '
                '<a href="relative_rate.pdf">Download PDF</a></p><h1>' + title + '</h1>' + body)
            catalog.append({"backend": backend, "modality": modality,
                            "directory": folder.name, "plots": entries})
    rows = []
    for backend in ("local", "smp", "ray"):
        cells = "".join(
            f'<td><a href="{backend}_{mod}/index.html">{names[mod]} charts</a> · '
            f'<a href="{backend}_{mod}/relative_rate.pdf">PDF</a></td>' for mod in names)
        rows.append(f"<tr><th>{backend.upper()}</th>{cells}</tr>")
    (output / "index.html").write_text(
        '<meta charset="utf-8"><title>Relative operator execution rate</title>'
        '<style>body{max-width:1100px;margin:40px auto;font-family:system-ui;line-height:1.7}'
        'td,th{padding:16px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
        '<h1>算子相对执行速率</h1>'
        '<p>按后端、模态分组。每张图包含多个算子；同一视图的算子放在一起，避免混入其他视图。'
        '每个算子以自身最小输入点为1；相对速率=(MiB/s)/(最小输入点MiB/s)，'
        '每轮分别归一化后计算3轮均值，不显示波动范围。横轴为实际字节量，线性刻度。</p>'
        '<table><tr><th>后端</th><th>图片</th><th>文本</th><th>图文混合</th></tr>'
        + "".join(rows) + '</table>'
        '<p>图文混合组为CLIP/BLIP的收集与批处理算子；图片读取按文件字节量，'
        '其他图片算子按解码后的实际处理字段计量。曲线反映worker内部计算，'
        '不包含通信；MiB/s为逻辑输入处理速率，不是物理内存带宽。</p>')
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (output / "README.md").write_text(
        "# Relative execution rate plots\n\n"
        "Rate means logical active-input MiB/s, not records/s. Each round is normalized "
        "by the same operator/backend/round at point 0. The curve is the mean of three "
        "ratios; no uncertainty bands or error bars are plotted. X uses actual bytes, on a linear scale.\n\n"
        "All 161 operator occurrences are retained independently; separate views are "
        "not pooled. Image/text/mixed classification is in relative_summary.csv. "
        "No measurement was rerun or changed. See index.html for all groups.\n")
    count = sum(len(c["plots"]) for c in catalog)
    covered = [i for c in catalog if c["backend"] == "local"
               for e in c["plots"] for i in e["operator_indices"]]
    assert sorted(covered) == list(range(len(operators)))
    print(json.dumps({"plots": count, "pdfs": len(catalog),
                      "operators_per_backend": len(covered),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
