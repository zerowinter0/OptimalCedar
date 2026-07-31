#!/usr/bin/env python3
"""Aggregate and plot the formal three-repeat cross-system matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


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
    ("redpajama_code", "RP-Code"),
    ("pile_hackernews", "HN"),
    ("pile_pubmed_abstracts", "PubMed"),
    ("pile_uspto_backgrounds", "USPTO"),
    ("pile_europarl", "EuroParl"),
]

SERIES = [
    ("pico", "PICO", "#0072B2", "///"),
    ("pytorch", "PyTorch", "#D55E00", "\\\\\\"),
    ("tensorflow", "tf.data", "#CC79A7", "---"),
    ("ray", "Ray Data", "#009E73", "xxx"),
    ("datajuicer", "Data-Juicer", "#E69F00", "..."),
    ("plumber", "Plumber", "#56B4E9", "+++"),
    ("fastflow", "FastFlow", "#000000", "ooo"),
]

EXPECTED_REPEATS = 3


def positive(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def read_optimizer_rows(path: Path) -> dict[str, dict]:
    output = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["optimizer"] != "dp_optimizer":
                continue
            if row["status"] != "success":
                raise ValueError(f"PICO baseline is unavailable: {row}")
            output[row["workload"]] = {
                "status": "success",
                "mean_execution_sec": float(row["mean_execution_sec"]),
                "sd_execution_sec": float(row["sd_execution_sec"]),
                "repeats": int(row["repeats"]),
                "samples": int(row["samples"]),
                "source": str(path),
                "measurement_kind": "exact measured execution",
            }
    missing = [name for name, _ in WORKLOADS if name not in output]
    if missing:
        raise ValueError(f"Optimizer TSV is missing PICO rows: {missing}")
    return output


def read_status(path: Path) -> tuple[str, str]:
    try:
        state, _, reason = path.read_text().rstrip("\n").partition("\t")
        return state, reason
    except OSError:
        return "not_run", "status file missing"


def aggregate_system(
    root: Path, workload: str, system: str, expected_samples: int
) -> dict:
    values = []
    measurement_kinds = set()
    statuses = []
    reasons = []
    sources = []
    for round_number in range(1, EXPECTED_REPEATS + 1):
        status_file = (
            root
            / "status"
            / workload
            / f"round{round_number}__{system}.tsv"
        )
        state, reason = read_status(status_file)
        statuses.append(state)
        if reason:
            reasons.append(reason)
        if state != "success":
            continue
        result = (
            root
            / "results"
            / workload
            / f"round{round_number}__{system}.json"
        )
        try:
            payload = json.loads(result.read_text())
            if int(payload["num_samples"]) != expected_samples:
                raise ValueError(
                    f"expected {expected_samples} samples, got "
                    f"{payload.get('num_samples')}"
                )
            value = float(payload["measured_time_sec"])
            if not positive(value):
                raise ValueError("non-positive measured time")
            values.append(value)
            measurement_kinds.add(
                payload.get("measurement_kind", "exact measured execution")
            )
            sources.append(str(result))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            statuses[-1] = "invalid"
            reasons.append(f"round {round_number}: {exc}")

    if len(values) == EXPECTED_REPEATS and set(statuses) == {"success"}:
        status = "success"
    elif set(statuses) == {"unsupported"}:
        status = "unsupported"
    elif set(statuses) == {"environment_unavailable"}:
        status = "environment_unavailable"
    elif "infeasible_timeout" in statuses:
        status = "infeasible_timeout"
    elif "not_run" in statuses:
        status = "not_run"
    else:
        status = "invalid"
    return {
        "status": status,
        "mean_execution_sec": statistics.mean(values) if status == "success" else None,
        "sd_execution_sec": statistics.stdev(values) if status == "success" else None,
        "execution_times_sec": values,
        "repeats": len(values),
        "samples": expected_samples,
        "measurement_kind": "; ".join(sorted(measurement_kinds)) or None,
        "round_statuses": statuses,
        "reasons": sorted(set(reasons)),
        "sources": sources,
    }


def build_report(run_root: Path, optimizer_tsv: Path) -> dict:
    pico = read_optimizer_rows(optimizer_tsv)
    workloads = {}
    for workload, _ in WORKLOADS:
        entities = {"pico": pico[workload]}
        for system, *_ in SERIES[1:]:
            entities[system] = aggregate_system(
                run_root,
                workload,
                system,
                pico[workload]["samples"],
            )
        baseline = pico[workload]["mean_execution_sec"]
        for item in entities.values():
            value = item.get("mean_execution_sec")
            item["speedup_vs_pico"] = (
                baseline / value
                if item["status"] == "success" and positive(value)
                else None
            )
        workloads[workload] = {
            "samples": pico[workload]["samples"],
            "entities": entities,
        }
    return {
        "schema_version": 1,
        "protocol": {
            "workers": 8,
            "cpu_budget": 64,
            "repeats": EXPECTED_REPEATS,
            "cell_timeout_sec": 3600,
            "excluded_workload": "pile_freelaw",
            "speedup_definition": "PICO execution time / system execution time",
            "datajuicer_note": (
                "retained-output throughput normalized from end-to-end wall time"
            ),
        },
        "workloads": workloads,
    }


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 8,
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
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def render(report: dict, output_dir: Path) -> None:
    configure()
    fig, ax = plt.subplots(figsize=(9.2, 3.05))
    width = 0.86 / len(SERIES)
    positive_values = [
        item["speedup_vs_pico"]
        for workload in report["workloads"].values()
        for item in workload["entities"].values()
        if positive(item.get("speedup_vs_pico"))
    ]
    lower = max(2 ** math.floor(math.log2(min(positive_values))) / 1.25, 0.015)
    upper = 2 ** math.ceil(math.log2(max(positive_values))) * 1.25
    for series_index, (entity, _, color, hatch) in enumerate(SERIES):
        offset = (series_index - (len(SERIES) - 1) / 2) * width
        for workload_index, (workload, _) in enumerate(WORKLOADS):
            item = report["workloads"][workload]["entities"][entity]
            x = workload_index + offset
            value = item.get("speedup_vs_pico")
            if positive(value):
                ax.bar(
                    x,
                    value,
                    width=width * 0.92,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.42,
                    hatch=hatch,
                    zorder=3,
                )
            else:
                marker = "x" if item["status"] != "unsupported" else "|"
                ax.plot(
                    x,
                    lower * 1.08,
                    marker=marker,
                    color="#777777",
                    markersize=4,
                    markeredgewidth=0.9,
                    zorder=4,
                )
    ax.set_yscale("log", base=2)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.axhline(1, color="#333333", linewidth=0.8)
    ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.55, len(WORKLOADS) - 0.45)
    ax.set_xticks(
        range(len(WORKLOADS)),
        [label for _, label in WORKLOADS],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Execution speedup vs. PICO")
    handles = [
        Patch(facecolor=color, edgecolor="#333333", hatch=hatch, label=label)
        for _, label, color, hatch in SERIES
    ]
    ax.legend(
        handles=handles,
        ncol=len(SERIES),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        handlelength=1.7,
        columnspacing=0.9,
    )
    fig.text(
        0.995,
        0.005,
        "| unsupported; × timeout/unavailable. Data-Juicer uses normalized retained-output throughput.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"cross_system_execution_speedup_vs_pico.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def write_tsv(report: dict, path: Path) -> None:
    fields = [
        "workload",
        "entity",
        "status",
        "mean_execution_sec",
        "sd_execution_sec",
        "repeats",
        "samples",
        "speedup_vs_pico",
        "measurement_kind",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for workload, _ in WORKLOADS:
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
                        "speedup_vs_pico": item.get("speedup_vs_pico"),
                        "measurement_kind": item.get("measurement_kind"),
                        "reasons": "; ".join(item.get("reasons", [])),
                    }
                )


def write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.tsv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((digest, path.stat().st_size, path.relative_to(root)))
    with (root / "MANIFEST.tsv").open("w", encoding="utf-8") as stream:
        stream.write("sha256\tbytes\tpath\n")
        for digest, size, relative in rows:
            stream.write(f"{digest}\t{size}\t{relative}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--optimizer-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.run_root, args.optimizer_tsv)
    (args.output_dir / "cross_system_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_tsv(report, args.output_dir / "cross_system_data.tsv")
    render(report, args.output_dir)
    write_manifest(args.run_root)
    print(args.output_dir)


if __name__ == "__main__":
    main()
