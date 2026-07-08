#!/usr/bin/env python3
"""Summarize Chapter 6 experiment outputs into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.json"))


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def workload_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_plan_cost",
        "_runtime",
        "_plan_only",
        "_dp_ablation",
        "_cache_epoch",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    if "_n" in stem and stem.endswith("_plan_cost"):
        return stem.split("_n", maxsplit=1)[0]
    return stem


def experiment_from_path(path: Path) -> str:
    parts = path.parts
    for name in (
        "optimizer_overhead",
        "plan_quality",
        "runtime",
        "ablation",
        "cache_epoch",
        "profile_sensitivity",
    ):
        if name in parts:
            return name
    return "unknown"


def flatten_optimizer_comparison(path: Path, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    workload = workload_from_path(path)
    experiment = experiment_from_path(path)
    for item in data.get("runs", []):
        row = {
            "experiment": experiment,
            "workload": workload,
            "name": item.get("optimizer") or item.get("condition"),
            "setup_time_sec": item.get("setup_time_sec"),
            "plan_cost": item.get("plan_cost"),
            "perf_time_sec": item.get("perf_time_sec"),
            "num_samples": item.get("num_samples"),
            "throughput_samples_per_sec": item.get("throughput_samples_per_sec"),
            "time_per_sample_us": item.get("time_per_sample_us"),
            "workload_skipped": item.get("workload_skipped", False),
            "skip_reason": item.get("skip_reason", ""),
            "error": item.get("error", ""),
            "source_json": str(path),
        }
        rows.append(row)
    return rows


def collect_json_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in load_json_files(results_dir):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("runs"), list):
            rows.extend(flatten_optimizer_comparison(path, data))
    return rows


def collect_overhead_csv(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*_synthetic_reorder.csv")):
        optimizer = path.stem.replace("_synthetic_reorder", "")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for item in reader:
                row = dict(item)
                row["experiment"] = "optimizer_overhead_synthetic"
                row["optimizer"] = optimizer
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, precision: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    numeric = safe_float(value)
    if numeric is not None:
        return f"{numeric:.{precision}f}"
    return str(value)


def best_by_metric(
    rows: List[Dict[str, Any]],
    experiment: str,
    metric: str,
    higher_is_better: bool,
) -> List[Dict[str, Any]]:
    candidates = []
    for row in rows:
        if row.get("experiment") != experiment:
            continue
        value = safe_float(row.get(metric))
        if value is None:
            continue
        candidates.append((value, row))
    candidates.sort(key=lambda item: item[0], reverse=higher_is_better)
    return [row for _, row in candidates[:10]]


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_No rows found._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(
    path: Path,
    json_rows: List[Dict[str, Any]],
    overhead_rows: List[Dict[str, Any]],
) -> None:
    parts: List[str] = []
    parts.append("# Chapter 6 Experiment Summary\n")
    parts.append(
        "This file is generated from raw outputs under the selected result directory.\n"
    )

    parts.append("## Synthetic Optimizer Overhead\n")
    overhead_table = []
    for row in overhead_rows:
        overhead_table.append(
            [
                row.get("optimizer", "-"),
                row.get("num_independent_ops", "-"),
                fmt(row.get("candidate_orders"), precision=0),
                fmt(row.get("dp_states"), precision=0),
                fmt(row.get("mean_seconds"), precision=6),
                row.get("status", "-"),
            ]
        )
    parts.append(
        markdown_table(
            [
                "optimizer",
                "ops",
                "candidate orders",
                "DP states",
                "mean sec",
                "status",
            ],
            overhead_table,
        )
    )

    parts.append("## Plan Quality\n")
    plan_rows = [
        row
        for row in json_rows
        if row.get("experiment") in {"plan_quality", "profile_sensitivity"}
    ]
    plan_table = []
    for row in plan_rows:
        plan_table.append(
            [
                row.get("experiment", "-"),
                row.get("workload", "-"),
                row.get("name", "-"),
                fmt(row.get("setup_time_sec"), precision=4),
                fmt(row.get("plan_cost"), precision=4),
                row.get("skip_reason", "-") or "-",
            ]
        )
    parts.append(
        markdown_table(
            ["experiment", "workload", "optimizer", "setup sec", "plan cost", "note"],
            plan_table,
        )
    )

    parts.append("## Runtime\n")
    runtime_rows = [
        row
        for row in json_rows
        if row.get("experiment") in {"runtime", "ablation", "cache_epoch"}
    ]
    runtime_table = []
    for row in runtime_rows:
        runtime_table.append(
            [
                row.get("experiment", "-"),
                row.get("workload", "-"),
                row.get("name", "-"),
                fmt(row.get("throughput_samples_per_sec"), precision=3),
                fmt(row.get("perf_time_sec"), precision=4),
                fmt(row.get("plan_cost"), precision=4),
                row.get("error", "")[:60] if row.get("error") else "-",
            ]
        )
    parts.append(
        markdown_table(
            [
                "experiment",
                "workload",
                "name",
                "throughput",
                "perf sec",
                "plan cost",
                "error",
            ],
            runtime_table,
        )
    )

    parts.append("## Quick Best-Of Checks\n")
    best_runtime = best_by_metric(
        json_rows, "runtime", "throughput_samples_per_sec", True
    )
    best_plan = best_by_metric(json_rows, "plan_quality", "plan_cost", False)
    parts.append("### Highest Runtime Throughput\n")
    parts.append(
        markdown_table(
            ["workload", "optimizer", "throughput"],
            [
                [
                    row.get("workload", "-"),
                    row.get("name", "-"),
                    fmt(row.get("throughput_samples_per_sec"), precision=3),
                ]
                for row in best_runtime
            ],
        )
    )
    parts.append("### Lowest Modeled Cost\n")
    parts.append(
        markdown_table(
            ["workload", "optimizer", "plan cost"],
            [
                [
                    row.get("workload", "-"),
                    row.get("name", "-"),
                    fmt(row.get("plan_cost"), precision=4),
                ]
                for row in best_plan
            ],
        )
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_rows = collect_json_rows(results_dir)
    overhead_rows = collect_overhead_csv(results_dir)
    write_csv(output_dir / "optimizer_comparison_summary.csv", json_rows)
    write_csv(output_dir / "synthetic_overhead_summary.csv", overhead_rows)
    write_markdown(output_dir / "chapter6_summary.md", json_rows, overhead_rows)

    print(f"Wrote {output_dir / 'optimizer_comparison_summary.csv'}")
    print(f"Wrote {output_dir / 'synthetic_overhead_summary.csv'}")
    print(f"Wrote {output_dir / 'chapter6_summary.md'}")


if __name__ == "__main__":
    main()
