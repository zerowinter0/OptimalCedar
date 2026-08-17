#!/usr/bin/env python3
"""Analyze controlled Cedar profile duration/repeat experiments."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import yaml


def coefficient_of_variation(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0 if all(value == 0 for value in values) else None
    return statistics.stdev(values) / abs(mean)


def percentile(values: Iterable[float], quantile: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    index = round((len(values) - 1) * quantile)
    return values[min(len(values) - 1, max(0, index))]


def common_mapping_cvs(
    profiles: list[dict[str, Any]], section: str, field: str
) -> list[float]:
    mappings = [profile[section][field] for profile in profiles]
    common = set.intersection(*(set(mapping) for mapping in mappings))
    result = []
    for key in common:
        cv = coefficient_of_variation(float(mapping[key]) for mapping in mappings)
        if cv is not None:
            result.append(cv)
    return result


def backend_rows(profile: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    rows: dict[tuple[str, int], dict[str, float]] = {}
    for variant, entries in profile.get("offloads", {}).items():
        for raw_pipe_id, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            row = {"throughput": float(entry["throughput"])}
            timing = entry.get("backend_compute")
            if isinstance(timing, dict):
                mean = float(timing.get("mean_ms_per_sample", math.nan))
                stderr = float(timing.get("stderr_ms_per_sample", math.nan))
                row["backend_mean"] = mean
                row["backend_rse"] = (
                    stderr / abs(mean)
                    if math.isfinite(mean) and mean != 0 and math.isfinite(stderr)
                    else math.nan
                )
                row["backend_count"] = float(timing.get("count", 0))
            rows[(variant, int(raw_pipe_id))] = row
    return rows


def summarize_cvs(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "maximum": max(values) if values else None,
    }


def summarize_group(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    throughputs = [float(profile["baseline"]["throughput"]) for profile in profiles]
    baseline = {
        "throughput_values": throughputs,
        "throughput_cv": coefficient_of_variation(throughputs),
    }
    for field in ("latencies", "wall_latencies", "input_sizes", "output_sizes"):
        baseline[f"{field}_cv"] = summarize_cvs(
            common_mapping_cvs(profiles, "baseline", field)
        )

    all_rows = [backend_rows(profile) for profile in profiles]
    common = set.intersection(*(set(rows) for rows in all_rows))
    throughput_cvs = []
    backend_mean_cvs = []
    for key in common:
        throughput_cv = coefficient_of_variation(rows[key]["throughput"] for rows in all_rows)
        if throughput_cv is not None:
            throughput_cvs.append(throughput_cv)
        means = [rows[key].get("backend_mean", math.nan) for rows in all_rows]
        if all(math.isfinite(value) for value in means):
            backend_cv = coefficient_of_variation(means)
            if backend_cv is not None:
                backend_mean_cvs.append(backend_cv)

    rse_values = [
        row["backend_rse"]
        for rows in all_rows
        for row in rows.values()
        if math.isfinite(row.get("backend_rse", math.nan))
    ]
    rse_le_10_fraction = (
        sum(value <= 0.10 for value in rse_values) / len(rse_values)
        if rse_values
        else None
    )
    backend = {
        "throughput_cv": summarize_cvs(throughput_cvs),
        "mean_compute_cost_cv": summarize_cvs(backend_mean_cvs),
        "within_run_rse": summarize_cvs(rse_values),
        "rse_le_10_percent_fraction": rse_le_10_fraction,
    }

    acceptance = {
        "baseline_throughput_cv_le_5_percent": (
            baseline["throughput_cv"] is not None
            and baseline["throughput_cv"] <= 0.05
        ),
        "backend_cost_median_cv_le_10_percent": (
            backend["mean_compute_cost_cv"]["median"] is not None
            and backend["mean_compute_cost_cv"]["median"] <= 0.10
        ),
        "backend_cost_p90_cv_le_20_percent": (
            backend["mean_compute_cost_cv"]["p90"] is not None
            and backend["mean_compute_cost_cv"]["p90"] <= 0.20
        ),
        "at_least_90_percent_backend_rse_le_10_percent": (
            rse_le_10_fraction is not None and rse_le_10_fraction >= 0.90
        ),
    }
    acceptance["sufficient"] = all(acceptance.values())
    return {"repeats": len(profiles), "baseline": baseline, "backend": backend, "acceptance": acceptance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(args.study_root.glob("duration_*/repeat_*/profiles/*.yaml")):
        duration = path.parents[2].name.removeprefix("duration_")
        with path.open(encoding="utf-8") as stream:
            profile = yaml.safe_load(stream)
        grouped.setdefault(path.stem, {}).setdefault(duration, []).append(profile)

    analysis = {
        "criteria": {
            "independent_repeats": 3,
            "baseline_throughput_cv_max": 0.05,
            "backend_cost_median_cv_max": 0.10,
            "backend_cost_p90_cv_max": 0.20,
            "backend_rse_max": 0.10,
            "backend_rse_required_fraction": 0.90,
        },
        "workloads": {
            workload: {
                duration: summarize_group(profiles)
                for duration, profiles in sorted(durations.items(), key=lambda item: float(item[0]))
            }
            for workload, durations in sorted(grouped.items())
        },
    }
    json_output = args.json_output or args.study_root / "analysis.json"
    markdown_output = args.markdown_output or args.study_root / "ANALYSIS.md"
    json_output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Profile reproducibility and duration study",
        "",
        "A duration is sufficient only when all four pre-registered acceptance criteria pass.",
        "",
        "| Workload | Seconds | Repeats | Baseline throughput CV | Backend cost CV (median / P90) | Backend RSE <=10% | Sufficient |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for workload, durations in analysis["workloads"].items():
        for duration, result in durations.items():
            baseline_cv = result["baseline"]["throughput_cv"]
            median_cv = result["backend"]["mean_compute_cost_cv"]["median"]
            p90_cv = result["backend"]["mean_compute_cost_cv"]["p90"]
            rse_fraction = result["backend"]["rse_le_10_percent_fraction"]
            fmt = lambda value: "n/a" if value is None else f"{100 * value:.1f}%"
            lines.append(
                f"| {workload} | {duration} | {result['repeats']} | {fmt(baseline_cv)} | "
                f"{fmt(median_cv)} / {fmt(p90_cv)} | {fmt(rse_fraction)} | "
                f"{'yes' if result['acceptance']['sufficient'] else 'no'} |"
            )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_output)
    print(markdown_output)


if __name__ == "__main__":
    main()
