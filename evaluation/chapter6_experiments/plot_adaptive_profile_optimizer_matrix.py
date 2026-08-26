#!/usr/bin/env python3
"""Plot formal baselines with the adaptive-profile PICO rerun substituted."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import plot_formal_seven_optimizer_matrix as canonical


BASE_WORKLOADS = [
    ("alpaca_cot", "Alpaca-CoT", 8),
    ("redpajama_code", "RP-Code", 17),
    ("pile_hackernews", "HN", 18),
    ("pile_pubmed_abstracts", "PubMed", 19),
    ("pile_uspto_backgrounds", "USPTO", 19),
    ("pile_europarl", "EuroParl", 19),
    ("stackexchange", "StackEx", 19),
]

VIDEO_WORKLOAD = [("general_video_refine", "GenerateVideo", 10)]

SUPPLEMENT_WORKLOADS = [
    ("commonvoice", "CommonVoice", 7),
    ("simclrv2_cache", "SimCLR [Cache]", 9),
]

WORKLOADS = sorted(
    BASE_WORKLOADS + VIDEO_WORKLOAD + SUPPLEMENT_WORKLOADS,
    key=lambda item: item[2],
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_subset(root: Path, workloads) -> dict[str, Any]:
    previous = canonical.WORKLOADS
    try:
        canonical.WORKLOADS = list(workloads)
        return canonical.load_matrix(root)
    finally:
        canonical.WORKLOADS = previous


def adaptive_dp_item(root: Path, workload: str, expected_samples: int) -> dict:
    workload_root = root / "formal_runs" / workload
    times = []
    observed_samples = []
    for repeat in (1, 2, 3):
        path = workload_root / "results" / f"round{repeat}__dp_optimizer.json"
        if not path.exists():
            continue
        result = read_json(path)
        times.extend(float(value) for value in result["epoch_run_times"])
        observed_samples.extend(
            int(value) for value in result["epoch_num_samples"]
        )

    setup_path = (
        workload_root / "warmup_results" / "plan_only__dp_optimizer.json"
    )
    setup_result = read_json(setup_path)
    setup_sec = float(setup_result["runs"][0]["setup_time_sec"])
    timeout_path = (
        workload_root / "results" / "round1__dp_optimizer.timeout.json"
    )

    if len(times) == 3:
        if len(set(observed_samples)) != 1:
            raise ValueError(
                f"Inconsistent adaptive samples for {workload}: "
                f"{observed_samples}"
            )
        if observed_samples[0] != expected_samples:
            raise ValueError(
                f"Adaptive run for {workload} used {observed_samples[0]} "
                f"samples; baseline matrix uses {expected_samples}"
            )
        status = "success"
    elif timeout_path.exists():
        status = "unified_task_timeout"
    else:
        raise ValueError(
            f"Adaptive PICO result for {workload} is neither complete nor "
            "a recorded timeout"
        )

    return {
        "label": "PICO",
        "status": status,
        "execution_times_sec": times,
        "mean_execution_sec": statistics.mean(times) if times else None,
        "sd_execution_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
        "repeats": len(times),
        "samples": expected_samples,
        "setup_sec": setup_sec,
        "source": str(workload_root),
        "protocol": "adaptive profile; W=8; unified one-hour first task",
    }


def load_matrix(args) -> dict[str, Any]:
    matrix = load_subset(args.baseline_root, BASE_WORKLOADS)
    matrix.update(load_subset(args.video_root, VIDEO_WORKLOAD))
    matrix.update(load_subset(args.supplement_root, SUPPLEMENT_WORKLOADS))
    for workload, _label, _count in WORKLOADS:
        expected_samples = int(matrix[workload]["metadata"]["samples"])
        matrix[workload]["optimizers"]["dp_optimizer"] = adaptive_dp_item(
            args.adaptive_root, workload, expected_samples
        )
    return matrix


def rename_outputs(output_dir: Path) -> None:
    mapping = {
        "formal_seven_optimizer_data": "adaptive_profile_optimizer_data",
    }
    for old, new in mapping.items():
        for suffix in ("pdf", "svg", "png", "json", "tsv"):
            source = output_dir / f"{old}.{suffix}"
            if source.exists():
                source.replace(output_dir / f"{new}.{suffix}")


def write_readme(args, output_dir: Path) -> None:
    text = f"""# Adaptive-profile optimizer comparison

This figure substitutes only the latest adaptive-profile PICO measurements;
all other optimizer cells remain immutable formal comparison data.

- Seven text/image baseline workloads: `{args.baseline_root}`
- GenerateVideo 5,000-sample baselines: `{args.video_root}`
- CommonVoice and SimCLR-cache baselines: `{args.supplement_root}`
- Adaptive-profile PICO replacement: `{args.adaptive_root}`

Every successful cell uses the same workload-specific sample count, W=8,
and three measured executions. StackExchange PICO is shown as a unified-task
timeout because planning plus the first execution exceeded one hour; no
execution time is imputed.

The PDF/PNG/SVG files are generated from the exported JSON/TSV data by
`evaluation/chapter6_experiments/plot_adaptive_profile_optimizer_matrix.py`.
The Draw.io file wraps the generated SVG for editable paper layout.
"""
    (output_dir / "adaptive_profile_optimizer_README.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    canonical.configure()
    matrix = load_matrix(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = canonical.WORKLOADS
    try:
        canonical.WORKLOADS = WORKLOADS
        canonical.draw_execution(
            matrix,
            args.output_dir,
            ncols=5,
            stem="adaptive_profile_optimizer_execution",
        )
        canonical.draw_overhead(
            matrix,
            args.output_dir,
            ncols=5,
            stem="adaptive_profile_optimizer_overhead",
        )
        canonical.export(matrix, args.output_dir)
    finally:
        canonical.WORKLOADS = previous
    rename_outputs(args.output_dir)
    write_readme(args, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
