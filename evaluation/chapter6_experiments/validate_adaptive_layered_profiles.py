#!/usr/bin/env python3
"""Validate and summarize formal adaptive layered Cedar profiles."""

import argparse
import json
from pathlib import Path

import yaml


def validate(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    layered = profile.get("layered_profile")
    if not isinstance(layered, dict):
        raise RuntimeError(f"{path}: missing layered_profile")
    if layered.get("method") != "fixed_legal_input_adaptive_microbenchmark":
        raise RuntimeError(f"{path}: unexpected layered profile method")
    pool = layered.get("input_pool", {}).get("samples_per_pipe", {})
    if not pool or min(int(value) for value in pool.values()) < 1:
        raise RuntimeError(f"{path}: empty legal-input reservoir")

    timings = layered.get("isolated_operator_costs", {})
    if not timings:
        raise RuntimeError(f"{path}: no isolated operator timings")
    converged = 0
    max_duration = 0
    for key, timing in timings.items():
        adaptive = timing.get("adaptive_profile")
        if not isinstance(adaptive, dict):
            raise RuntimeError(f"{path}: {key} lacks adaptive metadata")
        if int(timing.get("count", 0)) < 1:
            raise RuntimeError(f"{path}: {key} has no observations")
        if adaptive.get("unique_input_records", 0) < 1:
            raise RuntimeError(f"{path}: {key} has no replay inputs")
        if adaptive.get("converged"):
            converged += 1
        else:
            max_duration += 1

    boundary = profile.get("physical_model", {}).get("boundary", {})
    scaling = profile.get("physical_model", {}).get("scaling", {})
    if not boundary:
        raise RuntimeError(f"{path}: missing boundary calibration")
    if not scaling:
        raise RuntimeError(f"{path}: missing scaling calibration")
    return {
        "profile": str(path),
        "operator_variants": len(timings),
        "converged": converged,
        "max_duration": max_duration,
        "convergence_rate": converged / len(timings),
        "captured_pipes": len(pool),
        "captured_records": sum(int(value) for value in pool.values()),
        "boundary_backends": sorted(boundary),
        "scaling_backends": sorted(scaling),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        default="alpaca_cot,stackexchange,general_video_refine",
    )
    args = parser.parse_args()
    rows = [
        validate(args.profile_root / f"{name}.yaml")
        for name in args.workloads.split(",")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"{Path(row['profile']).stem}: "
            f"{row['converged']}/{row['operator_variants']} converged, "
            f"{row['captured_records']} fixed inputs"
        )


if __name__ == "__main__":
    main()
