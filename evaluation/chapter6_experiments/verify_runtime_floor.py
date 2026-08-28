#!/usr/bin/env python3
"""Reject formal workload results that are too short for stable comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--required-workloads", required=True)
    parser.add_argument("--minimum-seconds", type=float, default=1800.0)
    parser.add_argument("--required-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_seconds <= 0 or args.required_rounds <= 0:
        parser.error("runtime and round requirements must be positive")

    selected = {item for item in args.workloads.split(",") if item}
    required = {item for item in args.required_workloads.split(",") if item}
    report = {}
    failures = []
    for workload in sorted(selected & required):
        result_root = args.matrix_root / workload / "results"
        optimizer_medians = {}
        for first_round in sorted(result_root.glob("round1__*.json")):
            if first_round.name.endswith(
                (".timeout.json", ".skipped.json")
            ):
                continue
            optimizer = first_round.name.removeprefix(
                "round1__"
            ).removesuffix(".json")
            times = []
            for path in sorted(result_root.glob(f"round*__{optimizer}.json")):
                payload = json.loads(path.read_text())
                epoch_times = payload.get("epoch_run_times", [])
                if len(epoch_times) != 1:
                    raise RuntimeError(
                        f"{path} must contain exactly one formal epoch"
                    )
                times.append(float(epoch_times[0]))
            if len(times) == args.required_rounds:
                optimizer_medians[optimizer] = {
                    "times_sec": times,
                    "median_sec": statistics.median(times),
                }

        slowest_optimizer = None
        slowest_median = None
        if optimizer_medians:
            slowest_optimizer, slowest_payload = max(
                optimizer_medians.items(),
                key=lambda item: item[1]["median_sec"],
            )
            slowest_median = slowest_payload["median_sec"]
        passed = slowest_median is not None and slowest_median >= args.minimum_seconds
        report[workload] = {
            "minimum_seconds": args.minimum_seconds,
            "required_successful_rounds": args.required_rounds,
            "optimizer_medians": optimizer_medians,
            "slowest_available_optimizer": slowest_optimizer,
            "slowest_median_sec": slowest_median,
            "passed": passed,
        }
        if not passed:
            failures.append(workload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(
            "Formal runtime floor not met for: " + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
