#!/usr/bin/env python3
"""Verify that DP Pareto-dominates both policy two-stage plans."""

import argparse
import json
import re
from pathlib import Path


OPTIMIZERS = (
    "dp_optimizer",
    "pecan_two_stage_optimizer",
    "dj_two_stage_optimizer",
)
OBJECTIVE_RE = re.compile(
    r"DP objective: throughput_bottleneck "
    r"local_serial=([0-9.eE+-]+) parallel_bottleneck=([0-9.eE+-]+)"
)


def _read_cost(log_path: Path) -> dict[str, float]:
    matches = list(OBJECTIVE_RE.finditer(log_path.read_text(errors="replace")))
    if not matches:
        raise RuntimeError(f"No DP objective found in {log_path}")
    local_serial, parallel_bottleneck = map(float, matches[-1].groups())
    return {
        "local_serial": local_serial,
        "parallel_bottleneck": parallel_bottleneck,
        "score": max(local_serial, parallel_bottleneck),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {"criterion": "pareto_dominance", "workloads": {}}
    all_passed = True
    workload_dirs = sorted(
        path
        for path in args.matrix_root.iterdir()
        if path.is_dir() and (path / "logs").is_dir()
    )
    if not workload_dirs:
        raise RuntimeError(f"No workload plan logs found under {args.matrix_root}")

    for workload_dir in workload_dirs:
        costs = {
            optimizer: _read_cost(
                workload_dir / "logs" / f"plan__{optimizer}.log"
            )
            for optimizer in OPTIMIZERS
        }
        dp = costs["dp_optimizer"]
        comparisons = {}
        workload_passed = True
        for optimizer in OPTIMIZERS[1:]:
            other = costs[optimizer]
            tolerance = 1e-9
            no_worse = (
                dp["local_serial"] <= other["local_serial"] + tolerance
                and dp["parallel_bottleneck"]
                <= other["parallel_bottleneck"] + tolerance
            )
            strictly_better = (
                dp["local_serial"] < other["local_serial"] - tolerance
                or dp["parallel_bottleneck"]
                < other["parallel_bottleneck"] - tolerance
            )
            passed = no_worse and strictly_better
            comparisons[optimizer] = {
                "no_worse_in_both_coordinates": no_worse,
                "strictly_better_in_at_least_one_coordinate": strictly_better,
                "passed": passed,
            }
            workload_passed &= passed
        report["workloads"][workload_dir.name] = {
            "costs": costs,
            "comparisons": comparisons,
            "passed": workload_passed,
        }
        all_passed &= workload_passed

    report["passed"] = all_passed
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
