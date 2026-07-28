#!/usr/bin/env python3
"""Render the enlarged-data, reused-plan formal matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


OPTIMIZERS = (
    "optimizer",
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "pecan_optimizer",
)
SYSTEMS = ("pytorch", "tensorflow", "ray", "plumber", "fastflow")


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _status(root: Path, workload: str, entity: str, tag: str):
    path = root / "status" / workload / entity / f"{tag}.tsv"
    try:
        state, _, reason = path.read_text().rstrip("\n").partition("\t")
        return state, reason
    except OSError:
        return "not_run", "no status recorded"


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _prior_optimization_times(repo: Path):
    values = {}
    old = _json(
        repo
        / "evaluation/chapter6_experiments/formal_results/"
        "cross_system_w8_latest.json"
    ) or {}
    for workload, item in old.get("workloads", {}).items():
        for optimizer in OPTIMIZERS:
            record = item.get("entities", {}).get(f"cedar_{optimizer}", {})
            value = record.get("optimization_or_setup_time_sec")
            if _finite(value):
                values[(workload, optimizer)] = float(value)
    goal = _json(
        repo
        / "evaluation/chapter6_experiments/formal_results/"
        "dp_20pct_goal_latest.json"
    ) or {}
    for workload, item in goal.get("candidates", {}).items():
        for optimizer in OPTIMIZERS:
            value = (
                item.get("runs", {})
                .get(optimizer, {})
                .get("optimization_time_sec")
            )
            if _finite(value):
                values[(workload, optimizer)] = float(value)
    return values


def _mean_sd(values):
    if not values:
        return None, None
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else None
    )


def _fmt(mean, sd):
    if mean is None:
        return "N/A"
    if sd is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{sd:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    repo = args.repo_root.resolve()

    final_samples = {}
    for line in (root / "final_samples.tsv").read_text().splitlines():
        workload, samples, attempt = line.split("\t")
        final_samples[workload] = (int(samples), int(attempt))

    prior_opt = _prior_optimization_times(repo)
    report = {
        "schema_version": 1,
        "protocol": {
            "name": "enlarged_reused_plan_w8",
            "repeats": 3,
            "local_workers": 8,
            "cpu_budget": 64,
            "optimizer_timeout_sec": 3600,
            "execution_timeout_sec": 3600,
            "profile_reused": True,
            "plans_reused_except_missing_artifacts": True,
            "shrink_rule": "halve when more than two optimizer cells time out",
        },
        "workloads": {},
    }

    for workload, (samples, attempt) in final_samples.items():
        item = {"samples": samples, "attempt": attempt, "optimizers": {}, "systems": {}}
        for optimizer in OPTIMIZERS:
            times = []
            statuses = []
            for round_index in range(1, 4):
                tag = f"attempt{attempt}__round{round_index}"
                state, reason = _status(root, workload, optimizer, tag)
                statuses.append({"status": state, "reason": reason})
                payload = _json(
                    root
                    / "cedar"
                    / workload
                    / optimizer
                    / f"{tag}.json"
                )
                if state == "success" and payload:
                    run_times = payload.get("epoch_run_times", [])
                    counts = payload.get("epoch_num_samples", [])
                    if len(run_times) == 1 and counts == [samples]:
                        times.append(float(run_times[0]))
            mean, sd = _mean_sd(times)
            plan_wall = root / "plans" / workload / f"{optimizer}.wall_sec"
            try:
                optimization = float(plan_wall.read_text().strip())
                optimization_source = "materialized_missing_plan_in_this_run"
            except (OSError, ValueError):
                optimization = prior_opt.get((workload, optimizer))
                optimization_source = "prior_formal_plan_generation"
            item["optimizers"][optimizer] = {
                "execution_times_sec": times,
                "mean_execution_time_sec": mean,
                "stddev_execution_time_sec": sd,
                "optimization_time_sec": optimization,
                "optimization_time_source": optimization_source,
                "statuses": statuses,
                "valid": len(times) == 3,
            }

        raw_times = []
        raw_statuses = []
        for round_index in range(1, 4):
            tag = f"attempt{attempt}__round{round_index}"
            state, reason = _status(root, workload, "no_optimizer", tag)
            raw_statuses.append({"status": state, "reason": reason})
            payload = _json(
                root / "cedar" / workload / "no_optimizer" / f"{tag}.json"
            )
            if state == "success" and payload:
                times = payload.get("epoch_run_times", [])
                counts = payload.get("epoch_num_samples", [])
                if len(times) == 1 and counts == [samples]:
                    raw_times.append(float(times[0]))
        raw_mean, raw_sd = _mean_sd(raw_times)
        item["no_optimizer"] = {
            "execution_times_sec": raw_times,
            "mean_execution_time_sec": raw_mean,
            "stddev_execution_time_sec": raw_sd,
            "statuses": raw_statuses,
            "valid": len(raw_times) == 3,
        }

        for system in SYSTEMS:
            times = []
            statuses = []
            for round_index in range(1, 4):
                tag = f"attempt{attempt}__round{round_index}"
                state, reason = _status(root, workload, system, tag)
                statuses.append({"status": state, "reason": reason})
                payload = _json(
                    root / "systems" / workload / system / f"{tag}.json"
                )
                value = payload.get("measured_time_sec") if payload else None
                count = payload.get("num_samples") if payload else None
                if state == "success" and _finite(value) and count == samples:
                    times.append(float(value))
            mean, sd = _mean_sd(times)
            item["systems"][system] = {
                "execution_times_sec": times,
                "mean_execution_time_sec": mean,
                "stddev_execution_time_sec": sd,
                "statuses": statuses,
                "valid": len(times) == 3,
            }
        report["workloads"][workload] = item

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Enlarged-data reused-plan W=8 matrix",
        "",
        "Execution time excludes profile, plan optimization, cache warmup, and setup. Values are mean±sample SD over three round-robin repetitions.",
        "",
        "| workload | samples | Cedar | DJ | DP-Cedar | DP | Pecan | raw plan | PyTorch | TensorFlow | Ray | Plumber | FastFlow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workload, item in report["workloads"].items():
        cells = []
        for optimizer in OPTIMIZERS:
            record = item["optimizers"][optimizer]
            cells.append(_fmt(record["mean_execution_time_sec"], record["stddev_execution_time_sec"]))
        raw = item["no_optimizer"]
        cells.append(_fmt(raw["mean_execution_time_sec"], raw["stddev_execution_time_sec"]))
        for system in SYSTEMS:
            record = item["systems"][system]
            cells.append(_fmt(record["mean_execution_time_sec"], record["stddev_execution_time_sec"]))
        lines.append(f"| {workload} | {item['samples']} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Optimizer time",
        "",
        "Reused plans retain their prior formal optimization time; only previously missing plans are timed in this run.",
        "",
        "| workload | Cedar | DJ | DP-Cedar | DP | Pecan |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for workload, item in report["workloads"].items():
        values = []
        for optimizer in OPTIMIZERS:
            value = item["optimizers"][optimizer]["optimization_time_sec"]
            values.append("N/A" if value is None else f"{value:.3f}")
        lines.append(f"| {workload} | " + " | ".join(values) + " |")
    args.markdown_output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
