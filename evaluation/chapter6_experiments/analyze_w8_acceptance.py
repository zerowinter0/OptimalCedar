#!/usr/bin/env python3
"""Audit reduced single-run W=8 results against the acceptance gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


WORKLOAD_SAMPLES = {
    "coco": 5000,
    "commonvoice": 10000,
    "commonvoice_cache": 10000,
    "llava_pretrain": 5000,
    "redpajama_c4": 20000,
    "stackexchange": 10000,
    "simclrv2": 9469,
    "simclrv2_cache": 9469,
    "wikitext103": 100000,
    "wikitext103_cache": 100000,
}
COMPARATORS = ("dj_optimizer", "dp_cedar_optimizer")
DP_OPTIMIZER = "dp_optimizer"


def _load_runs(root: Path, workload: str, optimizer: str) -> Dict:
    files = sorted(
        (root / workload / "results").glob(f"round*__{optimizer}.json")
    )
    times: List[float] = []
    samples: List[int] = []
    declared: List[int] = []
    errors: List[str] = []
    for path in files:
        try:
            payload = json.loads(path.read_text())
            run_times = payload.get("epoch_run_times", [])
            run_samples = payload.get("epoch_num_samples", [])
            if len(run_times) != 1 or len(run_samples) != 1:
                errors.append(f"{path.name}: expected exactly one epoch")
                continue
            times.append(float(run_times[0]))
            samples.append(int(run_samples[0]))
            declared.append(int(payload["num_total_samples"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
    return {
        "files": [str(path) for path in files],
        "times": times,
        "samples": samples,
        "declared_samples": declared,
        "mean_sec": statistics.mean(times) if times else None,
        "stddev_sec": statistics.stdev(times) if len(times) >= 2 else None,
        "errors": errors,
    }


def audit(root: Path, expected_repeats: int = 1) -> Dict:
    workloads = {}
    within_five_percent = True
    faster_than_both = 0
    faster_than_either = 0
    valid_workloads = 0

    for workload, expected_samples in WORKLOAD_SAMPLES.items():
        runs = {
            optimizer: _load_runs(root, workload, optimizer)
            for optimizer in (*COMPARATORS, DP_OPTIMIZER)
        }
        reasons = []
        evidence_valid = True
        for optimizer, result in runs.items():
            if len(result["times"]) != expected_repeats:
                reasons.append(
                    f"{optimizer}: {len(result['times'])}/{expected_repeats} valid runs"
                )
                evidence_valid = False
            if result["errors"]:
                reasons.extend(f"{optimizer}: {item}" for item in result["errors"])
                evidence_valid = False
            if any(value != expected_samples for value in result["declared_samples"]):
                reasons.append(f"{optimizer}: declared sample count mismatch")
                evidence_valid = False
            if any(value != expected_samples for value in result["samples"]):
                observed = sorted(set(result["samples"]))
                reasons.append(
                    f"{optimizer}: processed {observed}, expected {expected_samples}"
                )
                evidence_valid = False

        observed_counts = {
            tuple(result["samples"]) for result in runs.values() if result["samples"]
        }
        if len(observed_counts) > 1:
            reasons.append("optimizers processed different sample counts")
            evidence_valid = False

        plan_path = root / workload / "plans" / f"{DP_OPTIMIZER}.yaml"
        # Formal reruns are deliberately workload-selective. A repository-wide
        # source mtime therefore cannot establish per-workload freshness: an
        # exact plan-generation-only change would incorrectly invalidate every
        # preserved execution. The runner deletes all artifacts for each
        # selected workload before regeneration, while this audit requires the
        # materialized DP plan and exact single-run result evidence.
        if not plan_path.exists():
            reasons.append("dp_optimizer plan is missing")
            evidence_valid = False

        ratios = {"dp_over_dj": None, "dp_over_dp_cedar": None}
        passes_limit = False
        beats_both = False
        beats_either = False
        means_available = all(runs[name]["mean_sec"] is not None for name in runs)
        if means_available:
            dp_mean = runs[DP_OPTIMIZER]["mean_sec"]
            dj_mean = runs["dj_optimizer"]["mean_sec"]
            cedar_mean = runs["dp_cedar_optimizer"]["mean_sec"]
            ratios = {
                "dp_over_dj": dp_mean / dj_mean,
                "dp_over_dp_cedar": dp_mean / cedar_mean,
            }
            passes_limit = all(value <= 1.05 for value in ratios.values())
            beats_both = dp_mean < dj_mean and dp_mean < cedar_mean
            beats_either = dp_mean < dj_mean or dp_mean < cedar_mean
            if ratios["dp_over_dj"] > 1.05:
                reasons.append(
                    f"dp/dj={ratios['dp_over_dj']:.4f} exceeds 1.05"
                )
            if ratios["dp_over_dp_cedar"] > 1.05:
                reasons.append(
                    "dp/dp_cedar="
                    f"{ratios['dp_over_dp_cedar']:.4f} exceeds 1.05"
                )
        else:
            reasons.append("runtime means unavailable")

        accepted = evidence_valid and passes_limit
        within_five_percent = within_five_percent and accepted
        if evidence_valid:
            valid_workloads += 1
            faster_than_both += int(beats_both)
            faster_than_either += int(beats_either)
        workloads[workload] = {
            "expected_samples": expected_samples,
            "evidence_valid": evidence_valid,
            "runs": runs,
            "ratios": ratios,
            "within_five_percent": accepted,
            "faster_than_both": evidence_valid and beats_both,
            "faster_than_either": evidence_valid and beats_either,
            "failure_reasons": reasons,
        }

    summary = {
        "valid_workloads": valid_workloads,
        "all_workloads_within_five_percent": within_five_percent,
        "faster_than_both_count": faster_than_both,
        "faster_than_either_count": faster_than_either,
    }
    summary["accepted"] = (
        valid_workloads == len(WORKLOAD_SAMPLES)
        and within_five_percent
        and faster_than_both >= 2
        and faster_than_either >= 5
    )
    return {"summary": summary, "workloads": workloads}


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def render_markdown(report: Dict) -> str:
    lines = [
        "# Reduced single-run W=8 dp_optimizer acceptance audit",
        "",
        "| workload | outputs | dp mean±sd (s) | dj mean±sd (s) | dp_cedar mean±sd (s) | dp/dj | dp/dp_cedar | valid | 5% gate | failure reason |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for workload, item in report["workloads"].items():
        def stat(name: str) -> str:
            result = item["runs"][name]
            if result["mean_sec"] is None:
                return "N/A"
            return f"{result['mean_sec']:.3f}±{_fmt(result['stddev_sec'])}"

        reason = "; ".join(item["failure_reasons"]) or "—"
        lines.append(
            f"| {workload} | {item['expected_samples']} | "
            f"{stat(DP_OPTIMIZER)} | {stat('dj_optimizer')} | "
            f"{stat('dp_cedar_optimizer')} | {_fmt(item['ratios']['dp_over_dj'])} | "
            f"{_fmt(item['ratios']['dp_over_dp_cedar'])} | "
            f"{'yes' if item['evidence_valid'] else 'no'} | "
            f"{'PASS' if item['within_five_percent'] else 'FAIL'} | {reason} |"
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            "## Raw epoch runtimes (seconds)",
            "",
            "| workload | dp_optimizer | dj_optimizer | dp_cedar_optimizer |",
            "|---|---|---|---|",
        ]
    )
    for workload, item in report["workloads"].items():
        def raw(name: str) -> str:
            return ", ".join(f"{value:.6f}" for value in item["runs"][name]["times"])

        lines.append(
            f"| {workload} | {raw(DP_OPTIMIZER) or 'N/A'} | "
            f"{raw('dj_optimizer') or 'N/A'} | "
            f"{raw('dp_cedar_optimizer') or 'N/A'} |"
        )
    lines.extend(
        [
            "",
            f"Valid workloads: {summary['valid_workloads']}/{len(WORKLOAD_SAMPLES)}.",
            f"Faster than both: {summary['faster_than_both_count']}/{len(WORKLOAD_SAMPLES)} (required ≥2).",
            f"Faster than either: {summary['faster_than_either_count']}/{len(WORKLOAD_SAMPLES)} (required ≥5).",
            f"Overall acceptance: {'PASS' if summary['accepted'] else 'FAIL'}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--expected-repeats", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.expected_repeats)
    markdown = render_markdown(report)
    print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)


if __name__ == "__main__":
    main()
