#!/usr/bin/env python3
"""Audit the single-run StackExchange 20% performance requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


OPTIMIZERS = ("dj_optimizer", "dp_cedar_optimizer", "dp_optimizer")
EXPECTED_SAMPLES = 10_000
MAX_RATIO = 0.8


def _load_run(root: Path, optimizer: str) -> Dict[str, Any]:
    files = sorted((root / "results").glob(f"round*__{optimizer}.json"))
    errors = []
    runtime = None
    samples = None
    declared = None
    if len(files) != 1:
        errors.append(f"expected one result, found {len(files)}")
    elif files:
        try:
            payload = json.loads(files[0].read_text())
            times = payload["epoch_run_times"]
            counts = payload["epoch_num_samples"]
            if len(times) != 1 or len(counts) != 1:
                errors.append("expected exactly one epoch")
            else:
                runtime = float(times[0])
                samples = int(counts[0])
                declared = int(payload["num_total_samples"])
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if samples != EXPECTED_SAMPLES or declared != EXPECTED_SAMPLES:
        errors.append(
            f"sample mismatch: processed={samples}, declared={declared}, "
            f"expected={EXPECTED_SAMPLES}"
        )
    return {
        "files": [str(path) for path in files],
        "runtime_sec": runtime,
        "processed_samples": samples,
        "declared_samples": declared,
        "errors": errors,
    }


def audit(root: Path) -> Dict[str, Any]:
    runs = {name: _load_run(root, name) for name in OPTIMIZERS}
    evidence_valid = all(not item["errors"] for item in runs.values())
    ratios = {"dp_over_dj": None, "dp_over_dp_cedar": None}
    if evidence_valid:
        dp = runs["dp_optimizer"]["runtime_sec"]
        ratios = {
            "dp_over_dj": dp / runs["dj_optimizer"]["runtime_sec"],
            "dp_over_dp_cedar": dp
            / runs["dp_cedar_optimizer"]["runtime_sec"],
        }
    accepted = evidence_valid and all(
        ratio <= MAX_RATIO for ratio in ratios.values()
    )
    reasons = [
        f"{name}: {error}"
        for name, item in runs.items()
        for error in item["errors"]
    ]
    if evidence_valid:
        for name, ratio in ratios.items():
            if ratio > MAX_RATIO:
                reasons.append(f"{name}={ratio:.4f} exceeds {MAX_RATIO:.1f}")
    return {
        "workload": "stackexchange",
        "protocol": {
            "local_workers": 8,
            "cpu_budget": 64,
            "repeats": 1,
            "expected_samples": EXPECTED_SAMPLES,
            "omitted_operator": "document_simhash_deduplicator",
            "max_ratio": MAX_RATIO,
        },
        "runs": runs,
        "ratios": ratios,
        "evidence_valid": evidence_valid,
        "accepted": accepted,
        "failure_reasons": reasons,
    }


def render(report: Dict[str, Any]) -> str:
    runs = report["runs"]
    ratios = report["ratios"]
    value = lambda name: (
        "N/A"
        if runs[name]["runtime_sec"] is None
        else f"{runs[name]['runtime_sec']:.6f}"
    )
    ratio = lambda name: (
        "N/A" if ratios[name] is None else f"{ratios[name]:.4f}"
    )
    reasons = "; ".join(report["failure_reasons"]) or "—"
    return "\n".join(
        [
            "# StackExchange reduced single-run W=8 20% acceptance audit",
            "",
            "| dp_optimizer (s) | dj_optimizer (s) | dp_cedar_optimizer (s) | dp/dj | dp/dp_cedar | valid | 20% gate |",
            "|---:|---:|---:|---:|---:|:---:|:---:|",
            f"| {value('dp_optimizer')} | {value('dj_optimizer')} | "
            f"{value('dp_cedar_optimizer')} | {ratio('dp_over_dj')} | "
            f"{ratio('dp_over_dp_cedar')} | "
            f"{'yes' if report['evidence_valid'] else 'no'} | "
            f"{'PASS' if report['accepted'] else 'FAIL'} |",
            "",
            f"Failure reason: {reasons}",
            "",
            "Protocol: 35,000 unique source documents, 10,000 measured outputs; W=8; CPU budget=64; one measured run; cache off.",
            "Omitted global operator: `document_simhash_deduplicator`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "stackexchange",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    markdown = render(report)
    print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)


if __name__ == "__main__":
    main()
