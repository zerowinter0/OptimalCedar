#!/usr/bin/env python3
"""Report whether DP beats the best competing Cedar optimizer by 20%."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


REGISTERED_CANDIDATE_WORKLOADS = (
    "pile_europarl",
    "redpajama_code",
    "pile_hackernews",
    "pile_pubmed_abstracts",
    "pile_freelaw",
    "pile_uspto_backgrounds",
)
OPTIMIZERS = (
    "optimizer",
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "pecan_optimizer",
)
DP = "dp_optimizer"
EXISTING_COMPETITOR_ENTITIES = (
    "cedar_optimizer",
    "cedar_dj_optimizer",
    "cedar_dp_cedar_optimizer",
    "cedar_pecan_optimizer",
)
THRESHOLD = 1.20
EXECUTION_TIMEOUT_SEC = 3600.0
FORMAL_UNAVAILABLE_STATUSES = {
    "optimizer_timeout",
    "infeasible_timeout",
    "skipped",
}


def _result_files(root: Path, workload: str, optimizer: str) -> list[Path]:
    output = []
    for path in (root / workload / "results").glob("*.json"):
        try:
            prefix, file_optimizer = path.stem.rsplit("__", 1)
        except ValueError:
            continue
        round_number = prefix.removeprefix("round")
        if (
            file_optimizer == optimizer
            and prefix.startswith("round")
            and round_number.isdigit()
        ):
            output.append(path)
    return sorted(output)


def _status_files(root: Path, workload: str, optimizer: str) -> list[Path]:
    output = []
    for path in (root / workload / "status").glob("*.json"):
        try:
            prefix, file_optimizer = path.stem.rsplit("__", 1)
        except ValueError:
            continue
        is_round = (
            prefix.startswith("round")
            and prefix.removeprefix("round").isdigit()
        )
        if file_optimizer == optimizer and (
            prefix == "plan" or is_round
        ):
            output.append(path)
    return sorted(output)


def _read_candidate(
    root: Path,
    workload: str,
    optimizer: str,
    expected_repeats: int = 3,
    expected_samples: int = 20_000,
    execution_timeout_sec: float = EXECUTION_TIMEOUT_SEC,
) -> Dict[str, Any]:
    files = _result_files(root, workload, optimizer)
    values = []
    samples = []
    errors = []
    statuses = []
    for path in files:
        try:
            payload = json.loads(path.read_text())
            epoch_times = payload["epoch_run_times"]
            epoch_samples = payload["epoch_num_samples"]
            if len(epoch_times) != 1 or len(epoch_samples) != 1:
                raise ValueError("expected exactly one measured epoch")
            values.append(float(epoch_times[0]))
            samples.append(int(epoch_samples[0]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
    status_files = _status_files(root, workload, optimizer)
    for path in status_files:
        try:
            payload = json.loads(path.read_text())
            status = payload.get("status", "failed")
            reason = payload.get("reason", "")
            statuses.append(
                {
                    "file": path.name,
                    "status": status,
                    "reason": reason,
                }
            )
            errors.append(
                f"{path.name}: {status}: {reason}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")

    setup = None
    plan_result = root / workload / "plan_results" / f"{optimizer}.json"
    if plan_result.exists():
        try:
            payload = json.loads(plan_result.read_text())
            run = next(
                item
                for item in payload["runs"]
                if item["optimizer"] == optimizer
            )
            setup = float(run["setup_time_sec"])
        except (OSError, KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{plan_result.name}: {exc}")
    else:
        errors.append("plan result missing")

    successful = (
        len(values) == expected_repeats
        and len(set(samples)) == 1
        and samples[0] == expected_samples
        and setup is not None
        and not errors
    )
    status_names = {item["status"] for item in statuses}
    formally_unavailable = (
        not successful
        and bool(status_names)
        and status_names <= FORMAL_UNAVAILABLE_STATUSES
        and (
            "optimizer_timeout" in status_names
            or "infeasible_timeout" in status_names
        )
    )
    outcome = (
        "success"
        if successful
        else "unavailable"
        if formally_unavailable
        else "invalid"
    )
    round_execution_timeouts = [
        item
        for item in statuses
        if item["file"].startswith("round")
        and item["status"] == "infeasible_timeout"
    ]
    execution_time_lower_bound_sec = (
        execution_timeout_sec
        if len(round_execution_timeouts) == expected_repeats
        and not values
        else None
    )
    source_infeasible = (
        len(values) == expected_repeats
        and len(set(samples)) == 1
        and 0 <= samples[0] < expected_samples
        and setup is not None
        and (
            not statuses
            or status_names == {"source_exhausted"}
        )
    )
    if source_infeasible:
        outcome = "source_infeasible"
    return {
        "execution_times_sec": values,
        "mean_execution_time_sec": statistics.mean(values) if values else None,
        "stddev_execution_time_sec": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ) if values else None,
        "processed_samples": samples,
        "optimization_time_sec": setup,
        "execution_time_lower_bound_sec": execution_time_lower_bound_sec,
        "outcome": outcome,
        "valid": successful,
        "formally_unavailable": formally_unavailable,
        "source_infeasible": source_infeasible,
        "statuses": statuses,
        "errors": errors,
    }


def _candidate_summary(
    root: Path,
    workload: str,
    expected_repeats: int = 3,
    expected_samples: int = 20_000,
    execution_timeout_sec: float = EXECUTION_TIMEOUT_SEC,
) -> Dict[str, Any]:
    runs = {
        optimizer: _read_candidate(
            root,
            workload,
            optimizer,
            expected_repeats,
            expected_samples,
            execution_timeout_sec,
        )
        for optimizer in OPTIMIZERS
    }
    workload_marker = (
        root / workload / "status" / "source_infeasible.json"
    )
    source_infeasible = workload_marker.exists() or any(
        run["source_infeasible"] for run in runs.values()
    )
    other_runtime_bounds = {}
    for name in OPTIMIZERS:
        if name == DP:
            continue
        run = runs[name]
        if run["outcome"] == "success":
            other_runtime_bounds[name] = run["mean_execution_time_sec"]
        elif run["execution_time_lower_bound_sec"] is not None:
            other_runtime_bounds[name] = run[
                "execution_time_lower_bound_sec"
            ]
    expected_outcomes = all(
        item["outcome"] in {"success", "unavailable"}
        for item in runs.values()
    )
    valid = (
        not source_infeasible
        and runs[DP]["outcome"] == "success"
        and bool(other_runtime_bounds)
        and expected_outcomes
    )
    best_other = None
    best_other_time = None
    speedup = None
    speedup_is_lower_bound = False
    best_other_is_lower_bound = False
    passes = False
    if valid:
        best_other = min(
            other_runtime_bounds,
            key=other_runtime_bounds.get,
        )
        best_other_time = other_runtime_bounds[best_other]
        best_other_is_lower_bound = (
            runs[best_other]["outcome"] != "success"
        )
        dp_time = runs[DP]["mean_execution_time_sec"]
        speedup = best_other_time / dp_time
        speedup_is_lower_bound = best_other_is_lower_bound
        passes = speedup >= THRESHOLD
    return {
        "runs": runs,
        "workload_outcome": (
            "source_infeasible" if source_infeasible else
            "success" if valid else "invalid"
        ),
        "valid": valid,
        "best_other_optimizer": best_other,
        "best_other_execution_time_sec": best_other_time,
        "best_other_is_lower_bound": best_other_is_lower_bound,
        "dp_speedup_over_best_other": speedup,
        "dp_speedup_is_lower_bound": speedup_is_lower_bound,
        "dp_at_least_20pct_faster": passes,
    }


def _existing_summaries(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    output = {}
    for workload, workload_item in report.get("workloads", {}).items():
        entities = workload_item.get("entities", {})
        dp = entities.get("cedar_dp_optimizer", {})
        dp_time = dp.get("execution_time_sec")
        competitors = {}
        for entity, item in entities.items():
            if entity not in EXISTING_COMPETITOR_ENTITIES:
                continue
            value = item.get("execution_time_sec")
            if item.get("status") == "success" and isinstance(
                value, (int, float)
            ) and math.isfinite(value):
                competitors[entity] = float(value)
        valid = (
            dp.get("status") == "success"
            and isinstance(dp_time, (int, float))
            and math.isfinite(dp_time)
            and bool(competitors)
        )
        best_name = min(competitors, key=competitors.get) if valid else None
        best_time = competitors.get(best_name) if best_name else None
        speedup = best_time / dp_time if valid else None
        output[workload] = {
            "valid": valid,
            "workload_outcome": "success" if valid else "invalid",
            "best_other_optimizer": best_name,
            "best_other_execution_time_sec": best_time,
            "best_other_is_lower_bound": False,
            "dp_execution_time_sec": dp_time if valid else None,
            "dp_speedup_over_best_other": speedup,
            "dp_speedup_is_lower_bound": False,
            "dp_at_least_20pct_faster": (
                speedup >= THRESHOLD if speedup is not None else False
            ),
            "source": str(path),
        }
    return output


def _discover_candidate_workloads(root: Path) -> list[str]:
    return [
        workload
        for workload in REGISTERED_CANDIDATE_WORKLOADS
        if (root / workload).is_dir()
    ]


def audit(
    candidate_roots: Union[Path, Iterable[Path]],
    existing_report: Path,
    require_all_registered: bool = False,
    expected_repeats: int = 3,
    expected_samples: int = 20_000,
    execution_timeout_sec: float = EXECUTION_TIMEOUT_SEC,
) -> Dict[str, Any]:
    if isinstance(candidate_roots, Path):
        roots = [candidate_roots]
    else:
        roots = list(candidate_roots)
    if not roots:
        raise ValueError("At least one candidate root is required")

    existing = _existing_summaries(existing_report)
    candidates = {}
    candidate_sources = {}
    for root in roots:
        workloads = _discover_candidate_workloads(root)
        if not workloads:
            raise ValueError(
                f"No registered candidate workload directories in {root}"
            )
        for workload in workloads:
            if workload in candidates:
                raise ValueError(
                    f"Duplicate candidate workload {workload} in "
                    f"{candidate_sources[workload]} and {root}"
                )
            candidates[workload] = _candidate_summary(
                root,
                workload,
                expected_repeats,
                expected_samples,
                execution_timeout_sec,
            )
            candidate_sources[workload] = root

    if require_all_registered:
        missing = sorted(
            set(REGISTERED_CANDIDATE_WORKLOADS) - set(candidates)
        )
        if missing:
            raise ValueError(
                "Missing pre-registered candidate workloads: "
                + ", ".join(missing)
            )

    combined = dict(existing)
    for workload, item in candidates.items():
        combined[workload] = {
            "valid": item["valid"],
            "best_other_optimizer": item["best_other_optimizer"],
            "best_other_execution_time_sec": item[
                "best_other_execution_time_sec"
            ],
            "best_other_is_lower_bound": item[
                "best_other_is_lower_bound"
            ],
            "dp_execution_time_sec": item["runs"][DP][
                "mean_execution_time_sec"
            ],
            "dp_speedup_over_best_other": item[
                "dp_speedup_over_best_other"
            ],
            "dp_speedup_is_lower_bound": item[
                "dp_speedup_is_lower_bound"
            ],
            "dp_at_least_20pct_faster": item[
                "dp_at_least_20pct_faster"
            ],
            "source": str(candidate_sources[workload]),
            "workload_outcome": item["workload_outcome"],
        }
    valid = [item for item in combined.values() if item["valid"]]
    wins = [
        item
        for item in combined.values()
        if item["dp_at_least_20pct_faster"]
    ]
    total = len(combined)
    fraction = len(wins) / total if total else 0.0
    minimum_wins_required = math.ceil(total * 3 / 10)
    additional_wins_needed = max(
        0,
        minimum_wins_required - len(wins),
    )
    target_met = total > 0 and additional_wins_needed == 0
    return {
        "protocol": {
            "threshold": THRESHOLD,
            "candidate_repeats": expected_repeats,
            "candidate_outputs": expected_samples,
            "execution_timeout_sec": execution_timeout_sec,
            "local_workers": 8,
            "cpu_budget": 64,
            "candidate_optimizer_set": list(OPTIMIZERS),
            "candidate_roots": [str(root) for root in roots],
        },
        "candidates": candidates,
        "combined": combined,
        "summary": {
            "total_workloads": total,
            "valid_workloads": len(valid),
            "dp_20pct_wins": len(wins),
            "fraction": fraction,
            "target_fraction": 0.30,
            "minimum_wins_required": minimum_wins_required,
            "additional_wins_needed": additional_wins_needed,
            "target_met": target_met,
        },
    }


def _fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def render(report: Dict[str, Any]) -> str:
    lines = [
        "# DP ≥20% workload audit",
        "",
        f"Candidate results use {report['protocol']['candidate_repeats']} "
        "round-robin repetitions. Execution time excludes optimization; "
        "optimization time is reported separately.",
        "",
        "| workload | outcome | DP (s) | best other | best other (s) | DP speedup | gate |",
        "|---|---|---:|---|---:|---:|:---:|",
    ]
    for workload, item in report["combined"].items():
        best_time = _fmt(item["best_other_execution_time_sec"])
        if item["best_other_is_lower_bound"]:
            best_time = f"≥{best_time}"
        speedup_value = item["dp_speedup_over_best_other"]
        if speedup_value is None:
            speedup = "N/A"
        else:
            speedup = f"{_fmt(speedup_value)}x"
            if item["dp_speedup_is_lower_bound"]:
                speedup = f"≥{speedup}"
        lines.append(
            f"| {workload} | {item['workload_outcome']} | "
            f"{_fmt(item['dp_execution_time_sec'])} | "
            f"{item['best_other_optimizer'] or 'N/A'} | "
            f"{best_time} | "
            f"{speedup} | "
            f"{'PASS' if item['dp_at_least_20pct_faster'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Candidate optimizer details",
            "",
            "| workload | optimizer | execution mean±sd (s) | optimization (s) | status |",
            "|---|---|---:|---:|---|",
        ]
    )
    for workload, candidate in report["candidates"].items():
        for optimizer, item in candidate["runs"].items():
            runtime = "N/A"
            if item["mean_execution_time_sec"] is not None:
                runtime = (
                    f"{item['mean_execution_time_sec']:.3f}±"
                    f"{item['stddev_execution_time_sec']:.3f}"
                )
            elif item["execution_time_lower_bound_sec"] is not None:
                runtime = (
                    f">={item['execution_time_lower_bound_sec']:.0f} "
                    f"({report['protocol']['candidate_repeats']}/"
                    f"{report['protocol']['candidate_repeats']} timed out)"
                )
            if item["valid"]:
                status = "valid"
            elif item["source_infeasible"]:
                observed = item["processed_samples"]
                status = (
                    "source_infeasible: "
                    f"{observed[0] if observed else 'unknown'}/"
                    f"{report['protocol']['candidate_outputs']} retained"
                )
            elif item["formally_unavailable"]:
                status = "unavailable: " + "; ".join(item["errors"])
            else:
                status = "; ".join(item["errors"]) or "invalid"
            lines.append(
                f"| {workload} | {optimizer} | {runtime} | "
                f"{_fmt(item['optimization_time_sec'])} | {status} |"
            )
    summary = report["summary"]
    lines.extend(
        [
            "",
            f"DP ≥20% wins: {summary['dp_20pct_wins']}/"
            f"{summary['total_workloads']} "
            f"({summary['fraction'] * 100:.1f}%).",
            f"Fully evaluable workloads: {summary['valid_workloads']}/"
            f"{summary['total_workloads']}. Formal timeouts remain in the "
            "denominator and are reported as unavailable.",
            f"Minimum wins required for 30%: "
            f"{summary['minimum_wins_required']}; additional wins needed: "
            f"{summary['additional_wins_needed']}.",
            f"30% target: {'PASS' if summary['target_met'] else 'FAIL'}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat to combine disjoint formal candidate batches.",
    )
    parser.add_argument(
        "--existing-report",
        type=Path,
        default=Path(__file__).resolve().parent
        / "formal_results"
        / "cross_system_w8_latest.json",
    )
    parser.add_argument(
        "--require-all-registered",
        action="store_true",
        help=(
            "Fail unless every workload frozen in "
            "REGISTERED_CANDIDATE_WORKLOADS is present across the supplied "
            "candidate roots."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--expected-samples", type=int, default=20_000)
    parser.add_argument(
        "--execution-timeout-sec",
        type=float,
        default=EXECUTION_TIMEOUT_SEC,
    )
    args = parser.parse_args()
    if args.expected_repeats <= 0:
        parser.error("--expected-repeats must be positive")
    if args.expected_samples <= 0:
        parser.error("--expected-samples must be positive")
    if args.execution_timeout_sec <= 0:
        parser.error("--execution-timeout-sec must be positive")
    report = audit(
        args.candidate_root,
        args.existing_report,
        require_all_registered=args.require_all_registered,
        expected_repeats=args.expected_repeats,
        expected_samples=args.expected_samples,
        execution_timeout_sec=args.execution_timeout_sec,
    )
    markdown = render(report)
    print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
