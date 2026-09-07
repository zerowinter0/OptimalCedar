"""Frozen-configuration three-round staged/joint comparison."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    sha256_file,
)
from evaluation.motivation_multimodal.pilot import (
    _require_protocol_environment,
    _run_worker,
    _shared_args,
)
from evaluation.motivation_multimodal.runner import STAGED_OPTIMIZERS


def _validate_pilot(pilot_root: Path) -> dict[str, Any]:
    grid = json.loads((pilot_root / "results.json").read_text(encoding="utf-8"))
    results = grid.get("results", [])
    if grid.get("status") != "complete" or len(results) != 27:
        raise ValueError("Formal execution requires all 27 terminal pilot configurations")
    if any(result.get("status") not in {"success", "failed"} for result in results):
        raise ValueError("Pilot grid contains a nonterminal configuration")
    selected = json.loads(
        (pilot_root / "selected_configuration.json").read_text(encoding="utf-8")
    )
    if selected.get("status") != "success":
        raise ValueError("Pilot selection did not produce a frozen configuration")
    threshold = Path(selected["threshold_path"])
    if sha256_file(threshold) != selected["threshold_sha256"]:
        raise ValueError("Selected pilot threshold hash no longer matches")
    return selected


def run_formal(
    fixture_root: Path,
    image_root: Path,
    pilot_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    _require_protocol_environment()
    selected = _validate_pilot(pilot_root)
    output_root.mkdir(parents=True, exist_ok=True)
    threshold_path = output_root / "selected_threshold.json"
    profile_path = output_root / "profile.yml"
    shutil.copyfile(selected["threshold_path"], threshold_path)
    shutil.copyfile(selected["result"]["profile"]["path"], profile_path)
    if sha256_file(threshold_path) != selected["threshold_sha256"]:
        raise RuntimeError("Threshold changed while freezing formal artifacts")

    calibration_path = fixture_root / "calibration.jsonl"
    formal_path = fixture_root / "formal.jsonl"
    calibration_args = _shared_args(
        calibration_path, threshold_path, image_root, 3000
    )
    optimizer_names = (*STAGED_OPTIMIZERS, "joint")
    plans = {}
    plan_paths = {
        name: output_root / f"{name}_plan.yml" for name in optimizer_names
    }
    for name in optimizer_names:
        plans[name] = _run_worker(
            [
                "plan",
                *calibration_args,
                "--profile",
                str(profile_path),
                "--optimizer",
                name,
                "--plan",
                str(plan_paths[name]),
            ],
            output_root / f"{name}_plan_result.json",
            output_root / f"{name}_plan.log",
            timeout_sec=3600,
        )
        plans[name]["path"] = plan_paths[name].name
    score_arguments = [
        "score",
        *calibration_args,
        "--profile",
        str(profile_path),
    ]
    for name in optimizer_names:
        score_arguments.extend(
            ["--named-plan", f"{name}={plan_paths[name]}"]
        )
    cedar_scores = _run_worker(
        score_arguments,
        output_root / "cedar_costs.json",
        output_root / "cedar_costs.log",
        timeout_sec=600,
    )
    executions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in optimizer_names
    }
    formal_args = _shared_args(formal_path, threshold_path, image_root, 3000)
    for repetition in range(3):
        order = (
            optimizer_names
            if repetition % 2 == 0
            else tuple(reversed(optimizer_names))
        )
        for name in order:
            run = _run_worker(
                ["execute", *formal_args, "--plan", str(plan_paths[name])],
                output_root / f"{name}_run_{repetition}.json",
                output_root / f"{name}_run_{repetition}.log",
                timeout_sec=3600,
            )
            executions[name].append(run)

    id_sets = [
        set(run["record_ids"])
        for name in optimizer_names
        for run in executions[name]
    ]
    equivalent = bool(id_sets) and all(ids == id_sets[0] for ids in id_sets[1:])
    if not equivalent:
        raise RuntimeError("Formal staged and joint plans are not output-equivalent")
    staged_medians = {
        name: statistics.median(run["seconds"] for run in executions[name])
        for name in STAGED_OPTIMIZERS
    }
    best_staged_name = min(
        staged_medians, key=lambda name: (staged_medians[name], name)
    )
    # Figure 2 consumes a stable "staged" alias representing the strongest
    # of the six sequential orders. The complete matrix remains archived.
    plans["staged"] = dict(plans[best_staged_name])
    executions["staged"] = list(executions[best_staged_name])
    cedar_costs = dict(cedar_scores["costs"])
    cedar_costs["staged"] = cedar_costs[best_staged_name]
    summary = {
        "schema_version": 1,
        "status": "success",
        "formal_protocol": True,
        "records": 3000,
        "repetitions": 3,
        "selected_key": selected["key"],
        "threshold_sha256": sha256_file(threshold_path),
        "profile_sha256": sha256_file(profile_path),
        "plans": plans,
        "executions": executions,
        "equivalent": equivalent,
        "output_count": len(id_sets[0]),
        "best_staged_name": best_staged_name,
        "staged_medians_sec": staged_medians,
        "staged_median_sec": staged_medians[best_staged_name],
        "joint_median_sec": statistics.median(
            run["seconds"] for run in executions["joint"]
        ),
        "cedar_costs": cedar_costs,
        "cedar_cost_model": cedar_scores["cost_model"],
        "metadata": command_metadata(),
    }
    atomic_write_json(output_root / "formal_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run_formal(
        args.fixture_root,
        args.image_root,
        args.pilot_root,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_key": summary["selected_key"],
                "staged_median_sec": summary["staged_median_sec"],
                "joint_median_sec": summary["joint_median_sec"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
