"""Bounded pilot grid and deterministic configuration selection."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    sha256_file,
)
from evaluation.motivation_multimodal.runner import STAGED_OPTIMIZERS


@dataclass(frozen=True, order=True)
class ThresholdConfig:
    p_retention: int
    q_retention: int
    a_retention: int
    c_retention: int = 80
    b_retention: int = 80

    @property
    def key(self) -> str:
        return (
            f"p{self.p_retention}-q{self.q_retention}-"
            f"a{self.a_retention}-c{self.c_retention}-b{self.b_retention}"
        )


@dataclass(frozen=True)
class SelectedConfiguration:
    key: str
    speedup: float
    output_count: int
    result: Mapping[str, Any]


def iter_threshold_configs(grid: str = "standard") -> Iterable[ThresholdConfig]:
    if grid == "standard":
        axes = ((20, 35, 50), (70, 80, 90), (40, 60, 80))
    elif grid == "distinct-staged":
        # Predeclared interaction grid: a selective image filter can move ahead
        # of text work when reorder runs first, whereas fusion-first may bind it
        # to the downstream GPU block.  Higher perplexity retention avoids
        # making the text filter unconditionally dominant.
        axes = ((50, 70, 90), (70, 90), (10, 20, 30))
    else:
        raise ValueError(f"Unknown pilot grid: {grid}")
    for p_retention, q_retention, a_retention in product(*axes):
        yield ThresholdConfig(p_retention, q_retention, a_retention)


def _staged_plan_groups(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for name in STAGED_OPTIMIZERS:
        digest = str(result["plans"][name]["sha256"])
        group = groups.setdefault(
            digest,
            {"optimizers": [], "cedar_costs": [], "pico_costs": [], "times": []},
        )
        group["optimizers"].append(name)
        group["cedar_costs"].append(float(result["cedar_costs"][name]))
        group["pico_costs"].append(float(result["pico_costs"][name]))
        group["times"].extend(
            float(run["seconds"])
            for run in result["executions"][name]
        )
    for group in groups.values():
        group["cedar_cost"] = statistics.median(group.pop("cedar_costs"))
        group["pico_cost"] = statistics.median(group.pop("pico_costs"))
        group["median_sec"] = statistics.median(group.pop("times"))
    return groups


def _ranking_witnesses(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Find staged pairs for which only PICO matches the full runtime order."""

    joint_time = float(result["joint_median_sec"])
    joint_cedar = float(result["cedar_costs"]["joint"])
    joint_pico = float(result["pico_costs"]["joint"])
    groups = list(_staged_plan_groups(result).items())
    witnesses = []
    for index, (left_hash, left) in enumerate(groups):
        for right_hash, right in groups[index + 1 :]:
            faster_hash, faster, slower_hash, slower = (
                (left_hash, left, right_hash, right)
                if left["median_sec"] < right["median_sec"]
                else (right_hash, right, left_hash, left)
            )
            if faster["median_sec"] == slower["median_sec"]:
                continue
            cedar_reverses_staged = faster["cedar_cost"] > slower["cedar_cost"]
            pico_orders_all_three = (
                joint_time < faster["median_sec"] < slower["median_sec"]
                and joint_pico < faster["pico_cost"] < slower["pico_cost"]
            )
            cedar_orders_joint_first = joint_cedar < min(
                faster["cedar_cost"], slower["cedar_cost"]
            )
            if cedar_reverses_staged and pico_orders_all_three and cedar_orders_joint_first:
                witnesses.append(
                    {
                        "faster_staged_hash": faster_hash,
                        "slower_staged_hash": slower_hash,
                        "faster_staged": faster,
                        "slower_staged": slower,
                    }
                )
    return witnesses


def _qualifies(result: Mapping[str, Any]) -> bool:
    try:
        joint = float(result["joint_median_sec"])
        staged_medians = result["staged_medians_sec"]
        cedar_costs = result["cedar_costs"]
        pico_costs = result["pico_costs"]
        if set(staged_medians) != set(STAGED_OPTIMIZERS):
            return False
        if set(cedar_costs) != {*STAGED_OPTIMIZERS, "joint"}:
            return False
        if set(pico_costs) != {*STAGED_OPTIMIZERS, "joint"}:
            return False
        groups = list(_staged_plan_groups(result).values())
        return (
            result["status"] == "success"
            and result["equivalent"] is True
            and int(result["joint_backend_count"]) >= 2
            and all(joint < float(value) for value in staged_medians.values())
            and float(cedar_costs["joint"])
            < min(float(cedar_costs[name]) for name in STAGED_OPTIMIZERS)
            and len(groups) >= 2
            and bool(_ranking_witnesses(result))
        )
    except (KeyError, TypeError, ValueError):
        return False


def select_configuration(
    results: Iterable[Mapping[str, Any]],
) -> SelectedConfiguration:
    candidates = []
    for result in results:
        if not _qualifies(result):
            continue
        staged = min(float(value) for value in result["staged_medians_sec"].values())
        joint = float(result["joint_median_sec"])
        speedup = (staged - joint) / staged
        candidates.append(
            SelectedConfiguration(
                key=str(result["key"]),
                speedup=speedup,
                output_count=int(result["output_count"]),
                result=result,
            )
        )
    if not candidates:
        raise ValueError("No pilot configuration satisfies all qualification rules")
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.speedup,
            -candidate.output_count,
            candidate.key,
        ),
    )[0]


def _mock_result(config: ThresholdConfig) -> dict[str, Any]:
    offset = list(iter_threshold_configs()).index(config)
    plan_hashes = {
        name: ("mock-plan-a" if index % 2 == 0 else "mock-plan-b")
        for index, name in enumerate(STAGED_OPTIMIZERS)
    }
    staged_times = {
        name: (10.0 if digest == "mock-plan-a" else 11.0)
        for name, digest in plan_hashes.items()
    }
    executions = {
        name: [{"seconds": value}] * 3
        for name, value in staged_times.items()
    }
    executions["joint"] = [{"seconds": 8.0 + offset / 100.0}] * 3
    return {
        "key": config.key,
        "status": "success",
        "equivalent": True,
        "joint_backend_count": 2,
        "plans": {
            **{
                name: {"sha256": digest}
                for name, digest in plan_hashes.items()
            },
            "joint": {"sha256": "mock-plan-joint"},
        },
        "executions": executions,
        "staged_median_sec": min(staged_times.values()),
        "staged_medians_sec": staged_times,
        "cedar_costs": {
            **{
                name: (2.0 if digest == "mock-plan-a" else 1.0)
                for name, digest in plan_hashes.items()
            },
            "joint": 0.5,
        },
        "pico_costs": {
            **{
                name: (1.0 if digest == "mock-plan-a" else 2.0)
                for name, digest in plan_hashes.items()
            },
            "joint": 0.5,
        },
        "joint_median_sec": 8.0 + offset / 100.0,
        "output_count": 10,
        "mock": True,
    }


def _require_protocol_environment() -> None:
    expected = {
        "CEDAR_PROFILE_TIME_SEC": "10",
        "CEDAR_PROFILE_RAY_ACTORS": "1",
        "CEDAR_PROFILE_SMP_PROCS": "1",
        "CEDAR_PROFILE_FILTER_SELECTIVITY": "1",
        "CEDAR_MATCH_PROFILE_RESOURCES": "1",
        "CEDAR_PROFILE_MATCH_CPU_BUDGET": "64",
        "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": os.environ.get("W", "1"),
    }
    mismatch = {
        key: {"expected": value, "actual": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Pilot resource environment mismatch: {mismatch}")


def _run_worker(
    arguments: list[str],
    result_path: Path,
    log_path: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "evaluation.motivation_multimodal.runner",
        *arguments,
        "--result",
        str(result_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def _shared_args(
    dataset_path: Path,
    threshold_path: Path,
    image_root: Path,
    num_samples: int,
) -> list[str]:
    return [
        "--dataset",
        str(dataset_path),
        "--threshold",
        str(threshold_path),
        "--image-root",
        str(image_root),
        "--num-samples",
        str(num_samples),
    ]


def _run_configuration(
    config: ThresholdConfig,
    fixture_root: Path,
    image_root: Path,
    calibration_root: Path,
    output_root: Path,
    pilot_num_samples: int,
    plan_only: bool = False,
) -> dict[str, Any]:
    config_root = output_root / "configurations" / config.key
    config_root.mkdir(parents=True, exist_ok=True)
    threshold_path = calibration_root / "thresholds" / f"{config.key}.json"
    calibration_path = fixture_root / "calibration.jsonl"
    pilot_path = fixture_root / "pilot.jsonl"
    profile_path = config_root / "profile.yml"
    optimizer_names = (*STAGED_OPTIMIZERS, "joint")
    plan_paths = {
        name: config_root / f"{name}_plan.yml" for name in optimizer_names
    }
    base = {
        "key": config.key,
        "threshold_path": str(threshold_path),
        "threshold_sha256": sha256_file(threshold_path),
        "status": "running",
        "metadata": command_metadata(),
    }
    atomic_write_json(config_root / "result.json", base)
    try:
        calibration_args = _shared_args(
            calibration_path, threshold_path, image_root, 500
        )
        profile_result = _run_worker(
            ["profile", *calibration_args, "--profile", str(profile_path)],
            config_root / "profile_result.json",
            config_root / "profile.log",
            timeout_sec=4 * 3600,
        )
        plans = {}
        for name in optimizer_names:
            path = plan_paths[name]
            plans[name] = _run_worker(
                [
                    "plan",
                    *calibration_args,
                    "--profile",
                    str(profile_path),
                    "--optimizer",
                    name,
                    "--plan",
                    str(path),
                ],
                config_root / f"{name}_plan_result.json",
                config_root / f"{name}_plan.log",
                timeout_sec=3600,
            )
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
        scores = _run_worker(
            score_arguments,
            config_root / "cedar_costs.json",
            config_root / "cedar_costs.log",
            timeout_sec=600,
        )
        staged_hashes = {
            plans[name]["sha256"] for name in STAGED_OPTIMIZERS
        }
        all_widths_are_one = all(
            all(int(width) == 1 for width in plans[name]["stage_widths"].values())
            for name in optimizer_names
        )
        cedar_joint_is_best = float(scores["costs"]["joint"]) < min(
            float(scores["costs"][name]) for name in STAGED_OPTIMIZERS
        )
        distinct_staged_costs = {
            round(float(scores["costs"][name]), 12)
            for name in STAGED_OPTIMIZERS
        }
        pico_joint_is_best = float(scores["pico_costs"]["joint"]) < min(
            float(scores["pico_costs"][name]) for name in STAGED_OPTIMIZERS
        )
        distinct_staged_pico_costs = {
            round(float(scores["pico_costs"][name]), 12)
            for name in STAGED_OPTIMIZERS
        }
        if not (
            len(staged_hashes) >= 2
            and len(distinct_staged_costs) >= 2
            and cedar_joint_is_best
            and pico_joint_is_best
            and len(distinct_staged_pico_costs) >= 2
            and all_widths_are_one
        ):
            result = {
                **base,
                "status": "screened_out",
                "profile": profile_result,
                "plans": plans,
                "cedar_costs": scores["costs"],
                "cedar_cost_model": scores["cost_model"],
                "pico_costs": scores["pico_costs"],
                "pico_cost_model": scores["pico_cost_model"],
                "screen": {
                    "distinct_staged_plans": len(staged_hashes),
                    "distinct_staged_costs": len(distinct_staged_costs),
                    "cedar_joint_is_best": cedar_joint_is_best,
                    "pico_joint_is_best": pico_joint_is_best,
                    "distinct_staged_pico_costs": len(distinct_staged_pico_costs),
                    "all_stage_widths_are_one": all_widths_are_one,
                },
            }
            atomic_write_json(config_root / "result.json", result)
            return result
        if plan_only:
            result = {
                **base,
                "status": "screen_passed",
                "profile": profile_result,
                "plans": plans,
                "cedar_costs": scores["costs"],
                "cedar_cost_model": scores["cost_model"],
                "pico_costs": scores["pico_costs"],
                "pico_cost_model": scores["pico_cost_model"],
                "screen": {
                    "distinct_staged_plans": len(staged_hashes),
                    "distinct_staged_costs": len(distinct_staged_costs),
                    "cedar_joint_is_best": cedar_joint_is_best,
                    "pico_joint_is_best": pico_joint_is_best,
                    "distinct_staged_pico_costs": len(distinct_staged_pico_costs),
                    "all_stage_widths_are_one": all_widths_are_one,
                },
            }
            atomic_write_json(config_root / "result.json", result)
            return result
        executions: dict[str, list[dict[str, Any]]] = {
            name: [] for name in optimizer_names
        }
        pilot_args = _shared_args(
            pilot_path, threshold_path, image_root, pilot_num_samples
        )
        for repetition in range(3):
            offset = repetition % len(optimizer_names)
            order = optimizer_names[offset:] + optimizer_names[:offset]
            for name in order:
                result = _run_worker(
                    [
                        "execute",
                        *pilot_args,
                        "--plan",
                        str(plan_paths[name]),
                    ],
                    config_root / f"{name}_run_{repetition}.json",
                    config_root / f"{name}_run_{repetition}.log",
                    timeout_sec=3600,
                )
                executions[name].append(result)

        all_ids = [
            set(run["record_ids"])
            for name in optimizer_names
            for run in executions[name]
        ]
        equivalent = bool(all_ids) and all(ids == all_ids[0] for ids in all_ids[1:])
        staged_medians = {
            name: statistics.median(run["seconds"] for run in executions[name])
            for name in STAGED_OPTIMIZERS
        }
        best_staged_name = min(
            staged_medians, key=lambda name: (staged_medians[name], name)
        )
        result = {
            **base,
            "status": "success",
            "profile": profile_result,
            "plans": plans,
            "executions": executions,
            "equivalent": equivalent,
            "output_count": len(all_ids[0]) if equivalent else 0,
            "joint_backend_count": len(plans["joint"]["backend_families"]),
            "best_staged_name": best_staged_name,
            "staged_medians_sec": staged_medians,
            "staged_median_sec": staged_medians[best_staged_name],
            "cedar_costs": scores["costs"],
            "cedar_cost_model": scores["cost_model"],
            "pico_costs": scores["pico_costs"],
            "pico_cost_model": scores["pico_cost_model"],
            "joint_median_sec": statistics.median(
                run["seconds"] for run in executions["joint"]
            ),
        }
        result["staged_plan_groups"] = _staged_plan_groups(result)
        result["ranking_witnesses"] = _ranking_witnesses(result)
    except BaseException as exc:
        result = {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    atomic_write_json(config_root / "result.json", result)
    return result


def run_real_pilot(
    fixture_root: Path,
    image_root: Path,
    output_root: Path,
    pilot_num_samples: int,
    grid: str,
    max_configs: int | None = None,
    plan_only: bool = False,
) -> None:
    _require_protocol_environment()
    from evaluation.motivation_multimodal.calibrate import run_calibration

    output_root.mkdir(parents=True, exist_ok=True)
    calibration_root = output_root / "calibration"
    calibration_summary = calibration_root / "calibration_summary.json"
    configs = list(iter_threshold_configs(grid))
    if max_configs is not None:
        configs = configs[:max_configs]
    if not calibration_summary.exists():
        run_calibration(
            fixture_root,
            image_root,
            calibration_root,
            configs=configs,
        )
    results = []
    for index, config in enumerate(configs, start=1):
        result_path = output_root / "configurations" / config.key / "result.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("status") in {
                "success",
                "failed",
                "screened_out",
                "screen_passed",
            }:
                results.append(prior)
                print(
                    f"pilot progress={index}/{len(configs)} "
                    f"key={config.key} resumed",
                    flush=True,
                )
                continue
        result = _run_configuration(
            config,
            fixture_root,
            image_root,
            calibration_root,
            output_root,
            pilot_num_samples,
            plan_only=plan_only,
        )
        results.append(result)
        atomic_write_json(
            output_root / "results.json",
            {"status": "running", "results": results},
        )
        print(
            f"pilot progress={index}/{len(configs)} "
            f"key={config.key} status={result['status']}",
            flush=True,
        )
    atomic_write_json(
        output_root / "results.json",
        {"status": "complete", "results": results},
    )
    if plan_only:
        atomic_write_json(
            output_root / "screening_summary.json",
            {
                "status": "complete",
                "screen_passed": sum(
                    item.get("status") == "screen_passed" for item in results
                ),
                "screened_out": sum(
                    item.get("status") == "screened_out" for item in results
                ),
                "results": results,
            },
        )
        return
    try:
        selected = select_configuration(results)
        selection = {
            "status": "success",
            "key": selected.key,
            "speedup": selected.speedup,
            "output_count": selected.output_count,
            "threshold_sha256": selected.result["threshold_sha256"],
            "threshold_path": selected.result["threshold_path"],
            "result": selected.result,
        }
    except ValueError as exc:
        selection = {"status": "no_match", "error": str(exc)}
    atomic_write_json(output_root / "selected_configuration.json", selection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("datasets/coco/val2017"),
    )
    parser.add_argument("--mock-scores", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--pilot-num-samples", type=int, default=500)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--grid",
        choices=("standard", "distinct-staged"),
        default="standard",
    )
    args = parser.parse_args()
    if args.pilot_num_samples <= 0:
        parser.error("--pilot-num-samples must be positive")
    if not args.mock_scores:
        run_real_pilot(
            args.fixture,
            args.image_root,
            args.output,
            args.pilot_num_samples,
            args.grid,
            max_configs=args.max_configs,
            plan_only=args.plan_only,
        )
        return
    configs = list(iter_threshold_configs())[: args.max_configs]
    results = [_mock_result(config) for config in configs]
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "results.json", {"results": results})
    try:
        selected = select_configuration(results)
        selection = {
            "key": selected.key,
            "speedup": selected.speedup,
            "output_count": selected.output_count,
        }
    except ValueError as exc:
        selection = {"error": str(exc)}
    atomic_write_json(args.output / "selection.json", selection)


if __name__ == "__main__":
    main()
