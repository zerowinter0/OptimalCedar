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


def iter_threshold_configs() -> Iterable[ThresholdConfig]:
    for p_retention, q_retention, a_retention in product(
        (20, 35, 50),
        (70, 80, 90),
        (40, 60, 80),
    ):
        yield ThresholdConfig(p_retention, q_retention, a_retention)


def _qualifies(result: Mapping[str, Any]) -> bool:
    try:
        staged = float(result["staged_median_sec"])
        joint = float(result["joint_median_sec"])
        cedar_staged = float(result["cedar_staged_cost"])
        cedar_joint = float(result["cedar_joint_cost"])
        return (
            result["status"] == "success"
            and result["equivalent"] is True
            and int(result["joint_backend_count"]) >= 2
            and joint < staged
            and cedar_staged < cedar_joint
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
        staged = float(result["staged_median_sec"])
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
    return {
        "key": config.key,
        "status": "success",
        "equivalent": True,
        "joint_backend_count": 2,
        "staged_median_sec": 10.0 + offset / 100.0,
        "joint_median_sec": 8.0 + offset / 100.0,
        "cedar_staged_cost": 1.0,
        "cedar_joint_cost": 2.0,
        "output_count": 10,
        "mock": True,
    }


def _require_protocol_environment() -> None:
    expected = {
        "CEDAR_PROFILE_TIME_SEC": "10",
        "CEDAR_PROFILE_RAY_ACTORS": "1",
        "CEDAR_PROFILE_SMP_PROCS": "1",
        "CEDAR_MATCH_PROFILE_RESOURCES": "1",
        "CEDAR_PROFILE_MATCH_CPU_BUDGET": "64",
        "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": "8",
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
) -> dict[str, Any]:
    config_root = output_root / "configurations" / config.key
    config_root.mkdir(parents=True, exist_ok=True)
    threshold_path = calibration_root / "thresholds" / f"{config.key}.json"
    calibration_path = fixture_root / "calibration.jsonl"
    pilot_path = fixture_root / "pilot.jsonl"
    profile_path = config_root / "profile.yml"
    staged_plan = config_root / "staged_plan.yml"
    joint_plan = config_root / "joint_plan.yml"
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
        for name, path in (("staged", staged_plan), ("joint", joint_plan)):
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
        scores = _run_worker(
            [
                "score",
                *calibration_args,
                "--profile",
                str(profile_path),
                "--staged-plan",
                str(staged_plan),
                "--joint-plan",
                str(joint_plan),
            ],
            config_root / "scores.json",
            config_root / "scores.log",
            timeout_sec=600,
        )

        executions: dict[str, list[dict[str, Any]]] = {
            "staged": [],
            "joint": [],
        }
        pilot_args = _shared_args(pilot_path, threshold_path, image_root, 500)
        for repetition in range(2):
            order = ("staged", "joint") if repetition == 0 else ("joint", "staged")
            for name in order:
                result = _run_worker(
                    [
                        "execute",
                        *pilot_args,
                        "--plan",
                        str(staged_plan if name == "staged" else joint_plan),
                    ],
                    config_root / f"{name}_run_{repetition}.json",
                    config_root / f"{name}_run_{repetition}.log",
                    timeout_sec=3600,
                )
                executions[name].append(result)

        staged_ids = [set(run["record_ids"]) for run in executions["staged"]]
        joint_ids = [set(run["record_ids"]) for run in executions["joint"]]
        all_ids = staged_ids + joint_ids
        equivalent = bool(all_ids) and all(ids == all_ids[0] for ids in all_ids[1:])
        result = {
            **base,
            "status": "success",
            "profile": profile_result,
            "plans": plans,
            "scores": scores,
            "executions": executions,
            "equivalent": equivalent,
            "output_count": len(all_ids[0]) if equivalent else 0,
            "joint_backend_count": len(plans["joint"]["backend_families"]),
            "staged_median_sec": statistics.median(
                run["seconds"] for run in executions["staged"]
            ),
            "joint_median_sec": statistics.median(
                run["seconds"] for run in executions["joint"]
            ),
            "cedar_staged_cost": scores["staged"]["cedar_cost"],
            "cedar_joint_cost": scores["joint"]["cedar_cost"],
            "pico_staged_cost": scores["staged"]["pico_cost"],
            "pico_joint_cost": scores["joint"]["pico_cost"],
        }
    except BaseException as exc:
        result = {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    atomic_write_json(config_root / "result.json", result)
    return result


def run_real_pilot(fixture_root: Path, image_root: Path, output_root: Path) -> None:
    _require_protocol_environment()
    from evaluation.motivation_multimodal.calibrate import run_calibration

    output_root.mkdir(parents=True, exist_ok=True)
    calibration_root = output_root / "calibration"
    calibration_summary = calibration_root / "calibration_summary.json"
    if not calibration_summary.exists():
        run_calibration(fixture_root, image_root, calibration_root)
    results = []
    for index, config in enumerate(iter_threshold_configs(), start=1):
        result_path = output_root / "configurations" / config.key / "result.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("status") in {"success", "failed"}:
                results.append(prior)
                print(f"pilot progress={index}/27 key={config.key} resumed", flush=True)
                continue
        result = _run_configuration(
            config,
            fixture_root,
            image_root,
            calibration_root,
            output_root,
        )
        results.append(result)
        atomic_write_json(
            output_root / "results.json",
            {"status": "running", "results": results},
        )
        print(
            f"pilot progress={index}/27 key={config.key} status={result['status']}",
            flush=True,
        )
    atomic_write_json(
        output_root / "results.json",
        {"status": "complete", "results": results},
    )
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
    args = parser.parse_args()
    if not args.mock_scores:
        if args.max_configs is not None:
            parser.error("--max-configs is available only with --mock-scores")
        run_real_pilot(args.fixture, args.image_root, args.output)
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
