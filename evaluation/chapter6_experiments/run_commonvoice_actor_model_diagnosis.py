#!/usr/bin/env python3
"""Run controlled CommonVoice plans that isolate backend and stage effects."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_ROOT = (
    REPO_ROOT
    / "outputs/chapter6_experiments/joint_actor_budget_formal_all_ten_v1"
)


def ray_context(width: int, batch_size: int) -> dict:
    return {
        "variant_type": "RAY",
        "n_actors": width,
        "max_inflight": 100,
        "max_prefetch": 100,
        "use_threads": True,
        "submit_batch_size": batch_size,
        "num_gpus": 0.0,
    }


def materialize_plans(formal_root: Path, output_root: Path) -> list[str]:
    source = formal_root / "formal_runs/commonvoice/plans/dp_optimizer.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    candidates = []
    for first_width, second_width in ((2, 4), (3, 3), (4, 2)):
        candidate = copy.deepcopy(payload)
        physical = candidate["physical_plan"]
        first = physical["pipes"][8]
        second = physical["pipes"][9]
        first["variant"] = "RAY"
        first["variant_ctx"] = ray_context(first_width, 5)
        second["variant"] = "RAY"
        second["variant_ctx"] = ray_context(second_width, 2)
        name = f"ray_split_{first_width}_{second_width}"
        (output_root / "plans" / f"{name}.yaml").write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        candidates.append(name)

    # Keep the backend and actor budget fixed while moving the Ray -> local
    # boundary.  These plans separate a compute-model error from batching,
    # marshalling, and local-suffix effects.
    gate_source = (
        REPO_ROOT
        / "outputs/chapter6_experiments/commonvoice_global_concurrency_plan_gate"
        / "commonvoice/plans/dp_optimizer.yaml"
    )
    gate_payload = yaml.safe_load(gate_source.read_text(encoding="utf-8"))
    for last_fused_id in (3, 2, 1):
        candidate = copy.deepcopy(gate_payload)
        physical = candidate["physical_plan"]
        physical["pipes"][8]["fused_pipes"] = list(
            range(6, last_fused_id - 1, -1)
        )
        physical["pipes"][8]["variant_ctx"] = ray_context(6, 1)
        suffix = list(range(last_fused_id - 1, -1, -1))
        graph = {7: "8", 8: str(suffix[0]) if suffix else "9"}
        for left, right in zip(suffix, suffix[1:]):
            graph[left] = str(right)
        if suffix:
            graph[suffix[-1]] = "9"
        graph[9] = ""
        physical["graph"] = graph
        name = f"ray_prefix_through_{last_fused_id}_batch1"
        (output_root / "plans" / f"{name}.yaml").write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        candidates.append(name)
    return candidates


def validate_result(path: Path, expected_samples: int) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("epoch_num_samples") != [expected_samples]:
        raise RuntimeError(f"Unexpected sample count in {path}")
    values = payload.get("epoch_run_times")
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(f"Unexpected timing result in {path}")
    return float(values[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs/chapter6_experiments/commonvoice_actor_model_diagnosis"
        ),
    )
    parser.add_argument("--repeats", type=int, choices=(1, 3), default=1)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument(
        "--candidates",
        nargs="*",
        help="Run only the named materialized candidates.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("plans", "results", "logs"):
        (args.output_root / directory).mkdir(exist_ok=True)
    candidates = materialize_plans(args.formal_root, args.output_root)
    if args.candidates:
        unknown = sorted(set(args.candidates) - set(candidates))
        if unknown:
            raise ValueError(f"Unknown candidates: {unknown}")
        candidates = list(args.candidates)
    summary = {name: [] for name in candidates}
    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(candidates)
        order = candidates[offset:] + candidates[:offset]
        for name in order:
            result = args.output_root / "results" / f"round{repeat}__{name}.json"
            log = args.output_root / "logs" / f"round{repeat}__{name}.log"
            command = [
                sys.executable,
                "evaluation/eval_cedar.py",
                "--dataset_file",
                "evaluation/pipelines/commonvoice/cedar_dataset.py",
                "--master_feature_config",
                str(args.output_root / "plans" / f"{name}.yaml"),
                "--num_total_samples",
                str(args.samples),
                "--num_epochs",
                "1",
                "--use_ray",
                "--ray_ip",
                "127.0.0.1:6379",
                "--results_path",
                str(result),
                "--dataset_kwargs",
                f"max_samples={args.samples}",
            ]
            with log.open("w", encoding="utf-8") as stream:
                stream.write("command: " + " ".join(command) + "\n")
                stream.flush()
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_sec,
                    check=True,
                )
            summary[name].append(validate_result(result, args.samples))
            print(f"{name} round {repeat}: {summary[name][-1]:.6f}s", flush=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
