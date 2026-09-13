"""Reject outputs that do not preserve the fixed local-GPU CLIP stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_OPTIMIZERS = {"optimizer", "simple_dp_optimizer", "dp_optimizer"}


def validate(
    path: Path,
    expected_optimizers: set[str] | None = None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    expected = expected_optimizers or DEFAULT_OPTIMIZERS
    observed = {run.get("optimizer") for run in runs}
    if observed != expected:
        raise RuntimeError(f"Unexpected optimizer set: {sorted(observed)}")

    for run in runs:
        name = run["optimizer"]
        plans = run.get("physical_plans_by_feature", {})
        if set(plans) != {"feature"}:
            raise RuntimeError(f"{name} did not persist exactly one feature plan")
        plan = plans["feature"]
        active = {str(pipe_id) for pipe_id in plan["graph"]}
        gpu_stages = [
            desc
            for pipe_id, desc in plan["pipes"].items()
            if str(pipe_id) in active and desc.get("execution_resource") == "cuda"
        ]
        if len(gpu_stages) != 1:
            raise RuntimeError(f"{name} has {len(gpu_stages)} active CUDA stages")
        stage = gpu_stages[0]
        if stage.get("name") != "FilterPipe_ClipPredicate":
            raise RuntimeError(f"{name} CUDA stage is not the CLIP filter: {stage}")
        if stage.get("variant") != "INPROCESS":
            raise RuntimeError(
                f"{name} CLIP filter is not fixed to local INPROCESS"
            )
        if stage.get("fused_pipes"):
            raise RuntimeError(f"{name} fused the fixed CLIP filter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--optimizers", nargs="+")
    args = parser.parse_args()
    expected = set(args.optimizers) if args.optimizers else None
    validate(args.results, expected_optimizers=expected)


if __name__ == "__main__":
    main()
