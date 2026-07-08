#!/usr/bin/env python3
"""Time SimCLRv2 cache optimizer plan construction.

This benchmark measures optimizer wall time only. It builds the SimCLRv2
cache workload graph and runs the requested optimizer against an existing
profile; it does not iterate over the dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cedar.compose.cedar_joint_optimizer import CedarJointOptimizer
from cedar.compose.optimizer import Optimizer, OptimizerOptions, PhysicalPlan
from cedar.sources import LocalFSSource
from evaluation.pipelines.simclrv2.cedar_cache_dataset import (
    DATASET_LOC,
    SimCLRV2Feature,
)


OPTIMIZERS = {
    "cedar_staged": Optimizer,
    "cedar_joint": CedarJointOptimizer,
}


def _set_log_level(level: str) -> None:
    numeric_level = getattr(logging, level.upper())
    logging.basicConfig(level=numeric_level)
    logging.getLogger().setLevel(numeric_level)
    for name in (
        "cedar",
        "cedar.compose",
        "cedar.compose.optimizer",
        "cedar.compose.cedar_joint_optimizer",
    ):
        logging.getLogger(name).setLevel(numeric_level)


def _plan_summary(plan: PhysicalPlan) -> Dict[str, Any]:
    pipe_descs = plan.pipe_descs
    graph = plan.graph
    return {
        "num_physical_pipes": len(pipe_descs),
        "num_graph_nodes": len(graph),
        "uses_cache": any(
            desc.name == "ObjectDiskCachePipe" for desc in pipe_descs.values()
        ),
        "uses_fusion": any(desc.name == "FusedPipe" for desc in pipe_descs.values()),
        "ray_pipes": sum(
            1
            for desc in pipe_descs.values()
            if getattr(desc.variant_type, "name", "") == "RAY"
        ),
    }


def _save_plan(plan: PhysicalPlan, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    plan_dict = plan.to_dict()
    for _, pipe_dict in plan_dict["pipes"].items():
        if "variant" not in pipe_dict:
            pipe_dict["variant"] = "INPROCESS"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump({"physical_plan": plan_dict}, f)


def _run_once(
    optimizer_name: str,
    profile_path: str,
    data_root: str,
    batch_size: int,
    num_samples: int,
    available_local_cpus: int,
    reorder_timeout_sec: Optional[float],
    disable_parallelism: bool,
    disable_offload: bool,
    disable_fusion: bool,
    disable_caching: bool,
    log_level: str,
    plan_output: Optional[str],
) -> Dict[str, Any]:
    _set_log_level(log_level)

    profile = Path(profile_path)
    if not profile.exists():
        raise FileNotFoundError(profile)

    train_path = Path(data_root) / "imagenette2" / "train"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    source = LocalFSSource(str(train_path), recursive=True)
    feature = SimCLRV2Feature(batch_size=batch_size)
    feature.apply(source)

    optimizer = OPTIMIZERS[optimizer_name]()
    feature.set_optimizer(optimizer)

    options = OptimizerOptions(
        enable_prefetch=False,
        est_throughput=None,
        available_local_cpus=available_local_cpus,
        enable_offload=not disable_offload,
        enable_reorder=True,
        enable_local_parallelism=not disable_parallelism,
        enable_fusion=not disable_fusion,
        enable_caching=not disable_caching,
        num_samples=num_samples,
        use_my_optimizer=6 if optimizer_name == "cedar_joint" else 0,
        reorder_timeout_sec=reorder_timeout_sec,
    )

    start = time.perf_counter()
    plan = feature.optimize(options, str(profile))
    elapsed = time.perf_counter() - start

    if plan_output:
        _save_plan(plan, Path(plan_output))

    result: Dict[str, Any] = {
        "optimizer": optimizer_name,
        "profile_path": str(profile),
        "data_root": str(data_root),
        "batch_size": batch_size,
        "num_samples_for_cache_cost": num_samples,
        "available_local_cpus": available_local_cpus,
        "disable_parallelism": disable_parallelism,
        "disable_offload": disable_offload,
        "disable_fusion": disable_fusion,
        "disable_caching": disable_caching,
        "reorder_timeout_sec": reorder_timeout_sec,
        "optimization_wall_time_sec": elapsed,
        "plan": _plan_summary(plan),
    }
    stats = getattr(optimizer, "integrated_optimizer_stats", None)
    if stats:
        result["integrated_optimizer_stats"] = stats
    return result


def _child_entry(kwargs: Dict[str, Any], queue: mp.Queue) -> None:
    try:
        queue.put(("ok", _run_once(**kwargs)))
    except BaseException:
        queue.put(("error", traceback.format_exc()))


def _run_once_with_timeout(kwargs: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child_entry, args=(kwargs, queue))
    proc.start()
    try:
        status, payload = queue.get(timeout=timeout_sec)
    except Empty as exc:
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(
            f"Optimizer run exceeded timeout of {timeout_sec:.3f}s"
        ) from exc

    proc.join()
    if status != "ok" or proc.exitcode != 0:
        raise RuntimeError(payload)
    return payload


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    times = [row["optimization_wall_time_sec"] for row in rows]
    return {
        "runs": rows,
        "summary": {
            "repeats": len(rows),
            "mean_sec": statistics.mean(times),
            "median_sec": statistics.median(times),
            "min_sec": min(times),
            "max_sec": max(times),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure optimizer-only wall time for the SimCLRv2 cache workload. "
            "Default optimizer is Cedar's native-pass joint enumeration "
            "(use_my_optimizer=6)."
        )
    )
    parser.add_argument(
        "--profiled_stats",
        default="/tmp/simclrv2_cache_profile.yml",
        help="YAML profile produced by SimCLRv2 cache profiling.",
    )
    parser.add_argument(
        "--data_root",
        default=str(REPO_ROOT / "evaluation" / DATASET_LOC),
        help="Directory containing imagenette2/train.",
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZERS),
        default="cedar_joint",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=9472,
        help="Dataset cardinality used by Cedar's cache cost model.",
    )
    parser.add_argument(
        "--available_local_cpus",
        type=int,
        default=max(mp.cpu_count(), 1),
    )
    parser.add_argument("--reorder_timeout_sec", type=float, default=None)
    parser.add_argument("--run_timeout_sec", type=float, default=1800.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--disable_parallelism", action="store_true")
    parser.add_argument("--disable_offload", action="store_true")
    parser.add_argument("--disable_fusion", action="store_true")
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument(
        "--log_level",
        default="ERROR",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Keep ERROR/CRITICAL for optimizer timing; INFO is very noisy.",
    )
    parser.add_argument(
        "--output_json",
        default="/tmp/simclrv2_cache_joint_optimization_time.json",
    )
    parser.add_argument(
        "--plan_output",
        default="",
        help="Optional path for saving the optimized plan from the last repeat.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for repeat_idx in range(args.repeats):
        plan_output = ""
        if args.plan_output and repeat_idx == args.repeats - 1:
            plan_output = args.plan_output
        kwargs = {
            "optimizer_name": args.optimizer,
            "profile_path": args.profiled_stats,
            "data_root": args.data_root,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "available_local_cpus": args.available_local_cpus,
            "reorder_timeout_sec": args.reorder_timeout_sec,
            "disable_parallelism": args.disable_parallelism,
            "disable_offload": args.disable_offload,
            "disable_fusion": args.disable_fusion,
            "disable_caching": args.disable_caching,
            "log_level": args.log_level,
            "plan_output": plan_output,
        }
        row = _run_once_with_timeout(kwargs, args.run_timeout_sec)
        row["repeat"] = repeat_idx + 1
        rows.append(row)
        print(
            "repeat {}/{}: optimizer={} time={:.6f}s".format(
                repeat_idx + 1,
                args.repeats,
                args.optimizer,
                row["optimization_wall_time_sec"],
            ),
            flush=True,
        )

    result = _summarize(rows)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    summary = result["summary"]
    print(
        "summary: repeats={repeats}, mean={mean_sec:.6f}s, "
        "median={median_sec:.6f}s, min={min_sec:.6f}s, max={max_sec:.6f}s".format(
            **summary
        )
    )
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
