#!/usr/bin/env python3
"""Run PICO/DP optimizer ablations for one Cedar workload.

The script keeps the normal Cedar DataSet path: it only builds different
OptimizerOptions and records setup time, modeled plan cost, cache warmup, and
optional workload throughput for each condition.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.eval_cedar import import_module_from_path
from evaluation.profiler import Profiler


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Condition:
    name: str
    optimizer_selector: int
    disable_prefetch: bool = False
    disable_offload: bool = False
    disable_parallelism: bool = False
    disable_reorder: bool = False
    disable_fusion: bool = False
    disable_caching: bool = False


CONDITIONS: Dict[str, Condition] = {
    "full_dp": Condition("full_dp", optimizer_selector=2),
    "stagewise_dp": Condition("stagewise_dp", optimizer_selector=4),
    "cedar_staged": Condition("cedar_staged", optimizer_selector=0),
    "dp_reorder_only_cedar_physical": Condition(
        "dp_reorder_only_cedar_physical", optimizer_selector=5
    ),
    "no_prefetch": Condition("no_prefetch", optimizer_selector=2, disable_prefetch=True),
    "no_reorder": Condition("no_reorder", optimizer_selector=2, disable_reorder=True),
    "no_fusion": Condition("no_fusion", optimizer_selector=2, disable_fusion=True),
    "no_offload": Condition("no_offload", optimizer_selector=2, disable_offload=True),
    "no_cache": Condition("no_cache", optimizer_selector=2, disable_caching=True),
    "no_parallelism": Condition(
        "no_parallelism", optimizer_selector=2, disable_parallelism=True
    ),
    "reorder_only_dp": Condition(
        "reorder_only_dp",
        optimizer_selector=2,
        disable_prefetch=True,
        disable_offload=True,
        disable_parallelism=True,
        disable_fusion=True,
        disable_caching=True,
    ),
}


def parse_dataset_kwargs(raw: Optional[str]) -> Dict[str, Optional[str]]:
    if not raw:
        return {}
    result: Dict[str, Optional[str]] = {}
    for pair in raw.split(","):
        parts = pair.split("=", maxsplit=1)
        if len(parts) == 1:
            result[parts[0]] = None
        else:
            result[parts[0]] = parts[1]
    return result


def summarize_profiler_results(results: Dict[str, Any]) -> Dict[str, Any]:
    run_times = results.get("epoch_run_times") or []
    sample_counts = results.get("epoch_num_samples") or []
    perf_time = float(sum(run_times))
    total_samples = int(sum(sample_counts))
    throughput = total_samples / perf_time if perf_time else 0.0
    return {
        "perf_time_sec": perf_time,
        "num_samples": total_samples,
        "throughput_samples_per_sec": throughput,
        "time_per_sample_us": (perf_time / total_samples * 1e6)
        if total_samples
        else 0.0,
        "raw_workload_results": results,
    }


def object_disk_cache_dir() -> Path:
    return Path("/tmp") / f"cedar_{os.getpid()}"


def clear_object_disk_cache_dir() -> None:
    cache_dir = object_disk_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def cleanup_runtime() -> None:
    gc.collect()
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception as exc:  # pragma: no cover - cleanup best effort
        logger.debug("Failed to shut down Ray cleanly: %s", exc)


def calculate_plan_costs(dataset: Any) -> Dict[str, float]:
    feature_plans = getattr(dataset, "feature_plans", None)
    if not feature_plans:
        raise RuntimeError("Dataset did not produce optimized feature plans.")

    costs: Dict[str, float] = {}
    features = getattr(dataset, "features", {})
    for feature_name, plan in feature_plans.items():
        feature = features.get(feature_name)
        if feature is None and features:
            feature = next(iter(features.values()))
        if feature is None or getattr(feature, "optimizer", None) is None:
            raise RuntimeError(f"Could not find optimizer for {feature_name}.")

        optimizer = feature.optimizer
        caching_on = optimizer._get_cache_pid(plan) is not None
        fused_blocks = [
            list(desc.fused_pipes)
            for desc in plan.pipe_descs.values()
            if getattr(desc, "fused_pipes", None)
            and len(getattr(desc, "fused_pipes", [])) > 1
        ]
        costs[feature_name] = optimizer.calculate_cost(
            plan.graph,
            physical_specs=plan.pipe_descs,
            fused_pipes=fused_blocks if fused_blocks else None,
            caching_on=caching_on,
            plan=plan,
        )
    return costs


def dataset_has_object_disk_cache(dataset: Any) -> bool:
    feature_plans = getattr(dataset, "feature_plans", None)
    if not feature_plans:
        return False
    for plan in feature_plans.values():
        for desc in getattr(plan, "pipe_descs", {}).values():
            if getattr(desc, "name", None) == "ObjectDiskCachePipe":
                return True
    return False


def copy_generated_plan(condition_name: str, plans_dir: Optional[Path]) -> Optional[str]:
    generated = Path("/tmp/cedar_optimized_plan.yml")
    if not plans_dir or not generated.exists():
        return None
    plans_dir.mkdir(parents=True, exist_ok=True)
    out_path = plans_dir / f"{condition_name}_plan.yml"
    shutil.copyfile(generated, out_path)
    return str(out_path)


def make_spec(
    args: argparse.Namespace,
    condition: Condition,
    run_samples: int,
) -> CedarEvalSpec:
    disable_parallelism = condition.disable_parallelism or (
        not args.enable_local_parallelism
    )
    disable_caching = condition.disable_caching or args.disable_caching
    return CedarEvalSpec(
        batch_size=args.batch_size,
        num_total_samples=run_samples,
        num_epochs=args.num_epochs,
        config=None,
        kwargs=parse_dataset_kwargs(args.dataset_kwargs),
        use_ray=args.use_ray,
        ray_ip=args.ray_ip,
        iteration_time=args.iteration_time,
        profiled_stats=args.profiled_stats,
        run_profiling=False,
        disable_optimizer=False,
        disable_controller=not args.enable_controller,
        disable_prefetch=condition.disable_prefetch,
        disable_offload=condition.disable_offload or args.disable_offload,
        disable_parallelism=disable_parallelism,
        disable_reorder=condition.disable_reorder,
        disable_fusion=condition.disable_fusion,
        disable_caching=disable_caching,
        use_my_optimizer=condition.optimizer_selector,
        generate_plan=False,
        reorder_timeout_sec=args.reorder_timeout_sec,
    )


def run_workload(dataset: Any, args: argparse.Namespace) -> Dict[str, Any]:
    profiler = Profiler(
        dataset,
        args.num_epochs,
        args.run_samples,
        args.batch_size,
        args.iteration_time,
    )
    profiler.run()
    return summarize_profiler_results(profiler.get_results())


def run_once(
    args: argparse.Namespace,
    dataset_getter: Any,
    condition: Condition,
    repeat_idx: int,
) -> Dict[str, Any]:
    clear_object_disk_cache_dir()
    spec = make_spec(args, condition, args.run_samples)
    setup_start = time.perf_counter()
    dataset = dataset_getter(spec)
    setup_time = time.perf_counter() - setup_start
    plan_costs = calculate_plan_costs(dataset)
    plan_path = copy_generated_plan(
        f"{condition.name}_repeat{repeat_idx + 1}", args.plans_dir_path
    )
    uses_cache = dataset_has_object_disk_cache(dataset)

    result: Dict[str, Any] = {
        "condition": condition.name,
        "optimizer_selector": condition.optimizer_selector,
        "repeat_idx": repeat_idx,
        "setup_time_sec": setup_time,
        "plan_cost": sum(plan_costs.values()),
        "plan_costs_by_feature": plan_costs,
        "plan_path": plan_path,
        "uses_cache": uses_cache,
        "disable_prefetch": condition.disable_prefetch,
        "disable_offload": spec.disable_offload,
        "disable_parallelism": spec.disable_parallelism,
        "disable_reorder": condition.disable_reorder,
        "disable_fusion": condition.disable_fusion,
        "disable_caching": spec.disable_caching,
    }

    if args.plan_only:
        result["workload_skipped"] = True
        result["skip_reason"] = "plan_only"
        return result

    warmup_result = None
    if uses_cache:
        warmup_start = time.perf_counter()
        warmup_result = run_workload(dataset, args)
        result["cache_warmup_wall_time_sec"] = time.perf_counter() - warmup_start
        result["cache_warmup_result"] = warmup_result

    workload_start = time.perf_counter()
    workload_result = run_workload(dataset, args)
    workload_wall_time = time.perf_counter() - workload_start
    result.update(workload_result)
    result["workload_wall_time_sec"] = workload_wall_time
    result["total_time_sec"] = setup_time + workload_wall_time
    result["cache_warmup_excluded"] = warmup_result is not None
    return result


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_condition(condition: str, repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(repeats) == 1:
        result = dict(repeats[0])
        result["num_repeats"] = 1
        result["repeat_results"] = repeats
        return result

    numeric_fields = [
        "setup_time_sec",
        "plan_cost",
        "perf_time_sec",
        "num_samples",
        "throughput_samples_per_sec",
        "time_per_sample_us",
        "workload_wall_time_sec",
        "total_time_sec",
        "cache_warmup_wall_time_sec",
    ]
    result = dict(repeats[0])
    result["condition"] = condition
    result["num_repeats"] = len(repeats)
    result["repeat_results"] = repeats
    for field in numeric_fields:
        vals = [
            item[field]
            for item in repeats
            if isinstance(item.get(field), (int, float))
        ]
        if vals:
            result[field] = mean(vals)
    result["cache_warmup_excluded"] = any(
        item.get("cache_warmup_excluded", False) for item in repeats
    )
    if any(item.get("error") for item in repeats):
        result["error"] = "one_or_more_repeats_failed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--dataset_func", default="get_dataset")
    parser.add_argument("--profiled_stats", required=True)
    parser.add_argument("--dataset_kwargs", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_total_samples", type=int, required=True)
    parser.add_argument("--data_num_total_samples", type=int, default=1000)
    parser.add_argument("--full_data_run", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--ray_ip", default="")
    parser.add_argument("--iteration_time", type=float, default=None)
    parser.add_argument("--enable_controller", action="store_true")
    parser.add_argument("--enable_local_parallelism", action="store_true")
    parser.add_argument("--disable_offload", action="store_true")
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument("--reorder_timeout_sec", type=float, default=None)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(CONDITIONS),
        default=[
            "full_dp",
            "stagewise_dp",
            "no_reorder",
            "no_fusion",
            "no_offload",
            "no_cache",
            "no_parallelism",
        ],
    )
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--plans_dir", default="")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--allow_torch_parallelism", action="store_true")
    args = parser.parse_args()
    if args.num_repeats < 1:
        raise ValueError("--num_repeats must be >= 1")
    args.run_samples = (
        args.num_total_samples
        if args.full_data_run
        else min(args.num_total_samples, args.data_num_total_samples)
    )
    args.plans_dir_path = Path(args.plans_dir) if args.plans_dir else None
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())

    if not args.allow_torch_parallelism:
        import torch

        logger.warning("Setting torch threads to 1")
        torch.set_num_threads(1)

    module = import_module_from_path(str(Path(args.dataset_file).resolve()))
    dataset_getter = getattr(module, args.dataset_func)

    runs: List[Dict[str, Any]] = []
    for condition_name in args.conditions:
        condition = CONDITIONS[condition_name]
        repeats: List[Dict[str, Any]] = []
        for repeat_idx in range(args.num_repeats):
            try:
                logger.info(
                    "Running condition %s repeat %d/%d",
                    condition_name,
                    repeat_idx + 1,
                    args.num_repeats,
                )
                repeats.append(run_once(args, dataset_getter, condition, repeat_idx))
            except BaseException:
                repeats.append(
                    {
                        "condition": condition_name,
                        "repeat_idx": repeat_idx,
                        "error": traceback.format_exc(),
                    }
                )
            finally:
                cleanup_runtime()
        runs.append(aggregate_condition(condition_name, repeats))

    results = {
        "dataset_file": args.dataset_file,
        "dataset_func": args.dataset_func,
        "profiled_stats": args.profiled_stats,
        "dataset_kwargs": args.dataset_kwargs,
        "num_epochs": args.num_epochs,
        "requested_num_total_samples": args.num_total_samples,
        "data_num_total_samples": args.run_samples,
        "num_repeats": args.num_repeats,
        "use_ray": args.use_ray,
        "ray_ip": args.ray_ip,
        "local_parallelism_enabled": args.enable_local_parallelism,
        "global_disable_offload": args.disable_offload,
        "global_disable_caching": args.disable_caching,
        "plan_only": args.plan_only,
        "runs": runs,
    }

    output_path = Path(args.results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=False)
        f.write("\n")

    print("======DP Ablation Summary======")
    for item in runs:
        if item.get("error"):
            print(f"{item['condition']}: ERROR {item['error']}")
            continue
        pieces = [
            f"{item['condition']}: setup={item.get('setup_time_sec', 0.0):.6f}s",
            f"cost={item.get('plan_cost', 0.0):.6f}",
        ]
        if not item.get("workload_skipped"):
            pieces.append(
                "throughput={:.3f} samples/s".format(
                    item.get("throughput_samples_per_sec", 0.0)
                )
            )
        print(", ".join(pieces))
    print(f"Wrote results: {output_path}")


if __name__ == "__main__":
    main()
