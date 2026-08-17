"""
Compare runtime performance between Cedar's original Optimizer, DjOptimizer,
DpOptimizer, DpTwoStageOptimizer, and DpCedarOptimizer.

The workload timer starts only after the dataset has been constructed, so the
profile/optimization/setup time is reported separately and is not included in
the workload performance numbers.
"""

import argparse
import contextlib
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import shutil
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import yaml

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.eval_cedar import _get_profiler as _get_workload_runner
from evaluation.eval_cedar import import_module_from_path


logger = logging.getLogger(__name__)


OPTIMIZERS = {
    "dj_optimizer": 3,
    "optimizer": 0,
    "dp_optimizer": 2,
    "dp_two_stage_optimizer": 4,
    # Backward-compatible CLI spellings; omitted from the default order.
    "dp_seperate_optimizer": 4,
    "dp_separate_optimizer": 4,
    "dp_cedar_optimizer": 5,
    "exp_optimizer": 7,
    "pecan_optimizer": 8,
    "pecan_two_stage_optimizer": 9,
    "dj_two_stage_optimizer": 10,
    "simple_dp_optimizer": 11,
}

DEFAULT_OPTIMIZER_ORDER = [
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "dp_two_stage_optimizer",
    "optimizer",
]

REQUIRED_PROFILE_KEYS = {
    "baseline": {"throughput", "latencies", "input_sizes", "output_sizes"},
    "disk_info": {"read_latency", "write_latency"},
}

DEFAULT_OPTIMIZER_TIME_LIMIT_SEC = 5 * 60
OPTIMIZER_RSS_LIMIT_BYTES = 405078136832


def _cache_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def _configure_object_disk_cache(
    args: argparse.Namespace, optimizer_name: str, reset: bool
) -> Path:
    results_path = Path(args.results_path).resolve()
    try:
        experiment_dir = results_path.parents[2]
    except IndexError:
        experiment_dir = results_path.parent
    workload_name = Path(args.dataset_file).resolve().parent.name
    namespace = (
        f"{_cache_component(workload_name)}__"
        f"{_cache_component(optimizer_name)}"
    )
    cache_root = experiment_dir / "cache"
    cache_dir = cache_root / namespace
    os.environ["CEDAR_CACHE_ROOT"] = str(cache_root)
    os.environ["CEDAR_CACHE_NAMESPACE"] = namespace
    os.environ.pop("CEDAR_CACHE_SHARD", None)
    if reset and cache_dir.exists():
        logger.info("Clearing persistent Cedar cache namespace %s", cache_dir)
        shutil.rmtree(cache_dir)
    logger.info(
        "Configured persistent Cedar cache namespace %s for workload=%s "
        "optimizer=%s",
        cache_dir,
        workload_name,
        optimizer_name,
    )
    return cache_dir


def _object_disk_cache_is_complete(dataset: Any) -> bool:
    cache_root = os.environ.get("CEDAR_CACHE_ROOT")
    namespace = os.environ.get("CEDAR_CACHE_NAMESPACE")
    if not cache_root or not namespace:
        return False
    cache_dir = Path(cache_root) / namespace
    expected_shards = (
        len(getattr(dataset, "features", {}))
        if getattr(dataset, "_iter_mode", "default") == "mp"
        else 1
    )
    manifests = list(cache_dir.glob("*/.manifest.json"))
    if len(manifests) != expected_shards:
        return False
    try:
        return all(json.load(open(path)).get("complete") is True for path in manifests)
    except (OSError, ValueError):
        return False


def _wait_for_object_disk_cache_complete(
    dataset: Any,
    timeout_sec: float = 60.0,
    poll_interval_sec: float = 0.1,
) -> bool:
    """Wait for multiprocessing cache workers to commit their manifests.

    The requested sample limit can be reached just before the worker processes
    finish handling end-of-stream and atomically commit their shard manifests.
    Treat that short shutdown window as asynchronous completion rather than a
    failed cache warmup.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        if _object_disk_cache_is_complete(dataset):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_interval_sec, remaining))


def _parse_dataset_kwargs(raw: Optional[str]) -> Dict[str, Optional[str]]:
    if not raw:
        return {}

    extra_kwargs = {}
    for pair in raw.split(","):
        res = pair.split("=")
        if len(res) == 1:
            key = res[0]
            value = None
        elif len(res) == 2:
            key = res[0]
            value = res[1]
        else:
            raise ValueError(f"Improperly formatted dataset_kwargs: {raw}")
        extra_kwargs[key] = value
    return extra_kwargs


def _load_profile(profiled_stats: str) -> Optional[Dict[str, Any]]:
    if not profiled_stats:
        return None

    path = Path(profiled_stats)
    if not path.exists():
        return None

    with path.open("r") as f:
        profile = yaml.safe_load(f)

    return profile if isinstance(profile, dict) else None


def _has_complete_profile(profiled_stats: str) -> bool:
    profile = _load_profile(profiled_stats)
    if profile is None:
        return False

    for section, keys in REQUIRED_PROFILE_KEYS.items():
        value = profile.get(section)
        if not isinstance(value, dict) or not keys.issubset(value):
            return False

    return isinstance(profile.get("offloads"), dict)


def _profile_has_offload_variant(profiled_stats: str, variant_name: str) -> bool:
    profile = _load_profile(profiled_stats)
    if profile is None:
        return False
    offloads = profile.get("offloads")
    return isinstance(offloads, dict) and variant_name in offloads


def _cleanup_runtime() -> None:
    gc.collect()
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception as e:
        logger.debug("Failed to shut down Ray cleanly: %s", e)


class _OptimizerResourceExceeded(RuntimeError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _get_current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        logger.debug("Failed to read /proc/self/status for RSS monitoring.")
    return 0


@contextlib.contextmanager
def _optimizer_resource_guard(
    time_limit_sec: int,
    rss_limit_bytes: int,
) -> Iterator[None]:
    exceeded: Dict[str, str] = {}
    stop_event = threading.Event()

    def _raise_from_signal(signum: int, frame: Any) -> None:
        if signum == signal.SIGALRM:
            raise _OptimizerResourceExceeded(
                "optimizer_time_limit_exceeded",
                f"optimizer exceeded {time_limit_sec}s time limit",
            )
        reason = exceeded.get("reason", "optimizer_memory_limit_exceeded")
        detail = exceeded.get(
            "detail",
            f"optimizer RSS exceeded {rss_limit_bytes / 1024**3:.1f}GB",
        )
        raise _OptimizerResourceExceeded(reason, detail)

    def _monitor_rss() -> None:
        while not stop_event.wait(1.0):
            rss_bytes = _get_current_rss_bytes()
            if rss_bytes > rss_limit_bytes:
                exceeded["reason"] = "optimizer_memory_limit_exceeded"
                exceeded["detail"] = (
                    "optimizer RSS exceeded "
                    f"{rss_limit_bytes / 1024**3:.1f}GB "
                    f"(current={rss_bytes / 1024**3:.3f}GB)"
                )
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    old_alarm_handler = signal.getsignal(signal.SIGALRM)
    old_usr1_handler = signal.getsignal(signal.SIGUSR1)
    old_alarm = signal.alarm(0)
    monitor = threading.Thread(target=_monitor_rss, daemon=True)
    signal.signal(signal.SIGALRM, _raise_from_signal)
    signal.signal(signal.SIGUSR1, _raise_from_signal)
    signal.alarm(time_limit_sec)
    monitor.start()
    try:
        yield
    finally:
        stop_event.set()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm_handler)
        signal.signal(signal.SIGUSR1, old_usr1_handler)
        if old_alarm:
            signal.alarm(old_alarm)
        monitor.join(timeout=1.0)


def _configure_optimizer_runtime(args: argparse.Namespace) -> None:
    if args.match_profile_resources:
        os.environ["CEDAR_MATCH_PROFILE_RESOURCES"] = "1"
        os.environ["CEDAR_PROFILE_MATCH_CPU_BUDGET"] = str(args.cpu_budget)
        logger.warning(
            "Enabled strict profile resource matching with unified "
            "CPU budget=%d.",
            args.cpu_budget,
        )
    else:
        os.environ.pop("CEDAR_MATCH_PROFILE_RESOURCES", None)
        os.environ.pop("CEDAR_PROFILE_MATCH_CPU_BUDGET", None)

    if args.fixed_local_workers_ablation is None:
        os.environ.pop("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", None)
    else:
        os.environ["CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"] = str(
            args.fixed_local_workers_ablation
        )
        logger.warning(
            "FIXED-LOCAL-WORKER ABLATION: forcing local_workers=%d; "
            "CPU budget may be intentionally underutilized.",
            args.fixed_local_workers_ablation,
        )

    if args.ray_available_parallelism is None:
        return

    import cedar.compose.constants as constants

    constants.RAY_AVAILABLE_PARALLELISM = args.ray_available_parallelism
    logger.warning(
        "Set Cedar RAY_AVAILABLE_PARALLELISM=%d for this comparison.",
        args.ray_available_parallelism,
    )


def _normalize_ray_args(args: argparse.Namespace) -> None:
    if args.ray_ip and not args.use_ray:
        logger.warning("--ray_ip was provided; enabling --use_ray automatically.")
        args.use_ray = True

    if args.use_ray and args.disable_offload:
        logger.warning(
            "Ray is configured, but --disable_offload is set. Optimizers will "
            "receive the Ray address, but RAY/TF_RAY offload variants remain disabled."
        )


def _verify_local_ray(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.use_ray:
        return {}

    address = None
    if args.ray_ip:
        if ":" in args.ray_ip:
            address = args.ray_ip
        else:
            address = f"ray://{args.ray_ip}:10001"
    logger.info("Verifying Ray resources at %s.", address or "local instance")
    # Keep Ray out of this process before Cedar forks its local workers. Ray
    # gRPC background state remains unsafe to fork even after ray.shutdown().
    spawn_ctx = mp.get_context("spawn")
    result_queue = spawn_ctx.Queue()
    verifier = spawn_ctx.Process(
        target=_query_ray_resources,
        args=(address, result_queue),
    )
    verifier.start()
    verifier.join(timeout=60)
    if verifier.is_alive():
        verifier.terminate()
        verifier.join(timeout=5)
        raise RuntimeError("Timed out while verifying Ray resources")
    if verifier.exitcode != 0:
        raise RuntimeError(
            f"Ray resource verifier exited with code {verifier.exitcode}"
        )
    try:
        status, payload = result_queue.get(timeout=5)
    except queue.Empty as exc:
        raise RuntimeError("Ray resource verifier returned no result") from exc
    finally:
        result_queue.close()
        result_queue.join_thread()
    if status == "error":
        raise RuntimeError(f"Failed to verify Ray resources: {payload}")
    logger.info("Local Ray resources: %s", payload)
    return payload


def _query_ray_resources(
    address: Optional[str], result_queue: mp.Queue
) -> None:
    """Query Ray in an isolated process so the benchmark remains fork-safe."""
    import ray

    try:
        if address:
            ray.init(address, ignore_reinit_error=True)
        else:
            ray.init(ignore_reinit_error=True)
        result_queue.put(("ok", dict(ray.cluster_resources())))
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))
    finally:
        if ray.is_initialized():
            ray.shutdown()


def _make_spec(
    args: argparse.Namespace,
    use_my_optimizer: int,
    reorder_timeout_sec: Optional[float] = None,
) -> CedarEvalSpec:
    data_num_total_samples = args.num_total_samples
    if not args.full_data_run:
        data_num_total_samples = min(
            args.num_total_samples,
            args.data_num_total_samples,
        )

    return CedarEvalSpec(
        args.batch_size,
        data_num_total_samples,
        args.num_epochs,
        None,
        _parse_dataset_kwargs(args.dataset_kwargs),
        args.use_ray,
        args.ray_ip,
        args.iteration_time,
        args.profiled_stats,
        False,
        False,
        not args.enable_controller,
        False,
        args.disable_offload,
        not args.enable_local_parallelism,
        False,
        False,
        args.disable_caching,
        use_my_optimizer,
        False,
        reorder_timeout_sec,
    )


def _summarize_results(
    name: str,
    setup_time_sec: float,
    workload_wall_time_sec: float,
    workload_results: Dict[str, Any],
    cache_warmup_wall_time_sec: float = 0.0,
) -> Dict[str, Any]:
    total_run_time = sum(workload_results["epoch_run_times"])
    total_samples = sum(workload_results["epoch_num_samples"])
    throughput = total_samples / total_run_time if total_run_time else 0.0
    time_per_sample_us = (
        (total_run_time / total_samples) * 1e6 if total_samples else 0.0
    )

    return {
        "optimizer": name,
        "setup_time_sec": setup_time_sec,
        "workload_wall_time_sec": workload_wall_time_sec,
        "total_time_sec": setup_time_sec + workload_wall_time_sec,
        "perf_time_sec": total_run_time,
        "num_samples": total_samples,
        "throughput_samples_per_sec": throughput,
        "time_per_sample_us": time_per_sample_us,
        "raw_workload_results": workload_results,
        "cache_warmup_wall_time_sec": cache_warmup_wall_time_sec,
        "cache_warmup_excluded": cache_warmup_wall_time_sec > 0.0,
    }


def _summarize_setup_only(
    name: str,
    setup_time_sec: float,
    skip_reason: str,
    plan_cost: Optional[float] = None,
    plan_costs_by_feature: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    result = {
        "optimizer": name,
        "setup_time_sec": setup_time_sec,
        "workload_wall_time_sec": 0.0,
        "total_time_sec": setup_time_sec,
        "perf_time_sec": 0.0,
        "num_samples": 0,
        "throughput_samples_per_sec": 0.0,
        "time_per_sample_us": 0.0,
        "raw_workload_results": None,
        "workload_skipped": True,
        "skip_reason": skip_reason,
    }
    if plan_cost is not None:
        result["plan_cost"] = plan_cost
        result["plan_costs_by_feature"] = plan_costs_by_feature or {}
    return result


def _calculate_plan_costs(
    dataset: Any, profiled_stats: Optional[str] = None
) -> Dict[str, float]:
    if getattr(dataset, "feature_plans", None) is None:
        raise RuntimeError("Dataset did not produce optimized feature plans.")

    costs: Dict[str, float] = {}
    features = getattr(dataset, "features", {})
    for f_name, plan in dataset.feature_plans.items():
        feature = features.get(f_name)
        if feature is None and features:
            feature = next(iter(features.values()))
        if feature is None or getattr(feature, "optimizer", None) is None:
            raise RuntimeError(f"Could not find optimizer for feature {f_name}.")

        optimizer = feature.optimizer
        if not hasattr(optimizer, "profiled_stats"):
            loaded_profile = _load_profile(profiled_stats or "")
            if loaded_profile is None:
                raise RuntimeError(
                    f"Optimizer for feature {f_name} does not retain profiled_stats, "
                    "and no readable profiled_stats file was provided."
                )
            optimizer.profiled_stats = loaded_profile
            optimizer._init_stats()
        caching_on = optimizer._get_cache_pid(plan) is not None
        fused_blocks = [
            list(desc.fused_pipes)
            for desc in plan.pipe_descs.values()
            if getattr(desc, "fused_pipes", None)
            and len(getattr(desc, "fused_pipes", [])) > 1
        ]
        fused_pipes = fused_blocks if fused_blocks else None
        costs[f_name] = optimizer.calculate_cost(
            plan.graph,
            physical_specs=plan.pipe_descs,
            fused_pipes=fused_pipes,
            caching_on=caching_on,
            plan=plan,
        )
    return costs


def _dataset_has_object_disk_cache(dataset: Any) -> bool:
    feature_plans = getattr(dataset, "feature_plans", None)
    if not feature_plans:
        return False

    for plan in feature_plans.values():
        for desc in getattr(plan, "pipe_descs", {}).values():
            if getattr(desc, "name", None) == "ObjectDiskCachePipe":
                return True
    return False


def _new_workload_runner_like(workload_runner: Any) -> Any:
    return workload_runner.__class__(
        workload_runner.dataset,
        workload_runner.num_epochs,
        workload_runner.num_total_samples,
        workload_runner.batch_size,
        workload_runner.iteration_time,
    )


def _close_workload_runner(workload_runner: Any) -> None:
    if workload_runner is None:
        return
    close = getattr(workload_runner, "close", None)
    if callable(close):
        close()
        return
    dataset = getattr(workload_runner, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _summarize_timeout(
    name: str,
    setup_time_sec: float,
    error: BaseException,
    reorder_timeout_sec: Optional[float],
) -> Dict[str, Any]:
    return {
        "optimizer": name,
        "setup_time_sec": setup_time_sec,
        "workload_wall_time_sec": 0.0,
        "total_time_sec": float("inf"),
        "perf_time_sec": float("inf"),
        "num_samples": 0,
        "throughput_samples_per_sec": 0.0,
        "time_per_sample_us": float("inf"),
        "raw_workload_results": None,
        "reorder_timeout_sec": reorder_timeout_sec,
        "timed_out": True,
        "error": str(error),
    }


def _summarize_optimizer_overhead(
    args: argparse.Namespace,
    name: str,
    setup_time_sec: float,
    error: _OptimizerResourceExceeded,
) -> Dict[str, Any]:
    return {
        "optimizer": name,
        "setup_time_sec": setup_time_sec,
        "workload_wall_time_sec": 0.0,
        "total_time_sec": float("inf"),
        "perf_time_sec": float("inf"),
        "num_samples": 0,
        "throughput_samples_per_sec": 0.0,
        "time_per_sample_us": float("inf"),
        "raw_workload_results": None,
        "workload_skipped": True,
        "skip_reason": error.reason,
        "optimizer_overhead_too_high": True,
        "optimizer_time_limit_sec": args.optimizer_time_limit_sec,
        "optimizer_rss_limit_bytes": OPTIMIZER_RSS_LIMIT_BYTES,
        "cedar_runs_last": False,
        "cedar_reorder_timeout_sec": args.cedar_reorder_timeout_sec,
        "cedar_runtime_timeout_enabled": (
            not args.disable_cedar_runtime_timeout and not args.skip_workload
        ),
        "cedar_timeout_multiplier": args.cedar_timeout_multiplier,
        "error": error.detail,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _average_repeat_results(
    optimizer_name: str,
    repeat_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(repeat_results) == 1:
        result = dict(repeat_results[0])
        result["num_repeats"] = 1
        result["repeat_results"] = repeat_results
        return result

    result = dict(repeat_results[0])
    result["optimizer"] = optimizer_name
    result["num_repeats"] = len(repeat_results)
    result["repeat_results"] = repeat_results
    result["raw_workload_results"] = None

    numeric_fields = (
        "setup_time_sec",
        "workload_wall_time_sec",
        "total_time_sec",
        "perf_time_sec",
        "num_samples",
        "throughput_samples_per_sec",
        "time_per_sample_us",
        "cache_warmup_wall_time_sec",
        "plan_cost",
    )
    for field in numeric_fields:
        values = [
            item[field]
            for item in repeat_results
            if field in item and isinstance(item[field], (int, float))
        ]
        if values:
            result[field] = _mean(values)

    result["statistics"] = {}
    # Student-t critical values for two-sided 95% confidence intervals.
    t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    for field in numeric_fields:
        values = [
            float(item[field])
            for item in repeat_results
            if field in item and isinstance(item[field], (int, float))
        ]
        if not values:
            continue
        mean = _mean(values)
        nonfinite_count = sum(not math.isfinite(value) for value in values)
        if nonfinite_count:
            # Infinity is an intentional result for optimizer timeout/resource
            # failures. Its variance and confidence interval are undefined, but
            # the repeated timeout itself must remain visible in the output.
            result["statistics"][field] = {
                "n": len(values),
                "finite_n": len(values) - nonfinite_count,
                "nonfinite_n": nonfinite_count,
                "mean": mean,
                "stddev": None,
                "ci95_low": None,
                "ci95_high": None,
            }
            continue
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        critical = t95.get(len(values) - 1, 1.96)
        half_width = critical * stddev / (len(values) ** 0.5)
        result["statistics"][field] = {
            "n": len(values),
            "mean": mean,
            "stddev": stddev,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        }

    if all("plan_costs_by_feature" in item for item in repeat_results):
        feature_names = set()
        for item in repeat_results:
            feature_names.update(item["plan_costs_by_feature"].keys())
        result["plan_costs_by_feature"] = {
            feature_name: _mean(
                [
                    item["plan_costs_by_feature"][feature_name]
                    for item in repeat_results
                    if feature_name in item["plan_costs_by_feature"]
                ]
            )
            for feature_name in sorted(feature_names)
        }

    result["cache_warmup_excluded"] = any(
        item.get("cache_warmup_excluded", False) for item in repeat_results
    )
    if any(item.get("workload_skipped", False) for item in repeat_results):
        result["workload_skipped"] = True
        result["skip_reason"] = repeat_results[0].get("skip_reason", "mixed")
    return result


def _workload_runtime_for_timeout(result: Dict[str, Any]) -> Optional[float]:
    if result.get("workload_skipped") or result.get("timed_out"):
        return None
    runtime = result.get("workload_wall_time_sec")
    if isinstance(runtime, (int, float)) and runtime > 0:
        return float(runtime)
    runtime = result.get("perf_time_sec")
    if isinstance(runtime, (int, float)) and runtime > 0:
        return float(runtime)
    return None


def _run_one(
    args: argparse.Namespace,
    optimizer_name: str,
    use_my_optimizer: int,
    reorder_timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    spec = _make_spec(args, use_my_optimizer, reorder_timeout_sec)
    workload_runner = None
    dataset = None
    cache_warmup_wall_time_sec = 0.0

    logger.info("Preparing dataset with %s...", optimizer_name)
    try:
        setup_start = time.perf_counter()
        try:
            with _optimizer_resource_guard(
                args.optimizer_time_limit_sec,
                OPTIMIZER_RSS_LIMIT_BYTES,
            ):
                if args.calculate_plan_cost:
                    module = import_module_from_path(
                        str(Path(args.dataset_file).resolve())
                    )
                    dataset_getter = getattr(module, args.dataset_func)
                    dataset = dataset_getter(spec)
                    plan_costs_by_feature = _calculate_plan_costs(dataset, args.profiled_stats)
                    plan_cost = sum(plan_costs_by_feature.values())
                    logger.info(
                        "%s optimized plan cost (calculate_cost) = %s",
                        optimizer_name,
                        plan_cost,
                    )
                elif args.skip_workload:
                    spec.generate_plan = True
                    module = import_module_from_path(
                        str(Path(args.dataset_file).resolve())
                    )
                    dataset_getter = getattr(module, args.dataset_func)
                    try:
                        dataset_getter(spec)
                    except SystemExit as e:
                        if e.code not in (0, None):
                            raise
                else:
                    workload_runner = _get_workload_runner(
                        args.dataset_file, args.dataset_func, spec
                    )
        except _OptimizerResourceExceeded as e:
            setup_time_sec = time.perf_counter() - setup_start
            logger.warning(
                "%s optimizer overhead too high during setup: %s",
                optimizer_name,
                e.detail,
            )
            return _summarize_optimizer_overhead(
                args,
                optimizer_name,
                setup_time_sec,
                e,
            )
        except MemoryError:
            setup_time_sec = time.perf_counter() - setup_start
            error = _OptimizerResourceExceeded(
                "optimizer_memory_limit_exceeded",
                "optimizer raised MemoryError during setup",
            )
            logger.warning(
                "%s optimizer overhead too high during setup: %s",
                optimizer_name,
                error.detail,
            )
            return _summarize_optimizer_overhead(
                args,
                optimizer_name,
                setup_time_sec,
                error,
            )
        except TimeoutError as e:
            setup_time_sec = time.perf_counter() - setup_start
            if optimizer_name == "optimizer":
                logger.warning(
                    "%s reorder exceeded %.6fs; treating optimizer time as infinity.",
                    optimizer_name,
                    reorder_timeout_sec if reorder_timeout_sec is not None else 0.0,
                )
                return _summarize_timeout(
                    optimizer_name, setup_time_sec, e, reorder_timeout_sec
                )
            raise

        setup_time_sec = time.perf_counter() - setup_start
        if args.calculate_plan_cost:
            logger.info(
                "Skipping workload execution for %s because --calculate_plan_cost "
                "was set; setup/optimization took %.6fs.",
                optimizer_name,
                setup_time_sec,
            )
            return _summarize_setup_only(
                optimizer_name,
                setup_time_sec,
                "calculate_plan_cost",
                plan_cost,
                plan_costs_by_feature,
            )

        if args.plan_only:
            logger.info(
                "Skipping workload execution for %s because --plan_only was set; "
                "setup/optimization took %.6fs.",
                optimizer_name,
                setup_time_sec,
            )
            return _summarize_setup_only(optimizer_name, setup_time_sec, "plan_only")

        if not workload_runner:
            raise RuntimeError(f"Could not create workload runner for {optimizer_name}.")

        logger.info(
            "Running workload with %s; setup/optimization took %.6fs "
            "and is excluded from perf time.",
            optimizer_name,
            setup_time_sec,
        )
        if (
            not args.disable_caching
            and _dataset_has_object_disk_cache(workload_runner.dataset)
        ):
            if _object_disk_cache_is_complete(workload_runner.dataset):
                logger.info(
                    "Reusing complete persistent cache for %s; no warmup needed.",
                    optimizer_name,
                )
            else:
                logger.info(
                    "Plan for %s contains ObjectDiskCachePipe; running one "
                    "unmeasured warmup pass to materialize its persistent cache.",
                    optimizer_name,
                )
                warmup_start = time.perf_counter()
                workload_runner.run()
                cache_warmup_wall_time_sec = time.perf_counter() - warmup_start
                if not _wait_for_object_disk_cache_complete(
                    workload_runner.dataset
                ):
                    raise RuntimeError(
                        f"Cache warmup for {optimizer_name} finished without "
                        "committing every worker shard"
                    )
                logger.info(
                    "Finished cache warmup for %s in %.6fs; measuring from "
                    "the committed cache.",
                    optimizer_name,
                    cache_warmup_wall_time_sec,
                )
                workload_runner = _new_workload_runner_like(workload_runner)

        if args.warmup_runs and args.disable_caching:
            logger.info("Running %d unmeasured warmup pass(es) for %s.", args.warmup_runs, optimizer_name)
            for _ in range(args.warmup_runs):
                workload_runner.run()
                workload_runner.epoch_run_times.clear()
                workload_runner.epoch_num_samples.clear()

        workload_start = time.perf_counter()
        workload_runner.run()
        workload_wall_time_sec = time.perf_counter() - workload_start
        return _summarize_results(
            optimizer_name,
            setup_time_sec,
            workload_wall_time_sec,
            workload_runner.get_results(),
            cache_warmup_wall_time_sec,
        )
    finally:
        _close_workload_runner(workload_runner)
        if dataset is not None and (
            workload_runner is None
            or dataset is not getattr(workload_runner, "dataset", None)
        ):
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        workload_runner = None
        dataset = None
        _cleanup_runtime()


def _write_results(path: str, results: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


def _print_summary(results: Dict[str, Any]) -> None:
    print("\n======Optimizer Performance Comparison======")
    for item in results["runs"]:
        repeat_text = ""
        if item.get("num_repeats", 1) > 1:
            repeat_text = " avg_over={num_repeats}".format(**item)
        if item.get("workload_skipped"):
            cost_text = ""
            if "plan_cost" in item:
                cost_text = ", plan_cost={plan_cost:.6f}".format(**item)
            print(
                "{optimizer}: setup/optimization={setup_time_sec:.6f}s"
                "{cost_text}, workload skipped ({skip_reason}){repeat_text}".format(
                    cost_text=cost_text, repeat_text=repeat_text, **item
                )
            )
            continue
        display_item = dict(item)
        display_item.setdefault("cache_warmup_wall_time_sec", 0.0)
        print(
            "{optimizer}: total={total_time_sec:.6f}s, perf={perf_time_sec:.6f}s, "
            "samples={num_samples}, throughput={throughput_samples_per_sec:.3f} "
            "samples/s, time_per_sample={time_per_sample_us:.3f}us, "
            "setup_excluded={setup_time_sec:.6f}s, "
            "cache_warmup_excluded={cache_warmup_wall_time_sec:.6f}s"
            "{repeat_text}".format(
                repeat_text=repeat_text,
                **display_item,
            )
        )

    by_name = {item["optimizer"]: item for item in results["runs"]}
    base = by_name.get("optimizer")
    dp = by_name.get("dp_optimizer")
    if not (base and dp):
        return

    if results.get("calculate_plan_cost"):
        if "plan_cost" in base and "plan_cost" in dp and dp["plan_cost"] > 0:
            cost_ratio = base["plan_cost"] / dp["plan_cost"]
            print(
                "dp_optimizer cost ratio by calculate_cost: {:.4f}x".format(
                    cost_ratio
                )
            )
        return

    if results.get("plan_only"):
        if dp["setup_time_sec"] > 0:
            speedup = base["setup_time_sec"] / dp["setup_time_sec"]
            print(
                "dp_optimizer speedup by setup/optimization time: {:.4f}x".format(
                    speedup
                )
            )
        return

    if base["perf_time_sec"] > 0 and dp["perf_time_sec"] > 0:
        speedup = base["perf_time_sec"] / dp["perf_time_sec"]
        throughput_ratio = (
            dp["throughput_samples_per_sec"]
            / base["throughput_samples_per_sec"]
            if base["throughput_samples_per_sec"]
            else 0.0
        )
        print(
            "dp_optimizer speedup by perf time: {:.4f}x; "
            "throughput ratio: {:.4f}x".format(speedup, throughput_ratio)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same Cedar workload with all optimizer passes enabled "
            "and compare dj_optimizer.py, optimizer.py, dp_cedar_optimizer.py, "
            "and dp_optimizer.py."
        )
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to Python file defining the Cedar dataset.",
    )
    parser.add_argument(
        "--dataset_func",
        type=str,
        default="get_dataset",
        help="Name of the dataset factory function.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_total_samples", type=int, required=True)
    parser.add_argument(
        "--data_num_total_samples",
        type=int,
        default=1000,
        help=(
            "Maximum samples to execute per optimizer in this comparison. "
            "The optimizer still uses the provided profile; this cap prevents "
            "large optimizer comparisons from becoming full dataset benchmarks. "
            "Use --full_data_run to execute --num_total_samples samples."
        ),
    )
    parser.add_argument(
        "--profiler_num_total_samples",
        dest="data_num_total_samples",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--profiled_stats",
        type=str,
        required=True,
        help="Path to the YAML profile used by both optimizers.",
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        help='Extra kwargs passed to dataset func, e.g. "split=train,path=dir"',
    )
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--ray_ip", type=str, default="")
    parser.add_argument(
        "--iteration_time",
        type=float,
        help="Optional simulated training iteration time per workload step.",
    )
    parser.add_argument(
        "--enable_controller",
        action="store_true",
        help="Enable Cedar controller. Disabled by default for fair optimizer comparison.",
    )
    parser.add_argument(
        "--enable_local_parallelism",
        action="store_true",
        help=(
            "Allow optimized plans to spawn local worker processes. Disabled by "
            "default so optimizer comparisons do not accidentally fork many "
            "Ray-initializing workers in small containers."
        ),
    )
    parser.add_argument(
        "--disable_offload",
        action="store_true",
        help=(
            "Disable offload backends such as RAY/TF_RAY. Useful for running "
            "workload comparisons inside memory-constrained local containers."
        ),
    )
    parser.add_argument(
        "--disable_caching",
        action="store_true",
        help="Disable optimizer cache insertion when constructing plans.",
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=0,
        help="Full unmeasured warmup passes before cache-off measurement.",
    )
    parser.add_argument(
        "--ray_available_parallelism",
        type=int,
        default=None,
        help=(
            "Override Cedar's Ray actor parallelism budget for this comparison. "
            "Lower values reduce Ray memory pressure but may reduce throughput."
        ),
    )
    parser.add_argument(
        "--match_profile_resources",
        action="store_true",
        help=(
            "Require a profile resource signature with one actor per Ray/SMP "
            "stage, then derive local workers from the final physical plan."
        ),
    )
    parser.add_argument(
        "--cpu_budget",
        type=int,
        default=mp.cpu_count(),
        help=(
            "Unified single-node CPU budget used by "
            "--match_profile_resources."
        ),
    )
    parser.add_argument(
        "--fixed_local_workers_ablation",
        type=int,
        default=None,
        help=(
            "Explicit ablation only: fix final local-worker count while "
            "retaining profile-matched Ray/SMP width and CPU validation."
        ),
    )
    parser.add_argument(
        "--full_data_run",
        action="store_true",
        help=(
            "Run the workload for the full --num_total_samples value instead "
            "of the bounded --data_num_total_samples comparison sample."
        ),
    )
    parser.add_argument(
        "--full_profiler_run",
        dest="full_data_run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help=(
            "Only construct the optimized plan and skip workload execution. "
            "By default, the script uses --profiled_stats to optimize and then "
            "executes the workload to measure runtime performance."
        ),
    )
    parser.add_argument(
        "--calculate_plan_cost",
        action="store_true",
        help=(
            "Construct each optimized plan, skip workload execution, and report "
            "Optimizer.calculate_cost for the produced plan."
        ),
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=sorted(OPTIMIZERS.keys()),
        default=DEFAULT_OPTIMIZER_ORDER,
        help=(
            "Optimizer implementations to run. Requested optimizers are always "
            "executed in DEFAULT_OPTIMIZER_ORDER."
        ),
    )
    parser.add_argument(
        "--optimizer_time_limit_sec",
        type=int,
        default=DEFAULT_OPTIMIZER_TIME_LIMIT_SEC,
        help=(
            "Wall-clock limit for optimizer setup/plan generation. Formal "
            "runners pass the same value as their outer plan timeout."
        ),
    )
    parser.add_argument(
        "--cedar_reorder_timeout_sec",
        type=float,
        default=None,
        help=(
            "Explicit timeout for Cedar optimizer reorder. When set, it is "
            "used even for --plan_only or --calculate_plan_cost runs."
        ),
    )
    parser.add_argument(
        "--disable_cedar_runtime_timeout",
        action="store_true",
        help=(
            "Disable the dynamic Cedar reorder timeout derived from other "
            "optimizers' measured workload runtime."
        ),
    )
    parser.add_argument(
        "--cedar_timeout_multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the measured non-Cedar workload runtime "
            "before using it as Cedar's reorder timeout."
        ),
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help=(
            "Number of repeated experiment groups to run for each optimizer. "
            "The reported optimizer result is the average across repeats, and "
            "per-repeat raw results are saved under repeat_results."
        ),
    )
    parser.add_argument(
        "--results_path",
        "-s",
        type=str,
        default="",
        help="Optional JSON output path for detailed comparison results.",
    )
    parser.add_argument(
        "--log_level",
        "-l",
        type=str,
        default="INFO",
    )
    parser.add_argument(
        "--allow_torch_parallelism",
        action="store_true",
        help="Allow torch to use its default thread parallelism.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())
    _normalize_ray_args(args)
    if args.num_repeats < 1:
        raise ValueError("--num_repeats must be >= 1")
    if args.cpu_budget < 1:
        raise ValueError("--cpu_budget must be >= 1")
    if args.fixed_local_workers_ablation is not None:
        if args.fixed_local_workers_ablation < 1:
            raise ValueError("--fixed_local_workers_ablation must be >= 1")
        if not args.match_profile_resources:
            raise ValueError(
                "--fixed_local_workers_ablation requires "
                "--match_profile_resources"
            )
    if args.cedar_timeout_multiplier <= 0:
        raise ValueError("--cedar_timeout_multiplier must be > 0")
    if args.optimizer_time_limit_sec <= 0:
        raise ValueError("--optimizer_time_limit_sec must be > 0")
    if (
        args.cedar_reorder_timeout_sec is not None
        and args.cedar_reorder_timeout_sec <= 0
    ):
        raise ValueError("--cedar_reorder_timeout_sec must be > 0")
    profile_complete = _has_complete_profile(args.profiled_stats)
    if args.use_ray and not args.disable_offload and not _profile_has_offload_variant(args.profiled_stats, "RAY"):
        raise RuntimeError(
            "Ray offload is enabled, but profiled_stats does not contain offloads.RAY. "
            "Run profiling with --use_ray first, or pass --disable_offload."
        )
    args.skip_workload = args.plan_only or args.calculate_plan_cost

    if not args.allow_torch_parallelism:
        logger.warning("Setting torch threads to 1")
        torch.set_num_threads(1)
    _configure_optimizer_runtime(args)
    ray_cluster_resources = _verify_local_ray(args)

    if args.use_ray:
        logger.warning(
            "Ray is enabled for all optimizer repeats (ray_ip=%s).",
            args.ray_ip or "local",
        )

    if args.plan_only:
        logger.warning(
            "--plan_only is set; skipping workload execution after optimizer plan generation."
        )
    if args.calculate_plan_cost:
        logger.warning(
            "--calculate_plan_cost is set; skipping workload execution and "
            "reporting calculate_cost for each optimized plan."
        )
    elif not args.full_data_run and args.data_num_total_samples < args.num_total_samples:
        logger.warning(
            "Capping workload execution to %d/%d samples per optimizer. "
            "Pass --full_data_run to run the full benchmark.",
            args.data_num_total_samples,
            args.num_total_samples,
        )
    if not args.enable_local_parallelism:
        logger.warning(
            "Local parallelism is disabled for this optimizer comparison. "
            "Pass --enable_local_parallelism to benchmark optimized multi-worker plans."
        )

    requested_optimizers = set(args.optimizers)
    optimizer_order = [
        name for name in DEFAULT_OPTIMIZER_ORDER if name in requested_optimizers
    ]
    # Experimental optimizers are deliberately omitted from the default
    # protocol, but an explicit --optimizers request must still execute them.
    optimizer_order.extend(
        name
        for name in args.optimizers
        if name in requested_optimizers and name not in optimizer_order
    )

    repeat_results_by_optimizer = {name: [] for name in optimizer_order}
    execution_order_by_repeat = []
    non_cedar_workload_runtimes = []
    for repeat_idx in range(args.num_repeats):
        offset = repeat_idx % len(optimizer_order)
        repeat_order = optimizer_order[offset:] + optimizer_order[:offset]
        execution_order_by_repeat.append(repeat_order)
        logger.info("Repeat %d optimizer order: %s", repeat_idx + 1, repeat_order)
        for optimizer_name in repeat_order:
            if not args.skip_workload and not args.disable_caching:
                # Treat repeats as independent cache trials.
                _configure_object_disk_cache(args, optimizer_name, reset=True)
            cedar_timeout_sec = None
            if (
                optimizer_name == "optimizer"
                and args.cedar_reorder_timeout_sec is not None
            ):
                cedar_timeout_sec = args.cedar_reorder_timeout_sec
            elif (
                optimizer_name == "optimizer"
                and not args.disable_cedar_runtime_timeout
                and not args.skip_workload
                and non_cedar_workload_runtimes
            ):
                cedar_timeout_sec = (
                    max(non_cedar_workload_runtimes)
                    * args.cedar_timeout_multiplier
                )
            logger.info(
                "Starting %s repeat %d/%d",
                optimizer_name,
                repeat_idx + 1,
                args.num_repeats,
            )
            item = _run_one(
                args,
                optimizer_name,
                OPTIMIZERS[optimizer_name],
                cedar_timeout_sec,
            )
            repeat_results_by_optimizer[optimizer_name].append(item)
            if optimizer_name != "optimizer":
                runtime = _workload_runtime_for_timeout(item)
                if runtime is not None:
                    non_cedar_workload_runtimes.append(runtime)

    runs = [
        _average_repeat_results(name, repeat_results_by_optimizer[name])
        for name in optimizer_order
    ]

    results = {
        "dataset_file": args.dataset_file,
        "dataset_func": args.dataset_func,
        "profiled_stats": args.profiled_stats,
        "all_optimizer_passes_enabled": True,
        "controller_enabled": args.enable_controller,
        "local_parallelism_enabled": args.enable_local_parallelism,
        "offload_enabled": not args.disable_offload,
        "ray_enabled": args.use_ray,
        "ray_ip": args.ray_ip,
        "ray_cluster_resources": ray_cluster_resources,
        "caching_enabled": not args.disable_caching,
        "ray_available_parallelism": args.ray_available_parallelism,
        "match_profile_resources": args.match_profile_resources,
        "fixed_local_workers_ablation": (
            args.fixed_local_workers_ablation
            if args.match_profile_resources
            else None
        ),
        "cpu_budget": (
            args.cpu_budget if args.match_profile_resources else None
        ),
        "optimizer_time_limit_sec": args.optimizer_time_limit_sec,
        "optimizer_rss_limit_bytes": OPTIMIZER_RSS_LIMIT_BYTES,
        "cedar_runs_last": False,
        "execution_order_policy": "round_robin",
        "execution_order_by_repeat": execution_order_by_repeat,
        "cedar_runtime_timeout_enabled": (
            not args.disable_cedar_runtime_timeout and not args.skip_workload
        ),
        "cedar_timeout_multiplier": args.cedar_timeout_multiplier,
        "profile_complete": profile_complete,
        "plan_only": args.plan_only,
        "calculate_plan_cost": args.calculate_plan_cost,
        "workload_skipped": args.skip_workload,
        "num_repeats": args.num_repeats,
        "requested_num_total_samples": args.num_total_samples,
        "data_num_total_samples": (
            args.num_total_samples
            if args.full_data_run
            else min(args.num_total_samples, args.data_num_total_samples)
        ),
        "perf_excludes_setup_and_optimization": True,
        "perf_excludes_cache_warmup": not args.disable_caching,
        "runs": runs,
    }

    _print_summary(results)
    if args.results_path:
        _write_results(args.results_path, results)
        logger.info("Wrote comparison results to %s", args.results_path)


if __name__ == "__main__":
    main()
