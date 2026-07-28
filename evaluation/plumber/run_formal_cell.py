"""Profile and benchmark one Plumber cell in a single callback registry."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import tensorflow as tf
from plumber_analysis import gen_util, pipeline_optimizer

from evaluation.plumber.profile_pipeline import _import_dataset
from evaluation.tf_utils import TFEvalSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--dataset-kwargs", type=json.loads, default={})
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--profile-samples", type=int, default=1000)
    parser.add_argument("--profile-seconds", type=int, default=10)
    parser.add_argument("--benchmark-seconds", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    module = _import_dataset(args.dataset_file)
    dataset = module.get_dataset(
        TFEvalSpec(
            batch_size=1,
            num_parallel_calls=1,
            num_total_samples=args.profile_samples,
            kwargs=args.dataset_kwargs,
        )
    )
    dataset = dataset.take(args.profile_samples)
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 64
    options.experimental_optimization.map_and_batch_fusion = True
    args.stats_file.parent.mkdir(parents=True, exist_ok=True)
    local_stats = Path.cwd() / f"plumber-stats-{uuid.uuid4().hex}.pb"
    local_temp = local_stats.with_name(f".{local_stats.name}")
    gen_util.add_analysis_to_dataset_options(
        options, hard_fail=True, stats_filename=local_stats.name
    )
    profile_start = time.perf_counter()
    try:
        profiled = dataset.with_options(options)
        gen_util.benchmark_and_profile_dataset(
            profiled, time_limit_s=args.profile_seconds
        )
        if local_stats.is_file():
            shutil.move(str(local_stats), str(args.stats_file.resolve()))
    finally:
        local_stats.unlink(missing_ok=True)
        local_temp.unlink(missing_ok=True)
    profile_time = time.perf_counter() - profile_start
    if not args.stats_file.is_file():
        raise RuntimeError(f"Plumber produced no stats file: {args.stats_file}")

    optimization_start = time.perf_counter()
    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(
        str(args.stats_file.resolve())
    )
    optimizer = pipeline_optimizer.DataPipelineOptimizer(
        plumber, calibrate_system=False, step_size=None
    )
    optimizer.apply_parallelism()
    if args.cache:
        optimizer.apply_cache(add_take_repeat=False)
    optimized = optimizer.instantiate_pipeline().batch(1)
    analyzed = tf.data.Options()
    gen_util.add_analysis_to_dataset_options(analyzed, hard_fail=True)
    optimized = optimized.with_options(analyzed)
    optimization_time = time.perf_counter() - optimization_start

    measurement_start = time.perf_counter()
    summary = gen_util.benchmark_dataset(
        optimized, time_limit_s=args.benchmark_seconds
    )
    measured_benchmark_time = time.perf_counter() - measurement_start
    throughput = float(summary["global_minibatch_rate"])
    payload = {
        "schema_version": 1,
        "system": "plumber",
        "num_samples": args.num_samples,
        "profile_time_sec": profile_time,
        "optimization_time_sec": optimization_time,
        "measured_benchmark_time_sec": measured_benchmark_time,
        "throughput_samples_per_sec": throughput,
        "measured_time_sec": args.num_samples / throughput if throughput > 0 else None,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
