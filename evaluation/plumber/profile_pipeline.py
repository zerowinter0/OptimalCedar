"""Collect a Plumber performance model for an OptimalCedar tf.data pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import tensorflow as tf
from plumber_analysis import gen_util

from evaluation.tf_utils import TFEvalSpec


REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_dataset(path: Path):
    rel = path.resolve().relative_to(REPO_ROOT)
    name = ".".join(rel.with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--dataset-kwargs", type=json.loads, default={})
    parser.add_argument("--profile-samples", type=int, default=1000)
    parser.add_argument("--profile-seconds", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--threadpool-size", type=int, default=64)
    args = parser.parse_args()

    module = _import_dataset(args.dataset_file)
    dataset = module.get_dataset(
        TFEvalSpec(
            batch_size=1,
            num_parallel_calls=args.parallelism,
            num_total_samples=args.profile_samples,
            kwargs=args.dataset_kwargs,
        )
    )
    dataset = dataset.take(args.profile_samples)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = (
        args.threadpool_size
    )
    options.experimental_optimization.map_and_batch_fusion = True
    stats_file = args.stats_file.resolve()
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    # Plumber prepends "." directly to the configured filename for its
    # temporary file, so an absolute path becomes an invalid "./abs/path".
    # Profile to a unique basename in cwd and move the completed snapshot.
    local_stats = Path.cwd() / f"plumber-stats-{uuid.uuid4().hex}.pb"
    local_temp = local_stats.with_name(f".{local_stats.name}")
    gen_util.add_analysis_to_dataset_options(
        options,
        hard_fail=True,
        stats_filename=local_stats.name,
    )
    try:
        dataset = dataset.with_options(options)
        gen_util.benchmark_and_profile_dataset(
            dataset,
            time_limit_s=args.profile_seconds,
        )
        if local_stats.is_file():
            local_stats.replace(stats_file)
    finally:
        local_stats.unlink(missing_ok=True)
        local_temp.unlink(missing_ok=True)
    if not stats_file.is_file():
        raise RuntimeError(
            f"Plumber profiling produced no model at {stats_file}"
        )


if __name__ == "__main__":
    main()
