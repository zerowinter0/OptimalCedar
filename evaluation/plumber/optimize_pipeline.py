"""Apply Plumber's native graph optimizer to a collected performance model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from plumber_analysis import gen_util, pipeline_optimizer


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--benchmark-seconds", type=int, default=42)
    parser.add_argument(
        "--skip-system-calibration",
        action="store_true",
        help="Skip Plumber filesystem/system calibration diagnostics.",
    )
    args = parser.parse_args()

    plumber = tf.data.experimental.analysis.PlumberPerformanceModel(
        str(args.stats_file.resolve())
    )
    optimizer = pipeline_optimizer.DataPipelineOptimizer(
        plumber,
        calibrate_system=not args.skip_system_calibration,
        step_size=None,
    )
    before = optimizer.get_performance_parameters()
    optimizer.apply_optimizations()
    dataset = optimizer.instantiate_pipeline()
    # benchmark_dataset interprets the first dimension as a batch dimension.
    # The reconstructed pipelines are normally unbatched, so batch explicitly
    # to make global_minibatch_rate a sample rate.
    dataset = dataset.batch(1)

    options = tf.data.Options()
    gen_util.add_analysis_to_dataset_options(options, hard_fail=True)
    dataset = dataset.with_options(options)
    summary = gen_util.benchmark_dataset(
        dataset,
        time_limit_s=args.benchmark_seconds,
    )

    result = {
        "schema_version": 1,
        "system": "plumber",
        "stats_file": str(args.stats_file.resolve()),
        "performance_parameters_before": _json_value(before),
        "benchmark_summary": _json_value(summary),
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
