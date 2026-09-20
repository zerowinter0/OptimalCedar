"""Execute one materialized physical plan and report its steady-state rate.

The plan is loaded through ``CedarEvalSpec.config`` (``DataSet(feature_config=...)``),
so no optimizer or profiling pass runs: the measured numbers describe exactly
the supplied plan.  Metric definitions match
``evaluation/compare_optimizer_perf.py`` so the numbers are comparable with the
campaign results: steady-state throughput = processed samples / sum of epoch
run times, with the dataset construction (Ray connect, worker startup, plan
load) reported separately as setup.

Usage (inside the container):
  python -u scripts/run_fixed_plan_throughput.py \
      --plan <plan.yaml> --label <name> --results-path <out.json> \
      --dataset-file <pipeline.py> --dataset-kwargs dataset_path=... \
      --batch-size 4 --num-epochs 1 --num-total-samples 0 \
      --profiled-stats <shared.yaml> --ray-ip 172.23.166.105:6379
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cedar_utils import CedarEvalSpec  # noqa: E402
from evaluation.eval_cedar import _get_profiler  # noqa: E402


def parse_kwargs(raw: str) -> dict:
    kwargs = {}
    for part in filter(None, (item.strip() for item in raw.split(","))):
        key, _, value = part.partition("=")
        kwargs[key] = value
    return kwargs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--dataset-func", default="get_dataset")
    parser.add_argument("--dataset-kwargs", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--num-total-samples",
        type=int,
        default=0,
        help="0 processes the whole input pass, like the campaign runs",
    )
    parser.add_argument("--profiled-stats", type=Path, required=True)
    parser.add_argument("--ray-ip", default="")
    parser.add_argument("--warmup-passes", type=int, default=0)
    args = parser.parse_args()

    logging.info("Running fixed plan %s (%s)", args.plan, args.label)
    spec = CedarEvalSpec(
        batch_size=args.batch_size,
        num_total_samples=args.num_total_samples,
        num_epochs=args.num_epochs,
        config=str(args.plan),
        kwargs=parse_kwargs(args.dataset_kwargs),
        use_ray=bool(args.ray_ip),
        ray_ip=args.ray_ip,
        profiled_stats=str(args.profiled_stats),
        disable_optimizer=True,
        disable_controller=True,
        disable_caching=True,
    )

    setup_start = time.perf_counter()
    runner = _get_profiler(str(args.dataset_file), args.dataset_func, spec)
    setup_time_sec = time.perf_counter() - setup_start
    try:
        for _ in range(args.warmup_passes):
            runner.run()
            runner.epoch_run_times.clear()
            runner.epoch_num_samples.clear()
        workload_start = time.perf_counter()
        runner.run()
        workload_wall_time_sec = time.perf_counter() - workload_start
        raw = runner.get_results()
    finally:
        runner.close()

    perf_time_sec = sum(raw["epoch_run_times"])
    num_samples = sum(raw["epoch_num_samples"])
    throughput = num_samples / perf_time_sec if perf_time_sec else 0.0
    summary = {
        "optimizer": args.label,
        "plan_path": str(args.plan),
        "plan_sha256": sha256(args.plan),
        "profile_path": str(args.profiled_stats),
        "profile_sha256": sha256(args.profiled_stats),
        "setup_time_sec": setup_time_sec,
        "workload_wall_time_sec": workload_wall_time_sec,
        "total_time_sec": setup_time_sec + workload_wall_time_sec,
        "perf_time_sec": perf_time_sec,
        "num_samples": num_samples,
        "throughput_samples_per_sec": throughput,
        "time_per_sample_us": (
            (perf_time_sec / num_samples) * 1e6 if num_samples else 0.0
        ),
        "epoch_run_times": list(raw["epoch_run_times"]),
        "epoch_num_samples": list(raw["epoch_num_samples"]),
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(summary, indent=2))
    print(
        f"{args.label}: samples={num_samples} perf={perf_time_sec:.3f}s "
        f"throughput={throughput:.2f} samples/s setup={setup_time_sec:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
