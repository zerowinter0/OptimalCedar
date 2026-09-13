"""Protocol B: every optimizer configures its own local worker count.

Unlike the fixed-W protocol (``--fixed_local_workers_ablation 8``), this run
lets each optimizer decide how many local workers to use under the same
64-local-CPU / 64-Ray-CPU budget:

  * Cedar          : its own rule (all available CPUs), clamped to the largest
                     worker count whose plan still fits the per-worker budget
                     (``CEDAR_LOCAL_WORKERS_MAX``).
  * Plumber-style  : ``CEDAR_PLUMBER_WORKER_SEARCH=1`` - extend its rate model
                     with the worker count, maximize W * min_i theta_i R_i.
  * PICO           : ``CEDAR_DP_WORKER_SEARCH=1`` - run the DP once per worker
                     count and keep the one minimizing cost(W) / W.

Every optimizer is evaluated on the same profile, samples, repeats and Ray
cluster; results and the chosen worker counts are written next to the logs.

Usage (inside the container):
  python -u tmp_analysis/run_worker_search_bench.py \
      --output outputs/worker_search_20260912 --workloads simclr \
      --samples 1000 --repeats 3
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
ADDRESS = "172.23.166.105:6379"
PROFILE_DIR = ROOT / "outputs/plumber_bench_20260912"
ENTRY = PROFILE_DIR / "entry.py"
DATASET = "evaluation/pipelines/target_pipeline"
WORKLOADS = {
    "simclr": "simclr/cedar_dataset.py",
    "blip": "blip/cedar_dataset.py",
    "clip": "clip/cedar_dataset.py",
    "dino": "dino/cedar_dataset.py",
    "swav": "swav/cedar_dataset.py",
}
OPTIMIZERS = ["optimizer", "plumber_optimizer", "dp_optimizer"]


def workload_args(workload, profile_dir):
    args = [f"workload={workload}"]
    manifests = [
        profile_dir / "datasets" / f"{workload}.jsonl",
        ROOT / "datasets/target_pipeline_bench" / f"{workload}.jsonl",
    ]
    if workload != "simclr":
        for manifest in manifests:
            if manifest.is_file():
                args.append(f"dataset_path={manifest}")
                break
    return str(ROOT / DATASET / WORKLOADS[workload]), args


def parse_choices(log_path):
    """Extract the worker count each optimizer chose from its log."""
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    choices = {}
    match = re.search(r"\[DpOptimizer\] Worker search selected W=(\d+)", text)
    if match:
        choices["dp_optimizer"] = int(match.group(1))
    match = re.search(r"\[Plumber\] Worker search picked W=(\d+)", text)
    if match:
        choices["plumber_optimizer"] = int(match.group(1))
    capped = re.findall(r"Capping local workers from (\d+) to (\d+)", text)
    if capped:
        choices["optimizer"] = int(capped[-1][1])
    else:
        used = re.findall(r"\[Parallelism\] Using (\d+) local workers", text)
        if used:
            choices["optimizer"] = int(used[-1])
    return choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", default=["simclr"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--worker-set", default="2,4,8,16,32")
    parser.add_argument("--cedar-workers-max", type=int, default=32)
    parser.add_argument("--optimizer-limit", type=float, default=1800.0)
    parser.add_argument("--cedar-timeout", type=float, default=600.0)
    parser.add_argument("--dp-search-time-limit", type=float, default=900.0)
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--dataset-kwargs",
        default="",
        help=(
            "Extra dataset factory kwargs, e.g. 'views=2' for dino/swav or "
            "'tokenizer_path=...' for clip."
        ),
    )
    parser.add_argument(
        "--fixed-workers",
        type=int,
        default=None,
        help=(
            "Run the fixed-W control protocol instead: pin every optimizer to "
            "this worker count and disable the per-optimizer worker search."
        ),
    )
    args = parser.parse_args()

    out = args.output if args.output.is_absolute() else (ROOT / args.output)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    extra_env = {
        "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
    }
    if args.fixed_workers is None:
        extra_env.update(
            {
                "CEDAR_LOCAL_WORKERS_MAX": str(args.cedar_workers_max),
                "CEDAR_PLUMBER_WORKER_SEARCH": "1",
                "CEDAR_DP_WORKER_SEARCH": "1",
                "CEDAR_WORKER_SEARCH_SET": str(args.worker_set),
                "CEDAR_DP_WORKER_SEARCH_TIME_LIMIT_SEC": str(
                    args.dp_search_time_limit
                ),
            }
        )
    summary = {
        "protocol": "worker_search",
        "worker_set": args.worker_set,
        "cedar_workers_max": args.cedar_workers_max,
        "samples": args.samples,
        "repeats": args.repeats,
        "workloads": {},
    }
    for workload in args.workloads:
        profile = args.profile_dir / f"{workload}_profile.yaml"
        if not profile.is_file():
            summary["workloads"][workload] = {"status": "no_profile"}
            print(f"[worker-search] {workload}: missing {profile}", flush=True)
            continue
        dataset_file, kwargs = workload_args(workload, args.profile_dir)
        if args.dataset_kwargs.strip():
            kwargs.extend(
                token for token in args.dataset_kwargs.split(",") if token
            )
        kwargs_flag = ["--dataset_kwargs", ",".join(kwargs)] if kwargs else []
        results = out / "results" / f"{workload}.json"
        log = out / "logs" / f"compare_{workload}.log"
        command = [
            sys.executable,
            "-u",
            str(ENTRY),
            "evaluation/compare_optimizer_perf.py",
            "--dataset_file",
            dataset_file,
            *kwargs_flag,
            "--batch_size",
            str(args.batch_size),
            "--num_total_samples",
            str(args.samples),
            "--full_data_run",
            "--profiled_stats",
            str(profile),
            "--use_ray",
            "--ray_ip",
            ADDRESS,
            "--enable_local_parallelism",
            "--disable_caching",
            "--match_profile_resources",
            "--cpu_budget",
            "64",
            "--ray_cpu_budget",
            "64",
            "--optimizers",
            *OPTIMIZERS,
            "--optimizer_time_limit_sec",
            str(int(args.optimizer_limit)),
            "--cedar_reorder_timeout_sec",
            str(int(args.cedar_timeout)),
            "--disable_cedar_runtime_timeout",
            "--num_repeats",
            str(args.repeats),
            "--results_path",
            str(results),
        ]
        if args.plan_only:
            command.append("--plan_only")
        if args.fixed_workers is not None:
            command.extend(
                ["--fixed_local_workers_ablation", str(args.fixed_workers)]
            )
            env_override = {
                "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": str(
                    args.fixed_workers
                )
            }
        else:
            env_override = {}
        env = dict(os.environ)
        env.update(extra_env)
        env.update(env_override)
        started = time.time()
        with log.open("w") as handle:
            code = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            ).returncode
        entry = {
            "status": "ok" if code == 0 else "failed",
            "exit_code": code,
            "wall_sec": round(time.time() - started, 1),
            "chosen_workers": parse_choices(log),
        }
        if results.is_file():
            payload = json.loads(results.read_text())
            entry["runs"] = {
                run["optimizer"]: {
                    "perf_time_sec": run.get("perf_time_sec"),
                    "plan_cost": run.get("plan_cost"),
                }
                for run in payload.get("runs", [])
            }
        summary["workloads"][workload] = entry
        (out / "status.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[worker-search] {workload} exit={code} "
            f"chosen={entry['chosen_workers']} log={log}",
            flush=True,
        )


if __name__ == "__main__":
    main()
