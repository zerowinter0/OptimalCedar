"""Run one workload's optimizer comparison with a bounded Cedar timeout."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
ADDRESS = "172.23.166.105:6379"
OPTIMIZERS = ["optimizer", "plumber_optimizer", "dp_optimizer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--dataset-kwargs", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cedar-timeout", type=float, default=600.0)
    parser.add_argument("--optimizer-limit", type=float, default=1800.0)
    args = parser.parse_args()

    profile = args.output / f"{args.workload}_profile.yaml"
    results = args.output / "results" / f"{args.workload}.json"
    log = args.output / "logs" / f"compare_{args.workload}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
        "CEDAR_MATCH_PROFILE_RESOURCES": "1",
        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": "8",
        "CEDAR_PROFILE_MATCH_CPU_BUDGET": "64",
        "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
    })
    cmd = [
        sys.executable, "-u", str(args.output / "entry.py"),
        "evaluation/compare_optimizer_perf.py",
        "--dataset_file", args.dataset_file,
        "--dataset_kwargs", args.dataset_kwargs,
        "--batch_size", str(args.batch_size),
        "--num_total_samples", str(args.samples),
        "--full_data_run",
        "--profiled_stats", str(profile),
        "--use_ray", "--ray_ip", ADDRESS,
        "--enable_local_parallelism",
        "--disable_caching",
        "--match_profile_resources",
        "--cpu_budget", "64",
        "--ray_cpu_budget", "64",
        "--fixed_local_workers_ablation", "8",
        "--optimizers", *OPTIMIZERS,
        "--optimizer_time_limit_sec", str(int(args.optimizer_limit)),
        "--cedar_reorder_timeout_sec", str(int(args.cedar_timeout)),
        "--num_repeats", str(args.repeats),
        "--results_path", str(results),
    ]
    with log.open("w") as handle:
        code = subprocess.run(
            cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
        ).returncode
    print(f"[compare_one] {args.workload} exit={code} log={log}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
