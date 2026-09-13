"""Run the same four-optimizer comparison under both profiles back to back."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
REF = ROOT / "outputs/simclrv2_four_remote_20260911"
NEW = ROOT / "outputs/simclrv2_scaling_20260911"
OUT = ROOT / "outputs/profile_ab_20260911"
ADDRESS = "172.23.166.105:6379"
DATA = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
NAMES = ["dp_optimizer", "old_dp_optimizer", "cm_optimizer", "optimizer"]


def status(state, **kwargs):
    (OUT / "status.json").write_text(
        json.dumps(
            dict(
                state=state,
                utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                **kwargs,
            ),
            indent=2,
        )
    )


def run(tag, profile):
    log_path = OUT / "logs" / (tag + ".log")
    env = dict(os.environ)
    env["CEDAR_RAY_PLACEMENT_RESOURCE"] = "cedar_remote"
    cmd = [
        sys.executable,
        "-u",
        str(REF / "entry.py"),
        "evaluation/compare_optimizer_perf.py",
        "--dataset_file",
        DATA,
        "--batch_size",
        "1",
        "--num_total_samples",
        "9469",
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
        "--fixed_local_workers_ablation",
        "8",
        "--optimizers",
        *NAMES,
        "--optimizer_time_limit_sec",
        "3600",
        "--cedar_reorder_timeout_sec",
        "3600",
        "--disable_cedar_runtime_timeout",
        "--num_repeats",
        "3",
        "--results_path",
        str(OUT / "results" / f"{tag}.json"),
    ]
    with log_path.open("w") as handle:
        code = subprocess.run(
            cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
        ).returncode
    print(f"[AB] {tag} exit={code} log={log_path}", flush=True)
    return code


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    status("old_profile")
    run("old_profile", REF / "profile.yaml")
    status("new_profile")
    run("new_profile", NEW / "profile.yaml")
    status("completed")


if __name__ == "__main__":
    main()
