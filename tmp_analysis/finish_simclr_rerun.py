"""Finish the SimCLRv2 redo once the wide-width calibration profile lands.

Steps (all offline, no online observation needed):

  1. wait for ``outputs/simclr_profile_wide_scaling/simclr_profile.yaml``;
  2. merge the transport and worker-contention calibrations from the previously
     calibrated profile into it;
  3. re-measure the four planner plans plus the unoptimized plan under the new
     profile (three repeats, no fixed worker count);
  4. score every plan with every system cost model and re-render both figure
     variants;
  5. write ``STATUS.json`` next to the results.

Run:  nohup python -u tmp_analysis/finish_simclr_rerun.py > tmp_analysis/simclr_rerun.log 2>&1 &
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
WIDE = ROOT / "outputs/simclr_profile_wide_scaling/simclr_profile.yaml"
CALIBRATED = ROOT / "outputs/pico_default_profiles/simclr_profile.yaml"
OUT = ROOT / "outputs/simclr_rerun_wide_20260913"
ENTRY = ROOT / "outputs/plumber_bench_20260912/entry.py"
ADDRESS = "172.23.166.105:6379"
DATASET = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
OPTIMIZERS = ["raydata_optimizer", "optimizer", "plumber_optimizer", "dp_optimizer"]
REPEATS = 3
SAMPLES = 1000


def log(message):
    print(f"[rerun {time.strftime('%H:%M:%S')}] {message}", flush=True)


def status(state, payload=None):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.json").write_text(
        json.dumps(
            {"state": state, "utc": time.strftime("%Y-%m-%d %H:%M:%S"), **(payload or {})},
            indent=2,
        )
    )


def wait_for_profile(timeout_sec=5400):
    started = time.time()
    while time.time() - started < timeout_sec:
        if WIDE.is_file() and WIDE.stat().st_size > 20000:
            return True
        time.sleep(30)
    return False


def merge_calibration():
    wide = yaml.safe_load(WIDE.read_text())
    calibrated = yaml.safe_load(CALIBRATED.read_text())
    source = calibrated.get("physical_model", {})
    target = wide.setdefault("physical_model", {})
    for key in ("transport", "worker_contention"):
        if key in source:
            target[key] = source[key]
    merged = OUT / "simclr_profile.yaml"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text(yaml.safe_dump(wide, sort_keys=False))
    return merged


def run(command, log_path, env, timeout=3600):
    started = time.time()
    try:
        with log_path.open("w") as handle:
            code = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env=env, timeout=timeout,
            ).returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        code, timed_out = None, True
    return code, timed_out, round(time.time() - started, 1)


def measure_plan(profile, optimizer, results):
    command = [
        sys.executable, "-u", str(ENTRY), "evaluation/compare_optimizer_perf.py",
        "--dataset_file", DATASET,
        "--batch_size", "4",
        "--num_total_samples", str(SAMPLES),
        "--full_data_run",
        "--profiled_stats", str(profile),
        "--use_ray", "--ray_ip", ADDRESS,
        "--enable_local_parallelism",
        "--disable_caching",
        "--match_profile_resources",
        "--cpu_budget", "64",
        "--ray_cpu_budget", "64",
        "--optimizers", optimizer,
        "--optimizer_time_limit_sec", "1800",
        "--cedar_reorder_timeout_sec", "900",
        "--disable_cedar_runtime_timeout",
        "--num_repeats", str(REPEATS),
        "--results_path", str(results),
    ]
    env = dict(os.environ)
    env["CEDAR_RAY_PLACEMENT_RESOURCE"] = "cedar_remote"
    return run(command, OUT / "logs" / f"{optimizer}.log", env)


def measure_unoptimized(profile):
    plan = ROOT / "tmp_analysis/plan_runs/plumber_serial_reconcile.yaml"
    times = []
    env = dict(os.environ)
    env.update(
        {
            "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
            "CEDAR_MATCH_PROFILE_RESOURCES": "1",
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": "8",
            "CEDAR_PROFILE_MATCH_CPU_BUDGET": "64",
            "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
        }
    )
    for index in range(REPEATS):
        command = [
            sys.executable, "-u", str(ENTRY), "evaluation/eval_cedar.py",
            "--dataset_file", DATASET,
            "--batch_size", "4",
            "--num_total_samples", str(SAMPLES),
            "--profiled_stats", str(profile),
            "--master_feature_config", str(plan),
            "--use_ray", "--ray_ip", ADDRESS,
            "--disable_controller", "--disable_caching",
        ]
        log_path = OUT / "logs" / f"unoptimized_{index}.log"
        code, timed_out, seconds = run(command, log_path, env)
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        value = None
        for line in text.splitlines():
            if line.startswith("Total time: "):
                try:
                    value = float(line.split()[2].rstrip("s"))
                except (IndexError, ValueError):
                    value = None
        log(f"unoptimized run {index}: exit={code} total={value}")
        if value:
            times.append(value)
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    status("waiting_for_wide_profile")
    if not wait_for_profile():
        status("failed: wide profile never appeared")
        return
    profile = merge_calibration()
    log(f"merged calibration into {profile}")
    status("measuring", {"profile": str(profile)})

    measured = {}
    for optimizer, key in (
        ("dp_optimizer", "pico"),
        ("optimizer", "cedar"),
        ("plumber_optimizer", "plumber"),
        ("raydata_optimizer", "raydata"),
    ):
        results = OUT / "results" / f"{optimizer}.json"
        code, timed_out, seconds = measure_plan(profile, optimizer, results)
        value = None
        if results.is_file():
            payload = json.loads(results.read_text())
            for entry in payload.get("runs", []):
                value = entry.get("perf_time_sec")
        log(f"{key}: exit={code} timed_out={timed_out} perf={value}")
        if value:
            measured[key] = value
        status("measuring", {"profile": str(profile), "measured": measured})

    unoptimized = measure_unoptimized(profile)
    if unoptimized:
        measured["unoptimized"] = unoptimized
    (OUT / "measured.json").write_text(json.dumps(measured, indent=2))
    status("scoring", {"measured": measured})

    env = dict(os.environ)
    env["CEDAR_MEASURED_JSON"] = str(OUT / "measured.json")
    matrix = ROOT / "outputs/plumber_bench_20260912/system_cost_matrix.json"
    for command in (
        [sys.executable, "-u", "tmp_analysis/score_plans_all_models.py",
         str(profile), str(matrix)],
        [sys.executable, "tmp_analysis/make_system_estimates_figure.py",
         "--basis", "baseline", "--matrix", str(matrix),
         "--out", "simclr_model_estimates"],
        [sys.executable, "tmp_analysis/make_system_estimates_figure.py",
         "--basis", "affine", "--matrix", str(matrix),
         "--out", "simclr_model_estimates_affine"],
    ):
        code, _, seconds = run(command, OUT / "logs" / "scoring.log", env, timeout=1800)
        log(f"{command[1]}: exit={code} wall={seconds}s")
    for suffix in ("pdf", "png"):
        source = ROOT / f"outputs/plumber_bench_20260912/figures/simclr_model_estimates.{suffix}"
        if source.is_file():
            shutil.copy2(
                source,
                ROOT / f"my_paper/69e75a0100d7b4afeb1cfc20/figures/simclr_model_estimates.{suffix}",
            )
            shutil.copy2(
                source,
                ROOT / f"outputs/plumber_bench_20260912/figures/simclr_model_estimates_trace.{suffix}",
            )
    status("completed", {"measured": measured})
    log("done")


if __name__ == "__main__":
    main()
