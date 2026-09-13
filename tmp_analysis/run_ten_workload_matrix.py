"""Ten-workload matrix: seven planners, one run per cell, fully offline.

Goal
----
Show where PICO's cost model + joint DP beats the alternatives on a *diverse*
set of pipelines, using the local host and the configured remote Ray node with
no artificial resource cap:

  * the local worker count W is **not** fixed: every planner configures the
    resources it optimises (Cedar's own rule is clamped to the largest
    feasible worker count by the harness);
  * `--cpu_budget` is the host's CPU count and `--ray_cpu_budget` the Ray
    node's, so both machines are usable;
  * one repeat per (workload, optimizer); the data volume is as large as the
    dataset allows while a single cell must stay under 60 minutes, enforced by
    an outer timeout of 3600 s and a 1800 s planning cap.
  * every planner uses the best cost model it has.  The baseline planners
    (Cedar, Plumber, Pecan, Data-Juicer, Simple-DP) price operators from the
    profile's trace latencies, which is the data their profilers expose; PICO
    keeps its full model (isolated affine operator costs plus the calibrated
    width / boundary / cross-host transport layers).  That is the realistic
    comparison: nobody is handed a model it does not have.

Workloads (ten, chosen for diversity across modalities and pipeline shapes)
-------------------------------------------------------------------------
  simclr                  image augmentation chain (9 operators)
  blip                    image-to-caption pipeline
  clip                    image + tokenizer pipeline
  dino                    multi-view image recipe (19 operators)
  alpaca_cot              short text filter chain
  pile_hackernews         long text filtering pipeline (18 operators)
  pile_pubmed_abstracts   Data-Juicer Hub recipe (text curation)
  pile_uspto_backgrounds  Data-Juicer Hub recipe (text curation)
  bloom_oscar             large text curation with model-based filters

Optimizers: DjTwoStage (Data-Juicer ordering + Cedar passes), PecanTwoStage,
Plumber-style widths, Ray Data-style placement, Cedar, PICO (dp_optimizer),
Simple-DP.

Run (offline, inside the container):
  nohup python -u tmp_analysis/run_ten_workload_matrix.py \
      --output outputs/pico_ten_workloads_trace_20260913 \
      > tmp_analysis/ten_workloads_trace.log 2>&1 &
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
ADDRESS = "172.23.166.105:6379"
REFERENCE = ROOT / "outputs/plumber_bench_20260912"
ENTRY = REFERENCE / "entry.py"
MANIFESTS = ROOT / "datasets/target_pipeline_bench"
TARGET = "evaluation/pipelines/target_pipeline"

# name -> (dataset file, [dataset kwargs], samples)
WORKLOADS = [
    ("simclr", f"{TARGET}/simclr/cedar_dataset.py", ["workload=simclr"], 8000),
    ("blip", f"{TARGET}/blip/cedar_dataset.py",
     ["workload=blip", f"dataset_path={MANIFESTS}/blip.jsonl"], 8000),
    ("clip", f"{TARGET}/clip/cedar_dataset.py",
     ["workload=clip", f"dataset_path={MANIFESTS}/clip.jsonl"], 8000),
    # ``dino`` is profiled fresh: the reuse copy was collected from a
    # different pipeline shape (75 pipes) and does not match this feature.
    ("dino", f"{TARGET}/dino/cedar_dataset.py",
     ["workload=dino", f"dataset_path={MANIFESTS}/dino.jsonl", "views=2"], 3000),
    ("alpaca_cot", "evaluation/pipelines/alpaca_cot/cedar_dataset.py", [], 20000),
    ("pile_hackernews", "evaluation/pipelines/pile_hackernews/cedar_dataset.py", [],
     20000),
    ("pile_pubmed_abstracts",
     f"{TARGET}/hub/pile_pubmed_abstracts/cedar_dataset.py", [], 20000),
    ("pile_uspto_backgrounds",
     f"{TARGET}/hub/pile_uspto_backgrounds/cedar_dataset.py", [], 20000),
    ("bloom_oscar", "evaluation/pipelines/bloom_oscar/cedar_dataset.py", [], 20000),
]

OPTIMIZERS = [
    "dj_two_stage_optimizer",
    "pecan_two_stage_optimizer",
    "plumber_optimizer",
    "raydata_optimizer",
    "optimizer",
    "dp_optimizer",
    "simple_dp_optimizer",
]

# Profiles produced by earlier runs, reused so every planner sees one profile.
REUSE_PROFILE = {
    "simclr": REFERENCE / "simclr_profile.yaml",
    "blip": REFERENCE / "blip_profile.yaml",
    "clip": REFERENCE / "clip_profile.yaml",
    # dino's reuse copy came from a different pipeline shape; profile it fresh.
    "alpaca_cot": ROOT / "outputs/plumber_bench_text_20260912/alpaca_cot_profile.yaml",
    "pile_hackernews": ROOT
    / "outputs/plumber_bench_text_20260912/pile_hackernews_profile.yaml",
}

CELL_TIMEOUT_SEC = 3600
PLANNER_TIME_LIMIT_SEC = 1800
CEDAR_REORDER_TIMEOUT_SEC = 900
PROFILE_TIME_SEC = 10.0
# Host-level cross-host payload calibration (per-worker rate, shared link
# rate, per-worker prefetch budget, overflow penalty).  It is a property of
# the machine pair, not of one workload, so every workload profile PICO uses
# carries the same measured values; without them the DP would fall back to the
# optimistic isolated-echo floor.
CROSS_HOST_CALIBRATION = ROOT / "outputs/cross_host_inplan_20260913/cross_host.json"
WORKER_CONTENTION_SOURCE = (
    ROOT / "outputs/pico_default_profiles/simclr_profile.yaml"
)


def merge_host_calibration(profile_path):
    """Copy host-level calibration blocks into one workload profile."""
    import yaml

    calibration = json.loads(CROSS_HOST_CALIBRATION.read_text())
    data = yaml.safe_load(profile_path.read_text())
    transport = data.setdefault("physical_model", {}).setdefault("transport", {})
    transport.setdefault(
        "cross_host_bytes_per_worker_per_sec",
        round(calibration["worker_payload_bytes_per_sec"], 1),
    )
    transport.setdefault("cross_host_link_bytes_per_sec", 200_000_000.0)
    transport.setdefault(
        "cross_host_prefetch_budget_bytes",
        calibration["prefetch_budget_bytes"],
    )
    transport.setdefault(
        "cross_host_oversized_window_penalty",
        round(calibration["overflow_penalty"], 4),
    )
    transport.setdefault(
        "cross_host_calibration_source",
        str(CROSS_HOST_CALIBRATION.relative_to(ROOT)),
    )
    if WORKER_CONTENTION_SOURCE.is_file():
        source = yaml.safe_load(WORKER_CONTENTION_SOURCE.read_text())
        contention = (
            source.get("physical_model", {}).get("worker_contention")
        )
        if isinstance(contention, dict):
            data["physical_model"].setdefault("worker_contention", contention)
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False))
    log_line(f"merged host calibration into {profile_path.name}")


def log_line(message):
    print(f"[matrix {time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_status(out, state, current, summary):
    (out / "status.json").write_text(
        json.dumps(
            {
                "state": state,
                "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "current": current,
                "cells": summary,
            },
            indent=2,
        )
    )


def run(command, log_path, env, timeout):
    started = time.time()
    try:
        with log_path.open("w") as handle:
            code = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            ).returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        code, timed_out = None, True
    return code, timed_out, round(time.time() - started, 1)


def profile_command(dataset_file, kwargs, samples, profile_path):
    command = [
        sys.executable, "-u", str(ENTRY), "evaluation/eval_cedar.py",
        "--dataset_file", dataset_file,
        "--batch_size", "4",
        "--num_total_samples", str(samples),
        "--run_profiling",
        "--profiled_stats", str(profile_path),
        "--use_ray", "--ray_ip", ADDRESS,
        "--disable_controller", "--disable_caching",
    ]
    if kwargs:
        command[4:4] = ["--dataset_kwargs", ",".join(kwargs)]
    return command


def profile_env():
    env = dict(os.environ)
    env.update(
        {
            "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
            "CEDAR_REUSE_BOUNDARY_MODEL": "0",
            "CEDAR_LAYERED_ADAPTIVE_PROFILE": "1",
            "CEDAR_PROFILE_TIME_SEC": str(PROFILE_TIME_SEC),
            "CEDAR_CM_SWEEP_TIME_SEC": str(PROFILE_TIME_SEC),
            # Keep the width set identical to the reused reference profiles:
            # the DP prices width-contention at its fixed-width reference
            # (default 8), so every workload must have been profiled with the
            # same widths for the comparison to be fair.
            "CEDAR_PROFILE_SCALING_WIDTHS": "1,2,4,8",
            "CEDAR_PROFILE_SCALING_TOP_K": "5",
            "CEDAR_PROFILE_SCALING_MAX_SEC": "10",
            "CEDAR_ADAPTIVE_PROFILE_MIN_SEC": "3",
            "CEDAR_ADAPTIVE_PROFILE_MAX_SEC": "30",
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": "8",
        }
    )
    return env


def cell_command(dataset_file, kwargs, samples, profile_path, optimizer, results):
    command = [
        sys.executable, "-u", str(ENTRY), "evaluation/compare_optimizer_perf.py",
        "--dataset_file", dataset_file,
        "--batch_size", "4",
        "--num_total_samples", str(samples),
        "--full_data_run",
        "--profiled_stats", str(profile_path),
        "--use_ray", "--ray_ip", ADDRESS,
        "--enable_local_parallelism",
        "--disable_caching",
        "--match_profile_resources",
        "--cpu_budget", str(os.cpu_count() or 64),
        "--ray_cpu_budget", "64",
        "--optimizers", optimizer,
        "--optimizer_time_limit_sec", str(PLANNER_TIME_LIMIT_SEC),
        "--cedar_reorder_timeout_sec", str(CEDAR_REORDER_TIMEOUT_SEC),
        "--disable_cedar_runtime_timeout",
        "--num_repeats", "1",
        "--results_path", str(results),
    ]
    if kwargs:
        command[4:4] = ["--dataset_kwargs", ",".join(kwargs)]
    return command


def cell_env():
    env = dict(os.environ)
    env.update({"CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote"})
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", default=None)
    parser.add_argument("--optimizers", nargs="+", default=None)
    parser.add_argument("--reprofile", action="store_true")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override the per-workload sample count (smoke tests only).",
    )
    args = parser.parse_args()

    out = args.output if args.output.is_absolute() else ROOT / args.output
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir(parents=True, exist_ok=True)
    if not (out / "entry.py").is_file():
        shutil.copy2(ENTRY, out / "entry.py")
    if not (out / "modules").is_dir():
        shutil.copytree(REFERENCE / "modules", out / "modules")

    workloads = [
        w for w in WORKLOADS if args.workloads is None or w[0] in args.workloads
    ]
    if args.samples is not None:
        workloads = [(n, d, k, args.samples) for n, d, k, _ in workloads]
    optimizers = args.optimizers or OPTIMIZERS
    summary = {}
    write_status(out, "starting", None, summary)

    for name, dataset_file, kwargs, samples in workloads:
        profile_path = out / "profiles" / f"{name}_profile.yaml"
        if not profile_path.is_file():
            if name in REUSE_PROFILE and REUSE_PROFILE[name].is_file() and not args.reprofile:
                shutil.copy2(REUSE_PROFILE[name], profile_path)
                log_line(f"{name}: reused profile {REUSE_PROFILE[name].name}")
            else:
                write_status(out, f"profile:{name}", None, summary)
                command = profile_command(dataset_file, kwargs, samples, profile_path)
                code, timed_out, seconds = run(
                    command,
                    out / "logs" / f"profile_{name}.log",
                    profile_env(),
                    CELL_TIMEOUT_SEC,
                )
                log_line(
                    f"{name}: profile exit={code} timed_out={timed_out} "
                    f"wall={seconds}s"
                )
                if code != 0 or not profile_path.is_file():
                    summary[f"{name}:profile"] = {
                        "status": "profile_failed",
                        "exit": code,
                        "timed_out": timed_out,
                        "wall_sec": seconds,
                    }
                    write_status(out, f"profile_failed:{name}", None, summary)
                    continue
        merge_host_calibration(profile_path)
        for optimizer in optimizers:
            key = f"{name}:{optimizer}"
            results = out / "results" / f"{name}__{optimizer}.json"
            log_path = out / "logs" / f"{name}__{optimizer}.log"
            write_status(out, f"run:{key}", key, summary)
            command = cell_command(
                dataset_file, kwargs, samples, profile_path, optimizer, results
            )
            code, timed_out, seconds = run(
                command, log_path, cell_env(), CELL_TIMEOUT_SEC
            )
            entry = {
                "status": "ok" if code == 0 else ("timeout" if timed_out else "failed"),
                "exit": code,
                "timed_out": timed_out,
                "wall_sec": seconds,
                "samples": samples,
                "log": str(log_path.relative_to(ROOT)),
            }
            if results.is_file():
                try:
                    payload = json.loads(results.read_text())
                    for run_entry in payload.get("runs", []):
                        entry["perf_time_sec"] = run_entry.get("perf_time_sec")
                        entry["setup_time_sec"] = run_entry.get("setup_time_sec")
                        entry["throughput"] = run_entry.get(
                            "throughput_samples_per_sec"
                        )
                        plans = run_entry.get("physical_plans_by_feature") or {}
                        entry["n_local_workers"] = (
                            next(iter(plans.values()), {}).get("n_local_workers")
                            if plans
                            else None
                        )
                except Exception as exc:  # noqa: BLE001
                    entry["results_error"] = str(exc)
            summary[key] = entry
            log_line(
                f"{key}: {entry['status']} wall={seconds}s "
                f"perf={entry.get('perf_time_sec')}"
            )
            write_status(out, f"done:{key}", key, summary)
    write_status(out, "completed", None, summary)
    log_line("all cells finished")


if __name__ == "__main__":
    main()
