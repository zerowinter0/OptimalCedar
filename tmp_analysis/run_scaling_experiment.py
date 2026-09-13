"""Regenerate the SimCLRv2 profile with layered (width + object) calibration.

Steps, all offline-friendly and idempotent:
  1. build the new output directory (code snapshot for the Ray runtime env);
  2. re-profile with CEDAR_LAYERED_ADAPTIVE_PROFILE=1 so the profile gains
     physical_model.scaling (width contention) and physical_model.object_boundary
     (real legal objects);
  3. self-check the new profile against the reference profile;
  4. rerun the four-optimizer comparison with the new profile;
  5. print the DP lane decomposition for every produced plan.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/simclrv2_scaling_20260911"
REF = ROOT / "outputs/simclrv2_four_remote_20260911"
ADDRESS = "172.23.166.105:6379"
DATA = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
NAMES = ["optimizer", "cm_optimizer", "old_dp_optimizer", "dp_optimizer"]

PROFILE_ENV = {
    "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
    "CEDAR_REUSE_BOUNDARY_MODEL": "0",
    "CEDAR_PROFILE_TIME_SEC": "10",
    "CEDAR_CM_SWEEP_TIME_SEC": "10",
    "CEDAR_LAYERED_ADAPTIVE_PROFILE": "1",
    "CEDAR_PROFILE_SCALING_WIDTHS": "1,2,4,8",
    "CEDAR_PROFILE_SCALING_TOP_K": "5",
    "CEDAR_PROFILE_SCALING_MAX_SEC": "10",
    "CEDAR_PROFILE_SCALING_RAY_BATCH_SIZE": "1",
}


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


def prepare_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    if not (OUT / "modules").exists():
        shutil.copytree(REF / "modules", OUT / "modules")
    for name in ("entry.py", "probe.py"):
        if not (OUT / name).exists():
            shutil.copy2(REF / name, OUT / name)


def run(tag, args, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    log_path = OUT / "logs" / (tag + ".log")
    with log_path.open("w") as handle:
        completed = subprocess.run(
            [sys.executable, "-u", str(OUT / "entry.py"), *args],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return completed.returncode, log_path


def load(path):
    return yaml.safe_load(Path(path).read_text())


def check_profile(new_path, ref_path):
    new = load(new_path)
    ref = load(ref_path)
    report = {"sections": {}, "boundary": {}, "scaling": {}}

    physical = new.get("physical_model", {})
    report["sections"] = {
        "physical_model_keys": sorted(physical),
        "object_boundary_variants": sorted(
            (physical.get("object_boundary") or {})
        ),
        "scaling_variants": sorted((physical.get("scaling") or {})),
        "cm_model": bool(new.get("cm_model")),
        "offload_variants": sorted(new.get("offloads", {})),
    }

    for variant in ("RAY", "SMP"):
        new_entry = (physical.get("boundary") or {}).get(variant, {})
        ref_entry = (ref.get("physical_model", {}).get("boundary") or {}).get(
            variant, {}
        )
        report["boundary"][variant] = {
            "new_fixed_latency_ms": new_entry.get("fixed_latency_ms"),
            "new_throughput_bytes_per_sec": new_entry.get(
                "throughput_bytes_per_sec"
            ),
            "new_r_squared": new_entry.get("r_squared"),
            "ref_fixed_latency_ms": ref_entry.get("fixed_latency_ms"),
            "ref_throughput_bytes_per_sec": ref_entry.get(
                "throughput_bytes_per_sec"
            ),
        }
        operators = (
            (physical.get("object_boundary") or {}).get(variant, {}).get(
                "operators", {}
            )
        )
        report["boundary"][variant]["object_boundary_pipes"] = sorted(
            operators, key=lambda x: int(x)
        )
        report["boundary"][variant]["object_boundary_sample"] = {
            pid: {
                "serialize_ms_per_sample": operators[pid].get(
                    "input_serialize_ms_per_sample"
                ),
                "deserialize_ms_per_sample": operators[pid].get(
                    "output_deserialize_ms_per_sample"
                ),
                "serialized_bytes_per_sample": operators[pid].get(
                    "input_serialized_bytes_per_sample"
                ),
            }
            for pid in list(sorted(operators, key=lambda x: int(x)))[:3]
        }

    for variant in ("RAY", "SMP"):
        entries = (physical.get("scaling") or {}).get(variant, {})
        for pid, entry in sorted(entries.items(), key=lambda kv: int(kv[0])):
            widths = entry.get("widths", {})
            report["scaling"].setdefault(variant, {})[pid] = {
                str(width): {
                    "converged": bool(
                        (timing.get("adaptive_profile") or {}).get("converged")
                    ),
                    "mean_ms_per_sample": timing.get("mean_ms_per_sample"),
                    "rse": (timing.get("adaptive_profile") or {}).get("rse"),
                }
                for width, timing in sorted(
                    widths.items(), key=lambda kv: int(kv[0])
                )
            }

    # Offload anchors moved with the layered isolated costs: keep both for对比.
    report["offload_anchors"] = {}
    for variant in ("RAY", "SMP"):
        new_off = new.get("offloads", {}).get(variant, {})
        ref_off = ref.get("offloads", {}).get(variant, {})
        report["offload_anchors"][variant] = {
            pid: {
                "new_mean_ms_per_sample": (
                    new_off.get(pid, {}).get("backend_compute", {}) or {}
                ).get("mean_ms_per_sample"),
                "ref_mean_ms_per_sample": (
                    ref_off.get(pid, {}).get("backend_compute", {}) or {}
                ).get("mean_ms_per_sample"),
            }
            for pid in sorted(set(new_off) | set(ref_off), key=int)
        }

    (OUT / "results" / "profile_check.json").write_text(
        json.dumps(report, indent=2)
    )
    return report


def print_report(report):
    print("===== profile self check =====", flush=True)
    print(json.dumps(report["sections"], indent=2), flush=True)
    for variant, item in report["boundary"].items():
        print(
            f"[boundary {variant}] new fixed={item['new_fixed_latency_ms']} "
            f"bw={item['new_throughput_bytes_per_sec']} r2={item['new_r_squared']} "
            f"| ref fixed={item['ref_fixed_latency_ms']} "
            f"bw={item['ref_throughput_bytes_per_sec']}",
            flush=True,
        )
        print(
            f"   object_boundary pipes={item['object_boundary_pipes']} "
            f"sample={item['object_boundary_sample']}",
            flush=True,
        )
    print("===== width scaling curves =====", flush=True)
    for variant, entries in report["scaling"].items():
        for pid, widths in entries.items():
            described = " ".join(
                f"{w}:{'ok' if v['converged'] else 'no'}"
                f"={None if v['mean_ms_per_sample'] is None else round(v['mean_ms_per_sample'], 4)}"
                for w, v in widths.items()
            )
            print(f"  {variant} pipe {pid}: {described}", flush=True)
    print("===== offload anchors old -> new (ms/sample) =====", flush=True)
    for variant, entries in report["offload_anchors"].items():
        for pid, item in entries.items():
            print(
                f"  {variant} pipe {pid}: "
                f"{item['ref_mean_ms_per_sample']} -> "
                f"{item['new_mean_ms_per_sample']}",
                flush=True,
            )


def main():
    prepare_dirs()
    profile_path = OUT / "profile.yaml"
    env_extra = dict(PROFILE_ENV)

    if "--resume-profile" not in sys.argv or not profile_path.exists():
        status("profiling")
        print("[driver] profiling with layered adaptive profile", flush=True)
        code, log_path = run(
            "profile",
            [
                "evaluation/eval_cedar.py",
                "--dataset_file",
                DATA,
                "--batch_size",
                "1",
                "--run_profiling",
                "--profiled_stats",
                str(profile_path),
                "--use_my_optimizer",
                "2",
                "--use_ray",
                "--ray_ip",
                ADDRESS,
                "--disable_controller",
                "--disable_caching",
            ],
            env_extra,
        )
        print(f"[driver] profile exit={code} log={log_path}", flush=True)
        if code != 0 or not profile_path.exists():
            status("failed", step="profile", log=str(log_path))
            return 1

    status("checking_profile")
    report = check_profile(profile_path, REF / "profile.yaml")
    print_report(report)

    status("comparison")
    print("[driver] running four-optimizer comparison", flush=True)
    code, log_path = run(
        "comparison",
        [
            "evaluation/compare_optimizer_perf.py",
            "--dataset_file",
            DATA,
            "--batch_size",
            "1",
            "--num_total_samples",
            "9469",
            "--full_data_run",
            "--profiled_stats",
            str(profile_path),
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
            str(OUT / "results" / "comparison.json"),
        ],
        {k: v for k, v in env_extra.items() if not k.startswith("CEDAR_PROFILE_SCALING")},
    )
    print(f"[driver] comparison exit={code} log={log_path}", flush=True)
    if code != 0:
        status("failed", step="comparison", log=str(log_path))
        return 1

    # lane decomposition under the new profile
    probe_code, probe_log = run(
        "probe_new_profile", ["tmp_analysis/probe.py", str(OUT)], {}
    )
    print(
        f"[driver] probe exit={probe_code} log={probe_log}",
        flush=True,
    )
    status("completed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:
        status("failed", step="driver")
        traceback.print_exc()
        raise
