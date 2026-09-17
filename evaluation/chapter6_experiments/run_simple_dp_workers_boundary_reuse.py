"""Run one W-conditioned boundary Simple-DP round with frozen profiles."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

from run_simple_dp_ablation_matrix import (
    REPO, WORKLOADS, config, prepare, run, sha, write_json,
)


PROFILE_SOURCES = {
    "simclrv2": REPO / "outputs/simple_dp_ablation_recovery_20260917/simclrv2/profiles/shared.yaml",
    "simclrv2_cache": REPO / "outputs/simple_dp_ablation_recovery_20260917/simclrv2_cache/profiles/shared.yaml",
    "commonvoice": REPO / "outputs/simple_dp_ablation_remaining_one_round_20260917/commonvoice/profiles/shared.yaml",
    "coco": REPO / "outputs/simple_dp_ablation_recovery_20260917/coco/profiles/shared.yaml",
    "llava_pretrain": REPO / "outputs/simple_dp_ablation_remaining_one_round_20260917/llava_pretrain/profiles/shared.yaml",
    "stackexchange": REPO / "outputs/simple_dp_ablation_remaining_one_round_20260917/stackexchange/profiles/shared.yaml",
}
SOURCE_INPUT_ROOT = REPO / "outputs/simple_dp_ablation_remaining_one_round_20260917/inputs"
DEFAULT_LABEL = "simple-dp+W+boundary"
DEFAULT_INTERNAL = "simple_dp_workers_boundary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--optimizer-internal", default=DEFAULT_INTERNAL)
    args = parser.parse_args()
    label = args.label
    internal = args.optimizer_internal
    root = args.output.resolve()
    modules, entry = prepare(root)

    # Use exactly the finite JSONL subsets from the source experiment.
    for workload in ("llava_pretrain", "stackexchange"):
        source = SOURCE_INPUT_ROOT / f"{workload}.jsonl"
        shutil.copy2(source, root / "inputs" / source.name)

    env = {k: v for k, v in os.environ.items() if not k.startswith("CEDAR_")}
    env.update(
        PYTHONPATH=str(modules), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        CEDAR_RAY_PLACEMENT_RESOURCE="cedar_remote",
        CEDAR_DATA_JUICER_ROOT=str(REPO / "data-juicer"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        CEDAR_PROFILE_BOUNDARY_MODEL="1", CEDAR_REUSE_BOUNDARY_MODEL="0",
        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC="240",
        CEDAR_WORKER_READY_TIMEOUT_SEC="600",
    )
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                "https_proxy", "all_proxy"):
        env.pop(key, None)

    source_profiles = {
        workload: {
            "path": str(PROFILE_SOURCES[workload]),
            "sha256": sha(PROFILE_SOURCES[workload]),
        }
        for workload in args.workloads
    }
    metadata = {
        "workloads": args.workloads,
        "methods": {label: internal},
        "repeats": 1,
        "input_records": {
            "simclrv2": 9469, "simclrv2_cache": 9469,
            "commonvoice": 15000, "coco": 5000,
            "llava_pretrain": 1000, "stackexchange": 2000,
        },
        "cpu_budget": 64,
        "ray_cpu_budget": 64,
        "ray_address": "172.23.166.105:6379",
        "fixed_W": None,
        "cell_timeout_sec": 3600,
        "profile_policy": "reuse_exact_frozen_profile_without_reprofiling",
        "profile_sources": source_profiles,
        "input_sha256": {
            p.name: sha(p) for p in (root / "inputs").glob("*.jsonl")
        },
        "command_by_workload": {
            workload: config(root, workload, 15000)
            for workload in args.workloads
        },
        "environment": {
            k: v for k, v in env.items()
            if k.startswith(("CEDAR_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_"))
        },
    }
    write_json(root / "metadata.json", metadata)
    (root / "runner.pid").write_text(str(os.getpid()))
    shutil.copy2(Path(__file__), root / "runner_source.py")
    state = {}

    for workload in args.workloads:
        work = root / workload
        for name in ("profiles", "plans", "results", "logs", "warmup_results", "cache"):
            (work / name).mkdir(parents=True, exist_ok=True)
        profile = work / "profiles/shared.yaml"
        shutil.copy2(PROFILE_SOURCES[workload], profile)
        profile_data = yaml.safe_load(profile.read_text())
        signature = profile_data.get("resource_config", {})
        if any(signature.get(key) != 1 for key in
               ("profile_local_workers", "ray_actors_per_stage", "smp_procs_per_stage")):
            raise RuntimeError(f"{workload}: profile is not width one")
        if sha(profile) != source_profiles[workload]["sha256"]:
            raise RuntimeError(f"{workload}: copied profile hash mismatch")

        state[workload] = {
            "profile": {
                "status": "reused", "path": str(profile),
                "source": str(PROFILE_SOURCES[workload]), "sha256": sha(profile),
            },
            "cells": [],
        }
        write_json(root / "status.json", state)
        common = config(root, workload, 15000) + [
            "--use_ray", "--ray_ip", metadata["ray_address"],
            "--profiled_stats", str(profile),
        ]
        result = work / f"results/round1__{internal}.json"
        cmd = [sys.executable, "-u", str(entry),
               str(modules / "evaluation/compare_optimizer_perf.py")]
        cmd += common + [
            "--full_data_run", "--enable_local_parallelism",
            "--match_profile_resources", "--cpu_budget", "64",
            "--ray_cpu_budget", "64", "--optimizers", internal,
            "--optimizer_time_limit_sec", "3600",
            "--cedar_reorder_timeout_sec", "3600",
            "--disable_cedar_runtime_timeout", "--num_repeats", "1",
            "--skip_pico_plan_cost", "--cache_root", str(work / "cache"),
            "--results_path", str(result),
        ]
        if not workload.endswith("_cache"):
            cmd.append("--disable_caching")
        state[workload]["active_cell"] = {
            "method": label, "round": 1, "started_unix": time.time()
        }
        write_json(root / "status.json", state)
        print(f"RUN {workload} {label}", flush=True)
        record = run(
            cmd, work / f"logs/round1__{internal}.log",
            dict(env, EXPERIMENT_SEED="20260917"), 3600,
        )
        state[workload].pop("active_cell", None)
        record.update(method=label, round=1, profile_sha256=sha(profile))
        if result.exists():
            try:
                payload = json.loads(result.read_text())
            except (ValueError, OSError) as exc:
                payload = {}
                record["result_parse_error"] = str(exc)
                if record["status"] != "timeout":
                    record["status"] = "failed"
            for measured in payload.get("runs", []):
                if measured.get("timed_out") or measured.get("skip_reason") == "optimizer_time_limit_exceeded":
                    record["status"] = "timeout"
                elif measured.get("workload_skipped"):
                    record["status"] = "failed"
                plans = measured.get("physical_plans_by_feature", {})
                (work / f"plans/round1__{internal}.yaml").write_text(
                    yaml.safe_dump(plans)
                )
                write_json(
                    work / f"warmup_results/round1__{internal}.json",
                    {k: v for k, v in measured.items()
                     if k.startswith("cache_warmup")},
                )
        state[workload]["cells"].append(record)
        write_json(root / "status.json", state)

    (root / "COMPLETE").write_text(
        "Single-round W-conditioned boundary experiment finished.\n"
    )


if __name__ == "__main__":
    main()
