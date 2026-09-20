"""Rerun only W boundary variants on the frozen 15k CommonVoice profile."""
import json
import os
import shutil
import sys
import time
from pathlib import Path
import yaml
from run_simple_dp_ablation_matrix import prepare, run, sha, write_json, REPO
from summarize_simple_dp_ablation import summarize

def main():
    source = REPO / "outputs/commonvoice_15000_no_reserve_20260918/matrix"
    root = REPO / "outputs/commonvoice_15000_w_boundary_smp_curve_20260918"
    methods = ["simple_dp_workers_boundary", "simple_dp_workers_width_boundary"]
    modules, entry = prepare(root)
    source_metadata = json.loads((source / "metadata.json").read_text())
    source_state = json.loads((source / "status.json").read_text())
    metadata = dict(source_metadata)
    profile_source = REPO / "outputs/smp_ray_transport_design_probe_20260918/shared_smp_curve.yaml"
    metadata.update(methods={m: m for m in methods}, repeats=1,
        profile_policy="reuse_compute_width_remote_ray_profile_add_measured_smp_aggregate_curve",
        profile_sources={"commonvoice": {"path": str(profile_source),
                                         "sha256": sha(profile_source)}},
        communication_model="(compute + fixed)/W + Ray_bytes/shared_remote_bandwidth + SMP_bytes/measured_aggregate_IPC_bandwidth(W)",
        source_run=str(source), cell_timeout_sec=7200)
    write_json(root / "metadata.json", metadata)
    (root / "runner.pid").write_text(str(os.getpid()) + "\n")
    shutil.copy2(Path(__file__), root / "runner_source.py")
    work = root / "commonvoice"
    for name in ["profiles", "results", "logs", "plans", "warmup_results", "cache"]:
        (work / name).mkdir(parents=True, exist_ok=True)
    profile = work / "profiles/shared.yaml"
    shutil.copy2(profile_source, profile)
    assert sha(profile) == sha(profile_source)
    from cedar.client.boundary_profiler import validate_remote_ray_boundary
    validate_remote_ray_boundary(yaml.safe_load(profile.read_text()))
    signature = yaml.safe_load(profile.read_text())["resource_config"]
    assert all(signature[k] == 1 for k in
               ["profile_local_workers", "ray_actors_per_stage", "smp_procs_per_stage"])
    env = {k: v for k, v in os.environ.items() if not k.startswith("CEDAR_")}
    env.update(source_metadata["environment"])
    env["PYTHONPATH"] = str(modules)
    env["EXPERIMENT_SEED"] = "20260917"
    env["CEDAR_RAY_REQUIRE_REMOTE"] = "1"
    env["CEDAR_RAY_PLACEMENT_RESOURCE"] = "cedar_remote"
    env["CEDAR_BOUNDARY_DIAGNOSTICS_DIR"] = str(root / "boundary_diagnostics")
    metadata["environment"] = {k:v for k,v in env.items()
        if k.startswith(("CEDAR_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_"))}
    write_json(root / "metadata.json", metadata)
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    state = {"commonvoice": {"profile": {
        "status": "reused", "source": str(profile_source),
        "sha256": sha(profile)}, "cells": []}}
    write_json(root / "status.json", state)
    original = next(c for c in source_state["commonvoice"]["cells"]
                    if c["method"] == "simple_dp_workers_boundary")["command"]
    for method in methods:
        cmd = [x.replace(str(source), str(root)) for x in original]
        cmd[cmd.index("--optimizers") + 1] = method
        cmd[cmd.index("--results_path") + 1] = str(work / "results" / ("round1__" + method + ".json"))
        cmd[cmd.index("--optimizer_time_limit_sec") + 1] = "7200"
        cmd[cmd.index("--cedar_reorder_timeout_sec") + 1] = "7200"
        state["commonvoice"]["active_cell"] = {
            "method": method, "round": 1, "started_unix": time.time()}
        write_json(root / "status.json", state)
        print("RUN", method, "profile_sha256", sha(profile), flush=True)
        record = run(cmd, work / "logs" / ("round1__" + method + ".log"), env, 7200)
        record.update(method=method, round=1, profile_sha256=sha(profile))
        result = Path(cmd[cmd.index("--results_path") + 1])
        if result.exists():
            payload = json.loads(result.read_text())
            for measured in payload.get("runs", []):
                if measured.get("timed_out") or measured.get("skip_reason") == "optimizer_time_limit_exceeded":
                    record["status"] = "timeout"
                elif measured.get("workload_skipped"):
                    record["status"] = "failed"
                (work / "plans" / ("round1__" + method + ".yaml")).write_text(
                    yaml.safe_dump(measured.get("physical_plans_by_feature", {})))
        state["commonvoice"].pop("active_cell", None)
        state["commonvoice"]["cells"].append(record)
        write_json(root / "status.json", state)
        summarize(root)
        print("DONE", method, record["status"], flush=True)
    (root / "COMPLETE").write_text("Finished two optimizer cells.\n")

if __name__ == "__main__":
    main()
