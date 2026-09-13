"""Calibrate the cross-host payload budget of a *running* stage.

The isolated echo sweep (``tmp_analysis/measure_ray_transport.py``) moves one
cached object per round trip, so it measures a latency no running plan
reaches.  The rate a plan actually sustains is bounded by the payload a stage
keeps resident: the Ray Data-style plan sizes its window in *records*
(``submit_batch_size * n_actors * 3``), which for multimodal records holds
hundreds of megabytes per worker.

This script sweeps that window (only ``max_inflight``/``max_prefetch`` change
between runs) by materializing a plan per window and executing each of them
with ``tmp_analysis/reconcile.py``, which reports wall-clock seconds for a
fixed number of records.  The fit is written to
``outputs/cross_host_inplan_20260913/cross_host.json`` and feeds
``physical_model.transport`` of the SimCLRv2 profiles.

Run (inside the container, offline):
  nohup python -u tmp_analysis/measure_cross_host_inplan.py \
      > tmp_analysis/cross_host_inplan.log 2>&1 &
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
PROFILE = ROOT / "outputs/simclr_rerun_wide2_20260913/simclr_profile.yaml"
BASE_PLAN = ROOT / "tmp_analysis/plans/simclr_bench/raydata_optimizer.yaml"
PLAN_DIR = ROOT / "tmp_analysis/plans/simclr_bench"
OUT = ROOT / "outputs/cross_host_inplan_20260913"
WINDOWS = (35, 64, 256, 400, 512, 630)
SAMPLES = 1000
BATCH = 4
WORKERS = 8
# Input (681 KB) plus output (238 KB) of the SimCLRv2 fused Ray stage.
PAYLOAD_BYTES_PER_RECORD = 919_000.0
# Window (records) above which the measured rate collapses.
PREFETCH_BUDGET_BYTES = 300_000_000.0


def plan_for_window(window: int) -> Path:
    payload = yaml.safe_load(BASE_PLAN.read_text())
    ctx = payload["physical_plan"]["pipes"]["11"]["variant_ctx"]
    ctx["max_inflight"] = window
    ctx["max_prefetch"] = window
    target = PLAN_DIR / f"raydata_win{window}.yaml"
    target.write_text(yaml.safe_dump(payload))
    return target


def run_sweep() -> dict:
    log_path = ROOT / "tmp_analysis/cross_host_inplan_runs.log"
    runs = {}
    for window in WINDOWS:
        plan = plan_for_window(window)
        env = dict(os.environ)
        env.update(
            {
                "CEDAR_RECONCILE_SAMPLES": str(SAMPLES),
                "CEDAR_RECONCILE_BATCH": str(BATCH),
                "CEDAR_PLAN_WORKERS": str(WORKERS),
            }
        )
        command = [
            sys.executable,
            "-u",
            str(ROOT / "tmp_analysis/reconcile.py"),
            str(PROFILE),
            f"win{window}={plan}",
        ]
        with log_path.open("a") as handle:
            handle.write(f"# window={window} plan={plan}\n")
            handle.flush()
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        eval_log = ROOT / "tmp_analysis/reconcile" / f"win{window}.log"
        total = None
        for line in eval_log.read_text(errors="replace").splitlines():
            if line.startswith("Total time: "):
                total = float(line.split()[2].rstrip("s"))
        runs[str(window)] = {
            "window_records": window,
            "seconds": total,
        }
        print(f"[cross-host] window={window} seconds={total}", flush=True)
    return runs


def main() -> None:
    runs = run_sweep()
    small = [
        entry["seconds"]
        for entry in runs.values()
        if entry["seconds"] and entry["window_records"] <= 256
    ]
    large = [
        entry["seconds"]
        for entry in runs.values()
        if entry["seconds"] and entry["window_records"] >= 400
    ]
    small_median = statistics.median(small)
    large_median = statistics.median(large)
    per_worker = (
        SAMPLES * PAYLOAD_BYTES_PER_RECORD / (WORKERS * small_median)
    )
    payload = {
        "schema_version": 1,
        "method": "in_plan_prefetch_window_sweep",
        "plan": str(BASE_PLAN.relative_to(ROOT)),
        "samples": SAMPLES,
        "payload_bytes_per_record": PAYLOAD_BYTES_PER_RECORD,
        "worker_payload_bytes_per_sec": per_worker,
        "small_window_median_sec": small_median,
        "large_window_median_sec": large_median,
        "overflow_penalty": large_median / small_median,
        "prefetch_budget_bytes": PREFETCH_BUDGET_BYTES,
        "note": (
            "Only max_inflight/max_prefetch change across runs.  The achieved "
            "payload rate is flat while the window fits in the worker prefetch "
            "budget, and drops by a constant factor past it."
        ),
        "runs": runs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_host.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
