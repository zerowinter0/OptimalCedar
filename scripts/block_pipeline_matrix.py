"""Experiment B: the same four block configurations inside the full pipeline.

Each configuration is executed as a *fixed plan* (no optimizer, no search) at
W=1 and W=64, three repeats, round-robin.  Everything except the block's
backend/fusion organisation and W is identical to the campaign: same dataset,
same batch size, same profile, same Ray cluster and CPU budget.

Usage (inside the container):
  python -u scripts/block_pipeline_matrix.py plans --base-plan <cedar.yaml> \
      --out-dir outputs/<run>/plans
  python -u scripts/block_pipeline_matrix.py run --run-dir outputs/<run> \
      --repeats 3 --num-samples 80000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    CONFIGS,
    SIMCLRV2_DATASET,
    load_base_plan,
    block_plan,
    write_plan,
)

DEFAULT_PROFILE = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml"
)
DEFAULT_BASE_PLAN = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/plans/round1__optimizer.yaml"
)
DATASET_FILE = ROOT / "evaluation/pipelines/simclrv2/cedar_dataset.py"
RAY_IP = "172.23.166.105:6379"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_plans(args: argparse.Namespace) -> int:
    base = load_base_plan(args.base_plan)
    out_dir = Path(args.out_dir)
    manifest = {}
    for workers in args.workers:
        for config in CONFIGS:
            plan = block_plan(base, config, workers)
            path = out_dir / f"{config}_w{workers}.yaml"
            write_plan(plan, path)
            manifest[f"{config}_w{workers}"] = {
                "path": str(path),
                "workers": workers,
                "sha256": sha256(path),
                "block_nodes": sorted(
                    p_id
                    for p_id, desc in plan["pipes"].items()
                    if desc.get("fused_pipes")
                    or (desc.get("variant") == "RAY")
                ),
            }
    (out_dir / "plans_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


ACTOR_RE = re.compile(
    r"Registering RayService for (?P<name>\S+) with (?P<count>\d+) actors"
)


def _actor_counts(log_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not log_path.exists():
        return counts
    for line in log_path.read_text(errors="ignore").splitlines():
        match = ACTOR_RE.search(line)
        if match:
            name = match.group("name")
            counts[name] = counts.get(name, 0) + int(match.group("count"))
    return counts


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    plans_dir = run_dir / "plans"
    logs_dir = run_dir / "pipeline_logs"
    results_dir = run_dir / "pipeline_results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cells: List[Dict[str, Any]] = []
    for config in CONFIGS:
        for workers in args.workers:
            cells.append({"config": config, "workers": workers})
    if args.cells:
        wanted = set(args.cells)
        filtered = [
            cell for cell in cells
            if f"{cell['config']}_w{cell['workers']}" in wanted
        ]
        if len(filtered) != len(wanted):
            missing = wanted - {
                f"{cell['config']}_w{cell['workers']}" for cell in filtered
            }
            raise SystemExit(f"unknown cells: {sorted(missing)}")
        cells = filtered

    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CEDAR_RAY_PLACEMENT_RESOURCE="cedar_remote",
        CEDAR_RAY_REQUIRE_REMOTE="1",
        CEDAR_RAY_PATH_TIMING="1",
        CEDAR_WORKER_READY_TIMEOUT_SEC="600",
        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC=os.environ.get(
            "CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC", "900"
        ),
        CEDAR_LOCAL_WORKERS_MAX="64",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pyarrow, torch, ray",  # fail fast on a broken environment
        ],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    rows: List[Dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        order = cells[repeat - 1:] + cells[: repeat - 1]
        for cell in order:
            config, workers = cell["config"], cell["workers"]
            plan_path = plans_dir / f"{config}_w{workers}.yaml"
            if not plan_path.exists():
                raise SystemExit(f"missing plan {plan_path}")
            label = f"{config}_w{workers}_r{repeat}"
            log_path = logs_dir / f"{label}.log"
            results_path = results_dir / f"{label}.json"
            command = [
                sys.executable, "-u", str(ROOT / "scripts/run_fixed_plan_throughput.py"),
                "--plan", str(plan_path),
                "--label", label,
                "--results-path", str(results_path),
                "--dataset-file", str(DATASET_FILE),
                "--dataset-kwargs", f"dataset_path={SIMCLRV2_DATASET}",
                "--batch-size", "4",
                "--num-epochs", str(args.epochs),
                "--num-total-samples", str(args.num_samples),
                "--profiled-stats", str(args.profile),
                "--ray-ip", RAY_IP,
            ]
            print(f"RUN {label}", flush=True)
            started = time.time()
            with log_path.open("wb") as handle:
                process = subprocess.run(
                    command, env=env, stdout=handle, stderr=subprocess.STDOUT
                )
            wall = time.time() - started
            row: Dict[str, Any] = {
                "repeat": repeat,
                "config": config,
                "workers": workers,
                "label": label,
                "returncode": process.returncode,
                "wall_time_sec": wall,
                "plan_path": str(plan_path),
                "plan_sha256": sha256(plan_path),
                "actors": json.dumps(_actor_counts(log_path), sort_keys=True),
            }
            if results_path.exists():
                payload = json.loads(results_path.read_text())
                row.update(
                    {
                        "num_samples": payload.get("num_samples"),
                        "perf_time_sec": payload.get("perf_time_sec"),
                        "throughput_samples_per_sec": payload.get(
                            "throughput_samples_per_sec"
                        ),
                        "setup_time_sec": payload.get("setup_time_sec"),
                        "total_time_sec": payload.get("total_time_sec"),
                        "epochs": len(payload.get("epoch_run_times", []) or []),
                    }
                )
            rows.append(row)
            print(
                "  -> rc=%s throughput=%s samples=%s wall=%.1fs"
                % (
                    row["returncode"],
                    row.get("throughput_samples_per_sec"),
                    row.get("num_samples"),
                    wall,
                ),
                flush=True,
            )

    fields = [
        "repeat", "config", "workers", "label", "returncode", "num_samples",
        "perf_time_sec", "throughput_samples_per_sec", "setup_time_sec",
        "total_time_sec", "wall_time_sec", "actors", "epochs", "plan_path",
        "plan_sha256",
    ]
    out_csv = run_dir / f"{args.out_name}.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {out_csv}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plans = sub.add_parser("plans")
    plans.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    plans.add_argument("--out-dir", type=Path, required=True)
    plans.add_argument("--workers", type=int, nargs="+", default=[1, 64])
    plans.set_defaults(func=cmd_plans)

    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--workers", type=int, nargs="+", default=[1, 64])
    run.add_argument("--num-samples", type=int, default=80000)
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument(
        "--cells", nargs="+", default=[],
        help="restrict to cells like L-U_w1 (recovery runs)",
    )
    run.add_argument("--out-name", default="pipeline_results")
    run.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
