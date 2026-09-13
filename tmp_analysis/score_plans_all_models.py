"""Score materialized plans with every system cost model.

Cost models only: each system's model predicts the per-record service time of
a given plan; nothing here searches for a plan.  Two profiling bases are shown
because the systems would in practice profile differently:

  baseline : per-pipe latencies measured inside a running pipeline (trace
             deltas).  This is what Cedar/Plumber/Pecan-style profilers expose.
  affine   : per-operator costs measured in isolation (our calibrated layer).

Comparing the two columns separates "bad profiled data" from "bad model form".

Usage:
  python -u tmp_analysis/score_plans_all_models.py \
      outputs/pico_default_profiles/simclr_profile.yaml
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", "8")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.optimizer import Optimizer, PhysicalPlan  # noqa: E402
from cedar.compose.system_cost_models import calculate_all  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    DATASET_LOC,
    SimCLRV2Feature,
)

PROFILE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else ROOT / "outputs/pico_default_profiles/simclr_profile.yaml"
)
PLAN_DIR = ROOT / "tmp_analysis/plans/simclr_bench"
PLANS = {
    "cedar": PLAN_DIR / "optimizer.yaml",
    "plumber": PLAN_DIR / "plumber_optimizer.yaml",
    "pico": PLAN_DIR / "dp_optimizer.yaml",
    "raydata": PLAN_DIR / "raydata_optimizer.yaml",
    # Declared order with every stage INPROCESS: the unoptimized plan.
    "unoptimized": ROOT / "tmp_analysis/plan_runs/plumber_serial_reconcile.yaml",
}
# Wall-clock seconds for 1000 records with the planner's own worker count.
MEASURED_SEC = {
    "cedar": 2.505,
    "plumber": 1.275,
    "pico": 0.612,
    "raydata": 13.732,
    "unoptimized": 3.052,
}
SAMPLES = 1000

# A finished measurement batch can override the wall-clock seconds per plan,
# so the scoring does not need to be edited by hand after every re-run.
_override = os.environ.get("CEDAR_MEASURED_JSON")
if _override and Path(_override).is_file():
    try:
        MEASURED_SEC.update(
            {
                key: float(value)
                for key, value in json.loads(Path(_override).read_text()).items()
                if key in MEASURED_SEC
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"warning: ignoring {_override}: {exc}")
WORKERS = 8


def build_feature(batch_size=4):
    data_dir = (
        Path(original_simclrv2.__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    feature = SimCLRV2Feature(batch_size=batch_size)
    feature.apply(LocalFSSource(str(data_dir / "imagenette2" / "train"), recursive=True))
    return feature


def options():
    return OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=9469,
        use_my_optimizer=2,
        reorder_timeout_sec=3600.0,
    )


def load_plan(path):
    data = yaml.safe_load(Path(path).read_text())
    payload = data.get("physical_plan", data)
    payload = payload.get("feature_r0", payload)
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for pipe in payload["pipes"].values():
        pipe["variant"] = pipe.get("variant") or "INPROCESS"
        pipe.setdefault("variant_ctx", {"variant_type": pipe["variant"]})
    return payload


def main():
    profile = yaml.safe_load(PROFILE.read_text())

    dp = DpOptimizer()
    build_feature().set_optimizer(dp)
    dp.run(str(PROFILE), options())

    cedar = Optimizer()
    build_feature().set_optimizer(cedar)
    cedar.run(str(PROFILE), options())

    # Per-record isolated operator costs, taken from the DP's own cost function
    # so that both profiling bases are expressed in the same units.
    affine_costs = {}
    for p_id in dp._dp_inner_ops:
        try:
            baseline_input = dp.profiled_stats["baseline"]["input_sizes"][p_id]
            affine_costs[p_id] = float(
                dp._calculate_pipe_cost(p_id, baseline_input, None)
            )
        except Exception:  # noqa: BLE001 - optional basis
            continue

    table = {}
    for name, path in PLANS.items():
        payload = load_plan(path)
        plan = PhysicalPlan.from_dict(payload)
        workers = int(payload.get("n_local_workers", 8) or 8)
        row = {
            # Aggregate per-record time: independent of how the planner split
            # records across workers, so plans with different W stay comparable.
            "measured_total_sec": MEASURED_SEC[name],
            "measured_aggregate_ms": MEASURED_SEC[name] * 1000.0 / SAMPLES,
            "workers": workers,
            "pico": round(dp.calculate_dp_objective_cost(plan=plan), 4),
            "baseline": {
                key: round(value.ms_per_record, 4)
                for key, value in calculate_all(
                    plan, profile, cedar, workers=WORKERS
                ).items()
            },
            "affine": {},
        }
        for key, model in __import__(
            "cedar.compose.system_cost_models", fromlist=["default_models"]
        ).default_models(cedar).items():
            model.cost_source = "affine"
            try:
                row["affine"][key] = round(
                    model.calculate_cost(
                        plan,
                        profile,
                        workers=WORKERS,
                        affine_costs=affine_costs,
                    ).ms_per_record,
                    4,
                )
            except Exception as exc:  # noqa: BLE001
                row["affine"][key] = f"error: {exc}"
        table[name] = row

    keys = ["pico", "cedar", "plumber", "pecan", "tfdata", "raydata",
            "fastflow", "aero"]
    header = f"{'plan':<8} {'measured':>9} " + " ".join(
        f"{k:>9}" for k in keys
    )
    print("\n=== profiled with pipeline traces (baseline) ===")
    print(header)
    for name, row in table.items():
        cells = [f"{row['pico']:>9.2f}"] + [
            f"{row['baseline'].get(k, float('nan')):>9.2f}" for k in keys[1:]
        ]
        print(f"{name:<8} {row['measured_aggregate_ms']:>9.3f} " + " ".join(cells))
    print("\n=== profiled with isolated operator costs (affine) ===")
    print(header)
    for name, row in table.items():
        cells = [f"{row['pico']:>9.2f}"] + [
            (
                f"{row['affine'][k]:>9.2f}"
                if isinstance(row["affine"].get(k), (int, float))
                else f"{'n/a':>9}"
            )
            for k in keys[1:]
        ]
        print(f"{name:<8} {row['measured_aggregate_ms']:>9.3f} " + " ".join(cells))

    print("\n=== plan ordering implied by each model (fastest first) ===")
    print(
        "measured ",
        " < ".join(
            sorted(table, key=lambda p: table[p]["measured_aggregate_ms"])
        ),
    )
    for key in keys:
        if key == "pico":
            ordering = sorted(table, key=lambda p: table[p]["pico"])
        else:
            ordering = sorted(
                table,
                key=lambda p: table[p]["baseline"].get(key, float("inf")),
            )
        winner = ordering[0]
        mark = "OK" if winner == "pico" else "wrong winner"
        print(f"{key:<9}", " < ".join(ordering), f"({mark})")

    out = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else ROOT / "outputs/plumber_bench_20260912/system_cost_matrix.json"
    )
    out.write_text(json.dumps(table, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
