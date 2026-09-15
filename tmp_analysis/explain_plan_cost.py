"""Break one materialized plan's cost model score into its stage terms.

For every stage of the plan this prints what the model charges: block compute
divided by the effective width, the boundary terms, the cross-host round trip
and the transport floor, plus the lane each term lands on.  It answers "why
does the model prefer this plan" without running the pipeline.

Usage (inside the container):
  python -u tmp_analysis/explain_plan_cost.py alpaca_cot <plan.yaml> [...]
"""

import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.optimizer import PhysicalPlan  # noqa: E402

from fit_stage_concurrency import WORKLOADS, build_feature  # noqa: E402

RUNS = [
    ROOT / "outputs/pico_drained_20260914",
]


def profile_path(workload: str) -> Path:
    for run in RUNS:
        candidate = run / "profiles" / f"{workload}_profile.yaml"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no profile for {workload}")


def main() -> int:
    workload = sys.argv[1]
    plans = sys.argv[2:]
    profile = profile_path(workload)
    feature = build_feature(workload, profile)
    optimizer = DpOptimizer()
    feature.set_optimizer(optimizer)
    _, _, samples = WORKLOADS[workload]
    optimizer.run(
        str(profile),
        OptimizerOptions(
            enable_prefetch=True,
            est_throughput=None,
            available_local_cpus=64,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=True,
            enable_fusion=True,
            num_samples=samples,
            use_my_optimizer=2,
            reorder_timeout_sec=1800.0,
        ),
    )
    for path in plans:
        plan = PhysicalPlan.from_dict(
            yaml.safe_load(Path(path).read_text())["physical_plan"]
        )
        workers = max(1, int(plan.n_local_workers or 1))
        print(f"\n=== {path}  workers={workers}")
        blocks = optimizer._dp_blocks_from_physical_plan(
            plan, optimizer._dp_inner_ops
        )
        for order, variant, _cache, parallelism in blocks:
            names = [optimizer._dp_inner_ops[i] for i in order]
            print(f"  stage {variant.name:<9} width={parallelism:<3} ops={names}")
        os.environ["CEDAR_DP_REPLAY_TRACE"] = "1"
        score = optimizer.calculate_dp_objective_cost(plan=plan)
        predicted = 1000.0 * workers / score if score > 0 else float("inf")
        print(f"  score={score:.4f} ms/record/worker  "
              f"predicted={predicted:.1f} rec/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
