"""Compare cost-model bases on the drained plan set of one workload.

Scores every measured plan twice — with the isolated per-actor basis and with
the stage-level curve basis — and prints measured vs predicted throughput, the
rank correlation and the top-1 agreement for each basis.

Usage (inside the container):
  python -u tmp_analysis/compare_cost_bases.py alpaca_cot
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import fit_stage_concurrency as fit  # noqa: E402
from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.optimizer import PhysicalPlan  # noqa: E402

BASES = [
    ("isolated per-actor", {"CEDAR_DP_STAGE_CURVE_COST": "0"}),
    ("stage-level curve", {"CEDAR_DP_STAGE_CURVE_COST": "1"}),
]


def main() -> None:
    workload = sys.argv[1] if len(sys.argv) > 1 else "alpaca_cot"
    plans = fit.collect_plans(workload)
    profile = None
    for run in fit.RUNS:
        candidate = run / "profiles" / f"{workload}_profile.yaml"
        if candidate.is_file():
            profile = candidate
            break
    feature = fit.build_feature(workload, profile)
    optimizer = DpOptimizer()
    feature.set_optimizer(optimizer)
    _, _, samples = fit.WORKLOADS[workload]
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
    names = sorted(plans)
    measured = [plans[name][1] for name in names]
    measured_best = names[measured.index(max(measured))]
    print(f"\n=== {workload}: measured (drained) best = {measured_best}")
    rows = {}
    for label, env in BASES:
        os.environ.update(env)
        scores, predicted = [], []
        for name in names:
            plan = PhysicalPlan.from_dict(plans[name][0])
            workers = max(1, int(plan.n_local_workers or 1))
            score = optimizer.calculate_dp_objective_cost(plan=plan)
            scores.append(score)
            predicted.append(1000.0 * workers / score if score > 0 else math.inf)
        rows[label] = (scores, predicted)
        rho = fit.spearman(measured, [-value for value in scores])
        model_best = names[scores.index(min(scores))]
        errors = [
            abs(math.log2(p / m)) for p, m in zip(predicted, measured)
        ]
        print(
            f"\n  basis = {label}: rho={rho:.2f} "
            f"model_best={model_best} "
            f"median|log2|={sorted(errors)[len(errors)//2]:.2f}"
        )
        print(
            f"  {'planner':<30}{'W':>4}{'measured':>10}{'predicted':>11}"
            f"{'ratio':>7}"
        )
        for name, plan_dict, throughput, score, value in zip(
            names, [plans[n][0] for n in names], measured, scores, predicted
        ):
            plan = PhysicalPlan.from_dict(plan_dict)
            workers = max(1, int(plan.n_local_workers or 1))
            print(
                f"  {name:<30}{workers:>4}{throughput:>10.0f}{value:>11.0f}"
                f"{value / throughput:>7.2f}"
            )


if __name__ == "__main__":
    main()
