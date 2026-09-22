"""Plan-only smoke test: staged ablation optimizers vs their DP counterparts."""
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    OldDpBoundaryOptimizer,
    SimpleDpBoundaryOptimizer,
    SimpleDpWorkersBoundaryOptimizer,
)
from cedar.compose.staged_ablation_optimizer import (  # noqa: E402
    StagedBoundaryOptimizer,
    StagedBoundaryAffineOptimizer,
    StagedWorkersBoundaryAffineOptimizer,
)
from score_plan_cost_models import build_feature, plan_chain  # noqa: E402

workload = sys.argv[1] if len(sys.argv) > 1 else "simclrv2"
profile_path = Path(sys.argv[2])
profile = yaml.safe_load(profile_path.read_text())
caching = workload.endswith("_cache")


def options(selector: int) -> OptimizerOptions:
    return OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=caching,
        num_samples=0,
        use_my_optimizer=selector,
        reorder_timeout_sec=7200.0,
    )


PAIRS = [
    ("staged-boundary", StagedBoundaryOptimizer, 31, OldDpBoundaryOptimizer, 28),
    ("staged-boundary-affine", StagedBoundaryAffineOptimizer, 32,
     SimpleDpBoundaryOptimizer, 21),
    ("staged-boundary-affine-W", StagedWorkersBoundaryAffineOptimizer, 33,
     SimpleDpWorkersBoundaryOptimizer, 25),
]

logging.disable(logging.WARNING)
rows = []
for label, staged_cls, staged_id, dp_cls, dp_id in PAIRS:
    runs = [("staged", staged_cls, staged_id)]
    if os.environ.get("SMOKE_SKIP_DP") != "1":
        runs.append(("dp", dp_cls, dp_id))
    for kind, cls, selector in runs:
        feature = build_feature(workload)
        optimizer = cls()
        feature.set_optimizer(optimizer)
        started = time.perf_counter()
        plan = optimizer.run(dict(profile), options(selector))
        elapsed = time.perf_counter() - started
        # price the plan with the matching DP cost model for a like-for-like
        # comparison
        pricer = dp_cls()
        pricer.init(optimizer.logical_pipes, optimizer.logical_graph)
        pricer.profiled_stats = dict(profile)
        pricer.options = options(dp_id)
        pricer._validate_stats()
        pricer._init_stats()
        pricer._prepare_dp_metadata(pricer._get_linear_inner_ops())
        score = pricer.calculate_dp_objective_cost(plan=plan)
        workers = max(1, int(plan.n_local_workers or 1))
        rows.append({
            "tier": label,
            "kind": kind,
            "W": workers,
            "plan_seconds": round(elapsed, 1),
            "score": round(score, 4),
            "score_per_W": round(score / workers, 4),
            "chain": plan_chain(plan),
        })

print(json.dumps(rows, indent=1, ensure_ascii=False))
