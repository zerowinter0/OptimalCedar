"""One staged tier, INFO logging, to see the staged decisions."""
import logging
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.staged_ablation_optimizer import (  # noqa: E402
    StagedBoundaryOptimizer,
    StagedBoundaryAffineOptimizer,
    StagedWorkersBoundaryAffineOptimizer,
)
from score_plan_cost_models import build_feature, plan_chain  # noqa: E402

workload, profile_path, tier = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
CLS = {
    "boundary": (StagedBoundaryOptimizer, 31),
    "affine": (StagedBoundaryAffineOptimizer, 32),
    "w": (StagedWorkersBoundaryAffineOptimizer, 33),
}[tier]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
)
profile = yaml.safe_load(profile_path.read_text())
feature = build_feature(workload)
optimizer = CLS[0]()
feature.set_optimizer(optimizer)
plan = optimizer.run(
    profile,
    OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=workload.endswith("_cache"),
        num_samples=0,
        use_my_optimizer=CLS[1],
        reorder_timeout_sec=7200.0,
    ),
)
print("PLAN:", plan_chain(plan))
