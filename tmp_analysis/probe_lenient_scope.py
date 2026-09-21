"""The lenient replay must never open up during a DP search.

Monkeypatches the block provider so every ``candidate_for_order`` call records
whether the optimizer was in lenient (plan-pricing) mode, then runs one real
PICO search and prints the plan it returns.

Usage (inside the container):
  python -u tmp_analysis/probe_lenient_scope.py coco
"""

import logging
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import BlockCandidateProvider, _BlockCostIndex  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)
from score_plan_cost_models import build_feature, plan_chain  # noqa: E402

CAMPAIGN = ROOT / "outputs/ultimate_eight_optimizers_fix_20260921"
SEEN = []


def lenient(optimizer) -> bool:
    """Read the flag without requiring the change to be present (HEAD check)."""
    getter = getattr(optimizer, "_dp_lenient_replay", None)
    return bool(getter()) if callable(getter) else False


def main() -> int:
    logging.disable(logging.INFO)
    workload = sys.argv[1] if len(sys.argv) > 1 else "coco"
    profile = yaml.safe_load((CAMPAIGN / workload / "profiles/shared.yaml").read_text())

    original_index = _BlockCostIndex.__init__
    original_candidate = BlockCandidateProvider.candidate_for_order

    def index_spy(self, optimizer, *args, **kwargs):
        SEEN.append(("index", lenient(optimizer)))
        return original_index(self, optimizer, *args, **kwargs)

    def candidate_spy(self, *args, **kwargs):
        SEEN.append(("candidate", lenient(self.optimizer)))
        return original_candidate(self, *args, **kwargs)

    _BlockCostIndex.__init__ = index_spy
    BlockCandidateProvider.candidate_for_order = candidate_spy

    feature = build_feature(workload)
    optimizer = SimpleDpWorkersWidthBoundaryOptimizer()
    feature.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=workload.endswith("_cache"),
        num_samples=0,
        use_my_optimizer=27,
        reorder_timeout_sec=7200.0,
    )
    start = time.time()
    error = None
    plan = None
    try:
        plan = optimizer.run(profile, options)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - start

    flags = {flag for _kind, flag in SEEN}
    print(f"workload            : {workload}")
    print(f"planning seconds    : {elapsed:.1f}")
    print(f"search calls        : {len(SEEN)} (index={sum(1 for k, _ in SEEN if k == 'index')}, "
          f"candidate={sum(1 for k, _ in SEEN if k == 'candidate')})")
    print(f"lenient flags seen  : {flags}   <- must be {{False}} during the search")
    print("call sequence       : " + ", ".join(
        f"{kind}:{'L' if flag else 'strict'}" for kind, flag in SEEN
    ))
    print(f"plan                : {plan_chain(plan) if plan is not None else '—'}")
    if error:
        print(f"run() error         : {error}")
    else:
        print(f"plan cost (replay)  : {optimizer.calculate_dp_objective_cost(plan=plan):.4f}")
    print(f"lenient flags after scoring: {getattr(optimizer, '_dp_replay_lenient', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
