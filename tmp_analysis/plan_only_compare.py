"""Plan once with the final optimizer and dump the plan + objective.

Used for the before/after comparison of a cost-model change: the campaign's
stored plans were produced by the old rule, so re-scoring them with the new
scorer and planning once with the new rule answers both halves of the question
(does the prediction change, does the selection change).

Usage (inside the container):
  python -u tmp_analysis/plan_only_compare.py <workload> <tag> [--no-w-search]
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import build_feature  # noqa: E402
from cedar.compose import OptimizerOptions, PhysicalPlan  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersBoundaryAffineReprOptimizer,
)

WORKLOAD = sys.argv[1] if len(sys.argv) > 1 else "simclrv2"
TAG = sys.argv[2] if len(sys.argv) > 2 else "unclamped"
PROFILE = ROOT / f"outputs/affine_repr_profile_20260924/{WORKLOAD}/shared.yaml"
OUT = ROOT / "outputs/pico_final_w_only_20260924"


def main() -> int:
    feature = build_feature(batch_size=4)
    optimizer = SimpleDpWorkersBoundaryAffineReprOptimizer()
    feature.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=False,
        num_samples=0,
        use_my_optimizer=39,
        reorder_timeout_sec=2400.0,
    )
    divergence = None
    try:
        plan = optimizer.run(str(PROFILE), options)
    except Exception as exc:  # noqa: BLE001
        divergence = str(exc)
        plan = optimizer.physical_plan
        print("DIVERGENCE:", divergence)
    evidence = getattr(optimizer, "_worker_search_evidence", None) or []
    for entry in evidence:
        print(
            "  W=%-3s score=%.6f plan_cost=%.6f status=%s"
            % (
                entry.get("workers"),
                float(entry.get("score") or 0.0),
                float(entry.get("plan_cost") or 0.0),
                entry.get("status"),
            )
        )
    print("selected W:", getattr(optimizer, "_dp_selected_workers", None),
          "selected limit:", getattr(optimizer, "_dp_selected_limit", None),
          "plan W:", plan.n_local_workers)
    plan_dict = plan.to_dict()
    objective = optimizer.calculate_dp_objective_cost(plan=plan)
    workers = max(1, int(plan.n_local_workers or 1))
    fused = [
        (p_id, desc.get("fused_pipes"))
        for p_id, desc in plan_dict["pipes"].items()
        if desc.get("fused_pipes")
    ]
    stages = [
        (desc.get("name"), desc.get("variant"))
        for desc in plan_dict["pipes"].values()
        if desc.get("variant") not in (None, "INPROCESS")
    ]
    target = OUT / f"plan_only_{TAG}.json"
    target.write_text(
        json.dumps(
            {
                "workload": WORKLOAD,
                "tag": TAG,
                "workers": workers,
                "objective": float(objective),
                "objective_per_worker": float(objective) / workers,
                "fused": fused,
                "parallel_stages": stages,
                "unpriced_backends": optimizer.repr_unpriced_backends(),
                "plan": plan_dict,
            },
            indent=1,
            default=float,
        )
    )
    print(
        f"{WORKLOAD}/{TAG}: W={workers} objective={float(objective):.4f} "
        f"objective/W={float(objective) / workers:.4f} fused={fused} stages={stages} "
        f"unpriced_backends={optimizer.repr_unpriced_backends()}"
    )
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
