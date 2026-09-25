"""Before/after comparison for the un-truncated backend compute cost.

For each workload:
  * count the (operator, backend) pairs whose measured backend compute time is
    *higher* than the local anchor (those were silently clamped before);
  * re-score the plan the campaign stored (produced under the clamped rule)
    with the new scorer;
  * plan once with the new rule and compare plan identity, W and objective.

Usage (inside the container):
  python -u tmp_analysis/backend_rule_compare.py
"""

import copy
import json
import os
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

OUT = ROOT / "outputs/pico_final_w_only_20260924"
WORKLOADS = ("simclrv2", "simclrv2_cache")


def make_optimizer(profile):
    feature = build_feature(batch_size=4)
    optimizer = SimpleDpWorkersBoundaryAffineReprOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = copy.deepcopy(profile)
    optimizer.options = OptimizerOptions(
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
        reorder_timeout_sec=600.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(optimizer._get_linear_inner_ops())
    return feature, optimizer


def normalized(d: dict):
    return {
        "graph": {int(k): v for k, v in d["graph"].items()},
        "pipes": {int(k): v for k, v in d["pipes"].items()},
        "n_local_workers": d["n_local_workers"],
    }


def main() -> int:
    os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
    os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
    os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")
    rows = []
    for workload in WORKLOADS:
        profile_path = ROOT / f"outputs/affine_repr_profile_20260924/{workload}/shared.yaml"
        results = OUT / workload / "results" / "main_fast.json"
        if not profile_path.exists() or not results.exists():
            rows.append({"workload": workload, "status": "missing profile or stored plan"})
            continue
        profile = yaml.safe_load(profile_path.read_text())
        feature, optimizer = make_optimizer(profile)
        # 1) how many backend prices used to be clamped
        clamped = []
        for backend, section in (profile.get("offloads") or {}).items():
            for p_id, entry in section.items():
                if not isinstance(entry, dict) or "backend_compute" not in entry:
                    continue
                try:
                    mean = float(entry["backend_compute"]["mean_ms_per_sample"])
                    local = optimizer._repr_compute_cost(
                        int(p_id), *optimizer._repr_declared_features(int(p_id))
                    )
                except Exception:  # noqa: BLE001
                    continue
                if mean > local > 0:
                    clamped.append(
                        {"backend": backend, "pipe": int(p_id), "mean": mean,
                         "local": local, "ratio": mean / local}
                    )
        # 2) re-score the stored (pre-change) plan with the new scorer
        data = json.loads(results.read_text())
        run = next((r for r in data.get("runs", []) if r["optimizer"] == "pico_final"), None)
        stored_plan = None
        stored_score = None
        if run:
            plans = run.get("physical_plans_by_feature") or {}
            key = "feature" if "feature" in plans else (sorted(plans)[0] if plans else None)
            if key:
                stored_plan = plans[key]
                plan = PhysicalPlan.from_dict(normalized(stored_plan))
                optimizer._dp_selected_workers = max(1, int(plan.n_local_workers or 1))
                stored_score = float(optimizer.calculate_dp_objective_cost(plan=plan))
        # 3) plan once under the new rule
        feature2, optimizer2 = make_optimizer(profile)
        new_plan = optimizer2.run(str(profile_path), optimizer2.options)
        new_score = float(optimizer2.calculate_dp_objective_cost(plan=new_plan))
        rows.append(
            {
                "workload": workload,
                "pairs_checked": len(
                    [
                        1
                        for backend, section in (profile.get("offloads") or {}).items()
                        for entry in section.values()
                        if isinstance(entry, dict) and "backend_compute" in entry
                    ]
                ),
                "pairs_previously_clamped": len(clamped),
                "worst_ratio": max((c["ratio"] for c in clamped), default=None),
                "clamped_examples": sorted(clamped, key=lambda c: -c["ratio"])[:5],
                "stored_plan_rescored_objective": stored_score,
                "new_plan_objective": new_score,
                "new_plan_workers": int(new_plan.n_local_workers or 1),
                "objective_delta_pct": (
                    (new_score - stored_score) / stored_score * 100.0
                    if stored_score
                    else None
                ),
                "plan_changed": (
                    json.dumps(stored_plan, sort_keys=True)
                    != json.dumps(new_plan.to_dict(), sort_keys=True)
                    if stored_plan
                    else None
                ),
            }
        )
        print(
            f"{workload}: clamped={len(clamped)} "
            f"stored_plan_rescored={stored_score} new_plan={new_score} "
            f"delta={rows[-1]['objective_delta_pct']} "
            f"changed={rows[-1]['plan_changed']}"
        )
    target = OUT / "backend_rule_compare.json"
    target.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
