"""Export the final staged/joint plans and score both with one scorer.

The campaign harness reports each cell's plan cost with *that cell's* optimizer,
so the staged and joint numbers are not comparable.  This script re-scores both
materialised plans with a freshly initialised final-PICO scorer (same profile,
same compute/boundary model, same W handling), writes the two plans, and pulls
the staged optimizer's stage-selection record out of its log.

Usage (inside the container):
  python -u tmp_analysis/staged_joint_export.py <workload> [cell]
"""

import ast
import json
import re
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
CELL = sys.argv[2] if len(sys.argv) > 2 else "staged_fast"
RESULTS = ROOT / f"outputs/pico_final_w_only_20260924/{WORKLOAD}/results/{CELL}.json"
PROFILE = ROOT / f"outputs/affine_repr_profile_20260924/{WORKLOAD}/shared.yaml"
OUT = ROOT / "outputs/pico_final_w_only_20260924"


def score_plan(profile: dict, plan_dict: dict, feature) -> dict:
    """Score one materialised plan with the final PICO's own objective."""
    optimizer = SimpleDpWorkersBoundaryAffineReprOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = profile
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
        reorder_timeout_sec=3600.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(optimizer._get_linear_inner_ops())
    # JSON round-tripping turns the int keys of ``PhysicalPlan.to_dict`` into
    # strings; ``from_dict`` keeps them as given, and a string-keyed graph then
    # fails the linear-source check against the int-keyed descriptors.
    normalized = {
        "graph": {int(k): v for k, v in plan_dict["graph"].items()},
        "pipes": {int(k): v for k, v in plan_dict["pipes"].items()},
        "n_local_workers": plan_dict["n_local_workers"],
    }
    plan = PhysicalPlan.from_dict(normalized)
    optimizer._dp_selected_workers = max(1, int(plan.n_local_workers or 1))
    value = optimizer.calculate_dp_objective_cost(plan=plan)
    return {"objective": float(value), "workers": int(plan.n_local_workers or 1)}


def main() -> int:
    data = json.loads(RESULTS.read_text())
    profile = yaml.safe_load(PROFILE.read_text())
    plans_dir = OUT / WORKLOAD / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    summary = {"workload": WORKLOAD, "cell": CELL, "plans": {}, "staged_stages": {}}
    feature = build_feature(batch_size=4)
    for run in data["runs"]:
        name = run["optimizer"]
        by_feature = run.get("physical_plans_by_feature") or {}
        key = "feature" if "feature" in by_feature else sorted(by_feature)[0]
        plan_dict = by_feature[key]
        target = plans_dir / f"{name}_final_plan.yaml"
        target.write_text(yaml.safe_dump({"physical_plan": plan_dict}))
        scored = score_plan(profile, plan_dict, feature)
        fused = [
            (p, d.get("fused_pipes"))
            for p, d in plan_dict["pipes"].items()
            if d.get("fused_pipes")
        ]
        stages = [
            (d.get("name"), d.get("variant"))
            for d in plan_dict["pipes"].values()
            if d.get("variant") not in (None, "INPROCESS")
        ]
        summary["plans"][name] = {
            "plan_file": str(target.relative_to(ROOT)),
            "hammer_objective": scored["objective"],
            "workers": scored["workers"],
            "fused": fused,
            "parallel_stages": stages,
            "throughput_samples_per_sec": run.get("throughput_samples_per_sec"),
            "harness_plan_cost": run.get("plan_cost"),
            "setup_time_sec": run.get("setup_time_sec"),
        }
    # Staged stage-selection record straight from the staged optimizer's log.
    log = OUT / WORKLOAD / "logs" / f"{CELL}.log"
    if log.exists():
        text = log.read_text(errors="replace")
        selected = re.search(r"selected W=(\d+), cost=([\d.eE+-]+)", text)
        evidence = re.search(r"worker search evidence=(\[.*\])\s*$", text, re.M)
        stats = re.findall(r"Exact search stats: (\{.*\})\s*$", text, re.M)
        summary["staged_stages"] = {
            "selected": (
                {"workers": int(selected.group(1)), "cost_ms_per_record": float(selected.group(2))}
                if selected
                else None
            ),
            "worker_search_evidence": (
                ast.literal_eval(evidence.group(1)) if evidence else None
            ),
            "dp_search_stats": [ast.literal_eval(item) for item in stats[:3]],
            "log": str(log.relative_to(ROOT)),
        }
    target = OUT / "staged_joint.json"
    target.write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(
        {
            name: {
                "objective": entry["hammer_objective"],
                "W": entry["workers"],
                "fused": entry["fused"],
                "throughput": entry["throughput_samples_per_sec"],
            }
            for name, entry in summary["plans"].items()
        },
        indent=1,
    ))
    print("staged selection:", summary["staged_stages"].get("selected"))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
