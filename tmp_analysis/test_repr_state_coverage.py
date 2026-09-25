"""Acceptance tests for the representation-state and coverage checks.

Three cases:
  1. a synthetic state conflict (two operators whose transitions do not
     commute) must be detected with a concrete counterexample;
  2. a fused-member curve that the profile never measured must be detected when
     a real materialized plan is validated;
  3. the real workloads' stored plans must pass the coverage check.

Usage (inside the container):
  python -u tmp_analysis/test_repr_state_coverage.py
"""

import copy
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

OUT = ROOT / "outputs/pico_final_w_only_20260924"
PROFILES = ROOT / "outputs/affine_repr_profile_20260924"


def load_optimizer(profile):
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
    return optimizer


def normalized(plan_dict):
    return {
        "graph": {int(k): v for k, v in plan_dict["graph"].items()},
        "pipes": {int(k): v for k, v in plan_dict["pipes"].items()},
        "n_local_workers": plan_dict["n_local_workers"],
    }


def stored_plan(workload, cell="main_fast"):
    data = json.loads((OUT / workload / "results" / f"{cell}.json").read_text())
    run = next(r for r in data["runs"] if r["optimizer"] == "pico_final")
    plans = run["physical_plans_by_feature"]
    key = "feature" if "feature" in plans else sorted(plans)[0]
    return plans[key]


def main() -> int:
    profile = yaml.safe_load((PROFILES / "simclrv2/shared.yaml").read_text())
    report = {"conflict_case": {}, "missing_curve_case": {}, "workloads": {}}

    # 1) synthetic conflict: make grayscale's transition depend on which of two
    #    operators ran first, by breaking commutativity with to_float.
    conflicting = copy.deepcopy(profile)
    model = conflicting["physical_model"]["compute_model"]
    transitions = model["class_transition"]
    klass = transitions["3"]["uint8:3ch"]          # grayscale on uint8 3ch
    transitions["3"][klass] = "uint8:3ch"          # no longer reduces channels
    transitions["3"]["float32:3ch"] = "uint8:3ch"  # and flips the dtype back
    optimizer = load_optimizer(conflicting)
    try:
        optimizer._repr_tables()
        report["conflict_case"] = {"detected": False}
    except RuntimeError as exc:
        report["conflict_case"] = {"detected": True, "error": str(exc)[:200]}

    # 2) fused-member curve removed: the plan fuses a member whose class curve
    #    the profile no longer contains.
    optimizer2 = load_optimizer(profile)
    plan_dict = stored_plan("simclrv2")
    fused = [
        desc.get("fused_pipes")
        for desc in plan_dict["pipes"].values()
        if desc.get("fused_pipes")
    ]
    member = fused[0][0]
    trace = optimizer2.assert_plan_covered(
        PhysicalPlan.from_dict(normalized(plan_dict))
    )
    used = next(item for item in trace if item["pipe"] == member)
    profile_missing = copy.deepcopy(profile)
    curves = profile_missing["physical_model"]["compute_model"]["operators"][
        str(member)
    ]["by_class"]
    # Remove exactly the representation class this member is priced in, so the
    # check has to notice the gap rather than a class the plan never uses.
    removed = used["class"]
    del curves[removed]
    optimizer3 = load_optimizer(profile_missing)
    plan = PhysicalPlan.from_dict(normalized(plan_dict))
    try:
        optimizer3.assert_plan_covered(plan)
        report["missing_curve_case"] = {"detected": False}
    except RuntimeError as exc:
        report["missing_curve_case"] = {
            "detected": True,
            "member": member,
            "removed_class": removed,
            "error": str(exc)[:200],
        }

    # 3) the shipped plans must pass
    # Only the SimCLRv2 feature is constructed here; LLaVA needs its own
    # dataset kwargs/feature builder, so its plan is validated in the campaign
    # path instead of being asserted with the wrong feature.
    for workload in ("simclrv2", "simclrv2_cache"):
        try:
            optimizer4 = load_optimizer(
                yaml.safe_load((PROFILES / workload / "shared.yaml").read_text())
            )
            plan = PhysicalPlan.from_dict(normalized(stored_plan(workload)))
            trace = optimizer4.assert_plan_covered(plan)
            report["workloads"][workload] = {
                "passed": True,
                "positions": len(trace),
                "state_report": optimizer4.repr_state_report(),
            }
        except Exception as exc:  # noqa: BLE001
            report["workloads"][workload] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}"[:200],
            }
    report["passed"] = (
        report["conflict_case"].get("detected")
        and report["missing_curve_case"].get("detected")
        and all(v.get("passed") for v in report["workloads"].values())
    )
    print(json.dumps(report, indent=1, default=str))
    (OUT / "repr_state_coverage.json").write_text(
        json.dumps(report, indent=1, default=str)
    )
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
