"""Check the four properties of a candidate SimCLRv2 plan against cedar-opt.

  1. final Cedar cost lower than cedar-opt's final Cedar cost
  2. measured throughput higher than cedar-opt's
  3. operator order different from cedar-opt's
  4. its reorder-only plan (no fusion, all local) costs MORE under Cedar's
     model than cedar-opt's reorder-only plan

Costs come from the real ``Optimizer.calculate_cost``; throughputs come from
the campaign result JSONs of the same profile/protocol.

Usage (inside the container):
  python -u scripts/verify_required_plan_properties.py \
      --workload-dir outputs/ultimate_eight_optimizers_fix_20260921/simclrv2 \
      --candidates simple_dp_boundary simple_dp old_dp_boundary \
                   simple_dp_workers_width_boundary
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.compose.optimizer import (  # noqa: E402
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
)
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402

from block_mechanism_common import build_feature  # noqa: E402
from score_simclrv2_orders import PIPE_NAMES, linear_plan  # noqa: E402


def load_plan(path: Path) -> PhysicalPlan:
    payload = yaml.safe_load(path.read_text())
    payload = payload.get("physical_plan", payload)
    if "feature" in payload:
        payload = payload["feature"]
    if "feature_r0" in payload:
        payload = payload["feature_r0"]
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for desc in payload["pipes"].values():
        desc.setdefault("variant", "INPROCESS")
        desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})
        desc.setdefault("execution_resource", "cpu")
    return PhysicalPlan.from_dict(payload)


def operator_order(plan: PhysicalPlan) -> List[int]:
    predecessors = {c for succ in plan.graph.values() for c in succ}
    starts = [p for p in plan.graph if p not in predecessors]
    node, order = starts[0], []
    while True:
        order.append(node)
        if not plan.graph[node]:
            break
        node = next(iter(plan.graph[node]))
    sequence: List[int] = []
    for p_id in order:
        desc = plan.pipe_descs[p_id]
        if desc.fused_pipes:
            sequence.extend(desc.fused_pipes)
        elif p_id in PIPE_NAMES:
            sequence.append(p_id)
    return sequence


def build_optimizer(profile: Dict[str, Any]) -> Optimizer:
    feature = build_feature(batch_size=4)
    optimizer = Optimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(
        enable_prefetch=True, est_throughput=None, available_local_cpus=64,
        enable_offload=True, enable_reorder=True, enable_local_parallelism=True,
        enable_fusion=True, enable_caching=False, num_samples=0,
        use_my_optimizer=0, reorder_timeout_sec=7200.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    return optimizer


def final_cost(optimizer: Optimizer, plan: PhysicalPlan) -> float:
    fused_blocks = [
        list(desc.fused_pipes)
        for desc in plan.pipe_descs.values()
        if desc.fused_pipes and len(desc.fused_pipes) > 1
    ]
    return optimizer.calculate_cost(
        plan.graph,
        physical_specs=plan.pipe_descs,
        fused_pipes=fused_blocks or None,
        caching_on=False,
        plan=plan,
    )


def reorder_only_cost(
    optimizer: Optimizer, sequence: List[int], profile: Dict[str, Any]
) -> float:
    """The same operator order with no fusion and every stage local."""
    order = [9, 8] + [p for p in sequence if p in (3, 6, 2, 5, 4, 7, 1)]
    plan = linear_plan(order, 1)
    return optimizer.calculate_cost(
        plan.graph, physical_specs=plan.pipe_descs, fused_pipes=None,
        caching_on=False, plan=plan,
    )


def throughput(workload_dir: Path, method: str) -> Optional[float]:
    path = workload_dir / "results" / f"round1__{method}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    runs = payload.get("runs") or []
    return runs[0].get("throughput_samples_per_sec") if runs else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--reference", default="optimizer")
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/"
                        "simclrv2/profiles/shared.yaml",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    optimizer = build_optimizer(profile)

    reference_plan = load_plan(
        args.workload_dir / "plans" / f"round1__{args.reference}.yaml"
    )
    reference_order = operator_order(reference_plan)
    reference_final = final_cost(optimizer, reference_plan)
    reference_reorder = reorder_only_cost(optimizer, reference_order, profile)
    reference_throughput = throughput(args.workload_dir, args.reference)

    print(
        f"reference {args.reference}: order={reference_order} "
        f"final_cost={reference_final:.4f} reorder_only={reference_reorder:.4f} "
        f"throughput={reference_throughput:.1f}"
    )
    results = []
    for candidate in args.candidates:
        plan_path = args.workload_dir / "plans" / f"round1__{candidate}.yaml"
        if not plan_path.exists():
            print(f"{candidate}: missing plan {plan_path}")
            continue
        plan = load_plan(plan_path)
        sequence = operator_order(plan)
        cand_final = final_cost(optimizer, plan)
        cand_reorder = reorder_only_cost(optimizer, sequence, profile)
        cand_throughput = throughput(args.workload_dir, candidate)
        checks = {
            "final_cost_lower": cand_final < reference_final,
            "throughput_higher": (
                cand_throughput is not None
                and reference_throughput is not None
                and cand_throughput > reference_throughput
            ),
            "order_differs": sequence != reference_order,
            "reorder_only_higher": cand_reorder > reference_reorder,
        }
        results.append(
            {
                "candidate": candidate,
                "plan_path": str(plan_path),
                "order": sequence,
                "final_cost_ms_per_record": cand_final,
                "reorder_only_cost_ms_per_record": cand_reorder,
                "throughput_samples_per_sec": cand_throughput,
                "checks": checks,
                "satisfies_all": all(checks.values()),
            }
        )
        flag = "OK " if all(checks.values()) else "no "
        print(
            f"{flag}{candidate}: order={sequence} final={cand_final:.4f} "
            f"reorder_only={cand_reorder:.4f} throughput={cand_throughput} "
            f"checks={checks}"
        )

    payload = {
        "reference": {
            "method": args.reference,
            "order": reference_order,
            "final_cost_ms_per_record": reference_final,
            "reorder_only_cost_ms_per_record": reference_reorder,
            "throughput_samples_per_sec": reference_throughput,
        },
        "candidates": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
