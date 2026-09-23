"""Cedar cost intermediates for the fusion-discount experiment (B and C).

Exports, with the real implementation:

  * for the real block: each member's Ray cost (after the Amdahl clip), the
    real serialized bytes in/out of every member, rho and the block cost the
    formula produces;
  * for the full-pipeline plans (U/P/F): the whole-plan cost, the block's
    contribution, and the block-external contribution, so the comparison can
    show that nothing outside the block changed.

Usage:
  python -u scripts/fusion_discount_cost_export.py \
      --run-dir outputs/<run> --block-pipes 7,6,5 \
      --plans expC/plans/U.yaml expC/plans/P.yaml expC/plans/F.yaml \
      --out outputs/<run>/cedar_costs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.compose.optimizer import PipeDesc  # noqa: E402
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402

from block_mechanism_common import load_base_plan  # noqa: E402
from cedar_block_cost_breakdown import load_plan  # noqa: E402
from scan_fusion_block_candidates import (  # noqa: E402
    NAMES,
    block_io,
    build_optimizer,
    plan_sizes,
    ray_cost,
)


def final_cost(optimizer, plan) -> float:
    fused_blocks = [
        list(desc.fused_pipes)
        for desc in plan.pipe_descs.values()
        if desc.fused_pipes and len(desc.fused_pipes) > 1
    ]
    return optimizer.calculate_cost(
        plan.graph, physical_specs=plan.pipe_descs,
        fused_pipes=fused_blocks or None, caching_on=False, plan=plan,
    )


def plan_cost_split(
    optimizer, plan, members: List[int], size_plan
) -> Dict[str, Any]:
    """Cedar's charged cost split into "inside the block" and "outside".

    The walk mirrors ``calculate_cost``: every path node is priced with its own
    desc; a materialized fused node is priced member-by-member and then
    multiplied by the I/O discount its member list implies.  Member input
    sizes come from ``size_plan`` (the same order without fusion), because a
    fused node hides its members from the plan's size walk while fusion only
    removes handoffs, not the data-size trajectory.
    """
    input_sizes, output_sizes = plan_sizes(optimizer, size_plan)
    source = optimizer._get_source_p_id()
    output = optimizer._get_output_p_id(plan.graph)
    path, _ = optimizer._get_critical_path(plan.graph, source, output, plan)
    member_set = set(members)
    total = 0.0
    block_charged = 0.0
    external = 0.0
    source_cost = float(
        optimizer._base_cost_map.get(optimizer._get_source_p_id(), 0.0)
    )
    total += source_cost
    external += source_cost
    nodes: List[Dict[str, Any]] = []
    for node in path[1:]:
        desc = plan.pipe_descs.get(node)
        if desc is None:
            continue
        is_fused = bool(desc.fused_pipes and len(desc.fused_pipes) > 1)
        if not is_fused and optimizer._is_optimizer_pipe(node, plan):
            # Prefetch/cache nodes carry zero service cost in calculate_cost.
            nodes.append(
                {
                    "kind": "optimizer_pipe",
                    "node": node,
                    "members": [node],
                    "variant": (
                        desc.variant_type.name if desc.variant_type else "INPROCESS"
                    ),
                    "charged_cost": 0.0,
                }
            )
            continue
        if is_fused:
            covers = list(desc.fused_pipes)
            io = block_io(covers, input_sizes, output_sizes)
            variant = desc.variant_type or PipeVariantType.INPROCESS
            member_costs = {}
            for p_id in covers:
                cost = (
                    ray_cost(optimizer, p_id, input_sizes[p_id])
                    if variant == PipeVariantType.RAY
                    else optimizer._calculate_pipe_cost(
                        p_id, input_sizes[p_id], None
                    )
                )
                member_costs[NAMES.get(p_id, str(p_id))] = cost
            member_sum = sum(member_costs.values())
            charge = member_sum * (io["rho_io_ratio"] or 0.0)
            nodes.append(
                {
                    "kind": "fused",
                    "node": node,
                    "members": covers,
                    "variant": variant.name,
                    "rho": io["rho_io_ratio"],
                    "member_costs": member_costs,
                    "member_cost_sum": member_sum,
                    "charged_cost": charge,
                }
            )
        else:
            covers = [node]
            variant = desc.variant_type or PipeVariantType.INPROCESS
            charge = (
                ray_cost(optimizer, node, input_sizes.get(node, 0.0))
                if variant == PipeVariantType.RAY
                else optimizer._calculate_pipe_cost(
                    node, input_sizes.get(node, 0.0), None
                )
            )
            nodes.append(
                {
                    "kind": "single",
                    "node": node,
                    "members": covers,
                    "variant": variant.name,
                    "charged_cost": charge,
                }
            )
        total += charge
        if set(covers) <= member_set:
            block_charged += charge
        else:
            external += charge

    block_io_bytes = block_io(members, input_sizes, output_sizes)
    member_costs = {
        NAMES.get(p, str(p)): ray_cost(optimizer, p, input_sizes[p])
        for p in members
    }
    return {
        "io_base_bytes": block_io_bytes["io_base_bytes_per_source_record"],
        "io_fused_bytes": block_io_bytes["io_fused_bytes_per_source_record"],
        "rho": block_io_bytes["rho_io_ratio"],
        "member_ray_costs_ms_per_record": member_costs,
        "member_ray_cost_sum_ms_per_record": sum(member_costs.values()),
        "member_input_bytes": {
            NAMES.get(p, str(p)): input_sizes[p] for p in members
        },
        "member_output_bytes": {
            NAMES.get(p, str(p)): output_sizes[p] for p in members
        },
        "charged_block_cost_ms_per_record": block_charged,
        "block_external_cost_ms_per_record": external,
        "walk_total_ms_per_record": total,
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--block-pipes", default="7,6,5")
    parser.add_argument("--plans", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/"
                        "simclrv2/profiles/shared.yaml",
    )
    parser.add_argument("--declared-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    feature, optimizer = build_optimizer(profile)
    members = [int(x) for x in args.block_pipes.split(",") if x]

    payload: Dict[str, Any] = {
        "profile_path": str(args.profile),
        "block_pipes": members,
        "block_member_names": [NAMES.get(p, str(p)) for p in members],
    }
    if args.declared_plan:
        declared = load_plan(args.declared_plan)
        payload["declared_order_block"] = plan_cost_split(
            optimizer, declared, members, declared
        )

    plans = []
    declared_size_plan = declared if args.declared_plan else None
    for index, raw in enumerate(args.plans):
        path = Path(raw)
        label = args.labels[index] if index < len(args.labels) else path.stem
        plan = load_plan(path)
        block = plan_cost_split(
            optimizer, plan, members,
            declared_size_plan if declared_size_plan is not None else plan,
        )
        total = final_cost(optimizer, plan)
        plans.append(
            {
                "label": label,
                "plan_path": str(path),
                "workers": plan.n_local_workers,
                "whole_plan_cost_ms_per_record": total,
                "block": block,
                "block_charged_cost_ms_per_record": block[
                    "charged_block_cost_ms_per_record"
                ],
                "block_external_cost_ms_per_record": block[
                    "block_external_cost_ms_per_record"
                ],
                "walk_vs_calculate_cost_delta": (
                    total - block["walk_total_ms_per_record"]
                ),
            }
        )
    payload["plans"] = plans
    if len(plans) >= 2:
        base = plans[0]["whole_plan_cost_ms_per_record"]
        for entry in plans[1:]:
            entry["predicted_speedup_vs_first"] = (
                base / entry["whole_plan_cost_ms_per_record"]
                if entry["whole_plan_cost_ms_per_record"] else None
            )
        external = [p["block_external_cost_ms_per_record"] for p in plans]
        payload["block_external_identical"] = max(external) - min(external) < 1e-9

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
