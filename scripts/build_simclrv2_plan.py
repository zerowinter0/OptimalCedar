"""Build an executable SimCLRv2 fixed plan from an order + segment structure.

Example (order crop,float,grayscale,blur,flip,jitter,normalize with a local
block over the first three and an SMP block over the rest):

  python -u scripts/build_simclrv2_plan.py \
      --order CFGBHJN --structure INPROCESS:6,7,3 SMP:2,5,4,1 \
      --workers 64 --out outputs/<run>/plans/CFGBHJN_local_smp.yaml \
      --score-out outputs/<run>/scores/CFGBHJN_local_smp.json

The plan is written in the format ``DataSet._load_config`` expects and scored
with Cedar's own model (final cost with the declared structure, and the
reorder-only cost of the same order without fusion).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import plan_payload, write_plan  # noqa: E402
from score_simclrv2_orders import PIPE_NAMES, parse_order  # noqa: E402
from search_simclrv2_order_fuse_plans import build_optimizer  # noqa: E402

SMP_CTX = {
    "variant_type": "SMP",
    "n_procs": 1,
    "max_inflight": 10,
    "max_prefetch": 10,
    "use_threads": True,
    "disable_torch_parallelism": True,
}
RAY_CTX = {
    "variant_type": "RAY",
    "n_actors": 1,
    "max_inflight": 100,
    "max_prefetch": 100,
    "use_threads": True,
    "submit_batch_size": 16,
    "num_gpus": 0.0,
}
INPROCESS_CTX = {"variant_type": "INPROCESS"}


def build_plan_dict(
    order: Sequence[int], structure: Sequence[tuple[str, List[int]]], workers: int
) -> Dict[str, Any]:
    nodes: List[Any] = [9, 8]
    pipes: Dict[int, Dict[str, Any]] = {
        p_id: {
            "name": PIPE_NAMES[p_id],
            "variant": "INPROCESS",
            "variant_ctx": dict(INPROCESS_CTX),
            "execution_resource": "cpu",
        }
        for p_id in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    }
    next_pid = 11
    for variant, members in structure:
        if len(members) == 1:
            node = members[0]
            pipes[node] = {
                "name": PIPE_NAMES[node],
                "variant": variant,
                "variant_ctx": dict(
                    {"INPROCESS": INPROCESS_CTX, "SMP": SMP_CTX,
                     "RAY": RAY_CTX}[variant]
                ),
                "execution_resource": "cpu",
            }
            nodes.append(node)
            continue
        node = next_pid
        next_pid += 1
        ctx = {"INPROCESS": INPROCESS_CTX, "SMP": SMP_CTX, "RAY": RAY_CTX}[variant]
        pipes[node] = {
            "name": "FusedPipe",
            "variant": variant,
            "variant_ctx": dict(ctx),
            "fused_pipes": list(members),
            "execution_resource": "cpu",
        }
        nodes.append(node)
    nodes.extend([0, 10])
    graph = {}
    for index, node in enumerate(nodes):
        graph[node] = [nodes[index + 1]] if index + 1 < len(nodes) else []
    return {"graph": graph, "pipes": pipes, "n_local_workers": workers}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", required=True)
    parser.add_argument(
        "--structure", nargs="+", required=True,
        help="segments as VARIANT:id,id,... (INPROCESS/SMP/RAY)",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-out", type=Path)
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/"
                        "simclrv2/profiles/shared.yaml",
    )
    args = parser.parse_args()

    order = parse_order(args.order)
    structure = []
    for raw in args.structure:
        variant, _, ids = raw.partition(":")
        structure.append((variant.upper(), [int(x) for x in ids.split(",") if x]))
    flat = [p for _variant, members in structure for p in members]
    if flat != list(order):
        raise SystemExit(f"structure {flat} does not match order {list(order)}")

    plan = build_plan_dict(order, structure, args.workers)
    write_plan(plan, args.out)

    profile = yaml.safe_load(args.profile.read_text())
    optimizer = build_optimizer(profile)
    from cedar.compose.optimizer import PhysicalPlan

    physical = PhysicalPlan.from_dict(plan_payload(plan))
    fused_blocks = [
        list(desc.fused_pipes)
        for desc in physical.pipe_descs.values()
        if desc.fused_pipes and len(desc.fused_pipes) > 1
    ]
    final_cost = optimizer.calculate_cost(
        physical.graph, physical_specs=physical.pipe_descs,
        fused_pipes=fused_blocks or None, caching_on=False, plan=physical,
    )
    from score_simclrv2_orders import linear_plan

    reorder_plan = linear_plan([9, 8] + list(order), 1)
    reorder_cost = optimizer.calculate_cost(
        reorder_plan.graph, physical_specs=reorder_plan.pipe_descs,
        fused_pipes=None, caching_on=False, plan=reorder_plan,
    )
    summary = {
        "order": args.order,
        "order_ids": list(order),
        "structure": ["%s:%s" % (v, ",".join(map(str, m))) for v, m in structure],
        "workers": args.workers,
        "plan_path": str(args.out),
        "final_cedar_cost_ms_per_record": final_cost,
        "reorder_only_cedar_cost_ms_per_record": reorder_cost,
    }
    print(json.dumps(summary, indent=2))
    if args.score_out:
        args.score_out.parent.mkdir(parents=True, exist_ok=True)
        args.score_out.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
