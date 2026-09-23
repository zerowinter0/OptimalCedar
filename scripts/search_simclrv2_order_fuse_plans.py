"""Construct SimCLRv2 plans whose *order* is expensive but whose fused plan is cheap.

For every candidate operator order this computes, with Cedar's own model:

  reorder_only   the same order with no fusion and every stage local
  final_local    one local fused block covering a contiguous window
  final_ray      the same window fused on Ray (reference only)

and reports them next to the cedar-opt reference (order 3,6,2,5,4,7,1 with the
B/H/J block fused on Ray).

Usage (inside the container):
  python -u scripts/search_simclrv2_order_fuse_plans.py \
      --orders GCBHJFN GCBHFJN GCFBHJN FCGBHJN ... \
      --json-out outputs/<run>/order_fuse_search.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from score_simclrv2_orders import LETTERS, PIPE_NAMES, parse_order  # noqa: E402

MAPPER_IDS = (3, 6, 2, 5, 4, 7, 1)


def _desc(p_id: int, variant: PipeVariantType, fused: Sequence[int] | None = None) -> PipeDesc:
    return PipeDesc(
        name=("FusedPipe" if fused else PIPE_NAMES[p_id]),
        variant_type=variant,
        variant_ctx=PipeVariantContextFactory.create_context(
            variant_type=variant
        ),
        fused_pipes=list(fused) if fused else None,
    )


def build_plan(
    order: Sequence[int],
    fused_window: Sequence[int] | None = None,
    fused_variant: PipeVariantType = PipeVariantType.INPROCESS,
    workers: int = 64,
) -> PhysicalPlan:
    """Linear plan: lister -> reader -> order (window optionally fused) -> batcher."""
    nodes: List[Any] = [9, 8]
    fused_members = list(fused_window or [])
    if fused_members:
        index = order.index(fused_members[0])
        nodes.extend(order[:index])
        nodes.append(("fused", tuple(fused_members)))
        nodes.extend(order[index + len(fused_members):])
    else:
        nodes.extend(order)
    nodes.extend([0, 10])

    graph: Dict[Any, set] = {}
    descs: Dict[Any, PipeDesc] = {}
    for position, node in enumerate(nodes):
        key = node if not isinstance(node, tuple) else "fused"
        succ = set()
        if position + 1 < len(nodes):
            nxt = nodes[position + 1]
            succ = {nxt if not isinstance(nxt, tuple) else "fused"}
        graph[key] = succ
        if isinstance(node, tuple):
            descs[key] = _desc(0, fused_variant, node[1])
        else:
            descs[key] = _desc(node, PipeVariantType.INPROCESS)
    return PhysicalPlan(graph=graph, pipe_descs=descs, n_local_workers=workers)


def cost(optimizer: Optimizer, plan: PhysicalPlan) -> float:
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


def windows(order: Sequence[int], min_size: int = 2) -> List[Tuple[int, ...]]:
    """Contiguous windows that may legally be fused (mappers only)."""
    out = []
    for size in range(min_size, len(order) + 1):
        for start in range(0, len(order) - size + 1):
            out.append(tuple(order[start:start + size]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", nargs="+", required=True)
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/"
                        "simclrv2/profiles/shared.yaml",
    )
    parser.add_argument("--min-reorder-cost", type=float, default=11.0)
    parser.add_argument("--max-final-cost", type=float, default=8.4753)
    parser.add_argument("--reference-order", default="GCBHJFN")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    optimizer = build_optimizer(profile)

    reference_order = parse_order(args.reference_order)
    reference_plan = build_plan(reference_order, (2, 5, 4),
                                PipeVariantType.RAY, args.workers)
    reference_final = cost(optimizer, reference_plan)
    reference_reorder = cost(optimizer, build_plan(reference_order, None))
    print(f"reference cedar plan: order={reference_order} final={reference_final:.4f} "
          f"reorder_only={reference_reorder:.4f}")

    rows: List[Dict[str, Any]] = []
    for raw in args.orders:
        order = parse_order(raw)
        reorder_cost = cost(optimizer, build_plan(order, None))
        if reorder_cost < args.min_reorder_cost:
            continue
        for window in windows(list(order)):
            plan = build_plan(order, window, PipeVariantType.INPROCESS,
                              args.workers)
            final_local = cost(optimizer, plan)
            if final_local >= args.max_final_cost:
                continue
            ray_plan = build_plan(order, window, PipeVariantType.RAY,
                                  args.workers)
            rows.append(
                {
                    "order": raw,
                    "order_ids": list(order),
                    "fused_window": list(window),
                    "reorder_only_cost": reorder_cost,
                    "final_local_cost": final_local,
                    "final_ray_cost": cost(optimizer, ray_plan),
                    "reorder_gain_vs_reference": reorder_cost - reference_reorder,
                    "final_gain_vs_reference": reference_final - final_local,
                }
            )
    rows.sort(key=lambda r: (-r["reorder_only_cost"], r["final_local_cost"]))
    print(
        f"\n{'order':<11}{'window':<16}{'reorder':>9}{'final_L':>9}{'final_R':>9}"
        f"{'Δreorder':>10}{'Δfinal':>9}"
    )
    for row in rows[:30]:
        print(
            f"{row['order']:<11}{','.join(map(str, row['fused_window'])):<16}"
            f"{row['reorder_only_cost']:>9.4f}{row['final_local_cost']:>9.4f}"
            f"{row['final_ray_cost']:>9.4f}{row['reorder_gain_vs_reference']:>10.4f}"
            f"{row['final_gain_vs_reference']:>9.4f}"
        )
    print(f"\n{len(rows)} candidate structures pass the two cost filters")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "reference": {
                        "order": reference_order,
                        "final_cost": reference_final,
                        "reorder_only_cost": reference_reorder,
                    },
                    "candidates": rows,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
