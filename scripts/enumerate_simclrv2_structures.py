"""Enumerate plan structures for a fixed operator order and price them.

For an order of the seven mapper pipes, every composition into contiguous
segments is generated, and every segment independently takes one of:

    INPROCESS        single stage (segment length 1)
    INPROCESS-FUSED  local fused block (segment length >= 2)
    RAY / RAY-FUSED  Ray stage or Ray fused block
    SMP / SMP-FUSED  SMP stage or SMP fused block

Every resulting plan is scored with Cedar's own ``calculate_cost`` so the
cheapest structures for a deliberately expensive order become visible.

Usage (inside the container):
  python -u scripts/enumerate_simclrv2_structures.py --order FCGBHJN \
      --max-final-cost 8.4753 --top 15 --json-out /tmp/structures.json
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

from cedar.compose.optimizer import PhysicalPlan, PipeDesc  # noqa: E402
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402

from score_simclrv2_orders import PIPE_NAMES, parse_order  # noqa: E402
from search_simclrv2_order_fuse_plans import build_optimizer  # noqa: E402

VARIANTS = {
    "INPROCESS": PipeVariantType.INPROCESS,
    "RAY": PipeVariantType.RAY,
    "SMP": PipeVariantType.SMP,
}


def compositions(n: int) -> List[Tuple[int, ...]]:
    """All compositions of n into positive parts (as cut positions)."""
    out = []
    for mask in range(1 << (n - 1)):
        cuts = [0]
        for position in range(n - 1):
            if mask & (1 << position):
                cuts.append(position + 1)
        cuts.append(n)
        out.append(tuple(cuts))
    return out


def build_plan(
    order: Sequence[int],
    cuts: Sequence[int],
    variants: Sequence[str],
    workers: int = 64,
) -> Tuple[PhysicalPlan, List[str]]:
    nodes: List[Any] = [9, 8]
    description: List[str] = []
    fused_index = 0
    for index in range(len(cuts) - 1):
        segment = tuple(order[cuts[index]:cuts[index + 1]])
        variant = variants[index]
        if len(segment) == 1 and not variant.endswith("FUSED"):
            nodes.append(segment[0])
            description.append(f"{PIPE_NAMES[segment[0]]}[{variant}]"
                               if variant != "INPROCESS"
                               else PIPE_NAMES[segment[0]])
        else:
            nodes.append((f"fused{fused_index}", segment, variant))
            fused_index += 1
            description.append(
                "Fused{%s}[%s]" % (",".join(map(str, segment)), variant)
            )
    nodes.extend([0, 10])

    graph: Dict[Any, set] = {}
    descs: Dict[Any, PipeDesc] = {}
    for position, node in enumerate(nodes):
        successor = set()
        if position + 1 < len(nodes):
            nxt = nodes[position + 1]
            successor = {nxt if not isinstance(nxt, tuple) else nxt[0]}
        if isinstance(node, tuple):
            key, members, variant = node
            graph[key] = successor
            descs[key] = PipeDesc(
                name="FusedPipe",
                variant_type=VARIANTS[variant],
                variant_ctx=PipeVariantContextFactory.create_context(
                    variant_type=VARIANTS[variant]
                ),
                fused_pipes=list(members),
            )
        else:
            graph[node] = successor
            descs[node] = PipeDesc(
                name=PIPE_NAMES[node],
                variant_type=PipeVariantType.INPROCESS,
                variant_ctx=PipeVariantContextFactory.create_context(
                    variant_type=PipeVariantType.INPROCESS
                ),
            )
    return PhysicalPlan(graph=graph, pipe_descs=descs, n_local_workers=workers), description


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", required=True)
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/"
                        "simclrv2/profiles/shared.yaml",
    )
    parser.add_argument("--max-final-cost", type=float, default=8.4753)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    optimizer = build_optimizer(profile)
    order = parse_order(args.order)
    n = len(order)

    rows: List[Dict[str, Any]] = []
    for cuts in compositions(n):
        segments = len(cuts) - 1
        for choice in itertools.product(VARIANTS, repeat=segments):
            variants = []
            for index, base in enumerate(choice):
                length = cuts[index + 1] - cuts[index]
                if length >= 2:
                    variants.append(base)  # fused block regardless of label
                else:
                    variants.append(base)
            plan, description = build_plan(order, cuts, variants, args.workers)
            fused_blocks = [
                list(desc.fused_pipes)
                for desc in plan.pipe_descs.values()
                if desc.fused_pipes and len(desc.fused_pipes) > 1
            ]
            final = optimizer.calculate_cost(
                plan.graph,
                physical_specs=plan.pipe_descs,
                fused_pipes=fused_blocks or None,
                caching_on=False,
                plan=plan,
            )
            if final >= args.max_final_cost:
                continue
            rows.append(
                {
                    "order": args.order,
                    "structure": " -> ".join(description),
                    "cuts": list(cuts),
                    "variants": list(variants),
                    "final_cost": final,
                }
            )
    rows.sort(key=lambda row: row["final_cost"])
    print(f"order {args.order}: {len(rows)} structures below {args.max_final_cost:.4f}")
    for row in rows[: args.top]:
        print(f"  {row['final_cost']:.4f}  {row['structure']}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows[:200], indent=2))
        print(f"wrote {args.json_out} (top 200)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
