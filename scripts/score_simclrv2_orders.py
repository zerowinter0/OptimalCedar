"""Price an explicit SimCLRv2 operator order with Cedar's own cost model.

The plan is built as a linear chain: LocalFSLister(9) -> ImageReader(8) ->
<ORDER> -> Batcher(0) -> Prefetcher(10), all INPROCESS (no fusion, all local),
and scored with ``Optimizer.calculate_cost`` from the campaign's shared profile.

Letters (as used in the experiment notes):
  L lister(9)  R reader(8)  G grayscale(3)  C crop(6)  B blur(2)
  H flip(5)    J jitter(4)  F to_float(7)   N normalize(1)  T batcher(0)

Usage (inside the container):
  python -u scripts/score_simclrv2_orders.py --order GCHBJFN GCBJHFN
  python -u scripts/score_simclrv2_orders.py --order 3,6,5,2,4,7,1
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

from cedar.compose.optimizer import (  # noqa: E402
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
)
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402

from block_mechanism_common import build_feature  # noqa: E402

DEFAULT_PROFILE = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml"
)

LETTERS = {
    "L": 9,
    "R": 8,
    "G": 3,
    "C": 6,
    "B": 2,
    "H": 5,
    "J": 4,
    "F": 7,
    "N": 1,
    "T": 0,
}
PIPE_NAMES = {
    9: "LocalFSListerPipe",
    8: "ImageReaderPipe",
    3: "MapperPipe_Grayscale",
    6: "MapperPipe_RandomResizedCrop",
    2: "MapperPipe_GaussianBlur",
    5: "MapperPipe_RandomHorizontalFlip",
    4: "MapperPipe_ColorJitter",
    7: "MapperPipe_to_float",
    1: "MapperPipe_Normalize",
    0: "BatcherPipe(batch_size=4)",
    10: "PrefetcherPipe",
}


def parse_order(raw: str) -> List[int]:
    tokens = [token for token in raw.replace("-", "").replace(",", " ").split() if token]
    if len(tokens) == 1 and all(ch in LETTERS for ch in tokens[0].upper()):
        return [LETTERS[ch] for ch in tokens[0].upper()]
    out = []
    for token in tokens:
        if token.upper() in LETTERS:
            out.append(LETTERS[token.upper()])
        else:
            out.append(int(token))
    return out


def build_optimizer(profile: Dict[str, Any]) -> Optimizer:
    feature = build_feature(batch_size=4)
    optimizer = Optimizer()
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
        use_my_optimizer=0,
        reorder_timeout_sec=7200.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    return optimizer


def linear_plan(order: List[int], workers: int = 1) -> PhysicalPlan:
    chain = order + [0, 10]
    graph = {p_id: {chain[index + 1]} for index, p_id in enumerate(chain[:-1])}
    graph[chain[-1]] = set()
    descs = {
        p_id: PipeDesc(
            name=PIPE_NAMES[p_id],
            variant_type=PipeVariantType.INPROCESS,
            variant_ctx=PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.INPROCESS
            ),
        )
        for p_id in chain
    }
    return PhysicalPlan(graph=graph, pipe_descs=descs, n_local_workers=workers)


def per_operator_costs(
    optimizer: Optimizer, plan: PhysicalPlan
) -> Dict[str, float]:
    """Cedar's contribution of each operator along this plan's path."""
    previous = optimizer.physical_plan
    optimizer.physical_plan = plan
    try:
        source = optimizer._get_source_p_id()
        output = optimizer._get_output_p_id(plan.graph)
        path, _ = optimizer._get_critical_path(plan.graph, source, output, plan)
        sizes = optimizer.profiled_stats["baseline"]["output_sizes"][path[0]]
        costs: Dict[str, float] = {}
        for p_id in path[1:]:
            if p_id == 0 or p_id == 10:
                continue
            costs[PIPE_NAMES[p_id]] = optimizer._calculate_pipe_cost(
                p_id, sizes, None
            )
            sizes = sizes * optimizer._data_size_ratio_map[p_id]
        return costs
    finally:
        optimizer.physical_plan = previous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", nargs="+", required=True,
                        help="letters (GCHBJFN) or comma-separated pipe ids")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    optimizer = build_optimizer(profile)
    baseline_declared = 1000.0 / profile["baseline"]["throughput"]

    rows = []
    for raw in args.order:
        ops = parse_order(raw)
        order = [9, 8] + ops
        plan = linear_plan(order, args.workers)
        cost = optimizer.calculate_cost(
            plan.graph,
            physical_specs=plan.pipe_descs,
            fused_pipes=None,
            caching_on=False,
            plan=plan,
        )
        rows.append(
            {
                "order": raw,
                "pipeline": " -> ".join(
                    PIPE_NAMES[p_id] for p_id in order + [0, 10]
                ),
                "cedar_cost_ms_per_record": cost,
                "workers": args.workers,
                "fusion": "none",
                "backend": "INPROCESS (all local)",
                "per_operator_ms": per_operator_costs(optimizer, plan),
            }
        )
    payload = {
        "profile_path": str(args.profile),
        "baseline_declared_order_cost_ms_per_record": baseline_declared,
        "results": rows,
    }
    header = f"{'order':<14}{'cedar cost (ms/record)':>24}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['order']:<14}{row['cedar_cost_ms_per_record']:>24.4f}")
    for row in rows:
        print(f"\n{row['order']}: {row['pipeline']}")
        print(
            "  per-operator: "
            + ", ".join(
                f"{name}={value:.4f}"
                for name, value in row["per_operator_ms"].items()
            )
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
