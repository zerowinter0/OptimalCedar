"""Scan legal contiguous operator blocks for the fusion-discount experiment.

For every contiguous window (length >= 2) of the SimCLRv2 declared order this
reports, using Cedar's own pricing:

  * whether every member can run on Ray and has a Ray profile entry,
  * each member's Ray cost (after the Amdahl inversion / zero clip),
  * the real serialized bytes in/out of the block and the I/O ratio rho,
  * the block cost the formula would produce (sum(member) * rho),
  * whether the window is usable for the fusion-discount experiment
    (>= 2 Ray-fusable members, member sum > 0, preferably all members > 0).

No execution happens here.

Usage (inside the container):
  python -u scripts/scan_fusion_block_candidates.py \
      --profile outputs/.../simclrv2/profiles/shared.yaml \
      --declared-plan outputs/.../plans/round1__unopti.yaml \
      --out outputs/<run>/candidate_blocks.json
"""
from __future__ import annotations

import argparse
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
    PipeDesc,
)
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402

from block_mechanism_common import build_feature, load_base_plan  # noqa: E402
from cedar_block_cost_breakdown import load_plan  # noqa: E402

MAPPER_IDS = (7, 6, 5, 4, 3, 2, 1)
NAMES = {
    7: "to_float",
    6: "RandomResizedCrop",
    5: "RandomHorizontalFlip",
    4: "ColorJitter",
    3: "Grayscale",
    2: "GaussianBlur",
    1: "Normalize",
}


def build_optimizer(profile: Dict[str, Any]):
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
    return feature, optimizer


def ray_cost(optimizer: Optimizer, p_id: int, input_size: float) -> float:
    desc = PipeDesc(
        name=None,
        variant_type=PipeVariantType.RAY,
        variant_ctx=PipeVariantContextFactory.create_context(
            variant_type=PipeVariantType.RAY
        ),
    )
    return optimizer._calculate_pipe_cost(p_id, input_size, desc)


def plan_sizes(optimizer: Optimizer, plan) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Input/output bytes of every pipe along a plan's chain."""
    previous = optimizer.physical_plan
    optimizer.physical_plan = plan
    try:
        return optimizer._calculate_size_map(plan.graph)
    finally:
        optimizer.physical_plan = previous


def block_io(members: Sequence[int], input_sizes: Dict[int, float],
             output_sizes: Dict[int, float]) -> Dict[str, Any]:
    """Same I/O accounting _calculate_cost_fused uses, applied to sizes."""
    io_base = input_sizes[members[0]]
    io_fused = input_sizes[members[0]]
    for p_id in members[1:]:
        io_base += input_sizes[p_id] * 2
    io_base += output_sizes[members[-1]]
    io_fused += output_sizes[members[-1]]
    return {
        "io_base_bytes_per_source_record": io_base,
        "io_fused_bytes_per_source_record": io_fused,
        "rho_io_ratio": io_fused / io_base if io_base else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--declared-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    feature, optimizer = build_optimizer(profile)
    declared = load_plan(args.declared_plan)
    input_sizes, output_sizes = plan_sizes(optimizer, declared)

    pipes = feature.logical_pipes
    base_rows = {}
    for p_id in MAPPER_IDS:
        entry = profile["offloads"].get("RAY", {}).get(p_id)
        base_rows[p_id] = {
            "pipe_id": p_id,
            "name": NAMES[p_id],
            "declared_input_bytes": input_sizes[p_id],
            "declared_output_bytes": output_sizes[p_id],
            "has_ray_profile": entry is not None,
            "can_mutate_to_ray": pipes[p_id].can_mutate_to(PipeVariantType.RAY),
            "inprocess_cost_ms_per_record": optimizer._calculate_pipe_cost(
                p_id, input_sizes[p_id], None
            ),
            "ray_cost_ms_per_record": ray_cost(optimizer, p_id, input_sizes[p_id]),
            "ray_throughput": (entry or {}).get("throughput"),
            "latency_share_f_i": optimizer._fractional_latencies.get(p_id),
            "amdahl_threshold": (
                1.0 / (1.0 - optimizer._fractional_latencies[p_id])
                if p_id in optimizer._fractional_latencies else None
            ),
        }
        if entry is not None:
            base_rows[p_id]["ray_total_speedup"] = (
                entry["throughput"] / profile["baseline"]["throughput"]
            )
            base_rows[p_id]["clipped_to_zero"] = (
                base_rows[p_id]["ray_total_speedup"]
                >= base_rows[p_id]["amdahl_threshold"]
            )

    windows: List[Dict[str, Any]] = []
    order = list(MAPPER_IDS)
    for size in range(2, len(order) + 1):
        for start in range(0, len(order) - size + 1):
            members = order[start:start + size]
            block_input = {p: input_sizes[p] for p in members}
            block_output = {p: output_sizes[p] for p in members}
            io = block_io(members, block_input, block_output)
            member_costs = {NAMES[p]: ray_cost(optimizer, p, input_sizes[p])
                            for p in members}
            member_sum = sum(member_costs.values())
            block_cost = member_sum * (io["rho_io_ratio"] or 0.0)
            usable = (
                all(base_rows[p]["can_mutate_to_ray"] and base_rows[p]["has_ray_profile"]
                    for p in members)
                and member_sum > 0
            )
            windows.append(
                {
                    "members": list(members),
                    "member_names": [NAMES[p] for p in members],
                    "size": size,
                    "io_base_bytes": io["io_base_bytes_per_source_record"],
                    "io_fused_bytes": io["io_fused_bytes_per_source_record"],
                    "rho": io["rho_io_ratio"],
                    "ray_member_costs_ms_per_record": member_costs,
                    "ray_member_cost_sum_ms_per_record": member_sum,
                    "formula_block_cost_ms_per_record": block_cost,
                    "all_members_nonzero": all(v > 0 for v in member_costs.values()),
                    "nonzero_members": sum(1 for v in member_costs.values() if v > 0),
                    "usable": usable,
                }
            )

    usable = [row for row in windows if row["usable"]]
    usable.sort(
        key=lambda row: (
            not row["all_members_nonzero"],
            -row["nonzero_members"],
            -row["ray_member_cost_sum_ms_per_record"],
        )
    )
    payload = {
        "profile_path": str(args.profile),
        "declared_plan": str(args.declared_plan),
        "declared_order": list(order),
        "member_table": base_rows,
        "windows": windows,
        "usable_sorted": usable,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print("member Ray costs (declared order):")
    print(f"{'pipe':<22}{'in B':>10}{'ray cost':>12}{'clipped':>9}"
          f"{'ray tput':>12}")
    for p_id in MAPPER_IDS:
        row = base_rows[p_id]
        print(f"{row['name']:<22}{row['declared_input_bytes']:>10.0f}"
              f"{row['ray_cost_ms_per_record']:>12.4f}"
              f"{str(row.get('clipped_to_zero')):>9}"
              f"{(row['ray_throughput'] or 0):>12.3f}")
    print("\nusable contiguous windows (>=2 Ray members, member sum > 0):")
    print(f"{'members':<28}{'#nz':>4}{'sum':>10}{'rho':>8}{'block':>10}")
    for row in usable[:15]:
        print(f"{str(row['member_names']):<28}{row['nonzero_members']:>4}"
              f"{row['ray_member_cost_sum_ms_per_record']:>10.4f}"
              f"{row['rho']:>8.4f}"
              f"{row['formula_block_cost_ms_per_record']:>10.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
