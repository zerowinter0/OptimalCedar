"""Export every intermediate Cedar's block model uses (experiment section 4).

All numbers come from the real implementation
(``cedar/compose/optimizer.py``): the Amdahl inversion in
``_calculate_pipe_cost`` and the fusion I/O discount in
``_calculate_cost_fused``.  Nothing is recomputed by hand.

Usage (inside the container):
  python -u scripts/cedar_block_cost_breakdown.py \
      --profile <shared.yaml> --out outputs/<run>/cedar_cost_breakdown.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    BLOCK_NAMES,
    BLOCK_ORDER,
    PIPE_BATCHER,
    PIPE_BLUR,
    PIPE_CROP,
    PIPE_FLIP,
    PIPE_GRAYSCALE,
    PIPE_IMAGE_READER,
    PIPE_JITTER,
    PIPE_NORMALIZE,
    PIPE_TO_FLOAT,
    block_plan,
    build_feature,
    load_base_plan,
    plan_payload,
)
from cedar.compose.optimizer import (  # noqa: E402
    Optimizer,
    OptimizerOptions,
    PipeDesc,
)
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_optimizer(profile: Dict[str, Any]):
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
    return feature, optimizer


def operator_row(
    optimizer: Optimizer, p_id: int, variant: PipeVariantType
) -> Dict[str, Any]:
    """Cedar's own single-operator offload computation, with intermediates."""
    profile = optimizer.profiled_stats
    baseline_tput = profile["baseline"]["throughput"]
    base_cost = optimizer._base_cost_map[p_id]
    fraction = optimizer._fractional_latencies[p_id]
    input_size = profile["baseline"]["input_sizes"][p_id]
    desc = PipeDesc(
        name=None,
        variant_type=variant,
        variant_ctx=PipeVariantContextFactory.create_context(
            variant_type=variant
        ),
    )
    row: Dict[str, Any] = {
        "pipe_id": p_id,
        "name": BLOCK_NAMES.get(p_id, str(p_id)),
        "variant": variant.name,
        "baseline_throughput_samples_per_sec": baseline_tput,
        "latency_share_f_i": fraction,
        "base_cost_ms_per_sample": base_cost,
        "baseline_input_bytes": input_size,
        "amdahl_threshold_total_speedup": 1.0 / (1.0 - fraction),
        "scaled_cost_before_amdahl_ms_per_sample": (
            input_size / profile["baseline"]["input_sizes"][p_id]
        )
        * base_cost,
    }
    if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY,
                   PipeVariantType.SMP):
        entry = profile["offloads"].get(variant.name, {}).get(p_id)
        if entry is None:
            row["status"] = "no_profile_entry"
            return row
        offload_tput = entry["throughput"]
        total_speedup = offload_tput / baseline_tput
        row["offload_throughput_samples_per_sec"] = offload_tput
        row["total_speedup"] = total_speedup
        row["clipped_to_zero"] = total_speedup >= 1.0 / (1.0 - fraction)
        row["cost_ms_per_sample"] = optimizer._calculate_pipe_cost(
            p_id, input_size, desc
        )
        if not row["clipped_to_zero"]:
            row["pipe_speedup"] = fraction / (
                (baseline_tput / offload_tput) - (1.0 - fraction)
            )
    else:
        row["cost_ms_per_sample"] = optimizer._calculate_pipe_cost(
            p_id, input_size, desc
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 64])
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    feature, optimizer = build_optimizer(profile)
    graph = optimizer.physical_plan.graph

    baseline_q0 = 1000.0 / profile["baseline"]["throughput"]
    input_size_map, output_size_map = optimizer._calculate_size_map(graph)

    operators = []
    for p_id in sorted(feature.logical_pipes):
        entry = {
            "pipe_id": p_id,
            "logical_name": feature.logical_pipes[p_id].get_logical_name(),
            "baseline_input_bytes": profile["baseline"]["input_sizes"].get(p_id),
            "baseline_output_bytes": profile["baseline"]["output_sizes"].get(p_id),
            "baseline_latency_ns": profile["baseline"]["latencies"].get(p_id),
        }
        if p_id in optimizer._base_cost_map:
            entry["latency_share_f_i"] = optimizer._fractional_latencies[p_id]
            entry["base_cost_ms_per_sample"] = optimizer._base_cost_map[p_id]
        operators.append(entry)

    single_offloads = {}
    for p_id, name in BLOCK_NAMES.items():
        single_offloads[name] = {}
        for variant in (PipeVariantType.INPROCESS, PipeVariantType.RAY,
                        PipeVariantType.SMP):
            row = operator_row(optimizer, p_id, variant)
            if variant is not PipeVariantType.INPROCESS:
                spec = {p_id: PipeDesc(
                    name=None,
                    variant_type=variant,
                    variant_ctx=PipeVariantContextFactory.create_context(
                        variant_type=variant
                    ),
                )}
                row["pipeline_cost_ms_per_sample"] = optimizer.calculate_cost(
                    graph, spec
                )
                row["pipeline_speedup_vs_baseline"] = (
                    baseline_q0 / row["pipeline_cost_ms_per_sample"]
                    if row["pipeline_cost_ms_per_sample"]
                    else None
                )
            single_offloads[name][variant.name] = row

    member_cost_map = {}
    for p_id in BLOCK_ORDER:
        member_cost_map[p_id] = optimizer._calculate_pipe_cost(
            p_id, input_size_map[p_id], None
        )
    io_base, io_fused = optimizer._calculate_cost_fused(
        {p_id: PipeDesc(None, PipeVariantType.INPROCESS) for p_id in BLOCK_ORDER},
        list(BLOCK_ORDER),
        input_size_map,
        output_size_map,
        member_cost_map,
    )

    base_plan = load_base_plan(args.base_plan)
    configs: Dict[str, Any] = {}
    for config in ("L-U", "L-F", "R-U", "R-F"):
        for workers in args.workers:
            plan = block_plan(base_plan, config, workers)
            from cedar.compose.optimizer import PhysicalPlan

            physical = PhysicalPlan.from_dict(plan_payload(plan))
            fused_blocks = [
                list(desc.fused_pipes)
                for desc in physical.pipe_descs.values()
                if desc.fused_pipes and len(desc.fused_pipes) > 1
            ]
            member_costs = {}
            fused_node_variant = None
            for desc in physical.pipe_descs.values():
                if desc.fused_pipes and set(desc.fused_pipes) == set(BLOCK_ORDER):
                    fused_node_variant = desc.variant_type
            for p_id in BLOCK_ORDER:
                # A fused block executes its members on the fused node's
                # backend; only unfused plans carry a variant per member.
                variant = (
                    fused_node_variant
                    if fused_node_variant is not None
                    else physical.pipe_descs[p_id].variant_type
                )
                if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
                    desc = PipeDesc(
                        name=None,
                        variant_type=variant,
                        variant_ctx=PipeVariantContextFactory.create_context(
                            variant_type=variant
                        ),
                    )
                    member_costs[p_id] = optimizer._calculate_pipe_cost(
                        p_id, input_size_map[p_id], desc
                    )
                else:
                    member_costs[p_id] = member_cost_map[p_id]
            block_io = optimizer._calculate_cost_fused(
                {p_id: PipeDesc(None, PipeVariantType.INPROCESS)
                 for p_id in BLOCK_ORDER},
                list(BLOCK_ORDER),
                input_size_map,
                output_size_map,
                member_costs,
            )
            full_cost = optimizer.calculate_cost(
                physical.graph,
                physical_specs=physical.pipe_descs,
                fused_pipes=fused_blocks or None,
                caching_on=False,
                plan=physical,
            )
            configs[f"{config}_w{workers}"] = {
                "config": config,
                "workers": workers,
                "member_costs_ms_per_sample": {
                    BLOCK_NAMES[p_id]: member_costs[p_id] for p_id in BLOCK_ORDER
                },
                "member_cost_sum_ms_per_sample": sum(member_costs.values()),
                # Cedar charges the summed member cost when the block runs as
                # separate pipes, and the I/O-discounted block cost when the
                # plan materialises the fused node.
                "block_cost_ms_per_sample": (
                    block_io[1] if fused_blocks else sum(member_costs.values())
                ),
                "io_base_bytes": io_base,
                "io_fused_bytes": io_fused,
                "rho_io_fused_over_base": io_fused / io_base if io_base else None,
                "fused_block_cost_ms_per_sample": block_io[1],
                "block_cost_sum_ms_per_sample": block_io[0],
                "full_plan_cost_ms_per_sample": full_cost,
                "plan_is_fused": bool(fused_blocks),
                "plan_fused_members": fused_blocks,
            }

    payload = {
        "profile_path": str(args.profile),
        "profile_sha256": sha256(args.profile),
        "base_plan_path": str(args.base_plan),
        "base_plan_sha256": sha256(args.base_plan),
        "git_commit": git_commit(),
        "units": {
            "cost": "ms per source record of one worker (Cedar's model has no W)",
            "io": "bytes per source record",
            "latency_share_f_i": "fraction of the profiled whole-pipeline latency",
        },
        "baseline_cost_q0_ms_per_sample": baseline_q0,
        "fractional_latency_sums_to_one": sum(
            optimizer._fractional_latencies.values()
        ),
        "operators": operators,
        "single_operator_offloads": single_offloads,
        "fusion_io_discount": {
            "members": list(BLOCK_ORDER),
            "member_costs_ms_per_sample": {
                str(p_id): member_cost_map[p_id] for p_id in BLOCK_ORDER
            },
            "io_base_bytes": io_base,
            "io_fused_bytes": io_fused,
            "rho": io_fused / io_base if io_base else None,
            "fused_block_cost_ms_per_sample": block_io[1],
            "block_cost_sum_ms_per_sample": block_io[0],
        },
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in (
        "baseline_cost_q0_ms_per_sample", "fusion_io_discount")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
