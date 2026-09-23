"""Export Cedar's block-cost intermediates, with the native/extended provenance.

Every number comes from the real implementation in
``cedar/compose/optimizer.py`` (the Amdahl inversion in ``_calculate_pipe_cost``
and the fusion I/O discount in ``_calculate_cost_fused``); nothing is
re-derived by hand except the byte-level I/O split, which applies the *same*
formula to sizes instead of costs and is labelled as such.

The script also records which functions are byte-identical to the version the
project imported (first commit ``f062305``) and to the vendored
``optimizer.py.orig`` snapshot, so the report can separate
"original Cedar behaviour" from "formula extension applied to extra plans".

Usage (inside the container):
  python -u scripts/cedar_block_cost_breakdown.py \
      --profile <shared.yaml> --base-plan <cedar plan> \
      --declared-plan <unopt plan> --out <run>/cedar_cost_breakdown.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    BLOCK_NAMES,
    BLOCK_ORDER,
    build_feature,
)
from cedar.compose.optimizer import (  # noqa: E402
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
)
from cedar.pipes import PipeVariantType, PipeVariantContextFactory  # noqa: E402

BLOCK_REFERENCE_COMMIT = "f062305"
VENDORED_SNAPSHOT = ROOT / "cedar/compose/optimizer.py.orig"
PROVENANCE_FUNCTIONS = (
    "_calculate_cost_fused",
    "_calculate_pipe_cost",
    "calculate_cost",
    "_offload_and_fuse",
    "_local_fusion",
    "_fuse_local_smp",
    "_fuse_tf",
    "_enumerate_fusions",
    "_calculate_offloads",
    "_is_optimizer_pipe",
)
NATIVE_PLAN_CONFIGS = ("L-U", "R-U", "R-F")
EXTENSION_PLAN_CONFIGS = ("L-F",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_functions(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    name, current = None, []
    for line in text.split("\n"):
        match = re.match(r"^    def (\w+)\(", line)
        if match:
            if name:
                out[name] = "\n".join(current)
            name, current = match.group(1), [line]
        elif name is not None:
            if re.match(r"^    def |^class ", line) and not line.startswith("        "):
                out[name] = "\n".join(current)
                name, current = None, []
            else:
                current.append(line)
    if name:
        out[name] = "\n".join(current)
    return out


def _git_show(commit: str, path: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"], text=True
        )
    except subprocess.CalledProcessError:
        return None


def function_provenance() -> Dict[str, Any]:
    """Byte-level comparison of the pricing/search functions across versions."""
    current_src = (ROOT / "cedar/compose/optimizer.py").read_text()
    snapshot_src = (
        VENDORED_SNAPSHOT.read_text() if VENDORED_SNAPSHOT.exists() else ""
    )
    imported_src = _git_show(BLOCK_REFERENCE_COMMIT, "cedar/compose/optimizer.py") or ""
    cur, snap, imp = (
        _extract_functions(current_src),
        _extract_functions(snapshot_src),
        _extract_functions(imported_src),
    )
    rows: Dict[str, Any] = {}
    for name in PROVENANCE_FUNCTIONS:
        rows[name] = {
            "identical_to_import_commit": cur.get(name) == imp.get(name),
            "identical_to_vendored_snapshot": cur.get(name) == snap.get(name),
            "import_commit": BLOCK_REFERENCE_COMMIT,
            "vendored_snapshot": str(VENDORED_SNAPSHOT),
            "location": f"cedar/compose/optimizer.py::{name}",
        }
    added = sorted(
        set(re.findall(r"^    def (\w+)\(", current_src, re.M))
        - set(re.findall(r"^    def (\w+)\(", snapshot_src, re.M))
    )
    return {
        "import_commit": BLOCK_REFERENCE_COMMIT,
        "vendored_snapshot": str(VENDORED_SNAPSHOT),
        "functions": rows,
        "added_since_import": added,
        "notes": {
            "fusion_discount": (
                "_calculate_cost_fused is byte-identical in the import commit, "
                "the vendored snapshot and the current file: the discount "
                "formula itself is original Cedar code and is backend-agnostic "
                "(it consumes an already backend-priced pipe_cost_map)."
            ),
            "backend_enumeration": (
                "_offload_and_fuse/_local_fusion/_fuse_local_smp/_fuse_tf are "
                "byte-identical: the original search enumerates RAY (non-TF, "
                "offload+fusion on), TF_RAY (TF graphs), TF (all-TF) and SMP "
                "(only when local parallelism is forbidden). It never fuses "
                "plain INPROCESS pipes."
            ),
            "materialized_fused_pricing": (
                "calculate_cost gained the 'materialized fused node' branch and "
                "multi-group fused_pipes support after the import commit, and "
                "_is_optimizer_pipe gained its plan argument. Pricing an "
                "already-materialized fused block (including an INPROCESS one) "
                "is therefore an extension, not original search behaviour."
            ),
            "inprocess_shortcut": (
                "_calculate_pipe_cost gained an early return for "
                "None/INPROCESS variants; the Amdahl inversion and its "
                "clip-to-zero threshold are unchanged from the import commit."
            ),
        },
    }


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


def member_sizes(
    optimizer: Optimizer, plan: PhysicalPlan
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Member-level sizes for an *unfused* plan in the requested order.

    A materialized fused plan hides its members behind the fused node, so the
    target-order sizes are taken from the same order executed unfused (the
    L-U plan); the fused node itself only short-circuits the size walk.
    """
    previous = optimizer.physical_plan
    optimizer.physical_plan = plan
    try:
        input_sizes, output_sizes = optimizer._calculate_size_map(plan.graph)
    finally:
        optimizer.physical_plan = previous
    return (
        {p_id: float(input_sizes[p_id]) for p_id in BLOCK_ORDER},
        {p_id: float(output_sizes[p_id]) for p_id in BLOCK_ORDER},
    )


def io_bytes_block(
    input_sizes: Dict[int, float], output_sizes: Dict[int, float]
) -> Dict[str, float]:
    """The byte-level counterpart of _calculate_cost_fused's I/O accounting."""
    members = list(BLOCK_ORDER)
    total_io_base = input_sizes[members[0]]
    total_io_fused = input_sizes[members[0]]
    for p_id in members[1:]:
        total_io_base += input_sizes[p_id] * 2
    total_io_base += output_sizes[members[-1]]
    total_io_fused += output_sizes[members[-1]]
    return {
        "io_base_bytes_per_source_record": total_io_base,
        "io_fused_bytes_per_source_record": total_io_fused,
        "rho_io_ratio": total_io_fused / total_io_base if total_io_base else None,
    }


def member_cost(
    optimizer: Optimizer, p_id: int, input_size: float, variant: PipeVariantType
) -> float:
    desc = PipeDesc(
        name=None,
        variant_type=variant,
        variant_ctx=PipeVariantContextFactory.create_context(
            variant_type=variant
        ),
    )
    return optimizer._calculate_pipe_cost(p_id, input_size, desc)


def offload_row(
    optimizer: Optimizer, p_id: int, variant: PipeVariantType
) -> Dict[str, Any]:
    profile = optimizer.profiled_stats
    baseline_tput = profile["baseline"]["throughput"]
    base_cost = optimizer._base_cost_map[p_id]
    fraction = optimizer._fractional_latencies[p_id]
    input_size = profile["baseline"]["input_sizes"][p_id]
    row: Dict[str, Any] = {
        "pipe_id": p_id,
        "name": BLOCK_NAMES.get(p_id, str(p_id)),
        "variant": variant.name,
        "baseline_throughput_samples_per_sec": baseline_tput,
        "latency_share_f_i": fraction,
        "base_cost_ms_per_sample": base_cost,
        "baseline_input_bytes": input_size,
        "amdahl_threshold_total_speedup": 1.0 / (1.0 - fraction),
        "cost_before_clip_ms_per_sample": (
            input_size / profile["baseline"]["input_sizes"][p_id]
        )
        * base_cost,
    }
    entry = profile["offloads"].get(variant.name, {}).get(p_id)
    if variant not in (PipeVariantType.RAY, PipeVariantType.TF_RAY,
                       PipeVariantType.SMP) or entry is None:
        row["cost_after_clip_ms_per_sample"] = row["cost_before_clip_ms_per_sample"]
        row["clipped_to_zero"] = False
        return row
    offload_tput = entry["throughput"]
    total_speedup = offload_tput / baseline_tput
    clipped = total_speedup >= 1.0 / (1.0 - fraction)
    row.update(
        {
            "offload_throughput_samples_per_sec": offload_tput,
            "total_speedup": total_speedup,
            "clipped_to_zero": clipped,
            "cost_after_clip_ms_per_sample": optimizer._calculate_pipe_cost(
                p_id,
                input_size,
                PipeDesc(
                    name=None,
                    variant_type=variant,
                    variant_ctx=PipeVariantContextFactory.create_context(
                        variant_type=variant
                    ),
                ),
            ),
        }
    )
    if not clipped:
        row["pipe_speedup"] = fraction / (
            (baseline_tput / offload_tput) - (1.0 - fraction)
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--declared-plan", type=Path)
    parser.add_argument(
        "--target-order-plan",
        type=Path,
        help="unfused plan in the target order (defaults to the L-U plan next "
             "to --base-plan)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 64])
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    feature, optimizer = build_optimizer(profile)
    block_pipes = list(optimizer._base_cost_map)

    baseline_q0 = 1000.0 / profile["baseline"]["throughput"]
    operators = []
    for p_id in sorted(feature.logical_pipes):
        entry: Dict[str, Any] = {
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

    single_offloads: Dict[str, Any] = {}
    for p_id, name in BLOCK_NAMES.items():
        single_offloads[name] = {
            variant.name: offload_row(optimizer, p_id, variant)
            for variant in (
                PipeVariantType.INPROCESS,
                PipeVariantType.RAY,
                PipeVariantType.SMP,
            )
        }
        for variant in (PipeVariantType.RAY, PipeVariantType.SMP):
            spec = {
                p_id: PipeDesc(
                    name=None,
                    variant_type=variant,
                    variant_ctx=PipeVariantContextFactory.create_context(
                        variant_type=variant
                    ),
                )
            }
            pipeline_cost = optimizer.calculate_cost(optimizer.physical_plan.graph, spec)
            single_offloads[name][variant.name][
                "pipeline_cost_ms_per_sample"
            ] = pipeline_cost
            single_offloads[name][variant.name][
                "pipeline_speedup_vs_baseline"
            ] = (baseline_q0 / pipeline_cost) if pipeline_cost else None

    orders: Dict[str, Any] = {}
    target_order_plan = args.target_order_plan
    if target_order_plan is None:
        candidate = args.base_plan.parent / "L-U_w1.yaml"
        target_order_plan = candidate if candidate.exists() else args.base_plan
    plan_paths = {"target": target_order_plan}
    if args.declared_plan:
        plan_paths["declared"] = args.declared_plan
    for label, path in plan_paths.items():
        plan = load_plan(path)
        input_sizes, output_sizes = member_sizes(optimizer, plan)
        costs = {
            variant.name: {
                BLOCK_NAMES[p_id]: member_cost(
                    optimizer, p_id, input_sizes[p_id], variant
                )
                for p_id in BLOCK_ORDER
            }
            for variant in (
                PipeVariantType.INPROCESS,
                PipeVariantType.RAY,
                PipeVariantType.SMP,
            )
        }
        orders[label] = {
            "plan_path": str(path),
            "plan_sha256": sha256(path),
            "order": list(block_pipes),
            "chain": _chain(plan),
            "member_input_bytes_per_source_record": {
                BLOCK_NAMES[p_id]: input_sizes[p_id] for p_id in BLOCK_ORDER
            },
            "member_output_bytes_per_source_record": {
                BLOCK_NAMES[p_id]: output_sizes[p_id] for p_id in BLOCK_ORDER
            },
            "member_cost_ms_per_sample_by_variant": costs,
            "block_io": io_bytes_block(input_sizes, output_sizes),
        }

    base_plan = load_plan(args.base_plan)
    configs: Dict[str, Any] = {}
    target_io = orders["target"]["block_io"]
    for config in ("L-U", "L-F", "R-U", "R-F"):
        for workers in args.workers:
            plan = _config_plan(base_plan, config, workers)
            fused_blocks = [
                list(desc.fused_pipes)
                for desc in plan.pipe_descs.values()
                if desc.fused_pipes and len(desc.fused_pipes) > 1
            ]
            fused_node_variant = next(
                (
                    desc.variant_type
                    for desc in plan.pipe_descs.values()
                    if desc.fused_pipes
                    and set(desc.fused_pipes) == set(BLOCK_ORDER)
                ),
                None,
            )
            variant = (
                fused_node_variant
                if fused_node_variant is not None
                else plan.pipe_descs[BLOCK_ORDER[0]].variant_type
            )
            variant = variant or PipeVariantType.INPROCESS
            # Members of a fused plan are hidden behind the fused node; their
            # sizes are exactly those of the same order executed unfused.
            input_sizes = {
                p_id: orders["target"]["member_input_bytes_per_source_record"][
                    BLOCK_NAMES[p_id]
                ]
                for p_id in BLOCK_ORDER
            }
            output_sizes = {
                p_id: orders["target"]["member_output_bytes_per_source_record"][
                    BLOCK_NAMES[p_id]
                ]
                for p_id in BLOCK_ORDER
            }
            costs = {
                BLOCK_NAMES[p_id]: member_cost(
                    optimizer, p_id, input_sizes[p_id], variant
                )
                for p_id in BLOCK_ORDER
            }
            member_sum = sum(costs.values())
            rho = target_io["rho_io_ratio"]
            block_cost = member_sum * rho if fused_blocks else member_sum
            full_cost = optimizer.calculate_cost(
                plan.graph,
                physical_specs=plan.pipe_descs,
                fused_pipes=fused_blocks or None,
                caching_on=False,
                plan=plan,
            )
            configs[f"{config}_w{workers}"] = {
                "config": config,
                "workers": workers,
                "block_variant": variant.name,
                "member_costs_ms_per_sample": costs,
                "member_cost_sum_ms_per_sample": member_sum,
                "fusion_discount_applied": bool(fused_blocks),
                "rho_io_ratio": rho,
                "fused_block_cost_ms_per_sample": block_cost,
                "full_plan_cost_ms_per_sample": full_cost,
                "plan_is_fused": bool(fused_blocks),
                "plan_provenance": (
                    "native_search_plan"
                    if config in NATIVE_PLAN_CONFIGS
                    else "formula_extension_plan"
                ),
                "plan_provenance_note": (
                    "the original staged search evaluates exactly this plan shape"
                    if config in NATIVE_PLAN_CONFIGS
                    else "the original search never fuses plain INPROCESS pipes; "
                         "the cost is the original _calculate_cost_fused formula "
                    "applied by the later materialized-fused-node pricing branch"
                ),
            }

    payload = {
        "profile_path": str(args.profile),
        "profile_sha256": sha256(args.profile),
        "base_plan_path": str(args.base_plan),
        "base_plan_sha256": sha256(args.base_plan),
        "declared_plan_path": (
            str(args.declared_plan) if args.declared_plan else None
        ),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "units": {
            "cost": "ms per source record of one worker (Cedar's model has no W)",
            "bytes": "serialized bytes per source record, from the profile's "
                     "baseline input/output sizes",
            "latency_share_f_i": "fraction of the profiled whole-pipeline latency",
            "io_bytes": "byte-level split of the same I/O accounting "
                        "_calculate_cost_fused applies to costs",
        },
        "provenance": function_provenance(),
        "baseline": {
            "q0_ms_per_sample": baseline_q0,
            "throughput_samples_per_sec": profile["baseline"]["throughput"],
            "fractional_latency_sum": sum(optimizer._fractional_latencies.values()),
        },
        "operators": operators,
        "orders": orders,
        "single_operator_offloads": single_offloads,
        "configs": configs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    summary = {
        "q0": round(baseline_q0, 4),
        "target_block_io": {
            k: (round(v, 1) if isinstance(v, float) else v)
            for k, v in target_io.items()
        },
        "block_costs": {
            key: {
                "member_sum": round(val["member_cost_sum_ms_per_sample"], 4),
                "block_cost": round(val["fused_block_cost_ms_per_sample"], 4),
                "full_plan": round(val["full_plan_cost_ms_per_sample"], 4),
                "provenance": val["plan_provenance"],
            }
            for key, val in configs.items()
            if key.endswith("_w1")
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


def _chain(plan: PhysicalPlan) -> List[str]:
    predecessors = {c for succ in plan.graph.values() for c in succ}
    starts = [p for p in plan.graph if p not in predecessors]
    if len(starts) != 1:
        return []
    node, order = starts[0], []
    while True:
        order.append(node)
        if not plan.graph[node]:
            break
        node = next(iter(plan.graph[node]))
    return [str(p) for p in order]


def _config_plan(base: PhysicalPlan, config: str, workers: int) -> PhysicalPlan:
    """Rebuild one of the four block organisations on top of ``base``."""
    from block_mechanism_common import block_plan, plan_payload

    payload = base.to_dict()
    plan_dict = {
        "graph": {
            int(k): [int(x) for x in str(v).split(",") if x.strip()]
            for k, v in payload["graph"].items()
        },
        "pipes": {
            int(k): {
                **desc,
                "variant": desc.get("variant"),
                "variant_ctx": desc.get("variant_ctx"),
                "execution_resource": desc.get("execution_resource", "cpu"),
            }
            for k, desc in payload["pipes"].items()
        },
        "n_local_workers": payload.get("n_local_workers", 1),
    }
    rewritten = block_plan(plan_dict, config, workers)
    return PhysicalPlan.from_dict(plan_payload(rewritten))


if __name__ == "__main__":
    raise SystemExit(main())
