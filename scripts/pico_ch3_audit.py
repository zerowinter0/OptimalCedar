"""Audit for PICO chapter 3: discount recomputation, model identity, windows.

Three sections:

  1. discount      - recompute Cedar's I/O ratio from *real* byte accounting
                     (the earlier fusion experiment hard-coded 0.5 for a
                      three-member equal-size block; the formula gives 1/3) and
                     recompute the derived rule predictions/errors;
  2. model_identity- itemise what "PICO" means here
                     (simple-dp-boundary-affine-W-width): local anchor, co-run
                     correction, backend anchors, affine response, width
                     scaling, fixed/byte boundary, W handling and the additive
                     score, each with file/function references;
  3. windows       - the measurement windows used by the reused experiments and
                     the actual coverage of the output-consistency checks.

Nothing here re-runs a measurement; the raw measurements are untouched.

Usage (inside the container):
  python -u scripts/pico_ch3_audit.py --run-dir outputs/pico_ch3_20260924 \
      --profile outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml \
      --out outputs/pico_ch3_20260924/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)
from cedar.compose.my_optimizer import MyOptimizer  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def location(func) -> Dict[str, Any]:
    try:
        source_file = inspect.getsourcefile(func)
        lines, start = inspect.getsourcelines(func)
    except (TypeError, OSError):
        return {}
    return {
        "function": getattr(func, "__qualname__", str(func)),
        "file": str(source_file),
        "line": start,
        "lines": len(lines),
    }


def cedar_discount(sizes_in: List[float], sizes_out: List[float]) -> Dict[str, Any]:
    """Cedar's own I/O accounting from _calculate_cost_fused, on bytes."""
    io_base = sizes_in[0]
    io_fused = sizes_in[0]
    for size in sizes_in[1:]:
        io_base += size * 2
    io_base += sizes_out[-1]
    io_fused += sizes_out[-1]
    return {
        "io_base_bytes": io_base,
        "io_fused_bytes": io_fused,
        "rho": io_fused / io_base if io_base else None,
    }


def discount_section(profile: Dict[str, Any]) -> Dict[str, Any]:
    baseline_in = profile["baseline"]["input_sizes"]
    baseline_out = profile["baseline"]["output_sizes"]

    # synthetic block: all three operators carry the same in/out tensor
    s = 244 * 244 * 4  # float32 (1, 244, 244)
    synthetic = {
        "tensor_bytes": s,
        "U_and_F_three_members": cedar_discount([s, s, s], [s, s, s]),
        "P_first_two_fused": cedar_discount([s, s], [s, s]),
        "previous_hardcoded_value": 0.5,
        "correction_note": (
            "The earlier fusion harness wrote 0.5 for the three-member block "
            "(4s in / 2s out). Cedar's formula charges 1 upload for the first "
            "member, 2x in/out for every inner member and 1 download for the "
            "last one: IO_U = s + 2s + 2s + s = 6s, IO_F = 2s, so rho = 1/3. "
            "The measurement values are unchanged; only derived quantities and "
            "their errors are recomputed."
        ),
    }

    real_three = cedar_discount(
        [baseline_in[7], baseline_in[6], baseline_in[5]],
        [baseline_out[7], baseline_out[6], baseline_out[5]],
    )
    real_two = cedar_discount(
        [baseline_in[7], baseline_in[6]], [baseline_out[7], baseline_out[6]]
    )
    return {
        "synthetic_block": synthetic,
        "real_block_to_float_crop_flip": {
            **real_three,
            "member_input_bytes": {
                "to_float": baseline_in[7],
                "crop": baseline_in[6],
                "flip": baseline_in[5],
            },
            "member_output_bytes": {
                "to_float": baseline_out[7],
                "crop": baseline_out[6],
                "flip": baseline_out[5],
            },
        },
        "real_block_to_float_crop": {
            **real_two,
            "member_input_bytes": {
                "to_float": baseline_in[7],
                "crop": baseline_in[6],
            },
            "member_output_bytes": {
                "to_float": baseline_out[7],
                "crop": baseline_out[6],
            },
        },
    }


def model_identity_section() -> Dict[str, Any]:
    pico = SimpleDpWorkersWidthBoundaryOptimizer
    components = {
        "class": {
            "name": pico.__name__,
            "selector": 27,
            "method_name": "simple_dp_workers_width_boundary",
            "local_label": "simple-dp-boundary-affine-W-width",
            **location(pico),
        },
        "local_compute_anchor": location(MyOptimizer._dp_affine_value),
        "co_run_correction": location(MyOptimizer._dp_co_run_factor),
        "backend_compute_anchor": location(
            SimpleDpWorkersWidthBoundaryOptimizer._calculate_pipe_cost
        ),
        "backend_shape_rescale": location(MyOptimizer._dp_affine_worker_cost),
        "width_scaling": location(DpOptimizer._dp_pipe_cost_at_parallelism),
        "candidate_widths": location(DpOptimizer._dp_candidate_parallelisms),
        "byte_boundary_plus_fixed": location(
            SimpleDpWorkersWidthBoundaryOptimizer._unscaled_boundary_transfer_ms
        ),
        "bundary_throughput_lookup": location(
            DpOptimizer._dp_boundary_throughput
        ),
        "worker_slices_and_W": location(
            SimpleDpWorkersWidthBoundaryOptimizer._worker_resource_groups
        ),
        "additive_objective": location(
            SimpleDpWorkersWidthBoundaryOptimizer._dp_accumulate_objective_cost
        ),
        "plan_replay_entry": location(
            SimpleDpWorkersWidthBoundaryOptimizer.calculate_dp_objective_cost
        ),
        "layer_pricing_entry": location(
            SimpleDpWorkersWidthBoundaryOptimizer._layered_boundary_cost_ms
        ),
    }
    return {
        "definition": (
            "PICO in this chapter is SimpleDpWorkersWidthBoundaryOptimizer "
            "(selector 27, method name simple_dp_workers_width_boundary): the "
            "joint DP that prices every operator as an isolated-measurement "
            "anchor rescaled by the fitted affine shape, adds a per-boundary "
            "fixed + byte/throughput term, searches W (replica count) and the "
            "stage width inside each W's CPU slice, and scores plans additively."
        ),
        "components": components,
        "explicitly_not_included": [
            "max-lane / max(Ray, SMP, local) objectives from PICO-Profilers "
            "style models",
            "overlap or non-overlap corrections",
            "GPU-specific lanes (LLaVA-style models)",
            "Cedar's Amdahl block clipping for operators inside an offloaded "
            "block (PICO prices compute anchors directly)",
        ],
        "score_units": (
            "objective = local_serial + ray_serial where ray_serial = W * "
            "sum(byte service per record); the system cost reported is "
            "objective / W (ms per source record). It is an additive estimate "
            "for ranking, not a wall-clock service-time prediction."
        ),
        "cedar_native_vs_extension": {
            "native": [
                "_calculate_pipe_cost (Amdahl inversion + clip)",
                "_calculate_cost_fused (I/O discount, backend agnostic)",
                "_offload_and_fuse / _local_fusion / _enumerate_fusions"
                " (backend enumeration; never INPROCESS fusion)",
            ],
            "extension": [
                "calculate_cost materialized-fused-node branch and multi-group "
                "fused_pipes (later additions)",
                "_calculate_pipe_cost INPROCESS early return (later addition)",
                "_is_optimizer_pipe plan argument (later addition)",
            ],
            "policy": (
                "INPROCESS fusion plans are only ever priced through the "
                "extension path and are never presented as original Cedar "
                "behaviour."
            ),
        },
    }


def window_section() -> Dict[str, Any]:
    return {
        "definitions": {
            "elapsed": (
                "caller monotonic clock, from before the first submit of a "
                "batch until the last stage's result is materialised; one "
                "batch in flight"
            ),
            "member_compute": (
                "wall time measured directly around each member callable "
                "inside the process that runs it (local process or Ray actor)"
            ),
            "other_overhead": (
                "batch elapsed minus the batch's summed member wall time; "
                "contains handoff, serialisation, scheduling and framework "
                "effects; it is NOT network time"
            ),
            "normalisation": (
                "batch-level sums first, then divide by the actual record "
                "count once; compute + other_overhead == elapsed is asserted "
                "for every batch"
            ),
            "non_additive": [
                "Ray client get_ms_per_sample contains actor compute, so it is "
                "never added to the member compute",
                "instrumented and uninstrumented runs are reported separately "
                "and never stacked into one bar",
            ],
        },
        "verification_coverage": {
            "fusion_harness_experiment_A": (
                "one batch (batch_size records) per cell at each compute "
                "level; the run reports max_abs_diff for F and P against U"
            ),
            "mechanism_harness_R_U_R_F": (
                "one batch (4 records) per configuration"
            ),
            "reorder_transfer_experiment": (
                "no bit-wise output comparison; it compares per-operator wall "
                "times and cost predictions only"
            ),
            "statement": (
                "Output equality is spot-checked per cell, not verified over "
                "the full run. This limits any claim that per-record values are "
                "identical beyond the checked batch."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    payload = {
        "profile_path": str(args.profile),
        "profile_sha256": sha256(args.profile),
        "dictionary": {},
        "discount": discount_section(profile),
        "model_identity": model_identity_section(),
        "measurement_windows": window_section(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    d = payload["discount"]
    print("discount recomputation:")
    print("  synthetic U/F three-member rho =",
          round(d["synthetic_block"]["U_and_F_three_members"]["rho"], 6),
          " (previously hard-coded 0.5)")
    print("  synthetic P (first two fused) rho =",
          round(d["synthetic_block"]["P_first_two_fused"]["rho"], 6))
    print("  real to_float->crop->flip rho =",
          round(d["real_block_to_float_crop_flip"]["rho"], 6),
          f"(IO {d['real_block_to_float_crop_flip']['io_base_bytes']:.0f} -> "
          f"{d['real_block_to_float_crop_flip']['io_fused_bytes']:.0f})")
    print("  real to_float->crop rho =",
          round(d["real_block_to_float_crop"]["rho"], 6))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
