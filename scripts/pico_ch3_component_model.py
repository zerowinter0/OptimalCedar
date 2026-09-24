"""Component model (compute + fixed/byte boundary) for the chapter-3 blocks.

This is an *independent* predictor: every parameter comes either from the
frozen profile (real block) or from a separate isolated profiling run of the
controlled block (synthetic block).  No target-plan timing is used.

Components per Ray stage:
    compute  = anchor_ms * (k*x + b)/(k*x_ref + b) * co_run_factor
    boundary = fixed_latency_ms + (in_bytes + out_bytes)/throughput * 1000

Subcommands
-----------
  profile-synthetic : measure the isolated per-record compute anchor of the
                      synthetic operator for each repeat level K (separate
                      profiling pass; frozen afterwards)
  predict           : build the component predictions for
                      (a) the synthetic U/F/P block,
                      (b) the real to_float->crop[->flip] block,
                      (c) nothing else (pipeline scoring lives in the other
                          script)

Usage:
  python -u scripts/pico_ch3_component_model.py profile-synthetic \
      --out outputs/pico_ch3_20260924/profile/synthetic_anchor.json \
      --levels 1 4 16 64
  python -u scripts/pico_ch3_component_model.py predict \
      --run-dir outputs/pico_ch3_20260924 \
      --profile <frozen profile> --anchor profile/synthetic_anchor.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def kernel_tensor() -> torch.Tensor:
    base = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor([v for row in base for v in row],
                        dtype=torch.float32).reshape(1, 1, 3, 3)


def run_op(data: torch.Tensor, repeats: int, kernel: torch.Tensor) -> torch.Tensor:
    value = data.unsqueeze(0) if data.dim() == 3 else data
    for _ in range(repeats):
        value = torch.nn.functional.conv2d(value, kernel, padding=1).clamp_(0.0, 1.0)
    return value.squeeze(0) if data.dim() == 3 else value


def profile_synthetic(args: argparse.Namespace) -> int:
    """Isolated per-record compute anchor per K, measured on the remote actor."""
    import ray

    sys.path.insert(0, str(ROOT))
    from cedar.pipes.ray_variant import (
        configure_remote_ray_experiment,
        get_ray_actor_options,
    )
    from cedar.service import RayActor

    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True)
    payload = torch.load(args.inputs)
    pool = [item.to(torch.float32) / 255.0 for item in payload["records"][:64]]

    @ray.remote(num_cpus=0)
    class AnchorActor(RayActor):
        def measure(self, records, levels, cpu):
            import os

            if cpu is not None:
                try:
                    os.sched_setaffinity(0, {int(cpu)})
                except (AttributeError, OSError):
                    pass
            torch.set_num_threads(1)
            out = {}
            for level in levels:
                for _ in range(5):
                    for item in records:
                        run_op(item, level, kernel_tensor())
                per_call = []
                for repeat in range(3):
                    for item in records:
                        started = time.perf_counter_ns()
                        run_op(item, level, kernel_tensor())
                        per_call.append((time.perf_counter_ns() - started) / 1e6)
                out[str(level)] = {
                    "mean_ms_per_record": statistics.fmean(per_call),
                    "median_ms_per_record": statistics.median(per_call),
                    "stdev_ms_per_record": statistics.stdev(per_call),
                    "calls": len(per_call),
                    "actor": "isolated Ray actor, single thread, CPU pin",
                    "cpu_pin": cpu,
                }
            return out

        def describe(self):
            import os

            return {"node_ip": ray.util.get_node_ip_address(),
                    "cpu_count": os.cpu_count()}

    actor = AnchorActor.options(**get_ray_actor_options(0.0)).remote(
        "pico_ch3_anchor"
    )
    measured = ray.get(actor.measure.remote(pool, args.levels, args.remote_cpu))
    where = ray.get(actor.describe.remote())
    result = {"levels": measured, "where": where,
              "inputs": str(args.inputs), "records": len(pool)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    ray.shutdown()
    return 0


def boundary_params(profile: Dict[str, Any], variant: str = "RAY") -> Dict[str, float]:
    boundary = profile["physical_model"]["boundary"][variant]
    return {
        "fixed_latency_ms": float(boundary["fixed_latency_ms"]),
        "throughput_bytes_per_sec": float(boundary["throughput_bytes_per_sec"]),
        "r_squared": boundary.get("r_squared"),
        "method": boundary.get("method"),
    }


def member_compute_from_profile(
    profile: Dict[str, Any], p_id: int, input_bytes: float, variant: str = "RAY"
) -> Dict[str, float]:
    entry = profile["offloads"][variant].get(p_id) or profile["offloads"][variant].get(str(p_id))
    anchor = float(entry["backend_compute"]["mean_ms_per_sample"])
    affine = profile["physical_model"]["operator_affine"]["operators"][str(p_id)]
    k = float(affine["k_ms_per_byte"])
    b = float(affine["b_ms"])
    x_ref = float(affine["x_reference_bytes"])
    shape = ((k * input_bytes + b) / (k * x_ref + b)) if (k * x_ref + b) else 1.0
    factors = profile.get("calibration", {}).get("co_run_factors", {}) or {}
    co_run = float(factors.get(str(p_id), 1.0) or 1.0)
    return {
        "anchor_ms_per_record": anchor,
        "shape_factor": shape,
        "co_run_factor": co_run,
        "compute_ms_per_record": anchor * shape * co_run,
        "k_ms_per_byte": k,
        "b_ms": b,
        "x_reference_bytes": x_ref,
    }


def predict(args: argparse.Namespace) -> int:
    profile = yaml.safe_load(args.profile.read_text())
    anchors = json.loads(args.anchor.read_text())
    run_dir = args.run_dir
    boundary = boundary_params(profile)
    levels = sorted(int(k) for k in anchors["levels"])
    s = 244 * 244 * 4  # synthetic tensor bytes per record

    synthetic = {}
    for level in levels:
        compute = anchors["levels"][str(level)]["mean_ms_per_record"]
        # U: three stages -> three boundary crossings of the same payload
        per_crossing = (s + s) / boundary["throughput_bytes_per_sec"] * 1000.0
        fixed = boundary["fixed_latency_ms"]
        synthetic[str(level)] = {
            "member_compute_ms_per_record": compute,
            "per_crossing_byte_ms": per_crossing,
            "per_crossing_fixed_ms": fixed,
            "U": {
                "compute_total": 3 * compute,
                "boundary_total": 3 * (fixed + per_crossing),
                "predicted_total": 3 * compute + 3 * (fixed + per_crossing),
            },
            "F": {
                "compute_total": 3 * compute,
                "boundary_total": 1 * (fixed + per_crossing),
                "predicted_total": 3 * compute + (fixed + per_crossing),
            },
            "P": {
                "compute_total": 3 * compute,
                "boundary_total": 2 * (fixed + per_crossing),
                "predicted_total": 3 * compute + 2 * (fixed + per_crossing),
            },
        }

    # real block: to_float(7), crop(6), flip(5) in declared order
    baseline_in = profile["baseline"]["input_sizes"]
    baseline_out = profile["baseline"]["output_sizes"]
    members = [7, 6, 5]
    compute = {}
    for p_id in members:
        compute[p_id] = member_compute_from_profile(
            profile, p_id, baseline_in[p_id]
        )
    def crossing(in_bytes: float, out_bytes: float) -> Dict[str, float]:
        byte_ms = (in_bytes + out_bytes) / boundary["throughput_bytes_per_sec"] * 1000.0
        return {
            "fixed_ms": boundary["fixed_latency_ms"],
            "byte_ms": byte_ms,
            "total_ms": boundary["fixed_latency_ms"] + byte_ms,
        }

    real = {
        "boundary": boundary,
        "member_compute": {
            str(p): compute[p] for p in members
        },
        "U": {
            "boundaries": [crossing(baseline_in[p], baseline_out[p]) for p in members],
        },
        "P": {
            "boundaries": [
                crossing(baseline_in[7], baseline_out[6]),
                crossing(baseline_in[5], baseline_out[5]),
            ],
        },
        "F": {
            "boundaries": [crossing(baseline_in[7], baseline_out[5])],
        },
    }
    for org in ("U", "P", "F"):
        real[org]["compute_total"] = sum(
            compute[p]["compute_ms_per_record"] for p in members
        )
        real[org]["boundary_total"] = sum(
            b["total_ms"] for b in real[org]["boundaries"]
        )
        real[org]["predicted_total"] = (
            real[org]["compute_total"] + real[org]["boundary_total"]
        )

    payload = {
        "profile_path": str(args.profile),
        "anchor_path": str(args.anchor),
        "synthetic_tensor_bytes": s,
        "synthetic_block": synthetic,
        "real_block": real,
        "note": (
            "Component model: compute anchors + fixed/byte boundary terms. It is "
            "a serial service-time diagnostic built from the frozen profile "
            "(real block) or from an isolated profiling pass (synthetic block); "
            "no target-plan timing is used."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "synthetic": {
            k: {
                org: round(v[org]["predicted_total"], 3)
                for org in ("U", "P", "F")
            }
            for k, v in synthetic.items()
        },
        "real": {org: round(real[org]["predicted_total"], 3) for org in ("U", "P", "F")},
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prof = sub.add_parser("profile-synthetic")
    prof.add_argument("--inputs", type=Path,
                      default=ROOT / "outputs/simclrv2_fusion_offload_mechanism_20260923/"
                                     "inputs/block_inputs.pt")
    prof.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 64])
    prof.add_argument("--remote-cpu", type=int, default=8)
    prof.add_argument("--ray-ip", default="172.23.166.105:6379")
    prof.add_argument("--out", type=Path, required=True)
    prof.set_defaults(func=profile_synthetic)

    pred = sub.add_parser("predict")
    pred.add_argument("--run-dir", type=Path, required=True)
    pred.add_argument("--profile", type=Path, required=True)
    pred.add_argument("--anchor", type=Path, required=True)
    pred.add_argument("--out", type=Path, required=True)
    pred.set_defaults(func=predict)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
