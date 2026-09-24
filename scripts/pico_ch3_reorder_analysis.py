"""Chapter 3.2: frozen-parameter prediction on the reordered full plans.

Two experiments on the existing every-record-traced order-transfer run
(``unopt_order_transfer_repeats_traceall_20260921``, all plans W=4, every stage
INPROCESS, no fusion/offload):

  experiment 1 (isolated input response): the declared order's measured
      per-operator time is the common anchor; compare
        proportional rule  t_ref * (x_plan / x_declared)
        affine rule        t_ref * (k*x + b)/(k*x_decl + b)
      against the measured per-operator time in the reordered plan.

  experiment 2 (actual PICO): price each plan with the frozen profile through
      the real optimizer (`SimpleDpWorkersWidthBoundaryOptimizer`), i.e.
      compute from the fitted affine layer only (INPROCESS), and compare the
      plan totals/ranking against the measured totals.  No plan's own measured
      time is used as an anchor here.

Outputs: operator_results.csv, summary.csv, reorder_analysis.json.

Usage:
  python -u scripts/pico_ch3_reorder_analysis.py \
      --run-dir outputs/pico_ch3_20260924 \
      --trace-run outputs/unopt_order_transfer_repeats_traceall_20260921 \
      --profile <frozen profile>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.compose.optimizer import PhysicalPlan  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)

NAMES = {
    0: "T  Batcher", 1: "N  Normalize", 2: "B  Blur", 3: "G  Grayscale",
    4: "J  Jitter", 5: "H  Flip", 6: "C  Crop", 7: "F  to_float",
    8: "R  ImageReader",
}
SMALL_OPS = (0, 1, 8)
PLANS = (("declared", "declared"), ("pico", "PICO"), ("cedar", "Cedar"),
         ("old-dp", "old-dp"))


def load_trace(trace_run: Path, cell: str) -> Dict[str, Dict[int, float]]:
    sizes: Dict[int, List[float]] = {}
    process: Dict[int, List[float]] = {}
    observations: Dict[int, int] = {}
    for path in sorted(trace_run.glob(f"reconcile_{cell}_*/worker_*.json")):
        data = json.loads(path.read_text())
        for pid, value in (data.get("input_sizes") or {}).items():
            sizes.setdefault(int(pid), []).append(float(value))
            observations[int(pid)] = observations.get(int(pid), 0) + int(
                (data.get("observations") or {}).get(pid, 0)
            )
        for pid, value in (
            data.get("process_latency_ns_per_sample") or {}
        ).items():
            process.setdefault(int(pid), []).append(float(value))
    return {
        "bytes": {p: statistics.mean(v) for p, v in sizes.items() if v},
        "measured_ms": {
            p: statistics.mean(v) / 1e6 for p, v in process.items() if v
        },
        "observations": observations,
    }


def pico_compute(optimizer, p_id: int, input_bytes: float) -> float:
    from cedar.compose.my_optimizer import MyOptimizer

    return MyOptimizer._dp_affine_value(
        optimizer, p_id, input_bytes
    ) * MyOptimizer._dp_co_run_factor(optimizer, p_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trace-run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    affine = profile["physical_model"]["operator_affine"]["operators"]
    declared = load_trace(args.trace_run, "declared")

    optimizer = SimpleDpWorkersWidthBoundaryOptimizer()
    from block_mechanism_common import build_feature
    from cedar.compose.optimizer import OptimizerOptions

    feature = build_feature(batch_size=4)
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(
        enable_prefetch=True, est_throughput=None, available_local_cpus=64,
        enable_offload=True, enable_reorder=True, enable_local_parallelism=True,
        enable_fusion=True, enable_caching=False, num_samples=0,
        use_my_optimizer=27, reorder_timeout_sec=7200.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(optimizer._get_linear_inner_ops())

    operator_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    plans_payload: Dict[str, Any] = {}
    for cell, label in PLANS:
        trace = load_trace(args.trace_run, cell)
        rows = []
        for p_id in sorted(NAMES):
            if p_id not in trace["measured_ms"] or p_id not in declared["measured_ms"]:
                continue
            x_plan = trace["bytes"].get(p_id, float("nan"))
            x_ref = declared["bytes"].get(p_id, float("nan"))
            t_meas = trace["measured_ms"][p_id]
            t_ref = declared["measured_ms"][p_id]
            entry = affine.get(str(p_id)) or {}
            k = float(entry.get("k_ms_per_byte", 0.0) or 0.0)
            b = float(entry.get("b_ms", 0.0) or 0.0)
            denom = k * x_ref + b
            proportional = t_ref * (x_plan / x_ref) if x_ref else float("nan")
            affine_pred = (
                t_ref * ((k * x_plan + b) / denom) if denom > 0
                else proportional
            )
            pico_pred = pico_compute(optimizer, p_id, x_plan)
            row = {
                "plan": label,
                "operator": NAMES[p_id],
                "calls_expected": declared["observations"].get(p_id),
                "calls_measured": trace["observations"].get(p_id),
                "input_bytes_per_record": x_plan,
                "reference_input_bytes": x_ref,
                "reference_ms_per_record": t_ref,
                "proportional_pred_ms": proportional,
                "affine_pred_ms": affine_pred,
                "pico_pred_ms": pico_pred,
                "measured_ms": t_meas,
                "error_proportional_pct": (
                    (proportional - t_meas) / t_meas * 100 if t_meas else None
                ),
                "error_affine_pct": (
                    (affine_pred - t_meas) / t_meas * 100 if t_meas else None
                ),
                "error_pico_pct": (
                    (pico_pred - t_meas) / t_meas * 100 if t_meas else None
                ),
                "tiny_operator": p_id in SMALL_OPS,
            }
            rows.append(row)
            operator_rows.append(row)

        big = [r for r in rows if not r["tiny_operator"]]
        summary = {
            "plan": label,
            "operators": len(rows),
            "measured_total_ms": sum(r["measured_ms"] for r in rows),
            "proportional_total_ms": sum(r["proportional_pred_ms"] for r in rows),
            "affine_total_ms": sum(r["affine_pred_ms"] for r in rows),
            "pico_total_ms": sum(r["pico_pred_ms"] for r in rows),
            "mape_proportional_all_pct": statistics.fmean(
                [abs(r["error_proportional_pct"]) for r in rows]
            ),
            "mape_affine_all_pct": statistics.fmean(
                [abs(r["error_affine_pct"]) for r in rows]
            ),
            "mape_pico_all_pct": statistics.fmean(
                [abs(r["error_pico_pct"]) for r in rows]
            ),
            "mape_proportional_nontiny_pct": statistics.fmean(
                [abs(r["error_proportional_pct"]) for r in big]
            ),
            "mape_affine_nontiny_pct": statistics.fmean(
                [abs(r["error_affine_pct"]) for r in big]
            ),
            "mape_pico_nontiny_pct": statistics.fmean(
                [abs(r["error_pico_pct"]) for r in big]
            ),
        }
        summary_rows.append(summary)

        plan_path = args.trace_run / "plans" / f"{cell}.yaml"
        plan_score = None
        if plan_path.exists():
            payload = yaml.safe_load(plan_path.read_text())
            payload = payload.get("physical_plan", payload)
            payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
            payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
            for desc in payload["pipes"].values():
                desc.setdefault("variant", "INPROCESS")
                desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})
                desc.setdefault("execution_resource", "cpu")
            physical = PhysicalPlan.from_dict(payload)
            plan_score = optimizer.calculate_dp_objective_cost(plan=physical)
        plans_payload[label] = {
            "summary": summary,
            "pico_plan_objective": plan_score,
            "plan_path": str(plan_path),
        }

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "operator_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operator_rows[0].keys()))
        writer.writeheader()
        writer.writerows(operator_rows)
    with (args.run_dir / "reorder_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.run_dir / "reorder_analysis.json").write_text(
        json.dumps(plans_payload, indent=2)
    )
    print(f"{'plan':<9}{'meas total':>11}{'prop':>9}{'affine':>9}{'PICO':>9}"
          f"{'MAPE prop':>11}{'MAPE aff':>10}{'MAPE PICO':>11}")
    for row in summary_rows:
        print(f"{row['plan']:<9}{row['measured_total_ms']:>11.3f}"
              f"{row['proportional_total_ms']:>9.3f}"
              f"{row['affine_total_ms']:>9.3f}{row['pico_total_ms']:>9.3f}"
              f"{row['mape_proportional_nontiny_pct']:>10.1f}%"
              f"{row['mape_affine_nontiny_pct']:>9.1f}%"
              f"{row['mape_pico_nontiny_pct']:>10.1f}%")
    print("(MAPE columns use the six size-changing mappers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
