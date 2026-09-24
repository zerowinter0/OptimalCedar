"""Chapter 3.4: frozen-profile PICO scoring of the real U/P/F pipeline plans.

The three plans differ only in the fusion boundary of the
`to_float -> crop -> flip` block (everything else identical, W=1).  This script
prices each plan with

  * PICO   - the real optimizer objective (frozen profile, no target timing),
             reported as the objective and as objective / W (system cost),
  * Cedar  - `Optimizer.calculate_cost` from the previously exported
             `cedar_costs.json`,

and pairs them with the measured steady-state throughputs of the same plans
(`fusion_discount_20260923/expC`).  Predicted ratios are ratios of model costs;
they are ranking-oriented estimates, not wall-clock throughput predictions.

Usage:
  python -u scripts/pico_ch3_pipeline_scoring.py \
      --run-dir outputs/pico_ch3_20260924 \
      --fusion-run outputs/fusion_discount_20260923 \
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

from cedar.compose.optimizer import OptimizerOptions, PhysicalPlan  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)

from block_mechanism_common import build_feature  # noqa: E402
from cedar_block_cost_breakdown import load_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fusion-run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text())
    pico = SimpleDpWorkersWidthBoundaryOptimizer()
    feature = build_feature(batch_size=4)
    feature.set_optimizer(pico)
    pico.profiled_stats = profile
    pico.options = OptimizerOptions(
        enable_prefetch=True, est_throughput=None, available_local_cpus=64,
        enable_offload=True, enable_reorder=True, enable_local_parallelism=True,
        enable_fusion=True, enable_caching=False, num_samples=0,
        use_my_optimizer=27, reorder_timeout_sec=7200.0,
    )
    pico._validate_stats()
    pico._init_stats()
    pico._prepare_dp_metadata(pico._get_linear_inner_ops())

    cedar_costs = json.loads(
        (args.fusion_run / "cedar_costs.json").read_text()
    )
    cedar_by_label = {
        entry["label"]: entry for entry in cedar_costs["plans"]
    }
    throughput: Dict[str, List[float]] = {}
    for path in sorted((args.fusion_run / "expC/throughput").glob("*.json")):
        payload = json.loads(path.read_text())
        label = path.stem.rsplit("_r", 1)[0]
        throughput.setdefault(label, []).append(
            payload["throughput_samples_per_sec"]
        )

    rows: List[Dict[str, Any]] = []
    for label in ("U", "P", "F"):
        plan_path = args.fusion_run / "expC/plans" / f"{label}.yaml"
        plan = load_plan(plan_path)
        objective = pico.calculate_dp_objective_cost(plan=plan)
        workers = max(1, int(plan.n_local_workers or 1))
        rates = throughput.get(label, [])
        cedar = cedar_by_label.get(label, {})
        rows.append(
            {
                "plan": label,
                "workers": workers,
                "pico_objective": objective,
                "pico_system_cost_ms_per_record": objective / workers,
                "cedar_cost_ms_per_record": cedar.get(
                    "whole_plan_cost_ms_per_record"
                ),
                "cedar_block_cost_ms_per_record": cedar.get(
                    "block_charged_cost_ms_per_record"
                ),
                "cedar_block_external_ms_per_record": cedar.get(
                    "block_external_cost_ms_per_record"
                ),
                "measured_throughput_mean": (
                    statistics.fmean(rates) if rates else None
                ),
                "measured_throughput_stdev": (
                    statistics.stdev(rates) if len(rates) > 1 else 0.0
                ),
                "measured_repeats": len(rates),
            }
        )

    base = rows[0]
    for row in rows:
        row["pico_predicted_speedup_vs_U"] = (
            base["pico_system_cost_ms_per_record"]
            / row["pico_system_cost_ms_per_record"]
            if row["pico_system_cost_ms_per_record"] else None
        )
        row["cedar_predicted_speedup_vs_U"] = (
            base["cedar_cost_ms_per_record"] / row["cedar_cost_ms_per_record"]
            if row["cedar_cost_ms_per_record"] else None
        )
        row["measured_speedup_vs_U"] = (
            row["measured_throughput_mean"] / base["measured_throughput_mean"]
            if row["measured_throughput_mean"]
            and base["measured_throughput_mean"] else None
        )
        row["pico_speedup_error_pct"] = (
            (row["pico_predicted_speedup_vs_U"] - row["measured_speedup_vs_U"])
            / row["measured_speedup_vs_U"] * 100
            if row["measured_speedup_vs_U"] else None
        )
        row["cedar_speedup_error_pct"] = (
            (row["cedar_predicted_speedup_vs_U"] - row["measured_speedup_vs_U"])
            / row["measured_speedup_vs_U"] * 100
            if row["measured_speedup_vs_U"] else None
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "pipeline_scoring.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.run_dir / "pipeline_scoring.json").write_text(
        json.dumps(
            {
                "plans": rows,
                "note": (
                    "PICO objective is the optimizer's additive score; the "
                    "system cost is objective / W. Cedar cost comes from the "
                    "native calculate_cost walk. Predicted speedups are ratios "
                    "of model costs and are ranking-oriented, not wall-clock "
                    "throughput predictions."
                ),
                "measured_ranking": [
                    r["plan"] for r in sorted(
                        rows,
                        key=lambda r: -(r["measured_throughput_mean"] or 0.0),
                    )
                ],
                "pico_ranking": [
                    r["plan"] for r in sorted(
                        rows, key=lambda r: r["pico_system_cost_ms_per_record"]
                    )
                ],
                "cedar_ranking": [
                    r["plan"] for r in sorted(
                        rows, key=lambda r: r["cedar_cost_ms_per_record"]
                    )
                ],
            },
            indent=2,
        )
    )
    print(f"{'plan':<5}{'PICO obj':>10}{'PICO/W':>9}{'Cedar':>9}"
          f"{'pred PICO':>10}{'pred Cedar':>11}{'measured':>10}{'meas speedup':>13}")
    for row in rows:
        print(f"{row['plan']:<5}{row['pico_objective']:>10.4f}"
              f"{row['pico_system_cost_ms_per_record']:>9.4f}"
              f"{row['cedar_cost_ms_per_record']:>9.4f}"
              f"{row['pico_predicted_speedup_vs_U']:>10.3f}"
              f"{row['cedar_predicted_speedup_vs_U']:>11.3f}"
              f"{row['measured_throughput_mean']:>10.2f}"
              f"{row['measured_speedup_vs_U']:>13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
