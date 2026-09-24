"""Assemble the chapter-3 deliverable files from the individual analyses.

Writes into the unified result directory:
  protocol.json          planned cells / parameters / repeats / stop rules
  predictions.csv        every model prediction with its provenance
  measurements.csv       measured values with their round-level statistics
  boundary_results.csv   boundary paths, bytes and fixed/byte predictions
  summary.csv            error summaries and rankings
  figure_data.json       direct data for the chapter's figures/tables

Usage:
  python -u scripts/pico_ch3_assemble.py --run-dir outputs/pico_ch3_20260924
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--fusion-run", type=Path,
        default=Path("outputs/fusion_discount_20260923"),
    )
    args = parser.parse_args()
    run = args.run_dir
    fusion = args.fusion_run

    audit = json.loads((run / "audit.json").read_text())
    expa = json.loads((run / "expA_analysis.json").read_text())
    component = json.loads((run / "component_predictions.json").read_text())
    pipeline = json.loads((run / "pipeline_scoring.json").read_text())
    reorder = json.loads((run / "reorder_analysis.json").read_text())
    expb = read_csv(fusion / "expB/expB_summary.csv")

    # ---------------- protocol ----------------
    protocol = {
        "frozen_before_execution": True,
        "principle": (
            "Every model is profiled and frozen before the plans it predicts "
            "are executed; no target-plan timing is used to calibrate its own "
            "prediction. Reused experiments keep their original protocols."
        ),
        "profile": {
            "path": audit["profile_path"],
            "sha256": audit["profile_sha256"],
            "affine_operators": "physical_model.operator_affine.operators",
            "backend_anchors": "offloads.<BACKEND>.<pipe>.backend_compute",
            "boundary": "physical_model.boundary.<BACKEND>",
        },
        "experiment_A1_main": {
            "purpose": "fixed-rho vs measured fusion retention, uninstrumented",
            "cells": "U and F at K in {1,4,16,64}",
            "rounds": 3,
            "round_order": "rotated (recorded in expA1_main/expA1_run_order.json)",
            "batches_per_cell_per_round": 120,
            "batch_size": 4,
            "warmup_batches": 40,
            "instrumentation": "member timing disabled (end-to-end clock only)",
            "pins": "driver CPU 12, all remote actors CPU 8, one thread each",
            "stop_rule": "fixed at three rounds; no extra rounds added for results",
        },
        "experiment_A_instrumented": {
            "source": "outputs/fusion_discount_20260923/expA",
            "cells": "U and F at the same K, plus P at K=16",
            "rounds": 3,
            "instrumentation": "per-member wall and CPU timing enabled",
        },
        "synthetic_anchor_profile": {
            "purpose": "independent isolated compute anchor per K",
            "actor": "single remote Ray actor, CPU pin, one thread",
            "records": 64,
            "calls": 192,
            "path": "profile/synthetic_anchor.json",
        },
        "real_block": {
            "members": ["to_float(7)", "RandomResizedCrop(6)",
                        "RandomHorizontalFlip(5)"],
            "order": "declared order, parameters unchanged",
            "organisations": ["U (3 Ray stages)", "P (fuse 7+6)", "F (fuse 7+6+5)"],
            "measurement_source": str(fusion / "expB"),
            "rounds": 3,
        },
        "pipeline": {
            "plans": str(fusion / "expC/plans"),
            "workers": 1,
            "num_samples_per_round": 4000,
            "rounds": 3,
            "prediction_source": "frozen profile, no target timing",
        },
        "reorder": {
            "trace_run": "outputs/unopt_order_transfer_repeats_traceall_20260921",
            "plans": ["declared", "pico", "cedar", "old-dp"],
            "workers": 4,
            "all_inprocess": True,
            "per_record_tracing": True,
        },
    }
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2))

    # ---------------- predictions ----------------
    predictions: List[Dict[str, Any]] = []
    for entry in expa["cells"]:
        predictions.append(
            {
                "experiment": "A1_fusion_discount",
                "cell": f"K={entry['level']}",
                "model": "cedar_fixed_rho_rule",
                "prediction": entry["rule_T_F"],
                "unit": "ms per source record (serial)",
                "anchor": "measured T_U of the same K (rule check only)",
                "rho": entry["rho_corrected"],
                "source": "audit.json discount recomputation",
            }
        )
        predictions.append(
            {
                "experiment": "A_component",
                "cell": f"K={entry['level']}",
                "model": "component_model",
                "prediction": component["synthetic_block"][str(entry["level"])][
                    "F"
                ]["predicted_total"],
                "unit": "ms per source record (serial)",
                "anchor": "isolated per-K compute anchor + frozen boundary",
                "rho": None,
                "source": "component_predictions.json synthetic_block",
            }
        )
    for org in ("U", "P", "F"):
        predictions.append(
            {
                "experiment": "B_real_block",
                "cell": org,
                "model": "component_model",
                "prediction": component["real_block"][org]["predicted_total"],
                "unit": "ms per source record (serial)",
                "anchor": "frozen profile compute anchors + boundary",
                "rho": audit["discount"]["real_block_to_float_crop_flip"]["rho"]
                if org == "F" else None,
                "source": "component_predictions.json real_block",
            }
        )
    for row in pipeline["plans"]:
        predictions.append(
            {
                "experiment": "C_pipeline",
                "cell": row["plan"],
                "model": "pico_optimizer_objective_per_W",
                "prediction": row["pico_system_cost_ms_per_record"],
                "unit": "ms per source record (system cost; ranking-oriented)",
                "anchor": "frozen profile",
                "rho": None,
                "source": "pipeline_scoring.json",
            }
        )
        predictions.append(
            {
                "experiment": "C_pipeline",
                "cell": row["plan"],
                "model": "cedar_calculate_cost",
                "prediction": row["cedar_cost_ms_per_record"],
                "unit": "ms per source record (single worker)",
                "anchor": "frozen legacy profile entries",
                "rho": None,
                "source": "fusion_discount_20260923/cedar_costs.json",
            }
        )
    for label, payload in reorder.items():
        for key, model in (("pico_total_ms", "pico_affine_absolute"),
                           ("affine_total_ms", "affine_anchored_rule"),
                           ("proportional_total_ms", "proportional_anchored_rule")):
            predictions.append(
                {
                    "experiment": "D_reorder",
                    "cell": label,
                    "model": model,
                    "prediction": payload["summary"][key],
                    "unit": "ms per source record (sum over operators)",
                    "anchor": (
                        "frozen profile affine coefficients"
                        if model == "pico_affine_absolute"
                        else "declared-order measured anchor"
                    ),
                    "rho": None,
                    "source": "reorder_analysis.json",
                }
            )
    write_csv(
        run / "predictions.csv", predictions,
        ["experiment", "cell", "model", "prediction", "unit", "anchor", "rho",
         "source"],
    )

    # ---------------- measurements ----------------
    measurements: List[Dict[str, Any]] = []
    for entry in expa["cells"]:
        measurements += [
            {
                "experiment": "A1_fusion_discount",
                "cell": f"K={entry['level']}",
                "organisation": "U",
                "value": entry["T_U_ms_per_record"],
                "stdev_across_rounds": entry["T_U_stdev"],
                "rounds": 3,
                "unit": "ms per source record",
                "source": "expA1_main (member timing off)",
            },
            {
                "experiment": "A1_fusion_discount",
                "cell": f"K={entry['level']}",
                "organisation": "F",
                "value": entry["T_F_ms_per_record"],
                "stdev_across_rounds": entry["T_F_stdev"],
                "rounds": 3,
                "unit": "ms per source record",
                "source": "expA1_main (member timing off)",
            },
        ]
    for row in expb:
        rounds = sum(1 for r in expb if r["config"] == row["config"])
        measurements.append(
            {
                "experiment": "B_real_block",
                "cell": "to_float-crop-flip",
                "organisation": row["config"],
                "value": float(row["elapsed_ms_per_record_mean"]),
                "stdev_across_rounds": float(row["elapsed_ms_per_record_stdev"]),
                "rounds": rounds,
                "unit": "ms per source record",
                "source": str(fusion / "expB/expB_summary.csv"),
            }
        )
    for row in pipeline["plans"]:
        measurements.append(
            {
                "experiment": "C_pipeline",
                "cell": row["plan"],
                "organisation": row["plan"],
                "value": row["measured_throughput_mean"],
                "stdev_across_rounds": row["measured_throughput_stdev"],
                "rounds": row["measured_repeats"],
                "unit": "records per second (steady state)",
                "source": str(fusion / "expC/throughput"),
            }
        )
    for label, payload in reorder.items():
        measurements.append(
            {
                "experiment": "D_reorder",
                "cell": label,
                "organisation": label,
                "value": payload["summary"]["measured_total_ms"],
                "stdev_across_rounds": None,
                "rounds": 1,
                "unit": "ms per source record (sum over operators)",
                "source": "unopt_order_transfer_repeats_traceall_20260921",
            }
        )
    write_csv(
        run / "measurements.csv", measurements,
        ["experiment", "cell", "organisation", "value",
         "stdev_across_rounds", "rounds", "unit", "source"],
    )

    # ---------------- boundary results ----------------
    boundary_rows: List[Dict[str, Any]] = []
    real = component["real_block"]
    for org in ("U", "P", "F"):
        payload = real[org]
        for index, cross in enumerate(payload["boundaries"]):
            boundary_rows.append(
                {
                    "experiment": "B_real_block",
                    "organisation": org,
                    "boundary_index": index,
                    "fixed_ms_per_record": cross["fixed_ms"],
                    "byte_ms_per_record": cross["byte_ms"],
                    "total_ms_per_record": cross["total_ms"],
                    "boundary_params": json.dumps(real["boundary"]),
                }
            )
    for org in ("U", "F", "P"):
        per_cross = (
            (component["synthetic_tensor_bytes"] * 2)
            / real["boundary"]["throughput_bytes_per_sec"] * 1000.0
        )
        crossings = {"U": 3, "P": 2, "F": 1}[org]
        boundary_rows.append(
            {
                "experiment": "A_synthetic",
                "organisation": org,
                "boundary_index": "total",
                "fixed_ms_per_record": crossings
                * real["boundary"]["fixed_latency_ms"],
                "byte_ms_per_record": crossings * per_cross,
                "total_ms_per_record": crossings
                * (real["boundary"]["fixed_latency_ms"] + per_cross),
                "boundary_params": json.dumps(real["boundary"]),
            }
        )
    write_csv(
        run / "boundary_results.csv", boundary_rows,
        ["experiment", "organisation", "boundary_index", "fixed_ms_per_record",
         "byte_ms_per_record", "total_ms_per_record", "boundary_params"],
    )

    # ---------------- summary ----------------
    summary_rows: List[Dict[str, Any]] = []
    for entry in expa["cells"]:
        summary_rows.append(
            {
                "experiment": "A1_fusion_discount",
                "cell": f"K={entry['level']}",
                "model": "cedar_fixed_rho_rule",
                "measured": entry["T_F_ms_per_record"],
                "predicted": entry["rule_T_F"],
                "error_pct": entry["rule_error_pct"],
                "metric": "serial service time (F)",
            }
        )
        predicted_component = component["synthetic_block"][str(entry["level"])][
            "F"
        ]["predicted_total"]
        summary_rows.append(
            {
                "experiment": "A1_fusion_discount",
                "cell": f"K={entry['level']}",
                "model": "component_model",
                "measured": entry["T_F_ms_per_record"],
                "predicted": predicted_component,
                "error_pct": (predicted_component
                              - entry["T_F_ms_per_record"])
                / entry["T_F_ms_per_record"] * 100,
                "metric": "serial service time (F)",
            }
        )
    for org in ("U", "P", "F"):
        measured = next(
            (float(r["elapsed_ms_per_record_mean"]) for r in expb
             if r["config"] == org), None
        )
        if measured is None:
            continue
        predicted = component["real_block"][org]["predicted_total"]
        summary_rows.append(
            {
                "experiment": "B_real_block",
                "cell": "to_float-crop-flip",
                "model": "component_model",
                "measured": measured,
                "predicted": predicted,
                "error_pct": (predicted - measured) / measured * 100,
                "metric": f"serial service time ({org})",
            }
        )
    u_measured = next(
        (float(r["elapsed_ms_per_record_mean"]) for r in expb
         if r["config"] == "U"), None
    )
    rho_real = audit["discount"]["real_block_to_float_crop_flip"]["rho"]
    if u_measured:
        f_measured = next(
            (float(r["elapsed_ms_per_record_mean"]) for r in expb
             if r["config"] == "F"), None
        )
        summary_rows.append(
            {
                "experiment": "B_real_block",
                "cell": "to_float-crop-flip",
                "model": "cedar_fixed_rho_rule",
                "measured": f_measured,
                "predicted": u_measured * rho_real,
                "error_pct": (u_measured * rho_real - f_measured)
                / f_measured * 100,
                "metric": "serial service time (F), anchored on measured U",
            }
        )
    for row in pipeline["plans"]:
        summary_rows.append(
            {
                "experiment": "C_pipeline",
                "cell": row["plan"],
                "model": "pico_optimizer",
                "measured": row["measured_speedup_vs_U"],
                "predicted": row["pico_predicted_speedup_vs_U"],
                "error_pct": row["pico_speedup_error_pct"],
                "metric": "speedup vs U (measured throughput vs model cost)",
            }
        )
        summary_rows.append(
            {
                "experiment": "C_pipeline",
                "cell": row["plan"],
                "model": "cedar_calculate_cost",
                "measured": row["measured_speedup_vs_U"],
                "predicted": row["cedar_predicted_speedup_vs_U"],
                "error_pct": row["cedar_speedup_error_pct"],
                "metric": "speedup vs U (measured throughput vs model cost)",
            }
        )
    for label, payload in reorder.items():
        summary = payload["summary"]
        for model, key in (("proportional_anchored_rule", "proportional_total_ms"),
                           ("affine_anchored_rule", "affine_total_ms"),
                           ("pico_affine_absolute", "pico_total_ms")):
            summary_rows.append(
                {
                    "experiment": "D_reorder",
                    "cell": label,
                    "model": model,
                    "measured": summary["measured_total_ms"],
                    "predicted": summary[key],
                    "error_pct": (summary[key] - summary["measured_total_ms"])
                    / summary["measured_total_ms"] * 100,
                    "metric": "sum of per-operator compute",
                }
            )
    write_csv(
        run / "summary.csv", summary_rows,
        ["experiment", "cell", "model", "measured", "predicted", "error_pct",
         "metric"],
    )

    # ---------------- figure data ----------------
    figure = {
        "3_1_input_scaling_and_fusion_limits": {
            "fixed_rho_rule": {
                "rho_corrected": expa["rho_corrected"],
                "previous_hardcoded_rho": expa["previous_hardcoded_rho"],
                "cells": expa["cells"],
            },
            "instrument_comparison": {
                "note": (
                    "uninstrumented main run vs instrumented run of the same "
                    "cells; reported separately, never stacked"
                ),
                "instrument_effect_U_pct": [
                    {"K": c["level"], "effect_pct": c["instrument_effect_U_pct"]}
                    for c in expa["cells"]
                ],
                "instrument_effect_F_pct": [
                    {"K": c["level"], "effect_pct": c["instrument_effect_F_pct"]}
                    for c in expa["cells"]
                ],
            },
        },
        "3_2_affine_vs_proportional": {
            "reorder": reorder,
            "summary_rows": [
                r for r in summary_rows if r["experiment"] == "D_reorder"
            ],
        },
        "3_3_compute_vs_boundary": {
            "synthetic_component_model": component["synthetic_block"],
            "real_block": component["real_block"],
            "real_block_measurements": expb,
            "a1_measurements": expa["cells"],
        },
        "3_4_pipeline": {
            "plans": pipeline["plans"],
            "rankings": {
                "measured": pipeline["measured_ranking"],
                "pico": pipeline["pico_ranking"],
                "cedar": pipeline["cedar_ranking"],
            },
        },
    }
    (run / "figure_data.json").write_text(json.dumps(figure, indent=2))
    print("wrote protocol.json, predictions.csv, measurements.csv,")
    print("      boundary_results.csv, summary.csv, figure_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
