"""Chapter 3.4.1: corrected analysis of the controlled fusion experiment.

Uses the uninstrumented main run (`expA1_main`, member timing off, three
interleaved rounds per cell) and the earlier instrumented run for comparison.
Cedar's I/O ratio for the three equal-size members is recomputed from the real
bytes (1/3), replacing the 0.5 that the earlier harness hard-coded; only
derived predictions and errors change, the measurements are untouched.

Outputs: expA_corrected.csv, expA_analysis.json.

Usage:
  python -u scripts/pico_ch3_expA_analysis.py \
      --run-dir outputs/pico_ch3_20260924 \
      --instrumented-run outputs/fusion_discount_20260923/expA \
      --instrumented-off-run outputs/fusion_discount_20260923/expA_instrument_off \
      --audit outputs/pico_ch3_20260924/audit.json
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


def per_cell(rows: List[Dict[str, str]], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in rows:
        key = f"{row['config']}@{int(row['level'])}"
        out.setdefault(key, {"elapsed": [], "compute": [], "other": []})
        out[key]["elapsed"].append(float(row["elapsed_ms_per_record_mean"]))
        out[key]["compute"].append(float(row["compute_ms_per_record_mean"]))
        out[key]["other"].append(float(row["other_ms_per_record_mean"]))
    return {
        key: {
            "rounds": len(value["elapsed"]),
            "elapsed_mean": statistics.fmean(value["elapsed"]),
            "elapsed_stdev": (
                statistics.stdev(value["elapsed"]) if len(value["elapsed"]) > 1 else 0.0
            ),
            "compute_mean": statistics.fmean(value["compute"]),
            "other_mean": statistics.fmean(value["other"]),
        }
        for key, value in out.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--instrumented-run", type=Path, required=True)
    parser.add_argument("--instrumented-off-run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    rho = audit["discount"]["synthetic_block"]["U_and_F_three_members"]["rho"]
    rho_partial = audit["discount"]["synthetic_block"]["P_first_two_fused"]["rho"]

    main_cells = per_cell(read_csv(args.run_dir / "expA1_main" / "expA1_summary.csv"), "A1")
    instr_cells = per_cell(
        read_csv(args.instrumented_run / "expA_summary.csv"), "A"
    )
    off_cells = per_cell(
        read_csv(args.instrumented_off_run / "expA_off_summary.csv"), "off"
    )

    rows: List[Dict[str, Any]] = []
    for level in sorted({int(key.split("@")[1]) for key in main_cells}):
        u = main_cells.get(f"U@{level}")
        f = main_cells.get(f"F@{level}")
        if not u or not f:
            continue
        retention = f["elapsed_mean"] / u["elapsed_mean"]
        rule = u["elapsed_mean"] * rho
        rows.append(
            {
                "level": level,
                "T_U_ms_per_record": u["elapsed_mean"],
                "T_U_stdev": u["elapsed_stdev"],
                "T_F_ms_per_record": f["elapsed_mean"],
                "T_F_stdev": f["elapsed_stdev"],
                "retention_T_F_over_T_U": retention,
                "rho_corrected": rho,
                "rule_T_F": rule,
                "rule_error_pct": (rule - f["elapsed_mean"]) / f["elapsed_mean"] * 100,
                "member_share_U": u["compute_mean"] / u["elapsed_mean"]
                if u["elapsed_mean"] else None,
                "instrumented_T_U": instr_cells.get(f"U@{level}", {}).get("elapsed_mean"),
                "instrumented_T_F": instr_cells.get(f"F@{level}", {}).get("elapsed_mean"),
                "instrumented_retention": (
                    instr_cells.get(f"F@{level}", {}).get("elapsed_mean")
                    / instr_cells.get(f"U@{level}", {}).get("elapsed_mean")
                    if instr_cells.get(f"U@{level}", {}).get("elapsed_mean") else None
                ),
                "instrument_effect_U_pct": (
                    (instr_cells.get(f"U@{level}", {}).get("elapsed_mean", 0)
                     - u["elapsed_mean"]) / u["elapsed_mean"] * 100
                    if u["elapsed_mean"] else None
                ),
                "instrument_effect_F_pct": (
                    (instr_cells.get(f"F@{level}", {}).get("elapsed_mean", 0)
                     - f["elapsed_mean"]) / f["elapsed_mean"] * 100
                    if f["elapsed_mean"] else None
                ),
            }
        )

    payload = {
        "rho_corrected": rho,
        "rho_partial_corrected": rho_partial,
        "previous_hardcoded_rho": audit["discount"]["synthetic_block"][
            "previous_hardcoded_value"
        ],
        "source_of_rho": (
            "recomputed from Cedar's _calculate_cost_fused byte accounting on "
            "the real tensors (equal in/out size s): IO_U = 6s, IO_F = 2s"
        ),
        "cells": rows,
        "instrumented_cells": instr_cells,
        "instrument_off_single_round": off_cells,
        "partial_fusion": {
            "measured_instrumented": instr_cells.get("P@16"),
            "rho_partial": rho_partial,
            "note": (
                "Cedar applies the discount per fused group. Anchoring the "
                "partial plan on the measured U time requires splitting the "
                "handoff cost by stage; the measured stage scaling "
                "(other_P / other_U) is reported instead of inventing one."
            ),
            "other_ratio_P_over_U": (
                instr_cells.get("P@16", {}).get("other_mean", 0)
                / instr_cells.get("U@16", {}).get("other_mean", 1)
                if instr_cells.get("U@16") else None
            ),
        },
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "expA_analysis.json").write_text(json.dumps(payload, indent=2))
    with (args.run_dir / "expA_corrected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rho (corrected) = {rho:.6f} (previously {payload['previous_hardcoded_rho']})")
    print(f"{'K':>3}{'T_U':>9}{'T_F':>9}{'retain':>8}{'rule':>9}{'rule err':>10}"
          f"{'instr retain':>14}")
    for row in rows:
        print(f"{row['level']:>3}{row['T_U_ms_per_record']:>9.3f}"
              f"{row['T_F_ms_per_record']:>9.3f}{row['retention_T_F_over_T_U']:>8.3f}"
              f"{row['rule_T_F']:>9.3f}{row['rule_error_pct']:>9.1f}%"
              f"{(row['instrumented_retention'] or float('nan')):>14.3f}")
    print("wrote", args.run_dir / "expA_corrected.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
