"""Aggregate the representation-aware compute-model evidence into deliverables.

Reads the per-plan M1..M5 scoring, the per-run capture summaries and the model
matrix, then writes the small CSV/JSON files the chapter needs and prints the
headline tables (accuracy per plan, model ranking, selection regret).

Usage (inside the container):
  python -u tmp_analysis/build_affine_repr_deliverable.py
"""

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/affine_repr_model_20260924"
DIAG = ROOT / "outputs/affine_reorder_diagnosis_20260924"
MODELS = ("M1", "M2", "M3", "M4", "M5")

RUNS = {
    "a_f0": ["a_f0_r1", "a_f0_r2", "a_f0_r3"],
    "a_f1": ["a_f1_r1", "a_f1_r2", "a_f1_r3"],
    "a_f2": ["a_f2_r1", "a_f2_r2", "a_f2_r3"],
    "a_f3": ["a_f3_r1", "a_f3_r2", "a_f3_r3"],
    "a_f4": ["a_f4_r1", "a_f4_r2", "a_f4_r3"],
    "a_f5": ["a_f5_r1", "a_f5_r2", "a_f5_r3"],
    "pico": ["pico", "pico_r2", "pico_r3"],
    "cedar": ["cedar", "cedar_r2", "cedar_r3"],
    "old-dp": ["old-dp", "old-dp_r2", "old-dp_r3"],
    "v1": ["v1", "v1_r2", "v1_r3"],
    "v2": ["v2", "v2_r2", "v2_r3"],
}


def capture_stats(run: str):
    directory = ROOT / f"tmp_analysis/capture_{run}/capture"
    per_pipe = {}
    for path in sorted(directory.glob("*_op_capture.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for pid, entry in data["pipes"].items():
            per_pipe.setdefault(int(pid), []).append(entry)
    summary = {}
    for pid, entries in per_pipe.items():
        calls = sum(entry["calls"] for entry in entries)
        summary[pid] = {
            key: sum(entry[key] * entry["calls"] for entry in entries) / calls
            for key in ("mean_ms", "median_ms", "p10_ms", "p90_ms")
        }
        summary[pid]["calls"] = calls
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scoring = json.loads((DIAG / "m1_m5_plan_scores.json").read_text())

    # ------------------------------------------------------ per-run summaries
    run_rows = []
    per_plan_runs = {}
    for plan, runs in RUNS.items():
        totals = []
        for run in runs:
            stats = capture_stats(run)
            p10 = sum(entry["p10_ms"] for entry in stats.values())
            median = sum(entry["median_ms"] for entry in stats.values())
            mean = sum(entry["mean_ms"] for entry in stats.values())
            totals.append({"run": run, "p10": p10, "median": median, "mean": mean})
            run_rows.append(
                {
                    "plan": plan,
                    "run": run,
                    "p10_sum_ms": p10,
                    "median_sum_ms": median,
                    "mean_sum_ms": mean,
                }
            )
        per_plan_runs[plan] = totals

    # ------------------------------------------------------------- plan table
    plan_rows = []
    for plan, payload in scoring["plans"].items():
        totals = payload["totals"]
        runs = per_plan_runs.get(plan, [])
        measured_mean = totals["measured"]
        measured_p10 = statistics.fmean([item["p10"] for item in runs]) if runs else None
        row = {
            "plan": plan,
            "chain": " ".join(str(p) for p in payload["chain"]),
            "measured_mean_ms": measured_mean,
            "measured_p10_ms": measured_p10,
            "runs": len(runs),
            "run_mean_min": min((item["mean"] for item in runs), default=None),
            "run_mean_max": max((item["mean"] for item in runs), default=None),
        }
        for model in MODELS:
            row[model] = totals[model]
            row[f"{model}_ratio"] = totals[model] / measured_mean
        plan_rows.append(row)
    with (OUT / "plan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_rows[0]))
        writer.writeheader()
        writer.writerows(plan_rows)

    # --------------------------------------------------------- operator table
    operator_rows = []
    for plan, payload in scoring["plans"].items():
        for row in payload["rows"]:
            entry = {"plan": plan, **row}
            operator_rows.append(entry)
    with (OUT / "operator_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operator_rows[0]))
        writer.writeheader()
        writer.writerows(operator_rows)

    # ------------------------------------------------------------ predictions
    predictions = []
    for plan, payload in scoring["plans"].items():
        for row in payload["rows"]:
            for model in MODELS:
                predictions.append(
                    {
                        "plan": plan,
                        "operator": row["operator"],
                        "representation": row["representation"],
                        "elements": row["elements"],
                        "model": model,
                        "predicted_ms": row[model],
                        "measured_mean_ms": row["measured_mean_ms"],
                    }
                )
    with (OUT / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    with (OUT / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    # ------------------------------------------------------------------ model
    print(f"{'plan':9s} {'measured':>9s} " + " ".join(f"{m:>7s}" for m in MODELS))
    for row in plan_rows:
        print(
            f"{row['plan']:9s} {row['measured_mean_ms']:9.3f} "
            + " ".join(f"{row[m]:7.3f}" for m in MODELS)
        )
    print("\nmodel / measured ratio:")
    summary = {}
    for model in MODELS:
        ratios = [row[f"{model}_ratio"] for row in plan_rows]
        errors = [abs(value - 1.0) for value in ratios]
        summary[model] = {
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "mape": statistics.fmean(errors),
            "rmse": (statistics.fmean([(value - 1.0) ** 2 for value in ratios])) ** 0.5,
        }
        print(
            f"  {model}: min={summary[model]['min_ratio']:.2f} "
            f"max={summary[model]['max_ratio']:.2f} "
            f"MAPE={summary[model]['mape'] * 100:.1f}% "
            f"RMSE={summary[model]['rmse'] * 100:.1f}%"
        )

    # --------------------------------------------------- cost ranking/regret
    measured_order = sorted(plan_rows, key=lambda row: row["measured_mean_ms"])
    best = measured_order[0]
    print("\nselection (cost only; lower measured mean is better):")
    selection = {}
    for model in MODELS:
        chosen = min(plan_rows, key=lambda row: row[model])
        regret = (
            chosen["measured_mean_ms"] / best["measured_mean_ms"] - 1.0
        )
        selection[model] = {
            "chosen": chosen["plan"],
            "chosen_measured": chosen["measured_mean_ms"],
            "regret": regret,
            "predicted": chosen[model],
        }
        print(
            f"  {model}: picks {chosen['plan']:8s} "
            f"measured={chosen['measured_mean_ms']:.3f} regret={regret * 100:+.1f}%"
        )
    print(f"  best plan by measurement: {best['plan']} ({best['measured_mean_ms']:.3f})")

    figure = {
        "plans": plan_rows,
        "model_summary": summary,
        "selection": selection,
        "runs": run_rows,
        "run_dirs": RUNS,
    }
    (OUT / "figure_data.json").write_text(json.dumps(figure, indent=1, default=float))
    (OUT / "model_summary.json").write_text(
        json.dumps({"models": summary, "selection": selection}, indent=1, default=float)
    )
    print(f"\nwrote {OUT}/plan_summary.csv, operator_results.csv, "
          f"predictions.csv, measurements.csv, figure_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
