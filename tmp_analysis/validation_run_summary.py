"""Cross-run summary of the validation orders (three independent runs each).

The protocol requires repeats to be complete runs, not batches.  This prints,
per run, the per-operator in-pipeline p10 sum and the model predictions, so the
run-to-run spread can be compared with the model error.

Usage (inside the container):
  python -u tmp_analysis/validation_run_summary.py
"""

import json
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/affine_reorder_diagnosis_20260924"
RUNS = {
    "v1": ["v1", "v1_r2", "v1_r3"],
    "v2": ["v2", "v2_r2", "v2_r3"],
}


def load(run: str):
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
    return summary


def main() -> int:
    scoring = json.loads((OUT / "plan_scoring.json").read_text())
    result = {}
    for plan, runs in RUNS.items():
        rows = []
        print(f"== {plan} ({len(runs)} complete runs)")
        for run in runs:
            stats = load(run)
            p10 = sum(entry["p10_ms"] for entry in stats.values())
            median = sum(entry["median_ms"] for entry in stats.values())
            mean = sum(entry["mean_ms"] for entry in stats.values())
            rows.append(
                {
                    "run": run,
                    "p10_sum_ms": p10,
                    "median_sum_ms": median,
                    "mean_sum_ms": mean,
                    "operators": {
                        str(pid): entry for pid, entry in stats.items()
                    },
                }
            )
            print(
                f"  {run:8s} p10={p10:8.3f} median={median:8.3f} "
                f"mean={mean:8.3f}"
            )
        totals = scoring["orders"][plan]["totals"]
        p10s = [row["p10_sum_ms"] for row in rows]
        print(
            f"  cross-run p10 range: {min(p10s):.3f}-{max(p10s):.3f} "
            f"(spread {max(p10s) - min(p10s):.3f} ms)"
        )
        for model in ("M1", "M2", "M3", "M4"):
            ratio = [
                totals[model] / row["p10_sum_ms"] for row in rows
            ]
            print(
                f"  {model}: {totals[model]:8.3f} ms -> ratio "
                f"{min(ratio):.2f}-{max(ratio):.2f}"
            )
        result[plan] = rows
    target = OUT / "validation_runs.json"
    target.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
