"""Per-plan prediction error of the cost model, across every measured cell.

For every (workload, planner) cell we have the materialized plan, the model's
per-record score for that plan (``pico_plan_costs_by_feature``, per worker
lane), and the measured throughput.  Comparing them answers, plan by plan,
whether the model's throughput estimate is right:

    predicted_throughput = 1000 * workers / lane_ms   [records/s]

Usage (inside the container):
  python -u tmp_analysis/prediction_error_report.py [--json out.json]
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
RUNS = [
    # Drained full-pass measurements take precedence: stop-at-N runs report
    # the time to fill a deep in-flight window for buffered plans.
    ROOT / "outputs/pico_drained_20260914",
    ROOT / "outputs/pico_ten_workloads_20260913b",
    ROOT / "outputs/pico_djpecan_20260914",
    ROOT / "outputs/pico_missing_20260914",
]

# Records in each workload's data set: a drained full pass processes each one
# exactly once (verified with per-operator call counters: parse_and_format
# calls == file lines), while the harness's sample counter can be inflated by
# its per-batch accounting, so the audit derives throughput from these counts.
WORKLOAD_RECORDS = {
    "simclr": 9469,
    "blip": 1000,
    "clip": 1000,
    "dino": 1000,
    "alpaca_cot": 74771,
    "pile_hackernews": 100000,
    "pile_pubmed_abstracts": 100000,
    "pile_uspto_backgrounds": 100000,
    "bloom_oscar": 50000,
}
DRAINED_RUN = "pico_drained_20260914"

WORKLOADS = [
    "simclr",
    "blip",
    "clip",
    "dino",
    "alpaca_cot",
    "pile_hackernews",
    "pile_pubmed_abstracts",
    "pile_uspto_backgrounds",
    "bloom_oscar",
]


def collect():
    rows = []
    seen = set()
    # Drained measurements take precedence over stop-at-N ones for the same
    # (workload, planner) cell.
    for run in RUNS:
        results = run / "results"
        if not results.is_dir():
            continue
        for path in sorted(results.glob("*.json")):
            workload, _, planner = path.stem.partition("__")
            if workload not in WORKLOADS:
                continue
            try:
                entry = json.loads(path.read_text())["runs"][0]
            except Exception:  # noqa: BLE001
                continue
            perf = entry.get("perf_time_sec")
            samples = entry.get("num_samples")
            costs = entry.get("pico_plan_costs_by_feature") or {}
            if not perf or not samples or perf <= 0 or not costs:
                continue
            lane_ms = sum(costs.values()) / len(costs)
            plans = entry.get("physical_plans_by_feature") or {}
            workers = next(iter(plans.values()), {}).get("n_local_workers", 1)
            workers = max(1, int(workers or 1))
            if lane_ms <= 0:
                continue
            predicted = 1000.0 * workers / lane_ms
            records = WORKLOAD_RECORDS.get(workload)
            if DRAINED_RUN in str(path) and records:
                measured = records / perf
                samples = records
            else:
                measured = samples / perf
            key = (workload, planner)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "workload": workload,
                    "planner": planner,
                    "samples": int(samples),
                    "workers": workers,
                    "lane_ms": lane_ms,
                    "measured": measured,
                    "predicted": predicted,
                    "ratio": predicted / measured,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    rows = collect()
    print(
        f"{'workload':<22}{'planner':<30}{'W':>4}{'measured':>10}"
        f"{'predicted':>11}{'ratio':>8}"
    )
    for row in sorted(rows, key=lambda r: (r["workload"], -r["ratio"])):
        print(
            f"{row['workload']:<22}{row['planner']:<30}{row['workers']:>4}"
            f"{row['measured']:>10.0f}{row['predicted']:>11.0f}"
            f"{row['ratio']:>8.2f}"
        )

    print("\nper-workload summary (|log2 ratio|, count within 1.25x):")
    for workload in WORKLOADS:
        subset = [r for r in rows if r["workload"] == workload]
        if not subset:
            continue
        errors = [abs(math.log2(r["ratio"])) for r in subset]
        within = sum(1 for r in subset if 0.8 <= r["ratio"] <= 1.25)
        median = sorted(errors)[len(errors) // 2]
        worst = max(subset, key=lambda r: abs(math.log2(r["ratio"])))
        print(
            f"  {workload:<22} n={len(subset):<2} median|log2|={median:4.2f} "
            f"within1.25x={within}/{len(subset)}  worst={worst['planner']} "
            f"(x{worst['ratio']:.2f})"
        )

    all_errors = [abs(math.log2(r["ratio"])) for r in rows]
    all_errors.sort()
    print(
        f"\noverall: n={len(rows)} median|log2|={all_errors[len(all_errors)//2]:.2f} "
        f"p90={all_errors[int(0.9 * len(all_errors))]:.2f} "
        f"max={all_errors[-1]:.2f}  within1.25x="
        f"{sum(1 for r in rows if 0.8 <= r['ratio'] <= 1.25)}/{len(rows)}"
    )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
