"""Rank the planners of one screened workload and check the target ratios.

    python tmp_analysis/screen_table.py <results.json> [source_records]

Prints, per planner, the harness epoch time, the throughput, and the two
ratios the study targets: PICO vs the best external system, and PICO vs the
``simple_dp_optimizer`` ablation.
"""

import json
import sys
from pathlib import Path

EXTERNAL = {
    "optimizer": "Cedar",
    "dj_optimizer": "DJ-Cedar",
    "pecan_optimizer": "Pecan-Cedar",
    "plumber_optimizer": "Plumber",
    "raydata_optimizer": "Ray-Data",
}
ABLATION = "simple_dp_optimizer"
PICO = "dp_optimizer"


def main() -> int:
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text())
    entries = {}
    for run in payload.get("runs", []):
        optimizer = run.get("optimizer")
        perf = run.get("perf_time_sec")
        raw = run.get("raw_workload_results") or {}
        batch = raw.get("batch_size") or 4
        samples = run.get("num_samples") or 0
        records = samples / batch if samples else None
        if perf is not None and (perf != perf or perf == float("inf")):
            perf = None
        plans = run.get("physical_plans_by_feature") or {}
        workers = (
            next(iter(plans.values()), {}).get("n_local_workers")
            if plans
            else None
        )
        entries[optimizer] = {
            "perf": perf,
            "records": records,
            "workers": workers,
            "rate": (records / perf) if (records and perf) else None,
        }

    def show(optimizer, label):
        entry = entries.get(optimizer)
        if entry is None or entry["rate"] is None:
            print(f"    {label:<14}{'—':>12}")
            return None
        print(
            f"    {label:<14}{entry['rate']:>12.1f}   perf={entry['perf']:.2f}s "
            f"W={entry['workers']}"
        )
        return entry["rate"]

    print(f"\n=== {path.stem}")
    print(f"    {'planner':<14}{'rec/s':>12}")
    external = []
    for optimizer, label in EXTERNAL.items():
        rate = show(optimizer, label)
        if rate:
            external.append((rate, label))
    ablation = show(ABLATION, "simple-DP")
    pico = show(PICO, "PICO")

    if external and pico:
        best_rate, best_label = max(external)
        print(
            f"    -> best external: {best_label} {best_rate:.1f} rec/s; "
            f"PICO speedup {pico / best_rate:.2f}x"
        )
    if ablation and pico:
        print(f"    -> PICO vs simple-DP: {pico / ablation:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
