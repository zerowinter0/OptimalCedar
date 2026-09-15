"""Rank the planners of a comparison matrix run, workload by workload.

    python tmp_analysis/summarize_formal_matrix.py <matrix output dir>

Throughput is ``records / perf_time_sec`` where ``records`` is the number of
samples the harness' own epoch counter reported divided by the batch size
(``evaluation/profiler.py`` counts one batch as ``batch_size`` samples), i.e.
the number of records that left the pipeline in that epoch.  The source count
per workload is printed as a cross-check.
"""

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
SOURCE_RECORDS = {
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
ORDER = [
    "dj_optimizer",
    "pecan_optimizer",
    "plumber_optimizer",
    "raydata_optimizer",
    "optimizer",
    "dp_optimizer",
    "simple_dp_optimizer",
]
SHORT = {
    "dj_optimizer": "dj-cedar",
    "pecan_optimizer": "pecan-cedar",
    "plumber_optimizer": "plumber",
    "raydata_optimizer": "ray-data",
    "optimizer": "cedar",
    "dp_optimizer": "PICO",
    "simple_dp_optimizer": "simple-dp",
}


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "outputs/pico_formal_20260915")
    if not out.is_absolute():
        out = ROOT / out
    cells = {}
    for path in sorted((out / "results").glob("*__*.json")):
        workload, _, optimizer = path.stem.partition("__")
        try:
            entry = json.loads(path.read_text())["runs"][0]
        except Exception:
            continue
        perf = entry.get("perf_time_sec")
        raw = entry.get("raw_workload_results") or {}
        batch = raw.get("batch_size") or 4
        samples = entry.get("num_samples") or 0
        records = samples / batch if samples else None
        plans = entry.get("physical_plans_by_feature") or {}
        workers = (
            next(iter(plans.values()), {}).get("n_local_workers")
            if plans
            else None
        )
        cells.setdefault(workload, {})[optimizer] = {
            "perf": perf,
            "records": records,
            "workers": workers,
            "rate": (records / perf) if (records and perf and perf > 0) else None,
            "plan_cost": entry.get("plan_cost"),
        }

    for workload in sorted(cells):
        per = cells[workload]
        source = SOURCE_RECORDS.get(workload)
        print(f"\n=== {workload}   (source records {source})")
        rows = []
        for optimizer in ORDER:
            entry = per.get(optimizer)
            if entry is None:
                rows.append((optimizer, None))
                continue
            rows.append((optimizer, entry))
        best = max(
            (entry["rate"] for _, entry in rows if entry and entry["rate"]),
            default=None,
        )
        print(
            f"    {'planner':<14}{'records':>9}{'perf_s':>10}{'rec/s':>10}"
            f"{'vs best':>9}{'W':>4}"
        )
        for optimizer, entry in rows:
            name = SHORT.get(optimizer, optimizer)
            if entry is None:
                print(f"    {name:<14}{'-':>9}{'-':>10}{'-':>10}{'-':>9}")
                continue
            rate = entry["rate"]
            perf = entry["perf"]
            records = entry["records"]
            ratio = (
                f"{rate / best:.2f}" if (rate and best) else "-"
            )
            perf_text = f"{perf:.2f}" if isinstance(perf, (int, float)) else str(perf)
            rate_text = f"{rate:.1f}" if rate else "-"
            records_text = f"{records:.0f}" if records else "-"
            print(
                f"    {name:<14}{records_text:>9}{perf_text:>10}{rate_text:>10}"
                f"{ratio:>9}{str(entry['workers'] or '-'):>4}"
            )
        if best:
            winner = max(
                (
                    (SHORT.get(name, name), entry["rate"])
                    for name, entry in rows
                    if entry and entry["rate"]
                ),
                key=lambda item: item[1],
            )
            print(f"    -> best: {winner[0]} ({winner[1]:.1f} rec/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
