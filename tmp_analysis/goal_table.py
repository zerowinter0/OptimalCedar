"""Consolidate every measured screening cell into the study's goal table.

    python tmp_analysis/goal_table.py [results.json ...]

With no arguments it scans the screening result directories used during the
session and reports, per workload:

  * PICO's throughput and the best *external* planner (everything except
    PICO's own ablation ``simple_dp_optimizer``),
  * PICO / best external  (target: >= 1.5x on eight workloads),
  * PICO / simple-DP      (target: >= 1.3x on at least three of those eight,
    and never below 1.0x).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    ROOT / "outputs/screen",
    Path("/tmp/screen_queue2"),
    Path("/tmp/screen_queue3"),
    Path("/tmp/screen_queue4"),
    Path("/tmp/screen_queue5"),
    Path("/tmp/screen_queue6"),
    Path("/tmp/screen_queue7"),
    Path("/tmp/screen_queue8"),
    Path("/tmp/screen_family2"),
    Path("/tmp/screen_family3"),
    Path("/tmp/screen_rerun"),
    Path("/tmp/screen_rerun2"),
    Path("/tmp/screen_rerun3"),
    Path("/tmp/screen_v2"),
    Path("/tmp/screen_v3"),
    Path("/tmp/screen_v4"),
    Path("/tmp/screen_video2"),
    Path("/tmp/screen_10k"),
    Path("/tmp/screen_10k2"),
    Path("/tmp/screen_r2"),
    Path("/tmp/screen_calib"),
    Path("/tmp/screen_final"),
    Path("/tmp/screen_wsweep"),
    Path("/tmp/screen_wsweep2"),
    Path("/tmp/screen_wsweep3"),
]
PICO = "dp_optimizer"
ABLATION = "simple_dp_optimizer"
NAMES = {
    "optimizer": "Cedar",
    "dj_optimizer": "DJ-Cedar",
    "pecan_optimizer": "Pecan-Cedar",
    "plumber_optimizer": "Plumber",
    "raydata_optimizer": "Ray-Data",
    "dp_optimizer": "PICO",
    "simple_dp_optimizer": "simple-DP",
}


def collect(paths):
    cells = {}
    for path in paths:
        workload = path.stem
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for run in payload.get("runs", []):
            planner = run.get("optimizer")
            perf = run.get("perf_time_sec")
            raw = run.get("raw_workload_results") or {}
            batch = raw.get("batch_size") or 4
            samples = run.get("num_samples") or 0
            if perf is None or (perf != perf) or perf == float("inf"):
                continue
            records = samples / batch if samples else None
            if not records:
                continue
            rate = records / perf
            previous = cells.setdefault(workload, {}).get(planner)
            if previous is None or rate > previous:
                cells[workload][planner] = rate
    return cells


def main() -> int:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = []
        for directory in SEARCH_DIRS:
            if directory.is_dir():
                paths.extend(sorted(directory.glob("*.json")))
    cells = collect(paths)
    if not cells:
        print("no measured workloads found")
        return 1

    rows = []
    for workload, planners in cells.items():
        pico = planners.get(PICO)
        if not pico:
            continue
        external = [
            (rate, planner)
            for planner, rate in planners.items()
            if planner not in (PICO, ABLATION)
        ]
        ablation = planners.get(ABLATION)
        if not external:
            continue
        best_rate, best_planner = max(external)
        rows.append(
            {
                "workload": workload,
                "pico": pico,
                "external": best_rate,
                "external_name": NAMES.get(best_planner, best_planner),
                "external_ratio": pico / best_rate,
                "ablation": ablation,
                "ablation_ratio": (pico / ablation) if ablation else None,
            }
        )

    rows.sort(key=lambda row: -row["external_ratio"])
    print(
        f"\n{'workload':<26}{'PICO':>9}{'best ext':>10}{'ext':>7}"
        f"{'simple-DP':>11}{'abl':>7}  target"
    )
    for row in rows:
        ablation_ratio = row["ablation_ratio"]
        target_a = row["external_ratio"] >= 1.5
        target_c = ablation_ratio is None or ablation_ratio >= 1.0
        target_b = ablation_ratio is not None and ablation_ratio >= 1.3
        flags = [
            "ext>=1.5" if target_a else "        ",
            "abl>=1.0" if target_c else "WORSE-ABL",
            "abl>=1.3" if target_b else "",
        ]
        ablation_value = row["ablation"] if row["ablation"] else float("nan")
        print(
            f"{row['workload']:<26}{row['pico']:>9.1f}{row['external']:>10.1f}"
            f"{row['external_ratio']:>7.2f}"
            f"{ablation_value:>11.1f}"
            f"{(ablation_ratio or float('nan')):>7.2f}  {' '.join(flags)}"
        )

    qualifiers = [row for row in rows if row["external_ratio"] >= 1.5]
    both = [
        row for row in qualifiers if (row["ablation_ratio"] or 0) >= 1.3
    ]
    worse = [
        row
        for row in rows
        if row["ablation_ratio"] is not None and row["ablation_ratio"] < 1.0
    ]
    print(
        f"\nworkloads with >=1.5x over the best external: {len(qualifiers)} "
        f"(target 8)"
    )
    print(f"of those, >=1.3x over simple-DP: {len(both)} (target 3)")
    print(f"rows where PICO is worse than simple-DP: {len(worse)}")
    for row in worse:
        print(f"  - {row['workload']}: {row['ablation_ratio']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
