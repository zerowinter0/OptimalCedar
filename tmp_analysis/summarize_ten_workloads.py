"""Print the ten-workload matrix from the runner's status.json.

Usage (inside the container):
  python tmp_analysis/summarize_ten_workloads.py outputs/pico_ten_workloads_20260913
"""

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/pico_ten_workloads_20260913")
payload = json.loads((OUT / "status.json").read_text())
cells = payload.get("cells", {})
print(f"state={payload.get('state')} current={payload.get('current')}")

workloads = sorted({key.split(":")[0] for key in cells if ":" in key})
optimizers = sorted({key.split(":", 1)[1] for key in cells if ":" in key})
short = {
    "dj_two_stage_optimizer": "DJ+Cedar",
    "pecan_two_stage_optimizer": "Pecan+Cedar",
    "plumber_optimizer": "Plumber",
    "raydata_optimizer": "RayData",
    "optimizer": "Cedar",
    "dp_optimizer": "PICO",
    "simple_dp_optimizer": "Simple-DP",
}
header = f"{'workload':<22}" + "".join(f"{short.get(o, o):>13}" for o in optimizers)
print(header)
for workload in workloads:
    row = f"{workload:<22}"
    best, best_value = None, float("inf")
    values = {}
    for optimizer in optimizers:
        entry = cells.get(f"{workload}:{optimizer}")
        if not entry or entry.get("perf_time_sec") is None:
            values[optimizer] = None
            continue
        total = entry["perf_time_sec"] + (entry.get("setup_time_sec") or 0.0)
        values[optimizer] = total
        if total < best_value:
            best, best_value = optimizer, total
    for optimizer in optimizers:
        value = values[optimizer]
        if value is None:
            row += f"{'-':>13}"
        else:
            mark = "*" if optimizer == best else " "
            row += f"{value:>12.2f}{mark}"
    print(row)

print("\nseconds = best execution + plan optimization per cell; * = fastest")
print("\nper-cell detail")
for key in sorted(cells):
    entry = cells[key]
    print(
        f"  {key:<45} {entry.get('status'):<8} "
        f"perf={entry.get('perf_time_sec')} setup={entry.get('setup_time_sec')} "
        f"W={entry.get('n_local_workers')} wall={entry.get('wall_sec')}s"
    )
