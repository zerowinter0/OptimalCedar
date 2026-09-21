"""Inspect the Batcher / Normalize / ImageReader rows of a reconcile run.

For each cell (declared, pico, cedar, old-dp) and each repeat it prints the
input bytes per record, the process-time per record and the wall segment per
record of the three pipes, so the doc can tell an attribution artifact (the
batcher's window covering the batch-assembly span) apart from a real plan
context effect (the same operator, same size, different cost).

Usage (inside the container):
  python -u tmp_analysis/batch_normalize_rows.py outputs/<run>
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
CELLS = ["declared", "pico", "cedar", "old-dp"]
NAMES = {0: "T Batcher", 1: "N Normalize", 8: "R ImageReader"}


def collect(run: Path, cell: str):
    rows = []
    for directory in sorted(run.glob(f"reconcile_{cell}*")):
        per = {}
        for path in sorted(directory.glob("worker_*.json")):
            data = json.loads(path.read_text())
            for pid in NAMES:
                entry = per.setdefault(pid, {"size": [], "proc": [], "wall": []})
                sizes = data.get("input_sizes") or {}
                if str(pid) in sizes:
                    entry["size"].append(float(sizes[str(pid)]))
                proc = (data.get("process_latency_ns_per_sample") or {}).get(str(pid))
                if proc is not None:
                    entry["proc"].append(float(proc) / 1e6)
                wall = (data.get("wall_latency_ns_per_sample") or {}).get(str(pid))
                if wall is not None:
                    entry["wall"].append(float(wall) / 1e6)
        rows.append((directory.name, per))
    return rows


def mean(values):
    return statistics.mean(values) if values else float("nan")


def main() -> int:
    run = ROOT / (sys.argv[1] if len(sys.argv) > 1 else
                  "outputs/unopt_order_transfer_repeats_20260921")
    print(f"run: {run}")
    for pid, name in NAMES.items():
        print(f"\n=== pipe {pid}  {name} ===")
        print(f"{'cell':<10}{'size B/rec':>12}{'process ms/rec':>17}{'wall ms/rec':>15}")
        for cell in CELLS:
            rows = collect(run, cell)
            size = mean([mean(r[1][pid]["size"]) for r in rows])
            proc = mean([mean(r[1][pid]["proc"]) for r in rows])
            wall = mean([mean(r[1][pid]["wall"]) for r in rows])
            procs = [mean(r[1][pid]["proc"]) for r in rows]
            spread = statistics.stdev(procs) if len(procs) > 1 else 0.0
            print(f"{cell:<10}{size:>12.0f}{proc:>12.4f}±{spread:<4.2f}{wall:>15.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
