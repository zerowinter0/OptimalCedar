"""Why do Normalize/Batcher differ between orders at identical input sizes?

Prints, per order, the wall segment and the process-time (CPU) segment of the
last stages plus the upstream interval, from the repeat traces of
outputs/unopt_order_transfer_repeats_20260921.
"""

import glob
import json
import statistics
from pathlib import Path

RUN = Path("/workspace/OptimalCedar/outputs/unopt_order_transfer_repeats_20260921")
CELLS = ["declared", "pico", "cedar", "old-dp"]
NAMES = {0: "T Batcher", 1: "N Normalize", 7: "F to_float", 8: "R ImageReader", 9: "source"}


def collect(cell, pid):
    walls, procs = [], []
    for directory in sorted(RUN.glob("reconcile_" + cell + "_r*")):
        for path in sorted(directory.glob("worker_*.json")):
            data = json.loads(path.read_text())
            values = (data.get("wall_latency_samples") or {}).get(str(pid)) or []
            if values:
                walls.append(statistics.mean(v / 1e6 for v in values))
            pv = (data.get("process_latency_ns_per_sample") or {}).get(str(pid))
            if pv:
                procs.append(float(pv) / 1e6)
    return walls, procs


def main() -> None:
    all_names = {
        0: "T Batcher", 1: "N Normalize", 2: "B Blur", 3: "G Grayscale",
        4: "J Jitter", 5: "H Flip", 6: "C Crop", 7: "F to_float",
        8: "R ImageReader", 9: "source",
    }
    print("per-operator wall segment / process-time segment (ms per record, mean over repeats)")
    header = "%-14s" % "operator" + "".join("%22s" % c for c in CELLS)
    print(header)
    totals = {cell: [0.0, 0.0] for cell in CELLS}
    for pid in sorted(all_names):
        row = "%-14s" % all_names[pid]
        for cell in CELLS:
            walls, procs = collect(cell, pid)
            wall = statistics.mean(walls) if walls else 0.0
            proc = statistics.mean(procs) if procs else 0.0
            totals[cell][0] += wall
            totals[cell][1] += proc
            row += "%11.3f/%-10.3f" % (wall, proc)
        print(row)
    print()
    print("sum over the 10 traced pipes, vs the per-worker record cycle (4/throughput)")
    for cell in CELLS:
        payload = json.loads((RUN / "results" / (cell + "_r1.json")).read_text())
        cycle = 4.0 / payload["throughput_samples_per_sec"] * 1000.0
        print(
            "  %-9s wall sum %6.2f ms, process sum %6.2f ms, per-worker cycle %6.2f ms"
            % (cell, totals[cell][0], totals[cell][1], cycle)
        )
    print()
    print("upstream pace and batcher accounting")
    for cell in CELLS:
        throughput_file = RUN / "results" / (cell + "_r1.json")
        payload = json.loads(throughput_file.read_text())
        samples = payload["num_samples"]
        perf = payload["perf_time_sec"]
        rate = payload["throughput_samples_per_sec"]
        workers = 4
        per_worker_interval = workers / rate * 1000.0
        batch4 = 3.0 * per_worker_interval
        _, batcher_cpu = collect(cell, 0)
        print(
            "  %-9s system %.1f rec/s -> per-worker record interval %.2f ms; "
            "batch-of-4 span %.2f ms; batcher CPU %.2f ms"
            % (cell, rate, per_worker_interval, batch4, statistics.mean(batcher_cpu))
        )


if __name__ == "__main__":
    main()
