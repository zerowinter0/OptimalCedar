"""Summarize the per-cell results of a campaign run root.

Usage (inside the container):
  python -u scripts/new_workloads_results.py outputs/<run> [workload ...]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run = ROOT / sys.argv[1]
    wanted = sys.argv[2:]
    for workload_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        workload = workload_dir.name
        if wanted and workload not in wanted:
            continue
        results = sorted((workload_dir / "results").glob("round1__*.json"))
        if not results:
            continue
        print(f"== {workload}")
        for path in results:
            method = path.name.replace("round1__", "").replace(".json", "")
            data = json.loads(path.read_text())
            run0 = (data.get("runs") or [data])[0]
            status = run0.get("status", "completed")
            throughput = run0.get("throughput_samples_per_sec") or 0.0
            setup = run0.get("setup_time_sec") or 0.0
            samples = run0.get("num_samples")
            print(f"   {method:34s} {status:12s} thr={throughput:9.1f} "
                  f"setup={setup:8.1f}s n={samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
