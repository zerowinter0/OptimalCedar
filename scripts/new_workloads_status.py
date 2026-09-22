"""Print the per-workload status of a campaign run root.

Usage (inside the container):
  python -u scripts/new_workloads_status.py outputs/<run>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run = ROOT / (sys.argv[1] if len(sys.argv) > 1 else
                  "outputs/new_workloads_validation_20260922")
    status_path = run / "status.json"
    if not status_path.exists():
        print(f"{run}: no status.json yet")
        return 0
    state = json.loads(status_path.read_text())
    print(f"run: {run}")
    for workload, entry in state.items():
        profile = (entry.get("profile") or {}).get("status", "?")
        cells = entry.get("cells", [])
        counts = {}
        for cell in cells:
            key = cell.get("status", "?")
            counts[key] = counts.get(key, 0) + 1
        finished = entry.get("finished")
        tail = f" finished={finished}" if finished else ""
        print(f"  {workload:18s} profile={profile:10s} "
              f"cells={len(cells):2d} {counts}{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
