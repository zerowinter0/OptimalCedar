"""Append one measured plan's row to a small-matrix summary table.

    python tmp_analysis/append_summary.py <summary.tsv> <workload> <planner> <stats.json>
"""

import json
import sys


def main() -> int:
    summary, workload, planner, path = sys.argv[1:5]
    try:
        data = json.load(open(path))
    except Exception as exc:  # noqa: BLE001 - a failed run has no stats
        print(f"summary: {planner}: no stats ({exc})")
        return 0
    row = (
        f"{workload}\t{planner}\t{data['records']}\t"
        f"{data['busy_window_sec']}\t{data['throughput']}\n"
    )
    with open(summary, "a") as handle:
        handle.write(row)
    print(row.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
