"""Per-operator busy-window statistics of one measured plan.

Reads the operator-call logs written by the injected ``sitecustomize`` hook
(``label duration_sec timestamp role``) and reports, per operator, how many
calls happened, how much CPU they consumed, and how wide the call window was.

The headline number is the *busy window*: the span between the first and the
last operator call of the whole run.  It deliberately ignores process start-up
and Ray actor creation, so it can be compared across plans whose cold-start
cost differs by tens of seconds.  Throughput is derived from the number of
records the sink-producing operator actually processed, not from the harness's
sample counter (which over-counts batched submissions).

Usage (inside the container):
    python tmp_analysis/plan_busy_analyze.py --sink FlaggedWordsFilter.__call__ \
        --label cedar local/*.log remote/*.log
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def parse(paths):
    stats = defaultdict(
        lambda: {"calls": 0, "cpu": 0.0, "first": float("inf"), "last": 0.0}
    )
    per_process = defaultdict(set)
    for path in paths:
        pid = os.path.basename(path).rsplit("_", 1)[-1].split(".")[0]
        with open(path, "r", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                label = parts[0]
                try:
                    duration = float(parts[1])
                    timestamp = float(parts[2])
                except ValueError:
                    continue
                entry = stats[label]
                entry["calls"] += 1
                entry["cpu"] += duration
                entry["first"] = min(entry["first"], timestamp)
                entry["last"] = max(entry["last"], timestamp)
                per_process[label].add(pid)
    return stats, per_process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="plan")
    parser.add_argument("--sink", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    files = []
    for pattern in args.paths:
        matched = glob.glob(pattern)
        files.extend(matched)
        if not matched and os.path.isfile(pattern):
            files.append(pattern)
    if not files:
        print(f"{args.label}: no operator-call logs found")
        return 1

    stats, per_process = parse(files)
    if not stats:
        print(f"{args.label}: logs are empty")
        return 1

    busy = max(entry["last"] - entry["first"] for entry in stats.values())
    cpu_total = sum(entry["cpu"] for entry in stats.values())
    sink = args.sink
    if sink is None:
        sink = max(stats, key=lambda name: stats[name]["calls"]).split(".")[0]
    sink_calls = sum(
        entry["calls"]
        for name, entry in stats.items()
        if name.split(".")[0] == sink
    )
    rate = sink_calls / busy if busy > 0 else float("nan")

    rows = []
    for name, entry in sorted(
        stats.items(), key=lambda item: -item[1]["cpu"]
    ):
        span = entry["last"] - entry["first"]
        rows.append(
            {
                "op": name,
                "processes": len(per_process[name]),
                "calls": entry["calls"],
                "cpu_sec": round(entry["cpu"], 3),
                "cpu_share": round(entry["cpu"] / cpu_total, 3)
                if cpu_total
                else 0.0,
                "mean_ms": round(1000.0 * entry["cpu"] / entry["calls"], 4),
                "span_sec": round(span, 3),
                "rate": round(entry["calls"] / span, 1) if span > 0 else None,
            }
        )

    result = {
        "label": args.label,
        "sink": sink,
        "records": sink_calls,
        "busy_window_sec": round(busy, 3),
        "throughput": round(rate, 1),
        "ms_per_record": round(1000.0 / rate, 4) if rate else None,
        "total_cpu_sec": round(cpu_total, 3),
        "mean_cores_busy": round(cpu_total / busy, 2) if busy else None,
        "processes": sum(len(v) for v in per_process.values()),
        "ops": rows,
    }

    if not args.quiet:
        print(
            f"{args.label:<14} records={sink_calls:<7} busy={busy:6.2f}s "
            f"rate={rate:8.1f} rec/s ms/rec={1000.0 / rate if rate else math.nan:7.3f} "
            f"cores={cpu_total / busy if busy else math.nan:5.1f} "
            f"procs={result['processes']}"
        )
        for row in rows:
            print(
                f"    {row['op']:<46} procs={row['processes']:<3} "
                f"calls={row['calls']:<7} cpu={row['cpu_sec']:8.2f}s "
                f"share={row['cpu_share']:.2f} mean={row['mean_ms']:8.3f}ms "
                f"rate={row['rate'] if row['rate'] is not None else float('nan'):8.1f}/s"
            )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
