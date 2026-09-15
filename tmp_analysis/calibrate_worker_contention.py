"""Write a same-plan worker sweep into ``physical_model.worker_contention``.

    python tmp_analysis/calibrate_worker_contention.py <profile.yaml> \
        --point 8:471.2 --point 16:485.7 --point 32:410.7

Each ``W:throughput`` pair is one measured run of the *same* plan with a fixed
local-worker count.  The DP's objective is a per-record service time for one
worker, so it implicitly credits ``W`` workers with ``W`` times the throughput.
Contention makes that false on a fixed host: the measurement is turned into the
inflation factor the model has to apply to its per-worker prediction,

    contention(W) = (W / throughput(W)) / min_V (V / throughput(V))

which is 1.0 at the best measured worker count and grows for worker counts that
add contention without adding throughput.  ``_dp_worker_contention_factor``
then reads the *running maximum* of these points, so the published curve is the
monotone envelope of a noisy measurement.  The term only exists in PICO's
objective: the Cedar-cost-model ablation scores plans with Cedar's original
cost function and therefore keeps assuming linear worker scaling.
"""

import argparse
import pathlib
import sys

import yaml


def build_points(pairs):
    measured = {}
    for item in pairs:
        workers_raw, throughput_raw = item.split(":", 1)
        workers = int(workers_raw)
        throughput = float(throughput_raw)
        if workers < 1 or not throughput > 0.0:
            raise ValueError(f"invalid point {item!r}")
        measured[workers] = throughput
    if len(measured) < 2:
        raise ValueError("at least two worker counts are required")
    per_worker = {w: w / r for w, r in measured.items()}
    best = min(per_worker.values())
    return {
        int(w): round(value / best, 4)
        for w, value in sorted(per_worker.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--point", action="append", required=True)
    parser.add_argument(
        "--note",
        default=(
            "same plan re-optimised at a fixed local-worker count on the "
            "formal 64-CPU budget; the DP multiplies its per-worker cost by "
            "this factor before comparing aggregate throughput"
        ),
    )
    args = parser.parse_args()

    path = pathlib.Path(args.profile)
    data = yaml.safe_load(path.read_text())
    points = build_points(args.point)
    data.setdefault("physical_model", {})["worker_contention"] = {
        "schema_version": 1,
        "method": "same_plan_worker_sweep",
        "note": args.note,
        "points": points,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"{path.name}: worker_contention points={points}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
