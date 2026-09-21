"""Per-operator input / Cedar estimate / measured time in the reordered plans.

Uses the every-record-traced run (``CEDAR_TRACE_FREQUENCY_SEC=0``) so the
inputs are exact means over the whole pass instead of a time-sampled subset,
and the batcher's window is its own stacking work after the attribution fix.

  x          : input bytes per source record at that operator in that order
  t_measured : process-time per record (ms) measured in that order
  cedar est  : t_declared * (x_reordered / x_declared)  -- Cedar's
               proportional-through-origin transfer of the declared profile
  affine est : t_declared * (k*x + b) / (k*x_declared + b) with the profile's
               fitted coefficients (shown for reference)

Usage (inside the container):
  python -u tmp_analysis/reordered_operator_table.py
"""

import json
import statistics
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
RUN = ROOT / "outputs/unopt_order_transfer_repeats_traceall_20260921"
PROFILE = ROOT / "outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml"
NAMES = {
    0: "T  Batcher",
    1: "N  Normalize",
    2: "B  Blur",
    3: "G  Grayscale",
    4: "J  Jitter",
    5: "H  Flip",
    6: "C  Crop",
    7: "F  to_float",
    8: "R  ImageReader",
}
REORDERED = [("pico", "PICO 顺序"), ("cedar", "cedar 顺序"), ("old-dp", "old-dp 顺序")]


def load(cell: str):
    sizes, process, observations = {}, {}, {}
    for path in sorted(RUN.glob(f"reconcile_{cell}_*/worker_*.json")):
        data = json.loads(path.read_text())
        for pid, value in (data.get("input_sizes") or {}).items():
            sizes.setdefault(int(pid), []).append(float(value))
            observations[int(pid)] = observations.get(int(pid), 0) + int(
                (data.get("observations") or {}).get(pid, 0)
            )
        for pid, value in (data.get("process_latency_ns_per_sample") or {}).items():
            process.setdefault(int(pid), []).append(float(value))
    return (
        {p: statistics.mean(v) for p, v in sizes.items() if v},
        {p: statistics.mean(v) / 1e6 for p, v in process.items() if v},
        observations,
    )


def main() -> int:
    profile = yaml.safe_load(PROFILE.read_text())
    affine = profile["physical_model"]["operator_affine"]["operators"]
    x_declared, t_declared, obs_declared = load("declared")
    print(f"declared order traced {obs_declared.get(8, 0)} reader records "
          f"({obs_declared.get(0, 0)} batcher records)\n")
    for cell, label in REORDERED:
        x_other, t_other, obs = load(cell)
        print(f"=== {label} ({RUN.name}/reconcile_{cell}_r1, "
              f"{obs.get(8, 0)} traced records) ===")
        print("| operator | input x (B/record) | cedar est (ms) | measured (ms) | "
              "err % | affine est (ms) | err % |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        cedar_errors, affine_errors = [], []
        for pid in sorted(NAMES):
            if pid not in t_other or pid not in t_declared:
                continue
            xd, td = x_declared[pid], t_declared[pid]
            xo, to = x_other[pid], t_other[pid]
            cedar = td * (xo / xd) if xd else float("nan")
            entry = affine.get(str(pid)) or {}
            k = float(entry.get("k_ms_per_byte", 0.0) or 0.0)
            b = float(entry.get("b_ms", 0.0) or 0.0)
            denom = k * xd + b
            aff = td * ((k * xo + b) / denom) if denom > 0 else cedar
            cedar_errors.append(abs((cedar - to) / to * 100))
            affine_errors.append(abs((aff - to) / to * 100))
            print(
                f"| {NAMES[pid]} | {xo:,.0f} | {cedar:.4f} | {to:.4f} | "
                f"{(cedar - to) / to * 100:+.1f}% | {aff:.4f} | "
                f"{(aff - to) / to * 100:+.1f}% |"
            )
        print(
            f"\nmean |error| over the 9 operators: cedar {statistics.mean(cedar_errors):.1f}%, "
            f"affine {statistics.mean(affine_errors):.1f}%; "
            f"over the 6 size-changing mappers: cedar "
            f"{statistics.mean([e for pid, e in zip(sorted(NAMES), cedar_errors) if pid in (2, 3, 4, 5, 6, 7)]):.1f}%, "
            f"affine "
            f"{statistics.mean([e for pid, e in zip(sorted(NAMES), affine_errors) if pid in (2, 3, 4, 5, 6, 7)]):.1f}%\n"
        )
        print()
    print("declared anchors (x B/record, measured ms/record):")
    for pid in sorted(NAMES):
        if pid in t_declared:
            print(f"  {NAMES[pid]:<14} {x_declared[pid]:>12,.0f} B  {t_declared[pid]:.4f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
