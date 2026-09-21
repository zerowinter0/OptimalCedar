"""Does Cedar's per-operator cost transfer to a new plan order?

Reads the CEDAR_RECONCILE_DIR traces of the four single-worker, all-INPROCESS
pipelines produced by scripts/run_unopt_order_transfer_20260921.sh.  For every
operator the traces give the input bytes per record (the size at which the
operator ran in that order) and the measured wall time per record.

Two model forms are then asked to predict a held-out order:

  cedar      t = k * x            (one measurement point, proportional through
                                   the origin -- what Cedar's profile does)
  affine     t = k * x + b        (two coefficients, the PICO operator model)

Usage (inside the container):
  python -u scripts/analyze_unopt_order_transfer.py
"""

import json
import statistics

import yaml
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
RUN = ROOT / "outputs/unopt_order_transfer_repeats_20260921"
if not RUN.exists():
    RUN = ROOT / "outputs/unopt_order_transfer_20260921"
CELLS = ["declared", "pico", "cedar", "old-dp"]
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


def load_cell(cell: str, metric: str = "process"):
    """Per repeat: {pipe: (input_bytes, ms_per_record)} for one order.

    ``metric='process'`` uses the worker's process-time delta per record (CPU
    time spent in that operator's segment); ``metric='wall'`` uses the wall
    segment, which also contains queueing.
    """
    runs = []
    dirs = sorted((RUN).glob(f"reconcile_{cell}*"))
    if not dirs:
        raise SystemExit(f"no trace for {cell}")
    for directory in dirs:
        samples, sizes, process = {}, {}, {}
        for path in sorted(directory.glob("worker_*.json")):
            data = json.loads(path.read_text())
            if metric == "process":
                for pid, value in (data.get("process_latency_ns_per_sample") or {}).items():
                    process.setdefault(int(pid), []).append(float(value))
            else:
                for pid, values in (data.get("wall_latency_samples") or {}).items():
                    samples.setdefault(int(pid), []).extend(values)
            for pid, value in (data.get("input_sizes") or {}).items():
                sizes.setdefault(int(pid), []).append(float(value))
        values_by_pipe = process if metric == "process" else samples
        point = {}
        for pid, values in values_by_pipe.items():
            if not values or not sizes.get(pid):
                continue
            point[pid] = (
                statistics.mean(sizes[pid]),
                statistics.mean(values) / 1e6,
            )
        runs.append(point)
    return runs


def fit_affine(points):
    """Least squares t = k*x + b."""
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(t for _, t in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * t for x, t in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-9:
        # every observation sits at the same input size: only the intercept is
        # identifiable
        return 0.0, sy / n
    k = (n * sxy - sx * sy) / denom
    b = (sy - k * sx) / n
    return k, b


def fit_proportional(points):
    """Least squares t = k*x through the origin."""
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * t for x, t in points)
    if abs(sxx) < 1e-9:
        return 0.0
    return sxy / sxx


def main() -> int:
    per_repeat = {cell: load_cell(cell) for cell in CELLS}
    reps = min(len(v) for v in per_repeat.values())
    pipes = sorted(
        set.intersection(
            *(
                set(pid for run in per_repeat[cell] for pid in run)
                for cell in CELLS
            )
        )
    )
    mean, spread = {}, {}
    for cell in CELLS:
        mean[cell], spread[cell] = {}, {}
        for pid in pipes:
            xs = [run[pid][0] for run in per_repeat[cell] if pid in run]
            ts = [run[pid][1] for run in per_repeat[cell] if pid in run]
            if not ts:
                continue
            mean[cell][pid] = (statistics.mean(xs), statistics.mean(ts))
            spread[cell][pid] = statistics.stdev(ts) if len(ts) > 1 else 0.0
    pipes = [pid for pid in pipes if mean["declared"].get(pid, (0, 0))[0] > 0]

    print(f"orders: {CELLS}")
    print(f"repeats per order: {reps}  (round-robin, same input: 4 workers x 592 batches x 4)")
    print()
    print("input size (bytes/record, identical across repeats)")
    header = f"{'operator':<14}" + "".join(f"{c:>12}" for c in CELLS)
    print(header)
    for pid in pipes:
        print(f"{NAMES.get(pid, str(pid)):<14}" + "".join(
            f"{mean[c][pid][0]:>12.0f}" for c in CELLS))
    print()
    print("measured process-time per record, ms (mean over repeats ± stdev)")
    print(header)
    for pid in pipes:
        row = f"{NAMES.get(pid, str(pid)):<14}"
        for cell in CELLS:
            m = mean[cell][pid][1]
            sd = spread[cell][pid]
            row += f"{m:>8.4f}±{sd:<3.2f}"
        print(row)
    print()
    reader = [pid for pid in pipes if NAMES.get(pid, "").endswith("ImageReader")]
    if reader:
        pid = reader[0]
        print("control operator (same position and size in every order)")
        for cell in CELLS:
            print("  %-9s %.4f ± %.4f ms" % (cell, mean[cell][pid][1], spread[cell][pid]))
        values = [mean[cell][pid][1] for cell in CELLS]
        pooled = statistics.mean([spread[cell][pid] for cell in CELLS])
        print("  spread across orders %.4f ms (%.1f%% of the mean), pooled run-to-run stdev %.4f ms"
              % (max(values) - min(values),
                 (max(values) - min(values)) / statistics.mean(values) * 100.0,
                 pooled))
        print()

    profile = yaml.safe_load(
        (ROOT / "outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml").read_text()
    )
    affine = profile["physical_model"]["operator_affine"]["operators"]

    print("transferring the DECLARED profile to the PICO-ordered plan (repeat means)")
    print("  cedar  : t = t_declared * (x / x_declared)")
    print("  affine : t = t_declared * (k*x + b) / (k*x_declared + b)   (profile k,b shape)")
    print(f"{'operator':<14}{'t pico':>9}{'±':>7}{'cedar est':>11}{'err %':>8}"
          f"{'affine est':>12}{'err %':>8}{'noise':>8}")
    cedar_errors, affine_errors, rows = [], [], []
    for pid in pipes:
        x_declared, t_declared = mean["declared"][pid]
        x_pico, t_pico = mean["pico"][pid]
        entry = affine.get(pid) or affine.get(str(pid)) or {}
        k = float(entry.get("k_ms_per_byte", 0.0) or 0.0)
        b = float(entry.get("b_ms", 0.0) or 0.0)
        cedar_est = t_declared * (x_pico / x_declared)
        denom = k * x_declared + b
        affine_est = t_declared * (
            ((k * x_pico + b) / denom) if denom > 0 else (x_pico / x_declared)
        )
        err_cedar = (cedar_est - t_pico) / t_pico * 100.0
        err_affine = (affine_est - t_pico) / t_pico * 100.0
        cedar_errors.append(abs(err_cedar))
        affine_errors.append(abs(err_affine))
        rows.append(dict(pipe=pid, name=NAMES.get(pid, str(pid)), x_declared=x_declared,
                         x_pico=x_pico, t_declared=t_declared, measured_pico=t_pico,
                         cedar_est=cedar_est, cedar_err_pct=err_cedar,
                         affine_est=affine_est, affine_err_pct=err_affine, k=k, b=b,
                         stdev_pico=spread["pico"][pid]))
        print(f"{NAMES.get(pid, str(pid)):<14}{t_pico:>9.4f}{spread['pico'][pid]:>7.2f}"
              f"{cedar_est:>11.4f}{err_cedar:>8.1f}{affine_est:>12.4f}{err_affine:>8.1f}"
              f"{spread['pico'][pid] / t_pico * 100.0:>7.1f}%")
    print()
    print("mean |error|: cedar proportional %.1f%%, affine %.1f%%"
          % (statistics.mean(cedar_errors), statistics.mean(affine_errors)))
    print("median |error|: cedar proportional %.1f%%, affine %.1f%%"
          % (statistics.median(cedar_errors), statistics.median(affine_errors)))
    print("max |error|: cedar proportional %.1f%%, affine %.1f%%"
          % (max(cedar_errors), max(affine_errors)))
    print("mean run-to-run noise (stdev/mean): %.1f%%"
          % statistics.mean(
              spread[c][pid] / mean[c][pid][1] * 100.0
              for c in CELLS for pid in pipes if mean[c].get(pid, (0, 0))[1] > 0
          ))
    print()
    print("leave-one-order-out over the four order means")
    cedar_loo, affine_loo = [], []
    for pid in pipes:
        points = [mean[cell][pid] for cell in CELLS]
        for held in range(len(CELLS)):
            train = [p for idx, p in enumerate(points) if idx != held]
            x_hold, t_hold = points[held]
            k_aff, b_aff = fit_affine(train)
            k_prop = fit_proportional(train)
            cedar_loo.append(abs(k_prop * x_hold - t_hold) / t_hold * 100.0)
            affine_loo.append(abs((k_aff * x_hold + b_aff) - t_hold) / t_hold * 100.0)
    print("  mean |error|: cedar %.1f%%, affine %.1f%%"
          % (statistics.mean(cedar_loo), statistics.mean(affine_loo)))
    print("  median |error|: cedar %.1f%%, affine %.1f%%"
          % (statistics.median(cedar_loo), statistics.median(affine_loo)))
    out = RUN / "analysis.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
