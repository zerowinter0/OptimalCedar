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
    """Return {pipe: (input_bytes, ms_per_record, observations)} for one cell.

    ``metric='process'`` uses the worker's process-time delta per record (CPU
    time spent in that operator's segment), which is the closest available
    stand-in for the operator's own compute; ``metric='wall'`` uses the wall
    segment, which also contains queueing and is therefore only reported for
    reference.
    """
    out = {}
    files = sorted((RUN / f"reconcile_{cell}").glob("worker_*.json"))
    if not files:
        raise SystemExit(f"no trace for {cell}")
    samples = {}
    sizes = {}
    process = {}
    for path in files:
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
    for pid, values in values_by_pipe.items():
        if not values or pid not in sizes or not sizes[pid]:
            continue
        out[pid] = (
            statistics.mean(sizes[pid]),
            statistics.mean(values) / 1e6,
            len(values),
        )
    return out


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
    data = {cell: load_cell(cell) for cell in CELLS}
    pipes = sorted(set.intersection(*(set(d) for d in data.values())))
    pipes = [pid for pid in pipes if data["declared"][pid][0] > 0]

    print("measured input size (bytes/record) per operator and order")
    header = f"{'operator':<14}" + "".join(f"{c:>12}" for c in CELLS)
    print(header)
    for pid in pipes:
        row = f"{NAMES.get(pid, str(pid)):<14}"
        for cell in CELLS:
            row += f"{data[cell][pid][0]:>12.0f}"
        print(row)
    print()
    print("measured wall time per record (ms) per operator and order")
    print(header)
    for pid in pipes:
        row = f"{NAMES.get(pid, str(pid)):<14}"
        for cell in CELLS:
            row += f"{data[cell][pid][1]:>12.4f}"
        print(row)
    print()

    profile = yaml.safe_load(
        (ROOT / "outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml").read_text()
    )
    affine = profile["physical_model"]["operator_affine"]["operators"]

    print("transferring the DECLARED (unoptimized) profile to the PICO-ordered plan")
    print("  cedar  : t = t_declared * (x / x_declared)          (proportional, through origin)")
    print("  affine : t = t_declared * (k*x + b) / (k*x_declared + b)   (profile k,b shape,")
    print("           anchored on the same declared measurement so only the shape differs)")
    print(
        f"{'operator':<14}{'x decl (B)':>12}{'x pico (B)':>12}{'t decl':>9}{'t pico':>9}"
        f"{'cedar est':>11}{'err %':>8}{'affine est':>12}{'err %':>8}"
    )
    cedar_errors, affine_errors, rows = [], [], []
    for pid in pipes:
        x_declared, t_declared, _ = data["declared"][pid]
        x_pico, t_pico, _ = data["pico"][pid]
        entry = affine.get(pid) or affine.get(str(pid)) or {}
        k = float(entry.get("k_ms_per_byte", 0.0) or 0.0)
        b = float(entry.get("b_ms", 0.0) or 0.0)
        cedar_est = t_declared * (x_pico / x_declared)
        denom = k * x_declared + b
        affine_ratio = ((k * x_pico + b) / denom) if denom > 0 else (x_pico / x_declared)
        affine_est = t_declared * affine_ratio
        err_cedar = (cedar_est - t_pico) / t_pico * 100.0
        err_affine = (affine_est - t_pico) / t_pico * 100.0
        cedar_errors.append(abs(err_cedar))
        affine_errors.append(abs(err_affine))
        rows.append(dict(pipe=pid, name=NAMES.get(pid, str(pid)), x_declared=x_declared,
                         x_pico=x_pico, t_declared=t_declared, measured_pico=t_pico,
                         cedar_est=cedar_est, cedar_err_pct=err_cedar,
                         affine_est=affine_est, affine_err_pct=err_affine, k=k, b=b))
        print(f"{NAMES.get(pid, str(pid)):<14}{x_declared:>12.0f}{x_pico:>12.0f}"
              f"{t_declared:>9.4f}{t_pico:>9.4f}{cedar_est:>11.4f}{err_cedar:>8.1f}"
              f"{affine_est:>12.4f}{err_affine:>8.1f}")
    print()
    print("mean |error|: cedar proportional %.1f%%, affine %.1f%%"
          % (statistics.mean(cedar_errors), statistics.mean(affine_errors)))
    print("median |error|: cedar proportional %.1f%%, affine %.1f%%"
          % (statistics.median(cedar_errors), statistics.median(affine_errors)))
    print("max |error|: cedar proportional %.1f%%, affine %.1f%%"
          % (max(cedar_errors), max(affine_errors)))
    print()
    print("same question with coefficients fitted from the four in-pipeline points")
    print("(leave-one-order-out: the table predicts each order from the other three)")
    print(f"{'operator':<14}{'cedar LOO err %':>17}{'affine LOO err %':>18}")
    cedar_loo, affine_loo = [], []
    for pid in pipes:
        points = [(data[cell][pid][0], data[cell][pid][1]) for cell in CELLS]
        for held in range(len(CELLS)):
            train = [p for idx, p in enumerate(points) if idx != held]
            x_hold, t_hold = points[held]
            k_aff, b_aff = fit_affine(train)
            k_prop = fit_proportional(train)
            cedar_loo.append(
                abs(k_prop * x_hold - t_hold) / t_hold * 100.0
            )
            affine_loo.append(
                abs((k_aff * x_hold + b_aff) - t_hold) / t_hold * 100.0
            )
    print(
        f"{'all operators':<14}{statistics.mean(cedar_loo):>17.1f}"
        f"{statistics.mean(affine_loo):>18.1f}"
    )
    print(
        "  median: cedar %.1f%%, affine %.1f%%"
        % (statistics.median(cedar_loo), statistics.median(affine_loo))
    )
    out = RUN / "analysis.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
