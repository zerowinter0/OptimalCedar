"""Core matrix for the affine-on-reordered-plans question.

Fit set: the payloads the *declared* SimCLRv2 plan really hands to its
operators (captured from a live run).  Validation set: three reordered plans
whose per-operator inputs and in-pipeline service times were captured the same
way.

Models
  M1 proportional-bytes    cost = C_declared * bytes / bytes_declared  (Cedar)
  M2 affine-bytes          cost = k_bytes * bytes + b_bytes           (PICO today)
  M3 affine-elements       cost = k_el * elements + b_el
  M4 affine-representation cost = k_(op,class) * elements + b_(op,class)

Both prediction routes are reported per operator:
  *predicted statistics* propagate the source payload through the per-operator
  ratios the profile measures (what a planner can know without running the
  target plan), and *measured statistics* come from the target plan's capture.
Comparing the two isolates a statistics-propagation error from a
cost-response error.

Usage (inside the container):
  python -u tmp_analysis/affine_reorder_matrix.py [out_dir]
"""

import json
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.utils.threading import limit_native_threadpools  # noqa: E402

_THREAD_LIMITER = limit_native_threadpools(1)  # noqa: F841
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)

from block_mechanism_common import build_feature  # noqa: E402

PROFILE = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2"
    / "profiles/shared.yaml"
)
PLANS = ROOT / "outputs/unopt_order_transfer_repeats_traceall_20260921/plans"
NAMES = {
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
}
SOURCE_PIPE = 8
VALIDATION = ("pico", "cedar", "old-dp")
CLASSES = ("uint8:3ch", "float32:3ch", "uint8:1ch", "float32:1ch")


# --------------------------------------------------------------------- utils
def representation_of(value) -> Tuple[str, int, int]:
    """(class, elements, pixels) of one payload."""
    import PIL.Image

    if isinstance(value, torch.Tensor):
        if value.dim() == 3:
            channels = int(value.shape[0])
            pixels = int(value.shape[1] * value.shape[2])
        else:
            channels, pixels = 1, int(value.numel())
        klass = f"{str(value.dtype).replace('torch.', '')}:{channels}ch"
        return klass, int(value.numel()), pixels
    if isinstance(value, PIL.Image.Image):
        width, height = value.size
        bands = len(value.getbands())
        return f"PIL{value.mode}", int(width * height * bands), int(width * height)
    return type(value).__name__, len(value) if hasattr(value, "__len__") else 0, 0


def rescale(value, factor: float):
    import PIL.Image

    if factor <= 0.0:
        return None
    if isinstance(value, PIL.Image.Image):
        width, height = value.size
        size = (
            max(1, int(round(width * factor))),
            max(1, int(round(height * factor))),
        )
        if size == value.size:
            return None
        return value.resize(size, PIL.Image.BILINEAR)
    if isinstance(value, torch.Tensor) and value.dim() >= 2:
        channel_last = value.dim() == 3 and value.shape[-1] in (1, 3, 4)
        moved = value.permute(2, 0, 1) if channel_last else value
        height, width = int(moved.shape[-2]), int(moved.shape[-1])
        size = (
            max(1, int(round(height * factor))),
            max(1, int(round(width * factor))),
        )
        if size == (height, width):
            return None
        floating = moved.dtype.is_floating_point
        out = F.interpolate(
            moved.unsqueeze(0).float(),
            size=size,
            mode="bilinear" if floating else "nearest",
            align_corners=False if floating else None,
        ).squeeze(0)
        if not floating:
            out = out.round().to(moved.dtype)
        if channel_last:
            out = out.permute(1, 2, 0)
        return out.contiguous()
    return None


def rebase_class(value, target_class: str, gray_fn, float_fn):
    """Move one payload into another representation class the plan can emit."""
    klass, _, _ = representation_of(value)
    if klass == target_class:
        return value
    dtype_target, channels_target = target_class.split(":")
    out = value
    if out.dim() != 3:
        return None
    channels = int(out.shape[0])
    if channels_target == "1ch" and channels == 3:
        out = gray_fn(out)
    elif channels_target == "3ch" and channels == 1:
        return None
    if dtype_target == "float32" and out.dtype != torch.float32:
        out = float_fn(out)
    elif dtype_target == "uint8" and out.dtype != torch.uint8:
        out = out.mul(255.0).round().clamp(0, 255).to(torch.uint8)
    return out.contiguous()


def time_callable(fn, value, warmup=2, calls=8, repeats=5):
    snapshot = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    for _ in range(warmup):
        fn(pickle.loads(snapshot))
    rates = []
    for _ in range(repeats):
        duration = 0.0
        for _ in range(calls):
            fresh = pickle.loads(snapshot)
            start = time.perf_counter()
            fn(fresh)
            duration += time.perf_counter() - start
            del fresh
        rates.append(calls / max(duration, 1e-9))
    return 1000.0 / statistics.median(rates)


def load_capture(run: str):
    directory = ROOT / f"tmp_analysis/capture_{run}/capture"
    measured, payloads = {}, {}
    for path in sorted(directory.glob("*_op_capture.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for pid, entry in data["pipes"].items():
            measured.setdefault(int(pid), []).append(entry["mean_ms"])
    for path in sorted(directory.glob("*_pipe_*.pkl")):
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001
            continue
        payloads.setdefault(int(blob["p_id"]), []).extend(blob["snapshots"])
    return (
        {pid: statistics.fmean(v) for pid, v in measured.items()},
        {pid: [pickle.loads(s) for s in v] for pid, v in payloads.items()},
    )


def plan_order(name: str) -> List[int]:
    data = yaml.safe_load((PLANS / f"{name}.yaml").read_text())
    graph = {int(k): v for k, v in data["physical_plan"]["graph"].items()}
    children = {
        int(child)
        for value in graph.values()
        for child in ([int(x) for x in value.split(",")] if value else [])
    }
    current = next(pid for pid in graph if pid not in children)
    order = []
    while True:
        order.append(current)
        value = graph.get(current)
        nxt = [int(x) for x in value.split(",")] if value else []
        if not nxt:
            break
        current = nxt[0]
    return order


def main() -> int:
    out_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "outputs/affine_reorder_diagnosis_20260924"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = yaml.safe_load(PROFILE.read_text())
    operators = profile["physical_model"]["operator_affine"]["operators"]
    baseline_in = {int(k): float(v) for k, v in profile["baseline"]["input_sizes"].items()}

    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    fns = {pid: pipes[pid].get_fused_callable() for pid in NAMES}
    gray_fn = fns[3]
    float_fn = fns[7]

    declared_a, declared_payloads = load_capture("declared")

    # ------------------------------------------------------- per-operator ratios
    # Each operator's behaviour on the payload it really received: class
    # transition, element ratio, byte ratio.  These ratios are plan independent
    # and are what a planner propagates.
    ratios: Dict[int, Dict[str, Any]] = {}
    for pid in list(NAMES) + [SOURCE_PIPE]:
        values = declared_payloads.get(pid) or []
        if not values:
            continue
        value = values[0]
        klass, elements, pixels = representation_of(value)
        entry: Dict[str, Any] = {
            "in_class": klass,
            "in_elements": elements,
            "in_pixels": pixels,
            "in_bytes": len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)),
        }
        if pid in NAMES:
            out = fns[pid](value)
            out_klass, out_elements, out_pixels = representation_of(out)
            out_bytes = len(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
            entry.update(
                {
                    "out_class": out_klass,
                    "element_ratio": out_elements / max(1, elements),
                    "pixel_ratio": out_pixels / max(1, pixels),
                    "byte_ratio": out_bytes / max(1, entry["in_bytes"]),
                }
            )
        ratios[pid] = entry

    # ---------------------------------------------------------------- fitting
    fit_rows: List[Dict[str, Any]] = []
    class_fit: Dict[int, Dict[str, Dict[str, float]]] = {}
    own_fit: Dict[int, Dict[str, float]] = {}
    for pid, name in NAMES.items():
        fn = fns[pid]
        base = (declared_payloads.get(pid) or [None])[0]
        if base is None:
            raise RuntimeError(f"no declared input captured for {name}")
        if not torch.is_tensor(base):
            base_class, base_elements, _ = representation_of(base)
            fit_rows.append(
                {
                    "pipe": pid,
                    "operator": name,
                    "class": base_class,
                    "elements": [base_elements],
                    "ms": [],
                }
            )
            continue
        table: Dict[str, Dict[str, float]] = {}
        for klass in CLASSES:
            try:
                candidate = rebase_class(base, klass, gray_fn, float_fn)
            except Exception:  # noqa: BLE001
                candidate = None
            if candidate is None:
                continue
            try:
                fn(candidate)
            except Exception:  # noqa: BLE001
                continue
            points = []
            for factor in (0.5, 2.0):
                scaled = rescale(candidate, factor)
                if scaled is None:
                    continue
                elements = representation_of(scaled)[1]
                points.append((float(elements), time_callable(fn, scaled)))
            points.sort()
            if len(points) < 2:
                continue
            (x0, y0), (x1, y1) = points[0], points[-1]
            k = max(0.0, (y1 - y0) / (x1 - x0))
            b = max(0.0, y0 - k * x0)
            table[klass] = {"k": k, "b": b, "points": points}
            fit_rows.append(
                {
                    "pipe": pid,
                    "operator": name,
                    "class": klass,
                    "k_elements": k,
                    "b_elements": b,
                    "points": points,
                }
            )
            print(
                f"fit {name:12s} {klass:12s} k={k:.3e} b={b:7.4f} "
                f"points={[(int(x), round(y, 3)) for x, y in points]}"
            )
        class_fit[pid] = table
        own_class = representation_of(base)[0]
        own = table.get(own_class)
        if own is None and table:
            own = next(iter(table.values()))
        own_fit[pid] = own or {"k": 0.0, "b": 0.0}

    # ------------------------------------------------------------- validation
    report: Dict[str, Any] = {
        "ratios": ratios,
        "fit": fit_rows,
        "orders": {},
    }
    for order in VALIDATION:
        a_times, payloads = load_capture(order)
        sequence = plan_order(order)
        # ---- propagated statistics (what a planner knows without the target run)
        # The reader's output is the input of the first operator of the
        # declared order, which is the payload every plan starts from.
        source = (declared_payloads.get(7) or [None])[0]
        src_bytes = len(pickle.dumps(source, protocol=pickle.HIGHEST_PROTOCOL))
        src_class, src_elements, _ = representation_of(source)
        prop_bytes, prop_elements = src_bytes, src_elements
        prop_class = src_class
        predicted: Dict[int, Dict[str, float]] = {}
        for pid in sequence:
            if pid in NAMES:
                predicted[pid] = {
                    "class": prop_class,
                    "bytes": prop_bytes,
                    "elements": prop_elements,
                }
            ratio = ratios.get(pid)
            if ratio and "element_ratio" in ratio:
                prop_bytes *= ratio["byte_ratio"]
                prop_elements *= ratio["element_ratio"]
                prop_class = ratio["out_class"]
        # ---- measured statistics + models
        rows = []
        totals = {
            "measured_A": 0.0,
            "measured_B": 0.0,
            "M1_A": 0.0,
            "M2": 0.0,
            "M3": 0.0,
            "M4": 0.0,
            "M2_prop": 0.0,
            "M3_prop": 0.0,
            "M4_prop": 0.0,
        }
        for pid, name in NAMES.items():
            values = payloads.get(pid) or []
            if not values:
                continue
            value = values[min(1, len(values) - 1)]
            klass, elements, _ = representation_of(value)
            size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            replay = time_callable(fns[pid], value)
            model = operators[str(pid)]
            ref = baseline_in[pid]
            m1 = declared_a.get(pid, float("nan")) * size / ref
            m2 = model["k_ms_per_byte"] * size + model["b_ms"]
            m3 = own_fit[pid]["k"] * elements + own_fit[pid]["b"]
            entry = class_fit[pid].get(klass)
            m4 = (
                entry["k"] * elements + entry["b"]
                if entry
                else float("nan")
            )
            guess = predicted.get(pid, {})
            p_class = guess.get("class")
            p_elements = guess.get("elements", float("nan"))
            p_bytes = guess.get("bytes", float("nan"))
            m2p = model["k_ms_per_byte"] * p_bytes + model["b_ms"]
            m3p = own_fit[pid]["k"] * p_elements + own_fit[pid]["b"]
            p_entry = class_fit[pid].get(p_class)
            m4p = (
                p_entry["k"] * p_elements + p_entry["b"]
                if p_entry
                else float("nan")
            )
            measured = a_times.get(pid, float("nan"))
            rows.append(
                {
                    "pipe": pid,
                    "operator": name,
                    "measured_class": klass,
                    "measured_elements": elements,
                    "measured_bytes": size,
                    "predicted_class": p_class,
                    "predicted_elements": p_elements,
                    "predicted_bytes": p_bytes,
                    "measured_A_ms": measured,
                    "measured_B_ms": replay,
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "M4": m4,
                    "M2_prop": m2p,
                    "M3_prop": m3p,
                    "M4_prop": m4p,
                }
            )
            for key, value_ in (
                ("measured_A", measured),
                ("measured_B", replay),
                ("M1_A", m1),
                ("M2", m2),
                ("M3", m3),
                ("M4", m4),
                ("M2_prop", m2p),
                ("M3_prop", m3p),
                ("M4_prop", m4p),
            ):
                totals[key] += value_
        report["orders"][order] = {"rows": rows, "totals": totals}
        print(f"\n== {order}  (ms / source record)")
        for row in rows:
            print(
                f"  {row['operator']:12s} meas={row['measured_class']:>11s}"
                f" el={row['measured_elements']:>7d} B={row['measured_bytes']:>8d}"
                f" | pred={str(row['predicted_class']):>11s}"
                f" el={row['predicted_elements']:>9.0f} B={row['predicted_bytes']:>9.0f}"
                f" | A={row['measured_A_ms']:8.4f} B={row['measured_B_ms']:8.4f}"
                f" M2={row['M2']:8.4f} M3={row['M3']:8.4f} M4={row['M4']:8.4f}"
            )
        print(
            "  TOTAL A={measured_A:.3f} B={measured_B:.3f} M1={M1_A:.3f} "
            "M2={M2:.3f} M3={M3:.3f} M4={M4:.3f} | propagated "
            "M2={M2_prop:.3f} M3={M3_prop:.3f} M4={M4_prop:.3f}".format(**totals)
        )

    target = out_dir / "affine_reorder_matrix.json"
    target.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
