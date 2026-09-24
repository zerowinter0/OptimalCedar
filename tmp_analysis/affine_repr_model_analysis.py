"""Fit three operator-cost models on the declared plan, score three reordered plans.

Fit set (never the target plans): the payloads the *declared* SimCLRv2 plan
really hands to its operators, captured from a live run.  Where an operator's
own input carries no size contrast the layered profiler rescales it spatially;
the same rule is used here, so every model is fitted with one protocol.

  M1 proportional-bytes   cost = C_declared * bytes / bytes_declared   (Cedar)
  M2 affine-bytes         cost = k * bytes + b                        (PICO today)
  M3 affine-elements      cost = k * elements + b, elements = C*H*W
  M4 affine-elements-repr cost = k_(op, channel class) * elements + b

Validation plans: the PICO / cedar / old-dp orders.  Their measured per-operator
service time comes from the diagnostic capture of the same runs (A), and the
replay of the captured payloads (B) is reported next to it so the framework
context can be separated from the payload itself.

Usage (inside the container):
  python -u tmp_analysis/affine_repr_model_analysis.py [out_dir]
"""

import json
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

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
NAMES = {
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
}
VALIDATION_ORDERS = ("pico", "cedar", "old-dp")


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
    return 1000.0 / statistics.median(rates), len(snapshot)


def representation_of(value):
    """(representation class, elements) of one payload.

    The class must separate dtype and channel count: a float32 single-channel
    tensor and a uint8 single-channel tensor have the same element count but
    are priced differently by real image operators, and a ``to_float`` call is
    free on the former and a real conversion on the latter.
    """
    import PIL.Image

    if isinstance(value, torch.Tensor):
        channels = int(value.shape[0]) if value.dim() == 3 else 1
        return f"{str(value.dtype).replace('torch.', '')}:{channels}ch", int(
            value.numel()
        )
    if isinstance(value, PIL.Image.Image):
        width, height = value.size
        bands = len(value.getbands())
        return f"PIL{value.mode}:{bands}ch", int(width * height * bands)
    if isinstance(value, (bytes, str)):
        return "text", len(value)
    return type(value).__name__, 0


def elements_of(value):
    return representation_of(value)[1]


def rescale(value, factor: float, depth: int = 0):
    """Same rule the layered profiler uses to make a size contrast."""
    import PIL.Image

    if depth > 3 or factor <= 0.0:
        return None
    if isinstance(value, dict):
        return None
    if isinstance(value, PIL.Image.Image):
        width, height = value.size
        size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
        if size == value.size:
            return None
        return value.resize(size, PIL.Image.BILINEAR)
    if isinstance(value, torch.Tensor) and value.dim() >= 2:
        channel_last = value.dim() == 3 and value.shape[-1] in (1, 3, 4)
        moved = value.permute(2, 0, 1) if channel_last else value
        height, width = int(moved.shape[-2]), int(moved.shape[-1])
        size = (max(1, int(round(height * factor))), max(1, int(round(width * factor))))
        if size == (height, width):
            return None
        floating = moved.dtype.is_floating_point
        resized = F.interpolate(
            moved.unsqueeze(0).float(),
            size=size,
            mode="bilinear" if floating else "nearest",
            align_corners=False if floating else None,
        ).squeeze(0)
        if not floating:
            resized = resized.round().to(moved.dtype)
        if channel_last:
            resized = resized.permute(1, 2, 0)
        return resized.contiguous()
    return None


def load_capture(run: str):
    """Per pipe: measured in-pipeline time, and the payloads it received."""
    directory = ROOT / f"tmp_analysis/capture_{run}/capture"
    measured, payloads, reps = {}, {}, {}
    for path in sorted(directory.glob("*_op_capture.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for pid, entry in data["pipes"].items():
            measured.setdefault(int(pid), []).append(entry["mean_ms"])
            reps.setdefault(int(pid), {}).update(entry["representations"])
    for path in sorted(directory.glob("*_pipe_*.pkl")):
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001
            # A worker can be torn down while it is flushing a snapshot; a
            # truncated file is skipped rather than aborting the analysis.
            continue
        payloads.setdefault(int(blob["p_id"]), []).extend(blob["snapshots"])
    return (
        {pid: statistics.fmean(v) for pid, v in measured.items()},
        {pid: [pickle.loads(s) for s in v] for pid, v in payloads.items()},
        reps,
    )


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "outputs/affine_reorder_diagnosis_20260924"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = yaml.safe_load(PROFILE.read_text())
    operators = profile["physical_model"]["operator_affine"]["operators"]
    references = {
        int(k): float(v)
        for k, v in profile["baseline"]["input_sizes"].items()
    }

    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    fns = {pid: pipes[pid].get_fused_callable() for pid in NAMES}

    declared_a, declared_payloads, declared_reps = load_capture("declared")

    # ---------------------------------------------------------------- fitting
    fit: Dict[int, Dict[str, Any]] = {}
    fit_rows = []
    for pid, name in NAMES.items():
        fn = fns[pid]
        pool = declared_payloads.get(pid) or []
        own = pool[0] if pool else None
        if own is None:
            raise RuntimeError(f"no captured input for pipe {pid}")
        own_class, own_elements = representation_of(own)

        # Spatial rescales of the operator's own legal input, exactly like the
        # layered profiler's counterfactual when the legal inputs lack contrast.
        points = []
        for factor in (0.5, 2.0):
            candidate = rescale(own, factor)
            if candidate is None:
                continue
            elements = elements_of(candidate)
            cost, size = time_callable(fn, candidate)
            points.append({"factor": factor, "elements": elements, "bytes": size, "ms": cost})
        points.sort(key=lambda item: item["elements"])

        # Per channel class: a payload of the declared plan that this operator
        # accepts.  Class 3 uses its own input; class 1 uses the single-channel
        # payload the same declared plan materialises (the grayscale output).
        class_points: Dict[str, Any] = {}
        donors: Dict[str, Any] = {own_class: own}
        for other_id, payloads in declared_payloads.items():
            for value in payloads:
                klass, _ = representation_of(value)
                if klass in donors:
                    continue
                try:
                    fn(value)
                except Exception:  # noqa: BLE001
                    continue
                donors[klass] = value
        for klass, base in donors.items():
            scaled = []
            for factor in (0.5, 2.0):
                candidate = rescale(base, factor)
                if candidate is None:
                    continue
                elements = elements_of(candidate)
                cost, _ = time_callable(fn, candidate)
                scaled.append((float(elements), cost))
            scaled.sort()
            if len(scaled) >= 2:
                (x0, y0), (x1, y1) = scaled[0], scaled[-1]
                k = max(0.0, (y1 - y0) / (x1 - x0))
                b = max(0.0, y0 - k * x0)
                class_points[klass] = {
                    "k": k,
                    "b": b,
                    "points": scaled,
                    "base": klass,
                }

        if len(points) >= 2:
            low, high = points[0], points[-1]
            k_el = max(
                0.0,
                (high["ms"] - low["ms"])
                / (high["elements"] - low["elements"]),
            )
            b_el = max(0.0, low["ms"] - k_el * low["elements"])
        else:
            k_el, b_el = 0.0, statistics.fmean(p["ms"] for p in points) if points else 0.0
        fit[pid] = {
            "name": name,
            "own_elements": own_elements,
            "own_class": own_class,
            "k_elements": k_el,
            "b_elements": b_el,
            "by_class": class_points,
        }
        fit_rows.append(
            {
                "pipe": pid,
                "operator": name,
                "own_class": own_class,
                "elements": [p["elements"] for p in points],
                "ms": [p["ms"] for p in points],
                "k_elements": k_el,
                "b_elements": b_el,
                "classes": sorted(class_points),
            }
        )
        print(
            f"fit {name:12s} own={own_elements:>7d} el ({own_class}) "
            f"k={k_el:.3e} b={b_el:7.4f} classes={sorted(class_points)}"
        )

    # ------------------------------------------------------------ validation
    report = {"fit": fit_rows, "orders": {}}
    for order in VALIDATION_ORDERS:
        a_times, payloads, reps = load_capture(order)
        rows = []
        totals = {
            "measured_A": 0.0,
            "measured_B": 0.0,
            "M1_proportional": 0.0,
            "M2_affine_bytes": 0.0,
            "M3_affine_elements": 0.0,
            "M4_representation": 0.0,
        }
        for pid, name in NAMES.items():
            values = payloads.get(pid) or []
            if not values:
                continue
            value = values[0]
            klass, elements = representation_of(value)
            size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            replay, _ = time_callable(fns[pid], value)
            model = operators[str(pid)]
            ref = references[pid]
            # M1: Cedar's proportional transfer of the declared measurement.
            declared_bytes = ref
            m1 = declared_a[pid] * size / declared_bytes
            # M2: the layered profile's fitted kx+b on serialized bytes.
            m2 = model["k_ms_per_byte"] * size + model["b_ms"]
            # M3: one kx+b per operator, fitted on elements, own representation.
            m3 = fit[pid]["k_elements"] * elements + fit[pid]["b_elements"]
            # M4: kx+b per (operator, channel class) fitted on elements.
            entry = fit[pid]["by_class"].get(klass)
            m4 = (
                entry["k"] * elements + entry["b"]
                if entry
                else float("nan")
            )
            measured = a_times.get(pid, float("nan"))
            rows.append(
                {
                    "pipe": pid,
                    "operator": name,
                    "representation": klass,
                    "elements": elements,
                    "bytes": size,
                    "measured_A_ms": measured,
                    "measured_B_ms": replay,
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "M4": m4,
                    "err_M1": (m1 - measured) / measured if measured else None,
                    "err_M2": (m2 - measured) / measured if measured else None,
                    "err_M3": (m3 - measured) / measured if measured else None,
                    "err_M4": (m4 - measured) / measured if measured else None,
                }
            )
            totals["measured_A"] += measured
            totals["measured_B"] += replay
            totals["M1_proportional"] += m1
            totals["M2_affine_bytes"] += m2
            totals["M3_affine_elements"] += m3
            totals["M4_representation"] += m4
        report["orders"][order] = {"rows": rows, "totals": totals}
        print(f"\n== {order} (ms per source record, summed over priced operators)")
        for row in rows:
            print(
                f"  {row['operator']:12s} {row['representation']:>10s} el={row['elements']:>8d} "
                f"B={row['bytes']:>8d} A={row['measured_A_ms']:8.4f} "
                f"B_replay={row['measured_B_ms']:8.4f} "
                f"M2={row['M2']:8.4f}({row['err_M2']*100:+6.1f}%) "
                f"M3={row['M3']:8.4f}({row['err_M3']*100:+6.1f}%) "
                f"M4={row['M4']:8.4f}({row['err_M4']*100:+6.1f}%)"
            )
        print(
            "  TOTAL A={measured_A:.3f} B={measured_B:.3f} "
            "M1={M1_proportional:.3f} M2={M2_affine_bytes:.3f} "
            "M3={M3_affine_elements:.3f} M4={M4_representation:.3f}".format(**totals)
        )
    target = out_dir / "affine_repr_model_analysis.json"
    target.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
