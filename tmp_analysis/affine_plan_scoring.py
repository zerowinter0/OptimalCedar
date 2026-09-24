"""Score reordered plans with four operator-cost models.

Fit set: ``tmp_analysis/bench_operator_matrix.py`` (interleaved rounds on the
declared plan's own record, four representation classes, three spatial scales;
k, b come from the two extreme scales, the middle scale is held out).

Validation set: real captured payloads and in-pipeline service times of the
declared / PICO / cedar / old-dp plans.

  M1 proportional-bytes      Cedar's rule, anchored on the declared measurement
  M2 affine-bytes            the layered profile's fitted kx+b on bytes (PICO today)
  M3 affine-elements         kx+b on elements, one pair per operator
  M4 affine-representation   kx+b on elements, one pair per (operator, class)

Every model is scored twice: with the statistics a planner can *propagate*
without the target run, and with the statistics the target run *actually* saw.

Usage (inside the container):
  python -u tmp_analysis/affine_plan_scoring.py
"""

import json
import pickle
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.utils.threading import limit_native_threadpools  # noqa: E402

_THREAD_LIMITER = limit_native_threadpools(1)  # noqa: F841
import torch  # noqa: E402

torch.set_num_threads(1)

from block_mechanism_common import build_feature  # noqa: E402

PROFILE = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2"
    / "profiles/shared.yaml"
)
PLANS = ROOT / "outputs/unopt_order_transfer_repeats_traceall_20260921/plans"
MATRIX = ROOT / "outputs/affine_reorder_diagnosis_20260924/operator_matrix.json"
NAMES = {
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
}
CLASSES = ("uint8:3ch", "float32:3ch", "uint8:1ch", "float32:1ch")
ORDERS = ("declared", "pico", "cedar", "old-dp", "v1", "v2")
VALIDATION_PLANS = ROOT / "tmp_analysis/validation_orders"


def class_of(value):
    if isinstance(value, torch.Tensor):
        channels = int(value.shape[0]) if value.dim() == 3 else 1
        return f"{str(value.dtype).replace('torch.', '')}:{channels}ch", int(
            value.numel()
        )
    return type(value).__name__, 0


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


def plan_order(name: str):
    path = PLANS / f"{name}.yaml"
    if not path.exists():
        path = VALIDATION_PLANS / f"{name}.yaml"
    data = yaml.safe_load(path.read_text())
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
    profile = yaml.safe_load(PROFILE.read_text())
    operators = profile["physical_model"]["operator_affine"]["operators"]
    baseline_in = {
        int(k): float(v) for k, v in profile["baseline"]["input_sizes"].items()
    }
    matrix = json.loads(MATRIX.read_text())
    fits = matrix["fits"]
    per_class = {
        (f["operator"], f["class"]): f for f in fits
    }
    per_operator = {}
    for name in NAMES.values():
        candidates = [
            f for f in fits
            if f["operator"] == name and f.get("k_elements") is not None
        ]
        if not candidates:
            continue
        # The profiler's own-representation fit: use the class the declared
        # plan actually delivers, i.e. the class of the operator's own input.
        per_operator[name] = candidates

    feature = build_feature(batch_size=4)
    fns = {pid: feature.logical_pipes[pid].get_fused_callable() for pid in NAMES}

    captures = {run: load_capture(run) for run in ORDERS}
    declared_a, declared_payloads = captures["declared"]
    source = (declared_payloads.get(7) or [None])[0]
    source_class, source_elements = class_of(source)
    source_bytes = len(pickle.dumps(source, protocol=pickle.HIGHEST_PROTOCOL))

    # Per-operator transition and ratio rules, measured on the declared plan's
    # own payloads: class transition, element ratio, byte ratio.
    rules = {}
    for pid, name in NAMES.items():
        base = (declared_payloads.get(pid) or [None])[0]
        transitions = {}
        for klass in CLASSES:
            probe = _probe_payload(base, klass)
            if probe is None:
                continue
            try:
                out = fns[pid](probe)
            except Exception:  # noqa: BLE001
                continue
            transitions[klass] = class_of(out)[0]
        in_class, in_elements = class_of(base)
        out = fns[pid](base)
        out_class, out_elements = class_of(out)
        rules[pid] = {
            "name": name,
            "in_class": in_class,
            "transition": transitions,
            "element_ratio": out_elements / max(1, in_elements),
            "byte_ratio": len(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
            / max(1, len(pickle.dumps(base, protocol=pickle.HIGHEST_PROTOCOL))),
        }

    report = {"rules": {str(k): v for k, v in rules.items()}, "orders": {}}
    for order in ORDERS:
        a_times, payloads = captures[order]
        sequence = plan_order(order)
        prop_class, prop_elements, prop_bytes = (
            source_class,
            float(source_elements),
            float(source_bytes),
        )
        propagated = {}
        for pid in sequence:
            if pid in NAMES:
                propagated[pid] = {
                    "class": prop_class,
                    "elements": prop_elements,
                    "bytes": prop_bytes,
                }
            rule = rules.get(pid)
            if rule:
                prop_class = rule["transition"].get(prop_class, prop_class)
                prop_elements *= rule["element_ratio"]
                prop_bytes *= rule["byte_ratio"]
        rows = []
        totals = {
            key: 0.0
            for key in ("A", "M1", "M2", "M3", "M4", "M2p", "M3p", "M4p")
        }
        for pid, name in NAMES.items():
            values = payloads.get(pid) or []
            if not values:
                continue
            value = values[min(1, len(values) - 1)]
            klass, elements = class_of(value)
            size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            guess = propagated.get(pid, {})
            model = operators[str(pid)]
            m1 = declared_a.get(pid, float("nan")) * size / baseline_in[pid]
            m2 = model["k_ms_per_byte"] * size + model["b_ms"]
            own = _pick_fit(per_operator.get(name), rules[pid]["in_class"])
            m3 = (own["k_elements"] * elements + own["b_elements"]) if own else float("nan")
            entry = per_class.get((name, klass))
            m4 = (
                entry["k_elements"] * elements + entry["b_elements"]
                if entry
                else float("nan")
            )
            m2p = model["k_ms_per_byte"] * guess.get("bytes", float("nan")) + model["b_ms"]
            m3p = (
                own["k_elements"] * guess.get("elements", float("nan")) + own["b_elements"]
                if own
                else float("nan")
            )
            p_entry = per_class.get((name, guess.get("class")))
            m4p = (
                p_entry["k_elements"] * guess.get("elements", float("nan"))
                + p_entry["b_elements"]
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
                    "predicted_class": guess.get("class"),
                    "predicted_elements": guess.get("elements"),
                    "predicted_bytes": guess.get("bytes"),
                    "A_ms": measured,
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "M4": m4,
                    "M2p": m2p,
                    "M3p": m3p,
                    "M4p": m4p,
                }
            )
            totals["A"] += measured
            for key, value_ in (
                ("M1", m1), ("M2", m2), ("M3", m3), ("M4", m4),
                ("M2p", m2p), ("M3p", m3p), ("M4p", m4p),
            ):
                totals[key] += value_
        report["orders"][order] = {"rows": rows, "totals": totals}
        print(f"\n== {order}")
        for row in rows:
            print(
                f"  {row['operator']:12s} meas {row['measured_class']:>11s} "
                f"{row['measured_elements']:>8d}el {row['measured_bytes']:>9d}B | "
                f"pred {str(row['predicted_class']):>11s} "
                f"{row['predicted_elements']:>8.0f}el {row['predicted_bytes']:>9.0f}B | "
                f"A={row['A_ms']:8.4f} M2={row['M2']:8.4f} M3={row['M3']:8.4f} "
                f"M4={row['M4']:8.4f} | prop M2={row['M2p']:8.4f} "
                f"M3={row['M3p']:8.4f} M4={row['M4p']:8.4f}"
            )
        print("  TOTAL " + " ".join(f"{k}={v:.3f}" for k, v in totals.items()))
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/plan_scoring.json"
    target.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {target}")
    return 0


def _pick_fit(candidates, klass):
    if not candidates:
        return None
    for entry in candidates:
        if entry["class"] == klass:
            return entry
    return candidates[0]


def _probe_payload(base, klass):
    if base is None or not isinstance(base, torch.Tensor) or base.dim() != 3:
        return None
    dtype, channels = klass.split(":")
    out = base
    floating = out.dtype.is_floating_point
    original_dtype = out.dtype
    if channels == "1ch" and out.shape[0] == 3:
        out = out.to(torch.float32).mean(dim=0, keepdim=True)
    elif channels == "3ch" and out.shape[0] == 1:
        out = out.repeat(3, 1, 1)
    if dtype == "uint8":
        out = out.to(torch.float32).mul(1.0 if floating else 1.0)
        if floating:
            out = out.mul(255.0)
        out = out.round().clamp(0, 255).to(torch.uint8)
    else:
        out = out.to(torch.float32)
        if original_dtype == torch.uint8 and floating is False:
            out = out.div(255.0)
    return out.contiguous()


if __name__ == "__main__":
    raise SystemExit(main())
