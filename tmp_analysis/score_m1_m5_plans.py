"""Score every candidate plan with M1..M5 against the measured mean service time.

Models
  M1  proportional bytes (Cedar's rule, anchored on the declared measurement)
  M2  byte affine  (physical_model.operator_affine)
  M3  element affine, one curve per operator (its declared representation)
  M4  representation-aware, through the origin (k_(i,z) * e)
  M5  representation-aware affine (k_(i,z) * e + b_(i,z))

Each model is evaluated twice: with the statistics a planner propagates
(``predicted``) and with the statistics the executed plan really produced
(``measured``).  The measured target is the mean per-call service time of each
operator inside the complete run, which is what a throughput objective needs.

Usage (inside the container):
  python -u tmp_analysis/score_m1_m5_plans.py <profile.yaml> <capture_prefix>=<plan.yaml> ...
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

from cedar.pipes.common import (  # noqa: E402
    payload_compute_scale,
    payload_representation_class,
)

NAMES = {
    0: "T_batcher",
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
    8: "R_reader",
}


def load_capture(runs):
    """Aggregate one or more capture runs into per-pipe summaries."""
    stats, payloads = {}, {}
    for run in runs:
        directory = ROOT / f"tmp_analysis/capture_{run}/capture"
        for path in sorted(directory.glob("*_op_capture.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            for pid, entry in data["pipes"].items():
                stats.setdefault(int(pid), []).append(entry)
        for path in sorted(directory.glob("*_pipe_*.pkl")):
            try:
                blob = pickle.loads(path.read_bytes())
            except Exception:  # noqa: BLE001
                continue
            for snapshot in blob["snapshots"]:
                try:
                    payloads.setdefault(int(blob["p_id"]), []).append(
                        (len(snapshot), pickle.loads(snapshot))
                    )
                except Exception:  # noqa: BLE001
                    continue
    summary = {}
    for pid, entries in stats.items():
        calls = sum(entry["calls"] for entry in entries)
        summary[pid] = {
            key: sum(entry[key] * entry["calls"] for entry in entries) / calls
            for key in ("mean_ms", "median_ms", "p10_ms", "p90_ms")
        }
        summary[pid]["calls"] = calls
        summary[pid]["runs"] = len(entries)
    return summary, payloads


def plan_order(name: str):
    for base in (
        ROOT / "outputs/unopt_order_transfer_repeats_traceall_20260921/plans",
        ROOT / "tmp_analysis/validation_orders",
        ROOT / "tmp_analysis/semantic_orders",
    ):
        path = base / f"{name}.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text())
            graph = {
                int(k): v for k, v in data["physical_plan"]["graph"].items()
            }
            children = {
                int(child)
                for value in graph.values()
                for child in ([int(x) for x in value.split(",")] if value else [])
            }
            node = next(p for p in graph if p not in children)
            chain = []
            while True:
                chain.append(node)
                value = graph.get(node)
                nxt = [int(x) for x in value.split(",")] if value else []
                if not nxt:
                    break
                node = nxt[0]
            return chain
    raise FileNotFoundError(name)


def transition(model, p_id, klass):
    return str(
        (model.get("class_transition", {}).get(str(p_id)) or {}).get(
            klass, klass
        )
    )


def propagate(model, chain):
    """Features a planner derives along an order: class + element multiplier."""
    sources = model.get("source_class"), float(model.get("source_elements") or 1.0)
    klass, elements = str(sources[0]), sources[1]
    ratios = model.get("element_ratio", {})
    features = {}
    for p_id in chain:
        if p_id in NAMES:
            features[p_id] = (elements, klass)
        elements *= float(ratios.get(str(p_id), 1.0) or 1.0)
        klass = transition(model, p_id, klass)
    return features


def curve(model, p_id, klass, variant):
    entry = (model.get("operators", {}) or {}).get(str(p_id)) or {}
    by_class = entry.get("by_class") or {}
    found = by_class.get(klass)
    if found is None:
        raise KeyError(f"{p_id}:{klass}")
    b = 0.0 if variant == "proportional" else float(found["b_ms"])
    return float(found["k_ms_per_element"]), b


def main() -> int:
    profile = yaml.safe_load(Path(sys.argv[1]).read_text())
    model = profile["physical_model"]["compute_model"]
    byte_affine = profile["physical_model"]["operator_affine"]["operators"]
    baseline_in = {
        int(k): float(v) for k, v in profile["baseline"]["input_sizes"].items()
    }
    declared_stats, _ = load_capture(["declared"])

    targets = []
    for spec in sys.argv[2:]:
        name, runs = spec.split("=", 1)
        targets.append((name, runs.split(",")))

    report = {"plans": {}, "models": {}}
    totals = {key: {} for key in ("measured", "M1", "M2", "M3", "M4", "M5")}
    for name, runs in targets:
        stats, payloads = load_capture(runs)
        chain = plan_order(name)
        predicted = propagate(model, chain)
        rows = []
        plan_totals = {key: 0.0 for key in totals}
        for p_id, op_name in NAMES.items():
            if p_id not in stats:
                continue
            values = payloads.get(p_id) or []
            if values:
                size, value = values[0]
                size = float(size)
                elements = float(payload_compute_scale(value) or 0.0)
                klass = str(payload_representation_class(value))
            else:
                elements, klass = predicted.get(p_id, (0.0, ""))
                size = 0.0
            measured = stats[p_id]["mean_ms"]
            reference = baseline_in[p_id]
            m1 = declared_stats[p_id]["mean_ms"] * (
                (size / reference) if size else 1.0
            )
            m2 = (
                byte_affine[str(p_id)]["k_ms_per_byte"] * size
                + byte_affine[str(p_id)]["b_ms"]
            )
            try:
                # M3 uses the operator's *own* (declared-plan) class for every
                # position: that is the variant the optimizer implements.
                own_class = str(
                    (model.get("operators", {}).get(str(p_id)) or {}).get(
                        "own_class"
                    )
                )
                k3, b3 = curve(model, p_id, own_class, "affine")
                m3 = k3 * elements + b3
            except KeyError:
                m3 = float("nan")
            try:
                k4, _ = curve(model, p_id, klass, "proportional")
                m4 = k4 * elements
            except KeyError:
                m4 = float("nan")
            try:
                k5, b5 = curve(model, p_id, klass, "affine")
                m5 = k5 * elements + b5
            except KeyError:
                m5 = float("nan")
            rows.append(
                {
                    "operator": op_name,
                    "representation": klass,
                    "elements": elements,
                    "measured_mean_ms": measured,
                    "measured_p10_ms": stats[p_id]["p10_ms"],
                    "M1": m1,
                    "M2": m2,
                    "M3": m3,
                    "M4": m4,
                    "M5": m5,
                }
            )
            for key, value in (
                ("measured", measured),
                ("M1", m1),
                ("M2", m2),
                ("M3", m3),
                ("M4", m4),
                ("M5", m5),
            ):
                plan_totals[key] += value
        report["plans"][name] = {
            "chain": chain,
            "rows": rows,
            "totals": plan_totals,
            "runs": runs,
        }
        for key, value in plan_totals.items():
            totals[key][name] = value
        print(f"\n== {name}  (runs: {','.join(runs)})")
        for row in rows:
            print(
                f"  {row['operator']:12s} {row['representation']:>12s} "
                f"e={row['elements']:>9.0f} measured={row['measured_mean_ms']:8.3f} "
                f"M2={row['M2']:8.3f} M3={row['M3']:8.3f} M4={row['M4']:8.3f} "
                f"M5={row['M5']:8.3f}"
            )
        print(
            "  TOTAL measured={measured:.3f} M1={M1:.3f} M2={M2:.3f} "
            "M3={M3:.3f} M4={M4:.3f} M5={M5:.3f}".format(**plan_totals)
        )

    print("\nmodel / measured ratio per plan:")
    for key in ("M1", "M2", "M3", "M4", "M5"):
        ratios = {
            name: totals[key][name] / totals["measured"][name]
            for name in totals["measured"]
            if totals["measured"][name]
        }
        report["models"][key] = ratios
        print(
            f"  {key}: "
            + "  ".join(f"{name}={value:.2f}" for name, value in ratios.items())
        )
    target = (
        ROOT / "outputs/affine_reorder_diagnosis_20260924/m1_m5_plan_scores.json"
    )
    target.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
