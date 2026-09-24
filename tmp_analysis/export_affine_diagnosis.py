"""Export the affine-reordering diagnosis into the shared deliverable files.

Inputs: the captured plan runs (per-operator in-pipeline distributions and real
payloads), the interleaved operator matrix with its per-class fits, and the
plan scoring produced by ``affine_plan_scoring.py``.

Usage (inside the container):
  python -u tmp_analysis/export_affine_diagnosis.py
"""

import csv
import json
import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/affine_reorder_diagnosis_20260924"
NAMES = {
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
}
PLAN_ORDER = {
    "declared": "9 8 7 6 5 4 3 2 1 0",
    "pico": "9 8 6 3 4 5 2 7 1 0",
    "cedar": "9 8 5 6 3 4 2 7 1 0",
    "old-dp": "9 7 6 5 4 3 2 1 0",
    "v1": "9 8 6 5 4 3 2 7 1 0",
    "v2": "9 8 7 3 6 5 4 2 1 0",
}


def load_capture(plan: str):
    directory = ROOT / f"tmp_analysis/capture_{plan}/capture"
    stats, reps = {}, {}
    for path in sorted(directory.glob("*_op_capture.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for pid, entry in data["pipes"].items():
            stats.setdefault(int(pid), []).append(entry)
            reps.setdefault(int(pid), {}).update(entry["representations"])
    rows = {}
    for pid, entries in stats.items():
        calls = sum(entry["calls"] for entry in entries)

        def weighted(key):
            return sum(entry[key] * entry["calls"] for entry in entries) / calls

        rows[pid] = {
            "calls": calls,
            "mean_ms": weighted("mean_ms"),
            "median_ms": weighted("median_ms"),
            "p10_ms": weighted("p10_ms"),
            "p90_ms": weighted("p90_ms"),
            "representation": max(
                reps[pid].items(), key=lambda item: item[1]
            )[0],
        }
    return rows


def payload_of(plan: str, pid: int):
    for path in sorted(
        (ROOT / f"tmp_analysis/capture_{plan}/capture").glob(f"*_pipe_{pid}.pkl")
    ):
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001
            continue
        if blob["snapshots"]:
            return pickle.loads(blob["snapshots"][0])
    return None


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scoring = json.loads((OUT / "plan_scoring.json").read_text())
    matrix = json.loads((OUT / "operator_matrix.json").read_text())

    # ---------------------------------------------------------- input metadata
    metadata_rows = []
    for plan in PLAN_ORDER:
        capture = load_capture(plan)
        for pid, name in NAMES.items():
            if pid not in capture:
                continue
            value = payload_of(plan, pid)
            if value is None:
                continue
            if hasattr(value, "numel"):
                shape = list(value.shape)
                dtype = str(value.dtype)
                elements = int(value.numel())
                contiguous = bool(value.is_contiguous())
            else:
                shape, dtype, elements, contiguous = [], type(value).__name__, 0, None
            metadata_rows.append(
                {
                    "plan": plan,
                    "operator": name,
                    "pipe": pid,
                    "representation": capture[pid]["representation"],
                    "dtype": dtype,
                    "shape": "x".join(str(dim) for dim in shape),
                    "contiguous": contiguous,
                    "elements": elements,
                    "payload_bytes": len(
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    ),
                }
            )
    write_csv(
        OUT / "input_metadata.csv",
        metadata_rows,
        [
            "plan", "operator", "pipe", "representation", "dtype", "shape",
            "contiguous", "elements", "payload_bytes",
        ],
    )

    # ---------------------------------------------------- operator diagnostics
    diagnostic_rows = []
    for plan in PLAN_ORDER:
        capture = load_capture(plan)
        rows = {row["operator"]: row for row in scoring["orders"][plan]["rows"]}
        for pid, name in NAMES.items():
            if pid not in capture:
                continue
            entry = rows.get(name, {})
            diagnostic_rows.append(
                {
                    "plan": plan,
                    "operator": name,
                    "representation": capture[pid]["representation"],
                    "calls": capture[pid]["calls"],
                    "in_pipeline_p10_ms": capture[pid]["p10_ms"],
                    "in_pipeline_median_ms": capture[pid]["median_ms"],
                    "in_pipeline_mean_ms": capture[pid]["mean_ms"],
                    "in_pipeline_p90_ms": capture[pid]["p90_ms"],
                    "M1_proportional": entry.get("M1"),
                    "M2_affine_bytes": entry.get("M2"),
                    "M3_affine_elements": entry.get("M3"),
                    "M4_affine_representation": entry.get("M4"),
                }
            )
    write_csv(
        OUT / "operator_diagnostics.csv",
        diagnostic_rows,
        [
            "plan", "operator", "representation", "calls",
            "in_pipeline_p10_ms", "in_pipeline_median_ms",
            "in_pipeline_mean_ms", "in_pipeline_p90_ms",
            "M1_proportional", "M2_affine_bytes", "M3_affine_elements",
            "M4_affine_representation",
        ],
    )

    # ------------------------------------------------------------ predictions
    prediction_rows = []
    for plan, payload in scoring["orders"].items():
        for row in payload["rows"]:
            for model, key in (
                ("M1_proportional", "M1"),
                ("M2_affine_bytes", "M2"),
                ("M3_affine_elements", "M3"),
                ("M4_affine_representation", "M4"),
            ):
                prediction_rows.append(
                    {
                        "plan": plan,
                        "operator": row["operator"],
                        "model": model,
                        "statistics": "measured",
                        "predicted_ms": row[key],
                    }
                )
            for model, key in (
                ("M2_affine_bytes", "M2p"),
                ("M3_affine_elements", "M3p"),
                ("M4_affine_representation", "M4p"),
            ):
                prediction_rows.append(
                    {
                        "plan": plan,
                        "operator": row["operator"],
                        "model": model,
                        "statistics": "propagated",
                        "predicted_ms": row[key],
                    }
                )
    write_csv(
        OUT / "predictions.csv",
        prediction_rows,
        ["plan", "operator", "model", "statistics", "predicted_ms"],
    )

    # ---------------------------------------------------------- plan summary
    summary_rows = []
    for plan, payload in scoring["orders"].items():
        capture = load_capture(plan)
        totals = payload["totals"]
        summary_rows.append(
            {
                "plan": plan,
                "plan_order": PLAN_ORDER.get(plan),
                "measured_p10_sum_ms": sum(
                    entry["p10_ms"] for entry in capture.values()
                ),
                "measured_median_sum_ms": sum(
                    entry["median_ms"] for entry in capture.values()
                ),
                "measured_mean_sum_ms": sum(
                    entry["mean_ms"] for entry in capture.values()
                ),
                **{key: value for key, value in totals.items()},
            }
        )
    fields = [
        "plan", "plan_order", "measured_p10_sum_ms", "measured_median_sum_ms",
        "measured_mean_sum_ms", "A", "M1", "M2", "M3", "M4",
        "M2p", "M3p", "M4p",
    ]
    write_csv(OUT / "plan_summary.csv", summary_rows, fields)

    # ------------------------------------------------------------- figure data
    figure = {
        "operator_matrix": matrix["rows"],
        "operator_fits": matrix["fits"],
        "plan_scoring": scoring["orders"],
        "plan_summary": summary_rows,
        "blur_geometry": json.loads((OUT / "blur_geometry.json").read_text()),
        "fresh_vs_reuse": json.loads((OUT / "fresh_vs_reuse.json").read_text()),
    }
    (OUT / "figure_data.json").write_text(json.dumps(figure, indent=1, default=float))
    print(f"wrote {OUT}/input_metadata.csv, operator_diagnostics.csv, "
          f"predictions.csv, plan_summary.csv, figure_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
