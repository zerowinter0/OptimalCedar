"""Interleaved operator benchmark: do bytes or elements predict the cost?

Payloads: the SimCLRv2 transforms are measured on the same record content in
four representation classes (uint8/float32 x 1/3 channels) at three spatial
scales each.  Every cell is measured in *interleaved rounds*; the reported
number is the median over rounds, with the inter-quartile range, so a single
slow period cannot dominate.

The fit uses only the two extreme scales (the profiler's own two-stratum rule)
under two size features -- serialized bytes and elements -- and the middle
scale is then a held-out point for both.

Usage (inside the container):
  python -u tmp_analysis/bench_operator_matrix.py [rounds]
"""

import gc
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.utils.threading import limit_native_threadpools  # noqa: E402

_THREAD_LIMITER = limit_native_threadpools(1)  # noqa: F841
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)

from block_mechanism_common import build_feature  # noqa: E402

NAMES = {
    1: "N_normalize",
    2: "B_blur",
    3: "G_grayscale",
    4: "J_jitter",
    5: "H_flip",
    6: "C_crop",
    7: "F_to_float",
}
SCALES = {"122": 0.5, "244": 1.0, "488": 2.0}
CLASSES = ("uint8:3ch", "float32:3ch", "uint8:1ch", "float32:1ch")


def load_crop_input():
    paths = sorted(
        (ROOT / "tmp_analysis/capture_declared/capture").glob("*_pipe_6.pkl")
    )
    for path in paths:
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001
            continue
        if blob["snapshots"]:
            return pickle.loads(blob["snapshots"][0])
    raise SystemExit("no captured crop input")


def to_class(value, klass: str):
    dtype, channels = klass.split(":")
    out = value
    if out.dim() != 3:
        return None
    if channels == "1ch" and out.shape[0] == 3:
        out = out.mean(dim=0, keepdim=True)
    if dtype == "uint8":
        out = out.mul(255.0).round().clamp(0, 255).to(torch.uint8)
    else:
        out = out.to(torch.float32)
    return out.contiguous()


def resize(value, factor: float):
    if factor == 1.0:
        return value.contiguous()
    height, width = int(value.shape[-2]), int(value.shape[-1])
    size = (max(1, int(round(height * factor))), max(1, int(round(width * factor))))
    floating = value.dtype.is_floating_point
    out = F.interpolate(
        value.unsqueeze(0).float(),
        size=size,
        mode="bilinear" if floating else "nearest",
        align_corners=False if floating else None,
    ).squeeze(0)
    if not floating:
        out = out.round().to(value.dtype)
    return out.contiguous()


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    feature = build_feature(batch_size=4)
    fns = {pid: feature.logical_pipes[pid].get_fused_callable() for pid in NAMES}
    base = to_class(load_crop_input(), "float32:3ch")

    cells = []
    for klass in CLASSES:
        class_base = to_class(base, klass)
        if class_base is None:
            continue
        for scale_name, factor in SCALES.items():
            payload = resize(class_base, factor)
            for pid, name in NAMES.items():
                try:
                    fns[pid](payload)
                except Exception:  # noqa: BLE001
                    continue
                cells.append(
                    {
                        "operator": name,
                        "pipe": pid,
                        "class": klass,
                        "scale": scale_name,
                        "elements": int(payload.numel()),
                        "bytes": len(
                            pickle.dumps(
                                payload, protocol=pickle.HIGHEST_PROTOCOL
                            )
                        ),
                        "payload": payload,
                        "samples": [],
                    }
                )
    for cell in cells:
        for _ in range(3):
            fns[cell["pipe"]](cell["payload"])
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(rounds):
            for cell in cells:
                fn = fns[cell["pipe"]]
                calls, duration = 3, 0.0
                for _ in range(calls):
                    start = time.perf_counter()
                    fn(cell["payload"])
                    duration += time.perf_counter() - start
                cell["samples"].append(1000.0 * duration / calls)
    finally:
        if was_enabled:
            gc.enable()
            gc.collect()

    rows = []
    for cell in cells:
        samples = sorted(cell["samples"])
        median = statistics.median(samples)
        p10 = samples[max(0, int(0.1 * (len(samples) - 1)))]
        rows.append(
            {
                "operator": cell["operator"],
                "pipe": cell["pipe"],
                "class": cell["class"],
                "scale": cell["scale"],
                "elements": cell["elements"],
                "bytes": cell["bytes"],
                "ms_median": median,
                "ms_p10": p10,
                "ms_p25": samples[len(samples) // 4],
                "ms_p75": samples[(3 * len(samples)) // 4],
            }
        )
    # Two-stratum fit per (operator, representation class): the profiler's own
    # rule, with the middle scale kept as a held-out point for both features.
    fits = []
    by_cell = {(r["operator"], r["class"], r["scale"]): r for r in rows}
    for operator in sorted({r["operator"] for r in rows}):
        for klass in CLASSES:
            low = by_cell.get((operator, klass, "122"))
            high = by_cell.get((operator, klass, "488"))
            mid = by_cell.get((operator, klass, "244"))
            if low is None or high is None:
                continue
            entry = {"operator": operator, "class": klass}
            for feature in ("elements", "bytes"):
                x0, x1 = low[feature], high[feature]
                y0, y1 = low["ms_p10"], high["ms_p10"]
                k = max(0.0, (y1 - y0) / max(1.0, x1 - x0))
                b = max(0.0, y0 - k * x0)
                entry[f"k_{feature}"] = k
                entry[f"b_{feature}"] = b
                if mid is not None:
                    predicted = k * mid[feature] + b
                    entry[f"heldout_ms_{feature}"] = mid["ms_p10"]
                    entry[f"pred_ms_{feature}"] = predicted
                    entry[f"err_{feature}"] = (
                        (predicted - mid["ms_p10"]) / mid["ms_p10"]
                        if mid["ms_p10"]
                        else None
                    )
            fits.append(entry)
    print("\nfits (two extreme scales; the middle scale is held out):")
    print(
        f"{'operator':12s} {'class':12s} {'k_el':>11s} {'b_el':>8s} "
        f"{'err_el':>8s} {'k_B':>11s} {'b_B':>8s} {'err_B':>8s}"
    )
    for entry in fits:
        def fmt(value, spec="11.3e"):
            return format(value, spec) if value is not None else "n/a"

        print(
            f"{entry['operator']:12s} {entry['class']:12s} "
            f"{fmt(entry['k_elements'])} {fmt(entry['b_elements'], '8.4f')} "
            f"{fmt(entry.get('err_elements'), '8.1%')} "
            f"{fmt(entry['k_bytes'])} {fmt(entry['b_bytes'], '8.4f')} "
            f"{fmt(entry.get('err_bytes'), '8.1%')}"
        )
    print(f"{'operator':12s} {'class':12s} {'scale':6s} {'elements':>9s} "
          f"{'bytes':>9s} {'p10 ms':>10s} {'median ms':>10s} {'IQR':>16s}")
    for row in rows:
        print(
            f"{row['operator']:12s} {row['class']:12s} {row['scale']:6s} "
            f"{row['elements']:9d} {row['bytes']:9d} {row['ms_p10']:10.4f} "
            f"{row['ms_median']:10.4f} "
            f"{row['ms_p25']:7.4f}-{row['ms_p75']:8.4f}"
        )
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/operator_matrix.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"rows": rows, "fits": fits}, indent=1))
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
