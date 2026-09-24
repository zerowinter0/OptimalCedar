"""Mean vs median of blur along the element axis, measured long enough to be stable.

The planning objective is a mean, so the shape of the mean curve decides
whether an affine fit on it is admissible.  Every payload is measured in
interleaved rounds for several seconds; both the median and the mean are kept.

Usage (inside the container):
  python -u tmp_analysis/blur_mean_curve.py [seconds_per_payload]
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


def load(run: str, pid: int, index: int = 0):
    for path in sorted(
        (ROOT / f"tmp_analysis/capture_{run}/capture").glob(f"*_pipe_{pid}.pkl")
    ):
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception:  # noqa: BLE001
            continue
        if len(blob["snapshots"]) > index:
            return pickle.loads(blob["snapshots"][index])
    return None


def resize(value, size):
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
    window = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    feature = build_feature(batch_size=4)
    blur = feature.logical_pipes[2].get_fused_callable()
    u8_244 = load("pico", 2)
    f32_244 = load("declared", 2)
    payloads = {
        "u8 1ch 61x61": resize(u8_244, (61, 61)),
        "u8 1ch 122x122": resize(u8_244, (122, 122)),
        "u8 1ch 187x250": resize(u8_244, (250, 187)),
        "u8 1ch 244x244": u8_244,
        "u8 1ch 375x500": resize(u8_244, (500, 375)),
        "f32 1ch 122x122": resize(f32_244, (122, 122)),
        "f32 1ch 244x244": f32_244,
        "f32 1ch 375x500": resize(f32_244, (500, 375)),
    }
    samples = {name: [] for name in payloads}
    for value in payloads.values():
        blur(value)
    was = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter()
        while time.perf_counter() - started < window:
            for name, value in payloads.items():
                t0 = time.perf_counter()
                blur(value)
                samples[name].append((time.perf_counter() - t0) * 1000.0)
    finally:
        if was:
            gc.enable()
            gc.collect()
    rows = []
    print(f"{'payload':20s} {'elements':>9s} {'p50':>8s} {'mean':>8s} {'p90':>8s} {'n':>6s}")
    for name, value in payloads.items():
        values = samples[name]
        ordered = sorted(values)
        row = {
            "payload": name,
            "elements": int(value.numel()),
            "p50": statistics.median(ordered),
            "mean": statistics.fmean(values),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "n": len(values),
        }
        rows.append(row)
        print(
            f"{name:20s} {row['elements']:9d} {row['p50']:8.3f} "
            f"{row['mean']:8.3f} {row['p90']:8.3f} {row['n']:6d}"
        )
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/blur_mean_curve.json"
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
