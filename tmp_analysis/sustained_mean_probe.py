"""Is the operator mean stable, or does it depend on how long we measure?

Runs each operator continuously for a fixed window and reports p10/p50/p90/mean
for successive slices.  If the mean grows with the window while the median does
not, the mean is dominated by intermittent interference, which is a property of
the machine's environment rather than of the payload.

Usage (inside the container):
  python -u tmp_analysis/sustained_mean_probe.py [seconds_per_operator]
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


def stats(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "p10": ordered[int(0.1 * (len(ordered) - 1))],
        "p50": statistics.median(ordered),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "mean": statistics.fmean(values),
    }


def main() -> int:
    window = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    feature = build_feature(batch_size=4)
    cases = {
        "B_blur(2) u8 1ch 244": (2, load("pico", 2)),
        "C_crop(6) u8 3ch": (6, load("pico", 6)),
        "J_jitter(4) u8 1ch 244": (4, load("pico", 4)),
        "G_grayscale(3) u8 3ch": (3, load("pico", 3)),
    }
    rows = []
    for name, (pid, payload) in cases.items():
        fn = feature.logical_pipes[pid].get_fused_callable()
        for _ in range(3):
            fn(payload)
        per_call = []
        was = gc.isenabled()
        gc.disable()
        started = time.perf_counter()
        try:
            while time.perf_counter() - started < window:
                t0 = time.perf_counter()
                fn(payload)
                per_call.append((time.perf_counter() - t0) * 1000.0)
        finally:
            if was:
                gc.enable()
                gc.collect()
        head = per_call[: max(10, len(per_call) // 10)]
        row = {
            "operator": name,
            "window_sec": window,
            "head": stats(head),
            "full": stats(per_call),
        }
        rows.append(row)
        print(
            f"{name:24s} first10%: p50={row['head']['p50']:7.3f} "
            f"mean={row['head']['mean']:8.3f} | full: p50={row['full']['p50']:7.3f} "
            f"mean={row['full']['mean']:8.3f} p90={row['full']['p90']:8.3f} "
            f"n={row['full']['n']}"
        )
    target = (
        ROOT / "outputs/affine_reorder_diagnosis_20260924/sustained_mean.json"
    )
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
