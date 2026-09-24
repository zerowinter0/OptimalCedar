"""Where does the heavy tail of in-pipeline operator time come from?

Two questions:
 1. Isolated: does blur get *slower* when torch is allowed many threads
    (oversubscription), which would make the fast single-thread path the p10?
 2. In-pipeline: does the per-call mean drop when the plan runs with one local
    worker instead of four?

Usage (inside the container):
  python -u tmp_analysis/heavy_tail_probe.py
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


def main() -> int:
    payload = load("pico", 2)
    feature = build_feature(batch_size=4)
    blur = feature.logical_pipes[2].get_fused_callable()
    rows = []
    print(f"{'threads':>8s} {'p10':>9s} {'median':>9s} {'mean':>9s}")
    for threads in (1, 2, 4, 8, 32, 64):
        torch.set_num_threads(threads)
        for _ in range(3):
            blur(payload)
        samples = []
        was = gc.isenabled()
        gc.disable()
        try:
            for _ in range(25):
                calls, duration = 3, 0.0
                for _ in range(calls):
                    start = time.perf_counter()
                    blur(payload)
                    duration += time.perf_counter() - start
                samples.append(1000.0 * duration / calls)
        finally:
            if was:
                gc.enable()
                gc.collect()
        samples.sort()
        row = {
            "threads": threads,
            "p10": samples[max(0, int(0.1 * (len(samples) - 1)))],
            "median": statistics.median(samples),
            "mean": statistics.fmean(samples),
        }
        rows.append(row)
        print(
            f"{threads:8d} {row['p10']:9.4f} {row['median']:9.4f} "
            f"{row['mean']:9.4f}"
        )
    torch.set_num_threads(1)
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/thread_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
