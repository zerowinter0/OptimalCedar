"""Is the heavy tail preemption (wall) or real work (CPU)?

Times the same operator calls with both the wall clock and the process CPU
clock.  If the CPU time has no tail while the wall clock does, the tail is the
shared machine stealing the core, not the operator's work.

Usage (inside the container):
  python -u tmp_analysis/wall_vs_cpu_probe.py
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


def main() -> int:
    feature = build_feature(batch_size=4)
    cases = {
        "B_blur (2)  u8 1ch 244 (pico)": (2, load("pico", 2), 120),
        "C_crop (6)  u8 3ch (pico)": (6, load("pico", 6), 120),
        "J_jitter (4) u8 1ch 244 (pico)": (4, load("pico", 4), 120),
        "G_grayscale (3) u8 3ch (pico)": (3, load("pico", 3), 120),
    }
    rows = []
    print(
        f"{'operator':34s} {'wall p50':>9s} {'wall mean':>10s} "
        f"{'cpu p50':>9s} {'cpu mean':>9s} {'cpu/wall':>9s}"
    )
    for name, (pid, payload, calls) in cases.items():
        fn = feature.logical_pipes[pid].get_fused_callable()
        for _ in range(3):
            fn(payload)
        wall, cpu = [], []
        was = gc.isenabled()
        gc.disable()
        try:
            for _ in range(calls):
                w0, c0 = time.perf_counter(), time.process_time()
                fn(payload)
                wall.append((time.perf_counter() - w0) * 1000.0)
                cpu.append((time.process_time() - c0) * 1000.0)
        finally:
            if was:
                gc.enable()
                gc.collect()
        row = {
            "operator": name,
            "wall_p50": statistics.median(wall),
            "wall_mean": statistics.fmean(wall),
            "wall_p10": sorted(wall)[int(0.1 * len(wall))],
            "wall_p90": sorted(wall)[int(0.9 * len(wall))],
            "cpu_p50": statistics.median(cpu),
            "cpu_mean": statistics.fmean(cpu),
        }
        rows.append(row)
        print(
            f"{name:34s} {row['wall_p50']:9.4f} {row['wall_mean']:10.4f} "
            f"{row['cpu_p50']:9.4f} {row['cpu_mean']:9.4f} "
            f"{row['cpu_mean'] / row['wall_mean']:9.2f}"
        )
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/wall_vs_cpu.json"
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
