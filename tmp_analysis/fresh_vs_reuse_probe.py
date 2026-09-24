"""Does the profiler's fresh-unpickle-per-call protocol inflate small-op cost?

The layered profiler unpickles a snapshot before every timed call; the
interleaved benchmark reuses one payload.  This probe measures both ways on the
same payloads, interleaved, in one process.

Usage (inside the container):
  python -u tmp_analysis/fresh_vs_reuse_probe.py [rounds]
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
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    feature = build_feature(batch_size=4)
    payloads = {
        "B_blur(2)  u8 1ch 244 (pico)": (2, load("pico", 2)),
        "B_blur(2)  f32 1ch 244 (decl)": (2, load("declared", 2)),
        "G_grayscale(3) u8 3ch (pico)": (3, load("pico", 3)),
        "J_jitter(4) u8 1ch 244 (pico)": (4, load("pico", 4)),
        "C_crop(6)  f32 3ch (decl)": (6, load("declared", 6)),
    }
    samples = {(name, mode): [] for name in payloads for mode in ("fresh", "reuse")}
    snapshots = {
        name: pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        for name, (_, value) in payloads.items()
    }
    for name, (pid, value) in payloads.items():
        fn = feature.logical_pipes[pid].get_fused_callable()
        for _ in range(3):
            fn(value)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(rounds):
            for name, (pid, value) in payloads.items():
                fn = feature.logical_pipes[pid].get_fused_callable()
                for mode in ("fresh", "reuse"):
                    calls, duration = 3, 0.0
                    for _ in range(calls):
                        target = pickle.loads(snapshots[name]) if mode == "fresh" else value
                        start = time.perf_counter()
                        fn(target)
                        duration += time.perf_counter() - start
                    samples[(name, mode)].append(1000.0 * duration / calls)
    finally:
        if was_enabled:
            gc.enable()
            gc.collect()
    rows = []
    print(f"{'payload':32s} {'fresh p10':>10s} {'reuse p10':>10s} {'fresh/reuse':>12s}")
    for name in payloads:
        fresh = sorted(samples[(name, "fresh")])
        reuse = sorted(samples[(name, "reuse")])
        f = fresh[max(0, int(0.1 * (len(fresh) - 1)))]
        r = reuse[max(0, int(0.1 * (len(reuse) - 1)))]
        rows.append({"payload": name, "fresh_p10_ms": f, "reuse_p10_ms": r})
        print(f"{name:32s} {f:10.4f} {r:10.4f} {f / r:12.2f}")
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/fresh_vs_reuse.json"
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
