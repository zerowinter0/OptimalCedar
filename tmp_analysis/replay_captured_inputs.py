"""B/C: replay the payloads a real plan handed to its operators.

For every captured snapshot this prints
  A  the direct callable time measured inside the running pipeline,
  B  the isolated replay time of the very same payload,
  C  the layered profile's own ``kx+b`` prediction for that payload size,
plus the payload representation, so a byte-only model can be compared against
what the operator actually received.

Usage (inside the container):
  python -u tmp_analysis/replay_captured_inputs.py <capture_dir> [profile.yaml]
"""

import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import build_feature  # noqa: E402

# The profiling driver and every local worker run with the native thread pools
# limited to one thread; the replay must match or it measures a parallel conv.
from cedar.utils.threading import limit_native_threadpools  # noqa: E402

_THREAD_LIMITER = limit_native_threadpools(1)  # noqa: F841
import torch  # noqa: E402

torch.set_num_threads(1)

NAMES = {
    0: "T batcher",
    1: "N normalize",
    2: "B blur",
    3: "G grayscale",
    4: "J jitter",
    5: "H flip",
    6: "C crop",
    7: "F to_float",
    8: "R reader",
}


def describe(value):
    import torch

    if isinstance(value, torch.Tensor):
        return (
            f"torch.{value.dtype} {tuple(value.shape)} "
            f"{'contig' if value.is_contiguous() else 'strided'}"
        )
    return type(value).__name__


def time_value(fn, value, calls=5, repeats=5):
    rates = []
    for _ in range(repeats):
        duration = 0.0
        for _ in range(calls):
            start = time.perf_counter()
            fn(value)
            duration += time.perf_counter() - start
        rates.append(calls / max(duration, 1e-9))
    return 1000.0 / statistics.median(rates)


def main() -> int:
    capture_dir = Path(sys.argv[1])
    profile_path = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2"
        "/profiles/shared.yaml"
    )
    profile = yaml.safe_load(profile_path.read_text())
    operators = profile["physical_model"]["operator_affine"]["operators"]
    references = profile["baseline"]["input_sizes"]

    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes

    baseline = {}
    for path in (capture_dir / "capture").glob("*_op_capture.json"):
        data = json.loads(path.read_text())
        for pid, entry in data["pipes"].items():
            baseline.setdefault(int(pid), []).append(entry["mean_ms"])

    rows = []
    print(f"capture: {capture_dir}")
    for path in sorted((capture_dir / "capture").glob("*_pipe_*.pkl")):
        blob = pickle.loads(path.read_bytes())
        p_id = int(blob["p_id"])
        fn = pipes[p_id].get_fused_callable()
        model = operators.get(str(p_id), operators.get(p_id))
        reference = float(
            references.get(str(p_id), references.get(p_id, 0.0))
        )
        y0 = (
            model["k_ms_per_byte"] * reference + model["b_ms"]
            if model
            else float("nan")
        )
        for snapshot in blob["snapshots"]:
            value = pickle.loads(snapshot)
            size = len(snapshot)
            replay = time_value(fn, value)
            affine = (
                model["k_ms_per_byte"] * size + model["b_ms"]
                if model
                else float("nan")
            )
            shape = describe(value)
            rows.append(
                {
                    "p_id": p_id,
                    "size": size,
                    "shape": shape,
                    "replay_ms": replay,
                    "affine_ms": affine,
                    "affine_abs_bytes": affine,
                }
            )
            print(
                f"  {NAMES[p_id]:12s} {shape:44s} bytes={size:>9d} "
                f"B_replay={replay:8.4f} ms  C_affine={affine:8.4f} ms  "
                f"err={(affine - replay) / replay * 100:7.1f}%"
            )

    print("\nper operator (mean over captured payloads):")
    for p_id in sorted({row["p_id"] for row in rows}):
        subset = [row for row in rows if row["p_id"] == p_id]
        replay = statistics.fmean(row["replay_ms"] for row in subset)
        affine = statistics.fmean(row["affine_ms"] for row in subset)
        a = statistics.fmean(baseline.get(p_id, [float("nan")]))
        print(
            f"  {NAMES[p_id]:12s} A_pipeline={a:8.4f} ms  B_replay={replay:8.4f} ms"
            f"  C_affine={affine:8.4f} ms  (A/B={a / replay:.2f}x,"
            f" C/B={affine / replay:.2f}x)"
        )
    out = capture_dir / "replay_summary.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
