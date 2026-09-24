"""Controlled representation matrix for the SimCLRv2 callables.

Same record content, same spatial size, only the *representation* changes
(float32 vs uint8, 3 channels vs 1, PIL vs tensor, full size vs half size).
Every cell is timed with the profiler's own protocol: native thread pools and
torch limited to one thread, fresh unpickle before the clock, repeated single
calls, median rate.

Usage (inside the container):
  python -u tmp_analysis/representation_matrix.py <capture_dir>
"""

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

NAMES = {2: "B blur", 3: "G grayscale", 4: "J jitter", 5: "H flip", 6: "C crop", 7: "F to_float"}


def load_snapshot(capture_dir: Path, p_id: int, index: int = 0):
    matches = sorted(capture_dir.glob(f"capture/*_pipe_{p_id}.pkl"))
    if not matches:
        raise FileNotFoundError(f"no capture for pipe {p_id} in {capture_dir}")
    blob = pickle.loads(matches[0].read_bytes())
    return pickle.loads(blob["snapshots"][index])


def to_uint8(value: torch.Tensor) -> torch.Tensor:
    if value.dtype == torch.uint8:
        return value
    return value.mul(255.0).round().clamp(0, 255).to(torch.uint8).contiguous()


def to_float(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float32).contiguous()


def half_size(value: torch.Tensor) -> torch.Tensor:
    src = value.unsqueeze(0).float()
    out = F.interpolate(src, scale_factor=0.5, mode="bilinear", align_corners=False)
    out = out.squeeze(0)
    return out.round().clamp(0, 255).to(torch.uint8) if value.dtype == torch.uint8 else out


def time_callable(fn, value, warmup=2, calls=10, repeats=5):
    snapshot = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    for _ in range(warmup):
        fn(pickle.loads(snapshot))
    rates = []
    for _ in range(repeats):
        duration = 0.0
        for _ in range(calls):
            fresh = pickle.loads(snapshot)
            start = time.perf_counter()
            fn(fresh)
            duration += time.perf_counter() - start
            del fresh
        rates.append(calls / max(duration, 1e-9))
    return 1000.0 / statistics.median(rates), len(snapshot)


def main() -> int:
    capture_dir = Path(sys.argv[1])
    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes

    crop_input = load_snapshot(capture_dir, 6)          # 3-channel uint8/strided
    blur_input = load_snapshot(capture_dir, 2)          # 1-channel input
    gray3 = crop_input.contiguous()

    variants = {
        "u8 3ch 244 (full)": to_uint8(gray3),
        "f32 3ch 244 (full)": to_float(gray3),
        "u8 1ch 244": to_uint8(blur_input),
        "f32 1ch 244": to_float(blur_input),
        "u8 1ch 122": half_size(to_uint8(blur_input)),
        "f32 1ch 122": to_float(half_size(to_uint8(blur_input))),
    }
    try:
        from PIL import Image

        variants["PIL L 244"] = Image.fromarray(
            to_uint8(blur_input)[0].numpy(), mode="L"
        )
    except Exception as exc:  # noqa: BLE001
        print("PIL variant unavailable:", exc)

    rows = []
    for p_id, op_name in NAMES.items():
        fn = pipes[p_id].get_fused_callable()
        for label, value in variants.items():
            try:
                cost, size = time_callable(fn, value)
            except Exception as exc:  # noqa: BLE001
                print(f"{op_name:12s} {label:20s} rejected: {type(exc).__name__}")
                continue
            rows.append(
                {
                    "operator": op_name,
                    "representation": label,
                    "bytes": size,
                    "ms": cost,
                }
            )
            print(
                f"{op_name:12s} {label:20s} bytes={size:>9d} "
                f"cost={cost:9.4f} ms"
            )
    out = ROOT / "tmp_analysis/representation_matrix.json"
    out.write_text(json.dumps({"capture": str(capture_dir), "rows": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
