"""Is the blur cost set by the byte count or by the representation?

Same record content, same spatial size, both real captured payloads and
synthetically re-classed ones, one measurement protocol.
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


def load(run: str, pid: int, index: int = 0):
    matches = sorted(
        (ROOT / f"tmp_analysis/capture_{run}/capture").glob(f"*_pipe_{pid}.pkl")
    )
    for match in matches:
        try:
            blob = pickle.loads(match.read_bytes())
        except Exception:  # noqa: BLE001
            continue
        if blob["snapshots"]:
            return pickle.loads(blob["snapshots"][index])
    raise SystemExit(f"no snapshot for {run}/{pid}")


def time_value(fn, value, calls=8, repeats=7):
    snapshot = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    rates = []
    for _ in range(repeats):
        duration = 0.0
        for _ in range(calls):
            fresh = pickle.loads(snapshot)
            start = time.perf_counter()
            fn(fresh)
            duration += time.perf_counter() - start
        rates.append(calls / max(duration, 1e-9))
    return 1000.0 / statistics.median(rates), len(snapshot)


def to_u8(value):
    return value.mul(255.0).round().clamp(0, 255).to(torch.uint8).contiguous()


def to_f32(value):
    return value.to(torch.float32).contiguous()


def half(value, dtype):
    out = F.interpolate(
        value.unsqueeze(0).float(),
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if dtype == torch.uint8:
        return out.round().clamp(0, 255).to(torch.uint8).contiguous()
    return out.contiguous()


def main() -> int:
    feature = build_feature(batch_size=4)
    blur = feature.logical_pipes[2].get_fused_callable()
    declared_f32 = load("declared", 2)
    pico_u8 = load("pico", 2)
    cedar_u8 = load("cedar", 2)
    old_f32_rgb = load("old-dp", 2)
    variants = {
        "declared f32 1ch 244 (real)": declared_f32,
        "declared->u8 1ch 244 (synthetic)": to_u8(declared_f32),
        "pico u8 1ch 244 (real)": pico_u8,
        "pico->f32 1ch 244 (synthetic)": to_f32(pico_u8),
        "cedar u8 1ch 244 (real)": cedar_u8,
        "declared->u8 1ch 122 (synthetic)": half(to_u8(declared_f32), torch.uint8),
        "pico u8 1ch 122 (real halved)": half(pico_u8, torch.uint8),
        "declared f32 1ch 122 (synthetic)": half(declared_f32, torch.float32),
        "old-dp f32 3ch 375x500 (real)": old_f32_rgb,
    }
    print(f"{'payload':36s} {'bytes':>9s} {'ms':>9s}")
    rows = []
    for name, value in variants.items():
        ms, size = time_value(blur, value)
        rows.append({"payload": name, "bytes": size, "ms": ms})
        print(f"{name:36s} {size:9d} {ms:9.4f}")
    target = ROOT / "tmp_analysis/focus_blur.json"
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
