"""Why do two payloads with the same class and element count differ 4x?

Interleaved, single process, two independent blocks: the real captured
payloads of each plan next to synthetic payloads rescaled to the same
geometry.  Nothing else runs in this process, so a stable ranking between the
block-1 and block-2 estimates is what makes a difference trustworthy.

Usage (inside the container):
  python -u tmp_analysis/blur_geometry_probe.py [rounds]
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
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    feature = build_feature(batch_size=4)
    blur = feature.logical_pipes[2].get_fused_callable()

    crop_input = load("declared", 6)          # float32 3ch 375x500
    declared_blur = load("declared", 2)       # float32 1ch 244x244
    pico_blur = load("pico", 2)               # uint8   1ch 244x244
    cedar_blur = load("cedar", 2)             # uint8   1ch 244x244

    def to_u8(value):
        return value.mul(255.0).round().clamp(0, 255).to(torch.uint8).contiguous()

    def to_f32(value):
        return value.to(torch.float32).contiguous()

    gray = crop_input.mean(dim=0, keepdim=True)
    variants = {
        "real pico   u8 1ch 244x244": pico_blur,
        "real cedar  u8 1ch 244x244": cedar_blur,
        "real decl   f32 1ch 244x244": declared_blur,
        "synthetic   f32 1ch 244x244": to_f32(pico_blur),
        "synthetic   u8 1ch 244x244": to_u8(declared_blur),
        "synthetic   u8 1ch 122x122": resize(pico_blur, (122, 122)),
        "synthetic   u8 1ch 187x250": resize(gray, (250, 187)).round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous(),
        "synthetic   u8 1ch 375x500": to_u8(gray),
        "synthetic   f32 1ch 122x122": resize(declared_blur, (122, 122)),
        "synthetic   f32 1ch 187x250": resize(declared_blur, (250, 187)),
        "synthetic   f32 1ch 375x500": resize(declared_blur, (500, 375)),
        "synthetic   u8 3ch 122x122": resize(to_u8(crop_input), (122, 122)),
        "synthetic   f32 3ch 122x122": resize(crop_input, (122, 122)),
    }
    for name, value in variants.items():
        for _ in range(3):
            blur(value)

    was_enabled = gc.isenabled()
    gc.disable()
    samples = {name: [] for name in variants}
    try:
        for block in range(2):
            for _ in range(rounds):
                for name, value in variants.items():
                    calls, duration = 3, 0.0
                    for _ in range(calls):
                        start = time.perf_counter()
                        blur(value)
                        duration += time.perf_counter() - start
                    samples[name].append(
                        {"block": block, "ms": 1000.0 * duration / calls}
                    )
    finally:
        if was_enabled:
            gc.enable()
            gc.collect()

    rows = []
    print(f"{'payload':30s} {'elements':>9s} {'bytes':>9s} "
          f"{'block1 p10':>11s} {'block2 p10':>11s} {'all p10':>9s} {'median':>9s}")
    for name, value in variants.items():
        values = samples[name]
        b1 = sorted(row["ms"] for row in values if row["block"] == 0)
        b2 = sorted(row["ms"] for row in values if row["block"] == 1)
        allv = sorted(row["ms"] for row in values)
        p10 = allv[max(0, int(0.1 * (len(allv) - 1)))]
        entry = {
            "payload": name,
            "elements": int(value.numel()),
            "bytes": len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)),
            "block1_p10": b1[max(0, int(0.1 * (len(b1) - 1)))],
            "block2_p10": b2[max(0, int(0.1 * (len(b2) - 1)))],
            "p10": p10,
            "median": statistics.median(allv),
        }
        rows.append(entry)
        print(
            f"{name:30s} {entry['elements']:9d} {entry['bytes']:9d} "
            f"{entry['block1_p10']:11.4f} {entry['block2_p10']:11.4f} "
            f"{entry['p10']:9.4f} {entry['median']:9.4f}"
        )
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/blur_geometry.json"
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
