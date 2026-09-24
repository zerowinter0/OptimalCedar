"""Does moving ``to_float`` preserve the augmentation, or change its values?

``to_float`` is a plain ``x.to(torch.float32)``: it does not rescale to [0, 1].
torchvision treats float images as unit-range, so the *same* transform can do
different things to a float32 payload (0..255) and to a uint8 payload (0..255).
This probe measures that difference per operator on one real record.

Usage (inside the container):
  python -u tmp_analysis/dtype_semantics_probe.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.utils.threading import limit_native_threadpools  # noqa: E402

_THREAD_LIMITER = limit_native_threadpools(1)  # noqa: F841
torch.set_num_threads(1)

from block_mechanism_common import (  # noqa: E402
    OP_SEED_OFFSET,
    SIMCLRV2_DATASET,
    build_feature,
)


def seed_for(record: int, stage: str) -> int:
    return (
        20260924 + record * 1_000_003 + OP_SEED_OFFSET.get(stage, 7)
    ) % (2**31 - 1)


def main() -> int:
    from cedar.pipes.io import read_image
    from torchvision.io import ImageReadMode

    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    jitter = pipes[4].get_fused_callable()
    crop = pipes[6].get_fused_callable()
    files = sorted(SIMCLRV2_DATASET.glob("**/*.JPEG"))[:4]
    rows = []
    for index, path in enumerate(files):
        image = read_image(str(path), mode=ImageReadMode.RGB)
        random.seed(seed_for(index, "crop"))
        torch.manual_seed(seed_for(index, "crop"))
        crop_u8 = crop(image)
        crop_f32 = crop(image.to(torch.float32))
        random.seed(seed_for(index, "jitter"))
        torch.manual_seed(seed_for(index, "jitter"))
        jit_u8 = jitter(crop_u8)
        random.seed(seed_for(index, "jitter"))
        torch.manual_seed(seed_for(index, "jitter"))
        jit_f32 = jitter(crop_f32)
        rows.append(
            {
                "record": path.name,
                "u8_input_range": [float(crop_u8.min()), float(crop_u8.max())],
                "f32_input_range": [float(crop_f32.min()), float(crop_f32.max())],
                "u8_output_range": [float(jit_u8.min()), float(jit_u8.max())],
                "f32_output_range": [float(jit_f32.min()), float(jit_f32.max())],
                "max_abs_diff": float(
                    (jit_u8.to(torch.float32) - jit_f32).abs().max()
                ),
                "mean_abs_diff": float(
                    (jit_u8.to(torch.float32) - jit_f32).abs().mean()
                ),
            }
        )
        print(
            f"{path.name[:28]:30s} f32 out range "
            f"[{rows[-1]['f32_output_range'][0]:.2f},"
            f"{rows[-1]['f32_output_range'][1]:.2f}]  u8 out range "
            f"[{rows[-1]['u8_output_range'][0]:.0f},"
            f"{rows[-1]['u8_output_range'][1]:.0f}]  mean|Δ|="
            f"{rows[-1]['mean_abs_diff']:.2f}"
        )
    target = (
        ROOT / "outputs/affine_reorder_diagnosis_20260924/dtype_semantics.json"
    )
    target.write_text(json.dumps(rows, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
