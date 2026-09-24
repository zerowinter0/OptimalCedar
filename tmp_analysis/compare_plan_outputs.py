"""Do the semantics-preserving plans produce comparable augmented records?

``a_f0`` runs the cast first (float32 pipeline) and ``a_f5`` runs it last
(uint8 pipeline).  Both apply crop -> flip -> jitter -> grayscale -> blur in
the same order with the same per-operator seed, so the only difference is the
element type each transform computes in.  This script quantifies that
difference per stage and on the final tensor; it does not claim bitwise
equality.

Usage (inside the container):
  python -u tmp_analysis/compare_plan_outputs.py [records]
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

STAGES = {
    "to_float": 7,
    "crop": 6,
    "flip": 5,
    "jitter": 4,
    "grayscale": 3,
    "blur": 2,
    "normalize": 1,
}
F32_ORDER = ["to_float", "crop", "flip", "jitter", "grayscale", "blur"]
U8_ORDER = ["crop", "flip", "jitter", "grayscale", "blur", "to_float"]


def seed_for(record: int, stage: str) -> int:
    offset = OP_SEED_OFFSET.get(stage, 7) + sum(
        ord(char) for char in stage
    )
    return (20260924 + record * 1_000_003 + offset) % (2**31 - 1)


def run(record, value, order, fns):
    trace = {}
    for stage in order:
        seed = seed_for(record, stage)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        value = fns[stage](value)
        trace[stage] = value
    return value, trace


def main() -> int:
    records = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    from cedar.pipes.io import read_image
    from torchvision.io import ImageReadMode

    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    fns = {name: pipes[p_id].get_fused_callable() for name, p_id in STAGES.items()}
    files = sorted(SIMCLRV2_DATASET.glob("**/*.JPEG"))[:records]
    rows = []
    for index, path in enumerate(files):
        reader = read_image(str(path), mode=ImageReadMode.RGB)
        f32_final, f32_trace = run(index, reader, F32_ORDER, fns)
        u8_final, u8_trace = run(index, reader.clone(), U8_ORDER, fns)
        record = {"record": path.name}
        for stage in ("crop", "flip", "jitter", "grayscale", "blur"):
            left = f32_trace[stage].to(torch.float32)
            right = u8_trace[stage].to(torch.float32)
            if left.shape != right.shape:
                record[stage] = {"shape_mismatch": [list(left.shape), list(right.shape)]}
                continue
            diff = (left - right).abs()
            record[stage] = {
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
            }
        rows.append(record)
    summary = {}
    for stage in ("crop", "flip", "jitter", "grayscale", "blur"):
        values = [
            row[stage]["max_abs"] for row in rows if "max_abs" in row.get(stage, {})
        ]
        means = [
            row[stage]["mean_abs"] for row in rows if "mean_abs" in row.get(stage, {})
        ]
        if values:
            summary[stage] = {
                "records": len(values),
                "max_abs_worst": max(values),
                "max_abs_mean": float(np.mean(values)),
                "mean_abs_mean": float(np.mean(means)),
            }
    for stage, entry in summary.items():
        print(
            f"{stage:10s} records={entry['records']:3d} "
            f"max|Δ| worst={entry['max_abs_worst']:8.4f} "
            f"mean(max|Δ|)={entry['max_abs_mean']:8.4f} "
            f"mean over pixels={entry['mean_abs_mean']:8.5f}"
        )
    target = (
        ROOT / "outputs/affine_reorder_diagnosis_20260924/plan_output_differences.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
