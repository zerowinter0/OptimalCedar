"""What representation does each operator actually receive in each order?

The declared SimCLRv2 order runs ``to_float`` right after the reader, so every
image transform below it is handed a float32 tensor.  A reordered plan that
moves ``to_float`` to the end hands the same transforms the reader's raw uint8
tensor instead.  This script walks a real imagenette2 record through both
orders with the pipeline's own callables and prints, per step, the payload
representation, its serialized size and the wall time of the call.

Usage (inside the container):
  python -u tmp_analysis/reorder_representation_probe.py
"""

import pickle
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    SIMCLRV2_DATASET,
    build_feature,
)

torch.set_num_threads(1)

# The two orders as pipe-id chains (ids follow the feature definition).
DECLARED = [7, 6, 5, 4, 3, 2, 1, 0]
PICO = [6, 3, 4, 5, 2, 7, 1, 0]
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
    import PIL.Image

    if isinstance(value, torch.Tensor):
        return (
            f"torch.Tensor dtype={value.dtype} shape={tuple(value.shape)} "
            f"contig={value.is_contiguous()}"
        )
    if isinstance(value, PIL.Image.Image):
        return f"PIL.Image mode={value.mode} size={value.size}"
    return type(value).__name__


def measure(fn, value, calls=8, repeats=5):
    rates = []
    for _ in range(repeats):
        duration = 0.0
        for _ in range(calls):
            start = time.perf_counter()
            value = fn(value)
            duration += time.perf_counter() - start
        rates.append(calls / max(duration, 1e-9))
    return value, 1000.0 / statistics.median(rates)


def run_order(fns, record, order, seed, names=NAMES):
    print(f"\n=== order {order} ===")
    value = record
    total = 0.0
    for step, pipe_id in enumerate(order):
        torch.manual_seed(seed + step)
        random.seed(seed + step)
        np.random.seed((seed + step) % (2**32 - 1))
        size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        if pipe_id not in fns:
            print(
                f"  {names[pipe_id]:12s} in={describe(value):52s} "
                f"bytes={size:>9d} (no single-value callable; skipped)"
            )
            continue
        fn = fns[pipe_id]
        before = describe(value)
        value, cost = measure(fn, value)
        total += cost
        print(
            f"  {names[pipe_id]:12s} in={before:52s} bytes={size:>9d} "
            f"cost={cost:8.4f} ms -> {describe(value)}"
        )
    print(f"  total in-process service = {total:.3f} ms/record")
    return total


def main() -> int:
    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    fns = {p_id: pipes[p_id].get_fused_callable() for p_id in range(1, 8)}

    from cedar.pipes.io import read_image
    from torchvision.io import ImageReadMode

    files = sorted(SIMCLRV2_DATASET.glob("**/*.JPEG"))
    if not files:
        files = sorted(SIMCLRV2_DATASET.glob("**/*.*"))
    print(f"dataset records: {len(files)} (first: {files[0].name})")
    record = read_image(str(files[0]), mode=ImageReadMode.RGB)
    print("reader output:", describe(record), "bytes=", len(
        pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    ))

    run_order(fns, record, DECLARED, seed=1234)
    run_order(fns, record, PICO, seed=1234)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
