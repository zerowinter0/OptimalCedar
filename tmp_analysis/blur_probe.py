"""Micro-benchmark the B/H/J callables on the captured block inputs."""
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    BLOCK_NAMES,
    BLOCK_ORDER,
    build_feature,
    record_seed,
)

import random  # noqa: E402
import numpy as np  # noqa: E402

torch.set_num_threads(1)
feature = build_feature(batch_size=4)
pipes = feature.logical_pipes
blob = torch.load(ROOT / "outputs/simclrv2_fusion_offload_mechanism_smoke/inputs/block_inputs.pt")
pool = blob["records"]

fns = {name: pipes[p_id].get_fused_callable()
       for p_id, name in BLOCK_NAMES.items()}

for name in ("B_blur", "H_flip", "J_jitter"):
    fn = fns[name]
    times = []
    for index in range(300):
        data = pool[index % len(pool)]
        seed = record_seed(index, name)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        started = time.perf_counter_ns()
        fn(data)
        times.append((time.perf_counter_ns() - started) / 1e6)
    times_sorted = sorted(times)
    print(
        "%-9s mean=%.3f median=%.3f p90=%.3f p99=%.3f max=%.3f ms" % (
            name, statistics.fmean(times), statistics.median(times),
            times_sorted[int(0.9 * len(times))],
            times_sorted[int(0.99 * len(times))], max(times),
        )
    )

fn = fns["B_blur"]
data = pool[0]
times = []
for index in range(200):
    started = time.perf_counter_ns()
    fn(data)
    times.append((time.perf_counter_ns() - started) / 1e6)
print("B_blur fixed input, no reseed: mean=%.3f median=%.3f max=%.3f" % (
    statistics.fmean(times), statistics.median(times), max(times)))

t = data.clone()
print("input dtype/contig:", t.dtype, t.is_contiguous(), t.shape)
times = []
for index in range(200):
    started = time.perf_counter_ns()
    out = torch.nn.functional.pad(t, (0, 0))
    times.append((time.perf_counter_ns() - started) / 1e6)
print("clone baseline: mean=%.4f max=%.4f" % (statistics.fmean(times), max(times)))
