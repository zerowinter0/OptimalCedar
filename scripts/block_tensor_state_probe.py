"""Does a freshly-materialised input tensor make the blur slower?

Within one remote actor process, alternate, per burst of 4 records:

  warm   blur on the tensor shipped to the actor once
  cold   blur on a fresh copy materialised inside the actor
         (new pages, like a payload that just arrived via deserialisation)

Reported as a diagnostic only; nothing here is merged into the main table.

Usage:
  python -u scripts/block_tensor_state_probe.py --run-dir <run> --repeats 60 \
      --out <run>/tensor_state_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    BLOCK_ORDER,
    build_feature,
    record_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--ray-ip", default="172.23.166.105:6379")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import ray

    os.environ.setdefault("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
    os.environ.setdefault("CEDAR_RAY_REQUIRE_REMOTE", "1")
    from cedar.pipes.ray_variant import (
        configure_remote_ray_experiment,
        get_ray_actor_options,
    )
    from cedar.service import RayActor

    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True)
    payload = torch.load(args.run_dir / "inputs" / "block_inputs.pt")
    pool = payload["records"]
    feature = build_feature(batch_size=4)
    blur = feature.logical_pipes[BLOCK_ORDER[0]].get_fused_callable()
    batches = [
        pool[start:start + 4] for start in range(0, len(pool) - 3, 4)
    ]

    @ray.remote(num_cpus=0)
    class StateActor(RayActor):
        def run(self, batches, blur, repeats, cpu):
            if cpu is not None:
                try:
                    os.sched_setaffinity(0, {int(cpu)})
                except (AttributeError, OSError):
                    pass
            torch.set_num_threads(1)

            def timed(item, key):
                seed = (20260923 + key * 1_000_003 + 11) % (2**31 - 1)
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                started = time.perf_counter_ns()
                blur(item)
                return (time.perf_counter_ns() - started) / 1e6

            for index in range(20):
                for key, item in enumerate(batches[index]):
                    timed(item, key)
            warm, cold = [], []
            for cycle in range(repeats):
                batch = batches[cycle % len(batches)]
                for key, item in enumerate(batch):
                    warm.append(timed(item, key))
                copies = [
                    torch.empty_like(item).copy_(item) for item in batch
                ]
                for key, item in enumerate(copies):
                    cold.append(timed(item, key))
            return {"warm": warm, "cold": cold}

    actor = StateActor.options(**get_ray_actor_options(0.0)).remote(
        "tensor_state_probe"
    )
    result = ray.get(actor.run.remote(batches, blur, args.repeats, args.cpu))
    ray.shutdown()
    summary = {
        "cpu_pin": args.cpu,
        "repeats": args.repeats,
        "warm_ms_mean": statistics.fmean(result["warm"]),
        "warm_ms_median": statistics.median(result["warm"]),
        "cold_ms_mean": statistics.fmean(result["cold"]),
        "cold_ms_median": statistics.median(result["cold"]),
        "cold_over_warm_mean": (
            statistics.fmean(result["cold"]) / statistics.fmean(result["warm"])
        ),
        "cold_over_warm_median": (
            statistics.median(result["cold"]) / statistics.median(result["warm"])
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
