"""Is the fused actor's lower member time a pacing effect?

R-U calls one member 4x, then the batch leaves the actor (serialise + handoff)
before the next member runs; R-F calls blur->flip->jitter per record back to
back.  This probe runs the *same* blur callable in one process in two pacings:

  continuous  blur(4 records) then flip(4) then jitter(4), repeated
  bursty      blur(4 records), then sleep for the same duration the other two
              members take in the continuous phase, repeated

If blur is slower in bursty pacing, part of the R-U/R-F "compute" difference is
the pacing of the caller (idle gaps, frequency ramp, allocator state), not the
fusion itself.  Reported as a separate diagnostic, never merged into the main
table.

Usage:
  python -u scripts/block_burst_idle_probe.py --run-dir outputs/<run> \
      --repeats 150 --out <run>/burst_idle_probe.json
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    BLOCK_NAMES,
    BLOCK_ORDER,
    build_feature,
    record_seed,
)


def _run(fn, data, key, name) -> float:
    seed = record_seed(key, name)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    started = time.perf_counter_ns()
    fn(data)
    return (time.perf_counter_ns() - started) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--remote", action="store_true",
                        help="run inside a remote Ray actor instead of locally")
    parser.add_argument(
        "--gap-ms", type=float, default=80.0,
        help="idle gap inserted between blur bursts in the bursty phase "
             "(the measured handoff per batch is ~80 ms for R-U, ~30 ms for R-F)",
    )
    parser.add_argument("--ray-ip", default="172.23.166.105:6379")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)

    payload = torch.load(args.run_dir / "inputs" / "block_inputs.pt")
    pool = payload["records"]
    feature = build_feature(batch_size=args.batch_size)
    pipes = feature.logical_pipes
    missing = None
    if args.remote:
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

        @ray.remote(num_cpus=0)
        class PacingActor(RayActor):
            def run(self, records, fns, repeats, cpu, gap_ms):
                if cpu is not None:
                    try:
                        os.sched_setaffinity(0, {int(cpu)})
                    except (AttributeError, OSError):
                        pass
                torch.set_num_threads(1)
                sizes = len(records)
                result = {
                    "continuous_blur_ms": [],
                    "bursty_blur_ms": [],
                    "other_members_ms": [],
                    "per_cycle": [],
                }
                # Warm both the process and the callable.
                for index in range(min(20, sizes)):
                    for key, item in enumerate(records[index]):
                        PacingActor._time(fns[0], item, key, "B_blur")
                burst = 4
                for cycle in range(repeats):
                    continuous = []
                    for burst_index in range(burst):
                        batch = records[(cycle * burst + burst_index) % sizes]
                        for key, item in enumerate(batch):
                            continuous.append(
                                PacingActor._time(fns[0], item, key, "B_blur")
                            )
                    gapped = []
                    for burst_index in range(burst):
                        time.sleep(gap_ms / 1000.0)
                        batch = records[(cycle * burst + burst_index) % sizes]
                        for key, item in enumerate(batch):
                            gapped.append(
                                PacingActor._time(fns[0], item, key, "B_blur")
                            )
                    result["continuous_blur_ms"].extend(continuous)
                    result["bursty_blur_ms"].extend(gapped)
                    result["per_cycle"].append(
                        {
                            "cycle": cycle,
                            "continuous_median": statistics.median(continuous),
                            "gapped_median": statistics.median(gapped),
                        }
                    )
                return result

            @staticmethod
            def _time(fn, data, key, name):
                seed = (20260923 + key * 1_000_003 + {"B_blur": 11, "H_flip": 23,
                                                      "J_jitter": 37}[name]) % (2**31 - 1)
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                started = time.perf_counter_ns()
                fn(data)
                return (time.perf_counter_ns() - started) / 1e6

        # Shape the pool as (n_batches, batch_size, ...) so the actor can index it.
        batches = []
        fns = [pipes[p_id].get_fused_callable() for p_id in BLOCK_ORDER]
        for start in range(0, len(pool) - args.batch_size + 1, args.batch_size):
            batches.append(pool[start:start + args.batch_size])
        actor = PacingActor.options(**get_ray_actor_options(0.0)).remote(
            "pacing_probe"
        )
        result = ray.get(
            actor.run.remote(
                batches, fns, args.repeats, args.cpu, args.gap_ms
            )
        )
        ray.shutdown()
    else:
        if args.cpu is not None:
            try:
                os.sched_setaffinity(0, {args.cpu})
            except (AttributeError, OSError):
                pass
        fns = [pipes[p_id].get_fused_callable() for p_id in BLOCK_ORDER]
        batches = [
            pool[start:start + args.batch_size]
            for start in range(0, len(pool) - args.batch_size + 1, args.batch_size)
        ]
        result = {"continuous_blur_ms": [], "bursty_blur_ms": [],
                  "other_members_ms": []}
        for index in range(min(20, args.repeats)):
            batch = batches[index % len(batches)]
            for offset, item in enumerate(batch):
                _run(fns[0], item, index * 4 + offset, "B_blur")
        for index in range(args.repeats):
            batch = batches[index % len(batches)]
            for offset, item in enumerate(batch):
                key = index * 4 + offset
                result["continuous_blur_ms"].append(
                    _run(fns[0], item, key, "B_blur")
                )
            for offset, item in enumerate(batch):
                key = index * 4 + offset
                result["other_members_ms"].append(
                    _run(fns[1], item, key, "H_flip")
                )
                result["other_members_ms"].append(
                    _run(fns[2], item, key, "J_jitter")
                )
        for index in range(args.repeats):
            batch = batches[index % len(batches)]
            for offset, item in enumerate(batch):
                key = index * 4 + offset
                result["bursty_blur_ms"].append(
                    _run(fns[0], item, key, "B_blur")
                )
            time.sleep(args.gap_ms / 1000.0)

    summary = {
        "where": "remote_actor" if args.remote else "driver_process",
        "cpu_pin": args.cpu,
        "repeats": args.repeats,
        "gap_ms": args.gap_ms,
        "continuous_blur_ms_mean": statistics.fmean(result["continuous_blur_ms"]),
        "continuous_blur_ms_median": statistics.median(result["continuous_blur_ms"]),
        "bursty_blur_ms_mean": statistics.fmean(result["bursty_blur_ms"]),
        "bursty_blur_ms_median": statistics.median(result["bursty_blur_ms"]),
        "bursty_over_continuous_mean": (
            statistics.fmean(result["bursty_blur_ms"])
            / statistics.fmean(result["continuous_blur_ms"])
        ),
        "other_members_ms_mean": (
            statistics.fmean(result["other_members_ms"])
            if result["other_members_ms"] else None
        ),
        "per_cycle_median_of_medians": {
            "continuous": statistics.median(
                [c["continuous_median"] for c in result.get("per_cycle", [])]
            ) if result.get("per_cycle") else None,
            "gapped": statistics.median(
                [c["gapped_median"] for c in result.get("per_cycle", [])]
            ) if result.get("per_cycle") else None,
        },
        "raw": {
            "continuous_blur_ms": result["continuous_blur_ms"][:50],
            "bursty_blur_ms": result["bursty_blur_ms"][:50],
            "per_cycle": result.get("per_cycle", [])[:20],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
