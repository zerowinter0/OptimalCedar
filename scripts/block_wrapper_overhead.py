"""Standalone cost of the measurement wrapper (reported separately).

Three nested variants of the same callable on the same inputs:

  raw        call the operator
  seeded     + per-(record, operator) RNG seeding
  instrument + seeding + wall/cpu clocks + event append + input describe

The deltas are *not* subtracted from the experiment; they only bound how much
of the reported per-call time the wrapper itself can contribute.

Usage:
  python -u scripts/block_wrapper_overhead.py --run-dir outputs/<run>
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


def _describe(data: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": type(data).__name__}
    if isinstance(data, torch.Tensor):
        info.update(
            {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "contiguous": bool(data.is_contiguous()),
                "strides": list(data.stride()),
            }
        )
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=300)
    parser.add_argument("--cpu", type=int, default=None)
    args = parser.parse_args()
    if args.cpu is not None:
        try:
            import os

            os.sched_setaffinity(0, {args.cpu})
        except (AttributeError, OSError):
            pass
    torch.set_num_threads(1)

    blob = torch.load(args.run_dir / "inputs" / "block_inputs.pt")
    pool = blob["records"]
    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes

    results: Dict[str, Any] = {"calls": args.calls, "operators": {}}
    for p_id in BLOCK_ORDER:
        name = BLOCK_NAMES[p_id]
        fn = pipes[p_id].get_fused_callable()
        # Interleave the three modes call by call: the operator itself is
        # bimodal, so paired differences are the only meaningful estimate.
        timings = {"raw": [], "seeded": [], "instrument": []}
        events: List[Any] = []
        for index in range(args.calls):
            data = pool[index % len(pool)]
            key = index
            started = time.perf_counter_ns()
            fn(data)
            timings["raw"].append((time.perf_counter_ns() - started) / 1e6)

            seed = record_seed(key, name)
            started = time.perf_counter_ns()
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            fn(data)
            timings["seeded"].append((time.perf_counter_ns() - started) / 1e6)

            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            wall_started = time.perf_counter_ns()
            cpu_started = time.process_time_ns()
            fn(data)
            wall_ns = time.perf_counter_ns() - wall_started
            cpu_ns = time.process_time_ns() - cpu_started
            events.append((key, wall_ns, cpu_ns, _describe(data)))
            timings["instrument"].append(wall_ns / 1e6)
        paired_seed = [
            s - r for r, s in zip(timings["raw"], timings["seeded"])
        ]
        paired_instrument = [
            i - r for r, i in zip(timings["raw"], timings["instrument"])
        ]
        results["operators"][name] = {
            mode: {
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p90_ms": sorted(values)[int(0.9 * len(values))],
            }
            for mode, values in timings.items()
        }
        results["operators"][name]["paired_seeding_ms_per_call"] = {
            "mean": statistics.fmean(paired_seed),
            "median": statistics.median(paired_seed),
            "stdev": statistics.stdev(paired_seed),
        }
        results["operators"][name]["paired_instrument_ms_per_call"] = {
            "mean": statistics.fmean(paired_instrument),
            "median": statistics.median(paired_instrument),
            "stdev": statistics.stdev(paired_instrument),
        }
        results["operators"][name]["seeding_ms_per_call"] = (
            results["operators"][name]["seeded"]["mean_ms"]
            - results["operators"][name]["raw"]["mean_ms"]
        )
        results["operators"][name]["instrument_minus_raw_ms_per_call"] = (
            results["operators"][name]["instrument"]["mean_ms"]
            - results["operators"][name]["raw"]["mean_ms"]
        )
        results["operators"][name]["events_recorded"] = len(events)
    path = args.run_dir / "wrapper_overhead.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
