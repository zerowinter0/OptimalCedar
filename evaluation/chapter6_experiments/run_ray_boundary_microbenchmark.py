#!/usr/bin/env python3
"""Measure the incremental cost of an additional Ray stage boundary."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import ray

from cedar.config import CedarContext
from cedar.pipes import MapperPipe, PipeVariantType, RayPipeVariantContext
from cedar.sources import IterSource


def identity(value):
    return value


def run_once(stages, actors, payload_bytes, warmup, samples):
    # Reuse one immutable payload so host memory does not scale with samples.
    payload = np.zeros(payload_bytes, dtype=np.uint8)
    source = IterSource([payload] * (warmup + samples))
    ctx = CedarContext()
    pipe = source.to_pipe()
    pipe.mutate(ctx, PipeVariantType.INPROCESS)
    ray_contexts = []
    for _ in range(stages):
        next_pipe = MapperPipe(pipe, identity)
        variant_ctx = RayPipeVariantContext(
            n_actors=actors,
            max_inflight=max(100, actors * 5),
            max_prefetch=max(100, actors * 5),
            submit_batch_size=1,
        )
        next_pipe.mutate(ctx, PipeVariantType.RAY, variant_ctx)
        ray_contexts.append(variant_ctx)
        pipe = next_pipe

    try:
        started = None
        count = 0
        for idx, _ in enumerate(pipe.pipe_variant):
            if idx + 1 == warmup:
                started = time.perf_counter()
            if idx >= warmup:
                count += 1
        elapsed = time.perf_counter() - started
        if count != samples:
            raise RuntimeError(f"Expected {samples} samples, received {count}")
        return {
            "stages": stages,
            "actors_per_stage": actors,
            "payload_bytes": payload_bytes,
            "samples": count,
            "elapsed_sec": elapsed,
            "throughput_samples_per_sec": count / elapsed,
            "effective_payload_gib_per_sec": (
                count * payload_bytes / elapsed / (1024**3)
            ),
        }
    finally:
        for variant_ctx in reversed(ray_contexts):
            variant_ctx.service.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray_address", default="127.0.0.1:6379")
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--payload_bytes", type=int, default=3358411)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    # Balanced alternating order limits drift from favoring either topology.
    order = [1, 2, 2, 1, 1, 2]
    runs = []
    try:
        for repeat, stages in enumerate(order, start=1):
            result = run_once(
                stages,
                args.actors,
                args.payload_bytes,
                args.warmup,
                args.samples,
            )
            result["repeat"] = repeat
            runs.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        ray.shutdown()

    summary = {}
    for stages in (1, 2):
        values = [
            run["throughput_samples_per_sec"]
            for run in runs
            if run["stages"] == stages
        ]
        summary[str(stages)] = {
            "mean_throughput_samples_per_sec": statistics.mean(values),
            "stddev_throughput_samples_per_sec": statistics.stdev(values),
        }
    one = summary["1"]["mean_throughput_samples_per_sec"]
    two = summary["2"]["mean_throughput_samples_per_sec"]
    summary["two_stage_vs_one_stage_throughput_ratio"] = two / one
    summary["incremental_boundary_time_fraction"] = one / two - 1
    config = vars(args).copy()
    config["output"] = str(config["output"])
    output = {"config": config, "order": order, "runs": runs, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
