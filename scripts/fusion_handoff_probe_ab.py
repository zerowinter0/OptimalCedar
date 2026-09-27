"""§3.3 diagnostic: why does the paired probe over-measure the stage handoff?

One long-lived actor per stage runs three modes in the *same* session, with the
same submit batch size the deployed plan uses:

  real    the operator actually runs (freshly allocated outputs, instrumented)
  pooled  no operator compute; returns one of the pre-built real output objects
          again and again (exactly what the profiler's paired probe does)
  fresh   no operator compute; returns a freshly allocated copy of that object

The client times the full serial round trip for each mode.  Comparing
``pooled`` with ``real - compute`` isolates the probe's framing cost, and
``fresh - pooled`` isolates the shared-buffer effect.  A second pooled pass with
the profiler's short warmup (2 batches) isolates the warm-up depth.

Usage (inside the container):
  python -u scripts/fusion_handoff_probe_ab.py --run-dir DIR \
      --inputs outputs/fusion_discount_20260923/expB
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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

import ray  # noqa: E402
from block_mechanism_common import build_feature, payload_bytes  # noqa: E402
from block_service_harness import _pin_cpu, _limit_threads  # noqa: E402
from cedar.pipes.ray_variant import (  # noqa: E402
    configure_remote_ray_experiment,
    get_ray_actor_options,
)
from fusion_cost_split_harness import BLOCK  # noqa: E402


def _actor_cls():
    @ray.remote(num_cpus=0)
    class StageAbActor:
        def __init__(self, name: str, fn: Any, seed_base: int, seed_offset: int):
            from cedar.utils.threading import limit_native_threadpools

            self._limiter = limit_native_threadpools(1)
            self.name = name
            self.fn = fn
            self.seed_base = seed_base
            self.seed_offset = seed_offset
            self.mode = "real"
            self.pool: List[Any] = []
            self.next_id = 0
            self.timings: List[Tuple[int, float]] = []
            self.record_timings = True

        def set_mode(self, mode: str):
            self.mode = mode
            return True

        def seed_pool(self, first_batch: List[Any]):
            """Build the canned response pool once, outside any timed window."""
            self.pool = [self.fn(item) for item in first_batch]
            return len(self.pool)

        def process(self, batch: Any) -> Any:
            out = []
            for item in batch:
                key = self.next_id
                self.next_id += 1
                if self.mode == "real":
                    seed = (
                        self.seed_base + key * 1_000_003 + self.seed_offset
                    ) % (2**31 - 1)
                    torch.manual_seed(seed)
                    if self.record_timings:
                        started = time.perf_counter_ns()
                        value = self.fn(item)
                        self.timings.append((key, time.perf_counter_ns() - started))
                    else:
                        value = self.fn(item)
                elif self.mode == "pooled":
                    value = self.pool[key % len(self.pool)]
                else:  # fresh
                    value = self.pool[key % len(self.pool)].clone()
                out.append(value)
            return out

        def reset_timings(self):
            self.timings.clear()
            self.next_id = 0
            return True

        def take_timings(self):
            return list(self.timings)

        def set_record_timings(self, enabled: bool):
            self.record_timings = bool(enabled)
            return True

        def set_affinity(self, cpu: int):
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except (AttributeError, OSError):
                return False
            return True

        def location(self):
            return {
                "ip": ray.util.get_node_ip_address(),
                "node_id": str(ray.get_runtime_context().get_node_id()),
            }

    return StageAbActor


def _run(actor, batch: List[Any], timeout: float = 240.0) -> float:
    started = time.perf_counter()
    future = actor.process.remote(batch)
    result = ray.get(future)
    elapsed = time.perf_counter() - started
    if len(result) != len(batch):
        raise RuntimeError("stage A/B actor returned the wrong batch size")
    return elapsed * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--inputs", default="outputs/fusion_discount_20260923/expB")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--submit-batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--short-warmup", type=int, default=2)
    parser.add_argument("--local-cpu", type=int, default=12)
    parser.add_argument("--remote-cpu", type=int, default=8)
    parser.add_argument("--ray-ip", default="172.23.166.105:6379")
    args = parser.parse_args()

    _limit_threads()
    _pin_cpu(args.local_cpu)
    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True, logging_level="ERROR")
    actor_cls = _actor_cls()
    options = get_ray_actor_options(0.0)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pool = list(
        torch.load(Path(args.inputs) / "inputs" / "block_inputs.pt")["records"]
    )
    feature = build_feature(batch_size=args.batch_size)
    callables = {
        name: feature.logical_pipes[p_id].get_fused_callable()
        for name, p_id in BLOCK
    }

    # Build each stage's real input pool by running the preceding operators
    # locally (untimed, outside any measurement window): the isolated probe
    # must see the same payload distribution as the chain.
    stage_inputs: Dict[str, List[Any]] = {}
    current: List[Any] = list(pool)
    for op_name, _p_id in BLOCK:
        stage_inputs[op_name] = list(current)
        fn = callables[op_name]
        current = [fn(item) for item in current]

    rows: List[Dict[str, Any]] = []
    for op_name, _p_id in BLOCK:
        inputs = stage_inputs[op_name]
        actor = actor_cls.options(**options).remote(
            op_name, callables[op_name], 20260923, 71
        )
        try:
            ray.get(actor.set_affinity.remote(args.remote_cpu))
            location = ray.get(actor.location.remote())
            ray.get(
                actor.seed_pool.remote(
                    [inputs[i % len(inputs)] for i in range(args.batch_size)]
                )
            )

            def measure(mode: str, warmup: int) -> Dict[str, float]:
                ray.get(actor.set_mode.remote(mode))
                ray.get(actor.reset_timings.remote())
                for index in range(max(1, warmup)):
                    batch = [
                        inputs[
                            (index * args.submit_batch_size + offset) % len(inputs)
                        ]
                        for offset in range(args.submit_batch_size)
                    ]
                    _run(actor, batch)
                durations: List[float] = []
                for index in range(args.batches):
                    batch = [
                        inputs[
                            ((warmup + index) * args.submit_batch_size + offset)
                            % len(inputs)
                        ]
                        for offset in range(args.submit_batch_size)
                    ]
                    durations.append(_run(actor, batch))
                per_sample = [
                    value / args.submit_batch_size for value in durations
                ]
                compute = []
                for _key, wall_ns in ray.get(actor.take_timings.remote()):
                    compute.append(wall_ns / 1e6 / args.submit_batch_size)
                return {
                    "service_ms_per_record": statistics.fmean(per_sample),
                    "service_stdev_ms_per_record": statistics.pstdev(per_sample)
                    if len(per_sample) > 1
                    else 0.0,
                    "member_compute_ms_per_record": (
                        statistics.fmean(compute) if compute else float("nan")
                    ),
                    "batches": len(per_sample),
                }

            real = measure("real", args.warmup)
            pooled_long = measure("pooled", args.warmup)
            fresh = measure("fresh", args.warmup)
            pooled_short = measure("pooled", args.short_warmup)
            for condition, values in (
                ("real_operator", real),
                ("pooled_response_long_warmup", pooled_long),
                ("fresh_response_long_warmup", fresh),
                ("pooled_response_short_warmup", pooled_short),
            ):
                rows.append(
                    {
                        "operator": op_name,
                        "condition": condition,
                        "submit_batch_size": args.submit_batch_size,
                        "actor_ip": location["ip"],
                        **values,
                        "real_minus_compute_ms_per_record": (
                            real["service_ms_per_record"]
                            - real["member_compute_ms_per_record"]
                            if condition == "real_operator"
                            else ""
                        ),
                    }
                )
            print(
                f"{op_name}: real={real['service_ms_per_record']:.2f} "
                f"(C={real['member_compute_ms_per_record']:.2f}) "
                f"pooled={pooled_long['service_ms_per_record']:.2f} "
                f"fresh={fresh['service_ms_per_record']:.2f} "
                f"pooled_short={pooled_short['service_ms_per_record']:.2f}",
                flush=True,
            )
        finally:
            try:
                ray.kill(actor)
            except Exception:  # noqa: BLE001
                pass
    fields = list(rows[0].keys())
    with (run_dir / "probe_ab.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (run_dir / "probe_ab.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {run_dir}/probe_ab.csv rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
