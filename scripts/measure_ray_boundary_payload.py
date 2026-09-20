"""Isolate the Ray boundary cost for the fused-stage payload sizes.

An idle remote actor echoes the batch, so the measurement contains the client
serialize+submit, the cross-host round trip (actor-side deserialization,
result serialization, transfer) and the client-side ray.get/deserialization,
with no operator compute and no queueing.

Usage (inside the container):
  python -u scripts/measure_ray_boundary_payload.py --ray-ip 172.23.166.105:6379
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-ip", default="172.23.166.105:6379")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--payload-bytes",
        type=int,
        nargs="+",
        default=[238562, 714852],
        help="per-sample serialized sizes seen at the fused stage (B)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import numpy as np
    import ray
    from ray import cloudpickle

    ray.init(args.ray_ip, ignore_reinit_error=True)

    placement = os.environ.get("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
    fraction = float(
        os.environ.get("CEDAR_RAY_PLACEMENT_RESOURCE_FRACTION", "0.001")
    )

    @ray.remote(num_cpus=1, resources={placement: fraction})
    class Echo:
        def process(self, batch):
            return batch

        def process_profiled(self, batch):
            started = time.perf_counter_ns()
            result = batch
            return result, time.perf_counter_ns() - started

        def where(self):
            return ray.util.get_node_ip_address()

    actor = Echo.remote()
    node_ip = ray.get(actor.where.remote())
    rows = []
    for payload_bytes in args.payload_bytes:
        n_float = max(1, payload_bytes // 4)
        sample = np.zeros(n_float, dtype=np.float32)
        batch = [sample.copy() for _ in range(args.batch_size)]
        # client-side marshal costs for one sample
        times = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            blob = cloudpickle.dumps(sample, protocol=5)
            middle = time.perf_counter_ns()
            cloudpickle.loads(blob)
            times.append((middle - started, time.perf_counter_ns() - middle))
        serialize_ms = statistics.median(t / 1e6 for t, _ in times)
        deserialize_ms = statistics.median(t / 1e6 for _, t in times)

        submit_ms, rtt_ms, compute_ms = [], [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter_ns()
            future = actor.process_profiled.remote(batch)
            t1 = time.perf_counter_ns()
            result, actor_ns = ray.get(future)
            t2 = time.perf_counter_ns()
            assert len(result) == args.batch_size
            submit_ms.append((t1 - t0) / 1e6 / args.batch_size)
            rtt_ms.append((t2 - t0) / 1e6 / args.batch_size)
            compute_ms.append(actor_ns / 1e6 / args.batch_size)
        rows.append(
            {
                "payload_bytes_per_sample": payload_bytes,
                "batch_size": args.batch_size,
                "actor_node_ip": node_ip,
                "client_serialize_ms_per_sample": round(serialize_ms, 4),
                "client_deserialize_ms_per_sample": round(deserialize_ms, 4),
                "submit_ms_per_sample": round(statistics.median(submit_ms), 4),
                "rtt_ms_per_sample": round(statistics.median(rtt_ms), 4),
                "actor_compute_ms_per_sample": round(
                    statistics.median(compute_ms), 5
                ),
                "return_path_ms_per_sample": round(
                    statistics.median(rtt_ms)
                    - statistics.median(submit_ms)
                    - statistics.median(compute_ms),
                    4,
                ),
                "repeats": args.repeats,
            }
        )
    print(f"actor node: {node_ip} (driver {ray.util.get_node_ip_address()})")
    header = (
        f"{'payload/sample':>15}{'serialize':>11}{'submit':>9}"
        f"{'rtt':>9}{'actor compute':>15}{'return path':>13}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['payload_bytes_per_sample']:>15}"
            f"{row['client_serialize_ms_per_sample']:>11.3f}"
            f"{row['submit_ms_per_sample']:>9.3f}"
            f"{row['rtt_ms_per_sample']:>9.3f}"
            f"{row['actor_compute_ms_per_sample']:>15.4f}"
            f"{row['return_path_ms_per_sample']:>13.3f}"
        )
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.output}")
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
