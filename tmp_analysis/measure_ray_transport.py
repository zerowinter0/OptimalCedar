"""Measure the cross-host round-trip budget that a Ray offload plan pays.

The modelled cost of a Ray stage is ``driver marshalling + lane service``; the
measurement here answers what the *transport* can actually sustain when every
local worker has its own actors, i.e. whether a plan that ships a payload per
record is limited by the link or by something else.

For a sweep of actor counts we run ``rounds`` echo round trips per actor with
the payload sizes seen by the SimCLRv2 Ray stage:

  identity   : 238 KB in  -> 238 KB out  (the boundary microbenchmark size)
  plan-like  : 714 KB in  -> 238 KB out  (grayscale->crop->...->normalize)

Reported per configuration: records/second, and aggregate MB/s counting both
directions, which is exactly the ``bytes/s`` budget the cost model needs.

Usage (inside the container):
  python -u tmp_analysis/measure_ray_transport.py \
      --actors 1,8,16,32,56 --rounds 20
"""

import argparse
import json
import time
from pathlib import Path

import ray

OUT = Path("/workspace/OptimalCedar/outputs/transport_microbench_20260912")


@ray.remote(num_cpus=1)
class EchoActor:
    def __init__(self, out_bytes: int):
        self.out_bytes = int(out_bytes)
        self._blob = b"\0" * self.out_bytes

    def round_trip(self, payload_ref) -> bytes:
        payload = bytes(payload_ref)
        # Touch the payload so it cannot be optimised away, then return a
        # payload of the requested size (the stage's output item).
        sink = payload[0] if payload else 0
        return self._blob if sink >= 0 else self._blob


def run_configuration(address, actors, rounds, in_bytes, out_bytes, warmup):
    handles = [EchoActor.remote(out_bytes) for _ in range(actors)]
    payload = ray.put(b"\0" * in_bytes)
    for handle in handles[: min(actors, 4)]:
        ray.get([handle.round_trip.remote(payload) for _ in range(warmup)])
    started = time.perf_counter()
    futures = []
    for handle in handles:
        futures.extend(
            handle.round_trip.remote(payload) for _ in range(rounds)
        )
    ray.get(futures)
    elapsed = time.perf_counter() - started
    records = actors * rounds
    bytes_moved = records * (in_bytes + out_bytes)
    return {
        "actors": actors,
        "rounds_per_actor": rounds,
        "records": records,
        "seconds": round(elapsed, 4),
        "records_per_second": round(records / elapsed, 1),
        "aggregate_mb_per_second": round(bytes_moved / elapsed / 1e6, 1),
        "per_record_ms": round(elapsed / records * 1000.0, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="172.23.166.105:6379")
    parser.add_argument("--actors", default="1,8,16,32,56")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--in-bytes", type=int, default=238144)
    parser.add_argument("--out-bytes", type=int, default=238144)
    parser.add_argument(
        "--plan-like",
        action="store_true",
        help="also measure the plan-like 714 KB -> 238 KB round trip",
    )
    args = parser.parse_args()

    ray.init(address=args.address, logging_level="ERROR")
    print("cluster:", ray.cluster_resources(), flush=True)
    actor_counts = [int(token) for token in args.actors.split(",") if token]
    configurations = [(args.in_bytes, args.out_bytes)]
    if args.plan_like:
        configurations.append((714432, 238144))
    results = []
    for in_bytes, out_bytes in configurations:
        print(f"\n=== round trip {in_bytes} B in -> {out_bytes} B out", flush=True)
        for actors in actor_counts:
            row = run_configuration(
                args.address,
                actors,
                args.rounds,
                in_bytes,
                out_bytes,
                args.warmup,
            )
            row["in_bytes"] = in_bytes
            row["out_bytes"] = out_bytes
            results.append(row)
            print(
                f"  actors={actors:>3} {row['records_per_second']:>9} rec/s "
                f"{row['aggregate_mb_per_second']:>8} MB/s "
                f"({row['per_record_ms']} ms/record)",
                flush=True,
            )
    ray.shutdown()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ray_transport.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "ray_transport.json")


if __name__ == "__main__":
    main()
