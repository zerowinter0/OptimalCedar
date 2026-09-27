"""Quiet-phase paired handoff probe, merged into a frozen profile.

The inline probe inside the big profiling run measures ~25-30% higher than the
same probe run in a dedicated, quiet session (see
``docs/experiments.md`` §4.18.1).  This tool re-runs the *same* no-compute
paired probe (real request batch in, real response objects out, deployed submit
batch size, serial one batch in flight) with the pipeline torn down, and merges
the values into the profile under ``physical_model.staged_handoff`` — which is
exactly what ``my_optimizer.stage_handoff_ms`` consumes.

Usage (inside the container):
  python -u scripts/freeze_handoff_probe.py \
      --base-profile outputs/.../shared.yaml \
      --inputs outputs/fusion_discount_20260923/expB \
      --out outputs/fusion_handoff_profile_quiet_20260927/simclrv2/shared.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ray  # noqa: E402
from block_mechanism_common import build_feature  # noqa: E402
from block_service_harness import _limit_threads, _pin_cpu  # noqa: E402
from cedar.compose.my_optimizer import MyOptimizer  # noqa: E402
from cedar.pipes.ray_variant import (  # noqa: E402
    configure_remote_ray_experiment,
    get_ray_actor_options,
)


def _probe_actor_cls():
    @ray.remote(num_cpus=0)
    class PairedProbeActor:
        def __init__(self, responses: List[Any]):
            from cedar.utils.threading import limit_native_threadpools

            self._limiter = limit_native_threadpools(1)
            self.responses = list(responses)
            self.index = 0

        def process(self, batch: Any) -> Any:
            out = []
            for _item in batch:
                out.append(self.responses[self.index % len(self.responses)])
                self.index += 1
            return out

        def location(self):
            return {
                "ip": ray.util.get_node_ip_address(),
                "node_id": str(ray.get_runtime_context().get_node_id()),
            }

        def set_affinity(self, cpu: int):
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except (AttributeError, OSError):
                return False
            return True

    return PairedProbeActor


def _chain(feature) -> List[int]:
    children: Dict[int, List[int]] = {}
    for child_id, child in feature.logical_pipes.items():
        for parent in getattr(child, "input_pipes", ()) or ():
            children.setdefault(parent.id, []).append(child_id)
    sources = [
        p_id
        for p_id, pipe in feature.logical_pipes.items()
        if not getattr(pipe, "input_pipes", None)
    ]
    ordered: List[int] = []
    seen: set = set()
    frontier = list(sources)
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        frontier.extend(sorted(children.get(current, [])))
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inputs", default="outputs/fusion_discount_20260923/expB")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-span", type=int, default=4)
    parser.add_argument("--pair-budget", type=int, default=32)
    parser.add_argument("--local-cpu", type=int, default=12)
    parser.add_argument("--remote-cpu", type=int, default=8)
    parser.add_argument("--ray-ip", default="172.23.166.105:6379")
    args = parser.parse_args()

    _limit_threads()
    _pin_cpu(args.local_cpu)
    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True, logging_level="ERROR")
    actor_cls = _probe_actor_cls()
    options = get_ray_actor_options(0.0)

    base_path = Path(args.base_profile).resolve()
    profile = yaml.safe_load(base_path.read_text())
    physical = profile["physical_model"]
    operators = (
        (physical.get("object_boundary") or {}).get("RAY") or {}
    ).get("operators") or {}

    feature = build_feature(batch_size=args.batch_size)
    records = list(
        torch.load(Path(args.inputs) / "inputs" / "block_inputs.pt")["records"]
    )
    # Walk the declared chain, materialising each stage's real input/output
    # pools locally (untimed) so every probe sees the chain's payload mix.
    chain = []
    for p_id in _chain(feature):
        if not (p_id in operators or str(p_id) in operators):
            continue
        pipe = feature.logical_pipes[p_id]
        if getattr(pipe, "fn", None) is None:
            # reader / batcher / prefetcher: no operator callable and no
            # payload of its own to probe.
            continue
        chain.append(p_id)
    pools: Dict[int, Dict[str, List[Any]]] = {}
    current: List[Any] = list(records[: args.batch_size * 4])
    for p_id in chain:
        fn = feature.logical_pipes[p_id].get_fused_callable()
        outputs = [fn(item) for item in current]
        pools[p_id] = {"in": list(current), "out": outputs}
        current = outputs
    print(f"chain={chain} pools={ {k: len(v['in']) for k, v in pools.items()} }", flush=True)

    def entry(p_id: int) -> Dict[str, Any]:
        return operators.get(p_id, operators.get(str(p_id))) or {}

    staged: Dict[str, Any] = {}
    pairs: List[Tuple[int, int]] = []
    for index, first_id in enumerate(chain):
        for last_id in chain[index:]:
            if chain.index(last_id) - index > args.max_span:
                continue
            pairs.append((first_id, last_id))
    for first_id, last_id in pairs[: args.pair_budget]:
        requests = pools[first_id]["in"]
        responses = pools[last_id]["out"]
        e_first = entry(first_id)
        e_last = entry(last_id)
        in_bytes = float(e_first.get("input_serialized_bytes_per_sample") or 0.0)
        out_bytes = float(e_last.get("output_serialized_bytes_per_sample") or 0.0)
        submit_batch = int(
            MyOptimizer._dp_ray_submit_batch_size(in_bytes, out_bytes)
        )
        actor = actor_cls.options(**options).remote(responses)
        try:
            ray.get(actor.set_affinity.remote(args.remote_cpu))
            location = ray.get(actor.location.remote())
            durations: List[float] = []

            def run(batch_index: int) -> float:
                batch = [
                    requests[
                        (batch_index * submit_batch + offset) % len(requests)
                    ]
                    for offset in range(submit_batch)
                ]
                started = time.perf_counter()
                result = ray.get(actor.process.remote(batch))
                elapsed = time.perf_counter() - started
                if len(result) != len(batch):
                    raise RuntimeError("paired probe returned wrong batch size")
                return elapsed * 1000.0 / submit_batch

            for index in range(max(1, args.warmup)):
                run(index)
            for index in range(args.batches):
                durations.append(run(index + args.warmup))
            staged[f"{first_id}->{last_id}"] = {
                "method": "paired_real_payload_handoff_serial_quiet",
                "first_pipe_id": first_id,
                "last_pipe_id": last_id,
                "submit_batch_size": submit_batch,
                "measured_batches": len(durations),
                "measured_input_records": len(durations) * submit_batch,
                "paired_probe_ms_per_sample": statistics.fmean(durations),
                "paired_probe_stdev_ms_per_sample": (
                    statistics.pstdev(durations) if len(durations) > 1 else 0.0
                ),
                "paired_probe_median_ms_per_sample": statistics.median(durations),
                "actor_location": location,
                "session": "quiet_dedicated",
            }
            print(
                f"  {first_id}->{last_id}: "
                f"{statistics.fmean(durations):.2f} ± "
                f"{statistics.pstdev(durations):.2f} ms/record "
                f"(submit_batch={submit_batch})",
                flush=True,
            )
        finally:
            try:
                ray.kill(actor)
            except Exception:  # noqa: BLE001
                pass
    physical.setdefault("staged_handoff", {})["RAY"] = staged
    physical.setdefault("staged_handoff", {})["provenance"] = {
        "tool": "scripts/freeze_handoff_probe.py",
        "base_profile": str(base_path),
        "session": "quiet_dedicated (pipeline torn down)",
        "chain": chain,
        "pairs": len(staged),
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(profile, sort_keys=False))
    print(f"wrote {out_path} with {len(staged)} quiet pairs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
