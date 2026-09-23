"""Experiment A: isolate the effect of Cedar's fixed I/O discount rho.

A synthetic three-operator block A -> B -> C over tensors of identical size,
dtype and layout.  Every operator repeats the same deterministic CPU transform
``repeats`` times and returns the last result, so:

  * the executed work is real and grows with ``repeats``,
  * the output representation never changes (shape/dtype/layout/bytes),
  * U and F produce bit-identical outputs.

Configurations (all on the remote Ray machine, one batch in flight):

  U  three independent Ray stages
  F  one fused Ray stage (all three members in one actor)
  P  fused A+B plus an independent C stage

Measured per batch: end-to-end wall time, per-member wall and CPU time, and the
derived other-overhead.  Cedar's own I/O ratio for such a block is
``rho = (in_first + out_last) / (in_first + 2*sum(inner in) + out_last)``.

Usage (inside the container):
  python -u scripts/fusion_discount_harness.py run \
      --run-dir outputs/<run>/expA --levels 1 4 16 64 --rounds 3 \
      --partial-level 4 --batches 120 --warmup 40
  python -u scripts/fusion_discount_harness.py run ... --instrument none
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.pipes import DataSample, PipeVariantType  # noqa: E402
from cedar.pipes.context import RayPipeVariantContext  # noqa: E402
from cedar.service import RayActor  # noqa: E402

OP_NAMES = ("A", "B", "C")
GROUPS = {
    "U": [("A",), ("B",), ("C",)],
    "P": [("A", "B"), ("C",)],
    "F": [("A", "B", "C")],
}


def limit_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    torch.set_num_threads(1)


def pin_cpu(cpu: Optional[int]) -> None:
    if cpu is None:
        return
    try:
        os.sched_setaffinity(0, {int(cpu)})
    except (AttributeError, OSError):
        pass


def make_kernel(repeats: int) -> List[float]:
    """A fixed 3x3 kernel, identical for every operator and configuration."""
    base = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return [value for row in base for value in row]


def output_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def load_inputs(run_dir: Path, batch_size: int, records: int) -> List[torch.Tensor]:
    """Fixed float32 tensors of identical shape/dtype/layout for A, B and C."""
    blob = torch.load(run_dir / "inputs" / "block_inputs.pt")
    pool = [
        item.to(torch.float32) / 255.0 for item in blob["records"][:records]
    ]
    return pool


# --------------------------------------------------------------------------
# actors: self-contained (the callable is built inside the actor)
# --------------------------------------------------------------------------
def actor_classes():
    import ray
    from cedar.pipes.ray_variant import get_ray_actor_options

    def kernel_tensor(values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 3, 3)

    def run_op(data: torch.Tensor, repeats: int, kernel: torch.Tensor,
               seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        value = data
        if value.dim() == 3:
            value = value.unsqueeze(0)
        for _ in range(repeats):
            value = torch.nn.functional.conv2d(
                value, kernel, padding=1
            ).clamp_(0.0, 1.0)
        return value.squeeze(0) if data.dim() == 3 else value

    def describe(data: torch.Tensor) -> Dict[str, Any]:
        return {
            "shape": "x".join(str(x) for x in data.shape),
            "dtype": str(data.dtype),
            "contiguous": bool(data.is_contiguous()),
            "strides": "x".join(str(x) for x in data.stride()),
            "bytes": output_bytes(data),
        }

    @ray.remote(num_cpus=0)
    class SynthMapperActor(RayActor):
        def __init__(self, name: str, op_name: str, repeats: int,
                     seed_offset: int, kernel: Sequence[float],
                     fn: Any = None) -> None:
            super().__init__(name)
            self.op_name = op_name
            self.repeats = int(repeats)
            self.seed_offset = int(seed_offset)
            self.kernel = kernel_tensor(kernel)
            self.fn = fn
            self.next_id = 0
            self.record_timings = True
            self.events: List[Tuple[int, int, int, int, Dict[str, Any]]] = []

        def _apply(self, item, seed):
            if self.fn is not None:
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                return self.fn(item)
            return run_op(item, self.repeats, self.kernel, seed)

        def process(self, data: Any) -> Any:
            out = []
            for item in data:
                key = self.next_id
                self.next_id += 1
                if self.record_timings:
                    wall = time.perf_counter_ns()
                    cpu = time.process_time_ns()
                    value = self._apply(
                        item,
                        (20260923 + key * 1_000_003 + self.seed_offset)
                        % (2**31 - 1),
                    )
                    self.events.append(
                        (key, 0, time.perf_counter_ns() - wall,
                         time.process_time_ns() - cpu, describe(item))
                    )
                else:
                    value = self._apply(
                        item,
                        (20260923 + key * 1_000_003 + self.seed_offset)
                        % (2**31 - 1),
                    )
                out.append(value)
            return out

        def take_events(self):
            return list(self.events)

        def reset_events(self):
            self.events.clear()
            self.next_id = 0
            return True

        def set_record_timings(self, enabled: bool):
            self.record_timings = bool(enabled)
            return True

        def set_affinity(self, cpu: int):
            pin_cpu(cpu)
            return True

    @ray.remote(num_cpus=0)
    class SynthFusedActor(RayActor):
        def __init__(self, name: str, ops: Sequence[Tuple[str, int, int]],
                     kernel: Sequence[float], fns: Any = None) -> None:
            super().__init__(name)
            self.ops = [(op, int(rep), int(off)) for op, rep, off in ops]
            self.kernel = kernel_tensor(kernel)
            self.fns = list(fns) if fns is not None else None
            self.next_id = 0
            self.record_timings = True
            self.events: List[Tuple[int, int, int, int, Dict[str, Any]]] = []

        def process(self, data: Any) -> Any:
            out = []
            for item in data:
                key = self.next_id
                self.next_id += 1
                value = item
                for index, (_op, repeats, offset) in enumerate(self.ops):
                    seed = (20260923 + key * 1_000_003 + offset) % (2**31 - 1)
                    fn = self.fns[index] if self.fns is not None else None
                    if self.record_timings:
                        wall = time.perf_counter_ns()
                        cpu = time.process_time_ns()
                        if fn is not None:
                            torch.manual_seed(seed)
                            random.seed(seed)
                            np.random.seed(seed % (2**32 - 1))
                            value = fn(value)
                        else:
                            value = run_op(value, repeats, self.kernel, seed)
                        self.events.append(
                            (key, index, time.perf_counter_ns() - wall,
                             time.process_time_ns() - cpu, describe(value))
                        )
                    else:
                        if fn is not None:
                            torch.manual_seed(seed)
                            random.seed(seed)
                            np.random.seed(seed % (2**32 - 1))
                            value = fn(value)
                        else:
                            value = run_op(value, repeats, self.kernel, seed)
                out.append(value)
            return out

        def take_events(self):
            return list(self.events)

        def reset_events(self):
            self.events.clear()
            self.next_id = 0
            return True

        def set_record_timings(self, enabled: bool):
            self.record_timings = bool(enabled)
            return True

        def set_affinity(self, cpu: int):
            pin_cpu(cpu)
            return True

    return SynthMapperActor, SynthFusedActor, get_ray_actor_options


def build_variant(kind: str, group: Sequence[str], levels: Dict[str, int],
                  batch_size: int, mapper_actor_cls, fused_actor_cls,
                  options, kernel: Sequence[float],
                  seed_offsets: Dict[str, int], name: str, cpu: int,
                  callables: Optional[Dict[str, Any]] = None):
    from cedar.pipes.map import RayMapperPipeVariant
    from cedar.pipes.optimize.fuse import (
        RayFusedOptimizerPipeVariant,
        Compose,
    )

    ctx = RayPipeVariantContext(
        n_actors=1, max_inflight=1, max_prefetch=1, use_threads=True,
        submit_batch_size=batch_size,
    )
    if len(group) == 1:
        op = group[0]

        class _Mapper(RayMapperPipeVariant):
            def _create_actor(self):
                return mapper_actor_cls.options(**options).remote(
                    self.name, op, levels[op], seed_offsets[op], kernel,
                    None if callables is None else callables[op],
                )

        return _Mapper(
            name, None,
            None if callables is None else callables[op], ctx
        )
    spec = [(op, levels[op], seed_offsets[op]) for op in group]

    class _Fused(RayFusedOptimizerPipeVariant):
        def _create_actor(self):
            return fused_actor_cls.options(**options).remote(
                self.name, spec, kernel,
                None if callables is None else [callables[op] for op in group],
            )

    class _Noop:
        def __call__(self, value: Any) -> Any:
            return value

    return _Fused(name, None, Compose([_Noop()]), ctx)


def run_batch(variants: Sequence[Any], records: Sequence[torch.Tensor]) -> List[Any]:
    payload = list(records)
    for variant in variants:
        samples = [DataSample(data=item) for item in payload]
        for sample in samples:
            variant._submit(sample)
        out = []
        while len(out) < len(samples):
            out.append(variant._get_next_result(timeout=120.0).data)
        payload = out
    return payload


def member_key(event_key: int, batch_size: int) -> int:
    return event_key // batch_size


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 64])
    run.add_argument("--partial-level", type=int, default=None)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--batches", type=int, default=120)
    run.add_argument("--warmup", type=int, default=40)
    run.add_argument("--pilot-batches", type=int, default=15)
    run.add_argument("--min-seconds", type=float, default=20.0)
    run.add_argument("--records", type=int, default=400)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--local-cpu", type=int, default=12)
    run.add_argument("--remote-cpu", type=int, default=8)
    run.add_argument("--ray-ip", default="172.23.166.105:6379")
    run.add_argument("--instrument", choices=("full", "none"), default="full")
    run.add_argument("--block", choices=("synthetic", "real"), default="synthetic")
    run.add_argument(
        "--real-pipes", default="7,6,5",
        help="mapper pipe ids of the real block, in declared order",
    )
    run.add_argument(
        "--real-names", nargs="+", default=["to_float", "crop", "flip"]
    )
    run.add_argument("--seed", type=int, default=20260923)
    run.add_argument("--label", default="expA",
                     help="prefix for the output CSV/JSON files")
    args = parser.parse_args()

    limit_threads()
    pin_cpu(args.local_cpu)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    import ray

    os.environ.setdefault("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
    os.environ.setdefault("CEDAR_RAY_REQUIRE_REMOTE", "1")
    from cedar.pipes.ray_variant import configure_remote_ray_experiment

    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True)
    SynthMapperActor, SynthFusedActor, get_ray_actor_options = actor_classes()
    mapper_actor_cls = SynthMapperActor
    fused_actor_cls = SynthFusedActor
    options = get_ray_actor_options(0.0)
    kernel = make_kernel(1)
    real_callables: Optional[Dict[str, Any]] = None
    if args.block == "real":
        from block_mechanism_common import build_feature

        real_pipes = [int(x) for x in args.real_pipes.split(",") if x]
        if len(real_pipes) != 3:
            raise SystemExit("--real-pipes must list exactly three pipes")
        global OP_NAMES
        OP_NAMES = tuple(args.real_names)
        GROUPS.clear()
        GROUPS.update({
            "U": [(OP_NAMES[0],), (OP_NAMES[1],), (OP_NAMES[2],)],
            "P": [(OP_NAMES[0], OP_NAMES[1]), (OP_NAMES[2],)],
            "F": [(OP_NAMES[0], OP_NAMES[1], OP_NAMES[2],)],
        })
        feature = build_feature(batch_size=args.batch_size)
        real_callables = {
            name: feature.logical_pipes[p_id].get_fused_callable()
            for name, p_id in zip(OP_NAMES, real_pipes)
        }
        seed_offsets = {OP_NAMES[0]: 71, OP_NAMES[1]: 53, OP_NAMES[2]: 23}
        pool = [
            item
            for item in torch.load(
                args.run_dir / "inputs" / "block_inputs.pt"
            )["records"][: args.records]
        ]
    else:
        seed_offsets = {"A": 11, "B": 23, "C": 37}
        pool = load_inputs(args.run_dir, args.batch_size, args.records)
    tensor = pool[0]
    spec = {
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_bytes_per_record": output_bytes(tensor),
        "batch_size": args.batch_size,
        "levels": args.levels,
        "partial_level": args.partial_level,
        "rounds": args.rounds,
        "warmup_batches": args.warmup,
        "measured_batches": args.batches,
        "instrument": args.instrument,
        "local_cpu": args.local_cpu,
        "remote_cpu": args.remote_cpu,
        "kernel": kernel,
        "rho_U": (output_bytes(tensor) * 2)
                 / (output_bytes(tensor) * 4),  # 3 members, in == out
        "rho_P": (output_bytes(tensor) * 2)
                 / (output_bytes(tensor) * 3),  # AB fused: in + inner + out
        "configs": {key: [list(group) for group in groups]
                    for key, groups in GROUPS.items()},
    }
    (args.run_dir / f"{args.label}_config.json").write_text(
        json.dumps(spec, indent=2)
    )

    def batch_inputs(index: int):
        start = index * args.batch_size
        records = [pool[(start + offset) % len(pool)]
                   for offset in range(args.batch_size)]
        return records

    raw_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    verification: Dict[str, Any] = {}
    run_order: List[Dict[str, Any]] = []

    stack = [("U", level) for level in args.levels] + \
            [("F", level) for level in args.levels]
    if args.partial_level is not None:
        stack.append(("P", args.partial_level))

    for round_index in range(args.rounds):
        order = stack[round_index % len(stack):] + stack[: round_index % len(stack)]
        run_order.append({"round": round_index + 1, "order": [f"{c}@{l}" for c, l in order]})
        for position, (config, level) in enumerate(order):
            groups = GROUPS[config]
            levels = {op: level for op in OP_NAMES}
            variants = []
            for index, group in enumerate(groups):
                variants.append(
                    build_variant(config, group, levels, args.batch_size,
                                  mapper_actor_cls, fused_actor_cls, options,
                                  kernel, seed_offsets,
                                  f"expA_{config}_{'-'.join(group)}", args.remote_cpu,
                                  real_callables)
                )
            actor_handles = []
            for variant in variants:
                actor_handles.extend(variant.variant_ctx.service._actors)
            for handle in actor_handles:
                ray.get(handle.set_affinity.remote(args.remote_cpu))
            for handle in actor_handles:
                ray.get(handle.set_record_timings.remote(args.instrument == "full"))

            # verification pass (before warmup, on the first batch only)
            verify_out = run_batch(variants, batch_inputs(0))
            key = f"{config}@{level}"
            if key not in verification:
                verification[key] = {
                    "shape": list(verify_out[0].shape),
                    "dtype": str(verify_out[0].dtype),
                    "outputs": [item.clone() for item in verify_out],
                }

            for _ in range(args.warmup):
                run_batch(variants, batch_inputs(0))
            for handle in actor_handles:
                ray.get(handle.reset_events.remote())
            for variant in variants:
                service = variant.variant_ctx.service
                service._path_timing_samples = 0
                service._path_timing_batches = 0
                service._path_submit_ns = 0.0
                service._path_get_ns = 0.0

            measured = args.batches
            gc.collect()
            gc.freeze()
            gc.disable()
            elapsed_list = []
            for index in range(measured):
                started = time.perf_counter_ns()
                run_batch(variants, batch_inputs(index))
                elapsed_list.append((time.perf_counter_ns() - started) / 1e6)
            gc.enable()

            per_batch = {op: [0.0] * measured for op in OP_NAMES}
            per_batch_cpu = {op: [0.0] * measured for op in OP_NAMES}
            for variant_index, handle in enumerate(actor_handles):
                group = groups[variant_index]
                for event_key, op_index, wall_ns, cpu_ns, meta in ray.get(
                    handle.take_events.remote()
                ):
                    batch_index = member_key(event_key, args.batch_size)
                    if batch_index >= measured:
                        continue
                    member = group[op_index]
                    per_batch[member][batch_index] += wall_ns / 1e6
                    per_batch_cpu[member][batch_index] += cpu_ns / 1e6
                    timing_rows.append(
                        {
                            "config": config,
                            "level": level,
                            "round": round_index + 1,
                            "batch_id": batch_index,
                            "record_key": event_key,
                            "operator": member,
                            "wall_ms": wall_ns / 1e6,
                            "cpu_ms": cpu_ns / 1e6,
                            **meta,
                        }
                    )
            for index, elapsed_ms in enumerate(elapsed_list):
                compute = sum(per_batch[op][index] for op in OP_NAMES)
                cpu_sum = sum(per_batch_cpu[op][index] for op in OP_NAMES)
                row = {
                    "config": config,
                    "level": level,
                    "round": round_index + 1,
                    "order_index": position,
                    "batch_id": index,
                    "source_records": args.batch_size,
                    "elapsed_ms_per_batch": elapsed_ms,
                    "compute_sum_ms_per_batch": compute,
                    "cpu_sum_ms_per_batch": cpu_sum,
                    "other_overhead_ms_per_batch": elapsed_ms - compute,
                    "elapsed_ms_per_record": elapsed_ms / args.batch_size,
                    "compute_sum_ms_per_record": compute / args.batch_size,
                    "other_overhead_ms_per_record": (elapsed_ms - compute) / args.batch_size,
                }
                for op in OP_NAMES:
                    row[f"compute_{op}_ms_per_batch"] = per_batch[op][index]
                    row[f"compute_{op}_ms_per_record"] = per_batch[op][index] / args.batch_size
                    row[f"cpu_{op}_ms_per_batch"] = per_batch_cpu[op][index]
                raw_rows.append(row)
            elapsed_all = [r["elapsed_ms_per_batch"] for r in raw_rows
                           if r["config"] == config and r["level"] == level
                           and r["round"] == round_index + 1]
            compute_all = [r["compute_sum_ms_per_batch"] for r in raw_rows
                           if r["config"] == config and r["level"] == level
                           and r["round"] == round_index + 1]
            other_all = [r["other_overhead_ms_per_batch"] for r in raw_rows
                         if r["config"] == config and r["level"] == level
                         and r["round"] == round_index + 1]
            summary_rows.append(
                {
                    "config": config,
                    "level": level,
                    "round": round_index + 1,
                    "order_index": position,
                    "measured_batches": measured,
                    "elapsed_ms_per_record_mean":
                        statistics.fmean(elapsed_all) / args.batch_size,
                    "elapsed_ms_per_record_stdev":
                        (statistics.stdev(elapsed_all) / args.batch_size
                         if len(elapsed_all) > 1 else 0.0),
                    "compute_ms_per_record_mean":
                        statistics.fmean(compute_all) / args.batch_size,
                    "other_ms_per_record_mean":
                        statistics.fmean(other_all) / args.batch_size,
                    "actors": len(actor_handles),
                    "gcs_disabled": True,
                }
            )
            print(
                f"RESULT {key} r{round_index + 1}: "
                f"elapsed={statistics.fmean(elapsed_all) / args.batch_size:.4f} "
                f"compute={statistics.fmean(compute_all) / args.batch_size:.4f} "
                f"other={statistics.fmean(other_all) / args.batch_size:.4f} "
                f"ms/record",
                flush=True,
            )
            for variant in variants:
                try:
                    variant.shutdown()
                except Exception:  # noqa: BLE001
                    pass

    # verification across configs at the same level
    checks = {}
    for level in args.levels:
        reference = verification.get(f"U@{level}", {}).get("outputs")
        for config in ("F", "P"):
            payload = verification.get(f"{config}@{level}")
            if payload is None or reference is None:
                continue
            max_diff = 0.0
            for left, right in zip(reference, payload["outputs"]):
                max_diff = max(max_diff, float((left - right).abs().max()))
            checks[f"{config}@{level}"] = {"max_abs_diff": max_diff}

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.run_dir / f"{args.label}_raw.csv", raw_rows)
    write_csv(args.run_dir / f"{args.label}_summary.csv", summary_rows)
    write_csv(args.run_dir / f"{args.label}_operator_timing.csv", timing_rows)
    (args.run_dir / f"{args.label}_verification.json").write_text(
        json.dumps(checks, indent=2)
    )
    (args.run_dir / f"{args.label}_run_order.json").write_text(
        json.dumps(run_order, indent=2)
    )
    ray.shutdown()
    print(f"wrote {args.run_dir}/{args.label}_*.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
