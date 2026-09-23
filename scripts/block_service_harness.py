"""Experiment A: serial (no-overlap) service time of the SimCLRv2 B/H/J block.

Four configurations of the same block, with the surrounding pipeline order
fixed to the Cedar plan's order:

    L-U  local        three independent native stages
    L-F  local        one native fused stage
    R-U  remote Ray   three actor stages, handed over through the driver
                      exactly like the runtime does (no actor->actor shortcut)
    R-F  remote Ray   one actor running the fused stage

Every batch is submitted only after the previous batch's result is fully
available (one batch in flight, no prefetch threads), actors are created and
warmed before measuring, and each operator is seeded per (record, operator) so
the random sequence does not depend on the configuration.

Usage (inside the container, after `source env/bin/activate`):
  python -u scripts/block_service_harness.py capture --run-dir <run>
  python -u scripts/block_service_harness.py service --run-dir <run>
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    BLOCK_NAMES,
    BLOCK_ORDER,
    CONFIGS,
    OP_SEED_OFFSET,
    PIPE_CROP,
    PIPE_GRAYSCALE,
    PIPE_IMAGE_READER,
    PIPE_LOCAL_FS,
    SEED_BASE,
    SEED_MODULUS,
    build_feature,
    current_record,
    payload_bytes,
    record_seed,
    set_current_record,
    tensor_nbytes,
)
from cedar.pipes import DataSample, PipeVariantType  # noqa: E402
from cedar.pipes.context import (  # noqa: E402
    InProcessPipeVariantContext,
    RayPipeVariantContext,
)
from cedar.pipes.map import (  # noqa: E402
    InProcessMapperPipeVariant,
)
from cedar.pipes.optimize.fuse import (  # noqa: E402
    Compose,
    InProcessFusedOptimizerPipeVariant,
)
from cedar.service import RayActor  # noqa: E402


def _limit_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _pin_cpu(cpu: Optional[int]) -> None:
    if cpu is None:
        return
    try:
        os.sched_setaffinity(0, {int(cpu)})
    except (AttributeError, OSError):
        pass


class TimedOp:
    """One native augmentation callable with per-record seeding and timing.

    ``events`` is filled inside whichever process executes the callable (the
    harness process for local stages, the Ray actor for remote ones); every
    entry is ``(record_key, wall_ns, cpu_ns, meta)`` where ``meta`` describes
    the input of that single call (shape/dtype/contiguity/stride/bytes), so
    per-batch attribution and per-input-representation checks are exact.
    """

    def __init__(self, name: str, fn: Any) -> None:
        self.name = name
        self.fn = fn
        self.record_timings = True
        self.events: List[Tuple[int, int, int, Dict[str, Any]]] = []
        self.bytes_in: List[int] = []
        self.bytes_out: List[int] = []
        self.measure_bytes = False

    @staticmethod
    def describe(data: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {"type": type(data).__name__}
        if isinstance(data, torch.Tensor):
            info.update(
                {
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "contiguous": bool(data.is_contiguous()),
                    "strides": list(data.stride()),
                    "element_size": int(data.element_size()),
                }
            )
        return info

    def __call__(self, data: Any) -> Any:
        record_id, _ = current_record()
        seed = record_seed(record_id, self.name)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if self.record_timings:
            wall_started = time.perf_counter_ns()
            cpu_started = time.process_time_ns()
            out = self.fn(data)
            wall_ns = time.perf_counter_ns() - wall_started
            cpu_ns = time.process_time_ns() - cpu_started
            self.events.append((record_id, wall_ns, cpu_ns, self.describe(data)))
        else:
            out = self.fn(data)
        if self.measure_bytes:
            self.bytes_in.append(payload_bytes(data) or 0)
            self.bytes_out.append(payload_bytes(out) or 0)
        return out

    def reset(self) -> None:
        self.events.clear()
        self.bytes_in.clear()
        self.bytes_out.clear()


def _timed_block_ops(feature) -> List[TimedOp]:
    pipes = feature.logical_pipes
    ops = []
    for p_id in BLOCK_ORDER:
        pipe = pipes[p_id]
        fn = pipe.get_fused_callable()
        ops.append(TimedOp(BLOCK_NAMES[p_id], fn))
    return ops


def _block_callables(feature) -> List[Tuple[str, Any]]:
    """The native callables as ``(name, fn)`` pairs, safe to ship to an actor."""
    pipes = feature.logical_pipes
    return [
        (BLOCK_NAMES[p_id], pipes[p_id].get_fused_callable())
        for p_id in BLOCK_ORDER
    ]


def _id_iter(samples: Sequence[DataSample], keys: Sequence[int]):
    for sample, key in zip(samples, keys):
        set_current_record(key)
        yield sample


def _id_iter_from(iterator: Iterable[DataSample]):
    """Bind a sequential record key to every sample a stage pulls."""
    for key, sample in enumerate(iterator):
        set_current_record(key)
        yield sample


# --------------------------------------------------------------------------
# capture: representative records entering B, from the real pipeline
# --------------------------------------------------------------------------
def cmd_capture(args: argparse.Namespace) -> int:
    _limit_threads()
    _pin_cpu(args.cpu)
    feature = build_feature(batch_size=args.batch_size)
    pipes = feature.logical_pipes
    source = pipes[PIPE_LOCAL_FS]._create_pipe_variant(
        PipeVariantType.INPROCESS, InProcessPipeVariantContext()
    )
    # Successors create their variant from their predecessor's current
    # variant, exactly like the runtime does when it materialises a plan.
    pipes[PIPE_LOCAL_FS].pipe_variant = source
    pipes[PIPE_LOCAL_FS].pipe_variant_type = PipeVariantType.INPROCESS
    reader = pipes[PIPE_IMAGE_READER]._create_pipe_variant(
        PipeVariantType.INPROCESS, InProcessPipeVariantContext()
    )
    pipes[PIPE_IMAGE_READER].pipe_variant = reader
    pipes[PIPE_IMAGE_READER].pipe_variant_type = PipeVariantType.INPROCESS
    records: List[Any] = []
    sizes: List[int] = []
    started = time.perf_counter()
    stage_gray = InProcessMapperPipeVariant(
        None, TimedOp("grayscale", pipes[PIPE_GRAYSCALE].get_fused_callable())
    )
    stage_crop = InProcessMapperPipeVariant(
        None, TimedOp("crop", pipes[PIPE_CROP].get_fused_callable())
    )
    reader._input_iter = _id_iter_from(source._iter_impl())
    stage_gray._input_iter = _id_iter_from(reader._iter_impl())
    stage_crop._input_iter = _id_iter_from(stage_gray._iter_impl())
    for sample in stage_crop._iter_impl():
        records.append(sample.data.detach().clone())
        sizes.append(payload_bytes(sample.data) or 0)
        if len(records) >= args.records:
            break
    elapsed = time.perf_counter() - started

    run_dir = Path(args.run_dir)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "records": records,
            "shape": list(records[0].shape),
            "dtype": str(records[0].dtype),
        },
        run_dir / "inputs" / "block_inputs.pt",
    )
    meta = {
        "records": len(records),
        "batch_size": args.batch_size,
        "shape": list(records[0].shape),
        "dtype": str(records[0].dtype),
        "payload_bytes_mean": statistics.fmean(sizes),
        "payload_bytes_median": statistics.median(sizes),
        "payload_bytes_min": min(sizes),
        "payload_bytes_max": max(sizes),
        "tensor_nbytes": tensor_nbytes(records[0]),
        "capture_seconds": elapsed,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "source_order": "LocalFSSource(recursive=True) over imagenette2/train",
        "pipeline_prefix": [
            "LocalFSListerPipe(9)",
            "ImageReaderPipe(8)",
            "Grayscale(3)",
            "RandomResizedCrop(6)",
        ],
    }
    (run_dir / "inputs" / "block_inputs_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(json.dumps(meta, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
# experiment A: serial service time
# --------------------------------------------------------------------------
def _ray_actor_classes():
    import ray
    from cedar.pipes.ray_variant import get_ray_actor_options

    @ray.remote(num_cpus=0)
    class TimedRayMapperActor(RayActor):
        @staticmethod
        def _describe(data: Any) -> Dict[str, Any]:
            info: Dict[str, Any] = {"type": type(data).__name__}
            if isinstance(data, torch.Tensor):
                info.update(
                    {
                        "shape": list(data.shape),
                        "dtype": str(data.dtype),
                        "contiguous": bool(data.is_contiguous()),
                        "strides": list(data.stride()),
                        "element_size": int(data.element_size()),
                    }
                )
            return info

        def __init__(
            self,
            name: str,
            op_name: str,
            fn: Any,
            seed_base: int,
            seed_modulus: int,
            seed_offset: int,
        ) -> None:
            super().__init__(name)
            self.op_name = op_name
            self.fn = fn
            self.seed_base = seed_base
            self.seed_modulus = seed_modulus
            self.seed_offset = seed_offset
            self.next_id = 0
            self.events: List[Tuple[int, int, int, int, Dict[str, Any]]] = []
            self.record_timings = True

        def process(self, data: Any) -> Any:
            out = []
            for item in data:
                key = self.next_id
                self.next_id += 1
                seed = (
                    self.seed_base + key * 1_000_003 + self.seed_offset
                ) % self.seed_modulus
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                if self.record_timings:
                    wall_started = time.perf_counter_ns()
                    cpu_started = time.process_time_ns()
                    value = self.fn(item)
                    meta = self._describe(item)
                    self.events.append(
                        (
                            key,
                            0,
                            time.perf_counter_ns() - wall_started,
                            time.process_time_ns() - cpu_started,
                            meta,
                        )
                    )
                else:
                    value = self.fn(item)
                out.append(value)
            return out

        def set_record_timings(self, enabled: bool):
            self.record_timings = bool(enabled)
            return True

        def take_events(self):
            return list(self.events)

        def reset_events(self):
            self.events.clear()
            self.next_id = 0
            return True

        def set_affinity(self, cpu: int):
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except (AttributeError, OSError):
                return False
            return True

    @ray.remote(num_cpus=0)
    class TimedRayFusedActor(RayActor):
        @staticmethod
        def _describe(data: Any) -> Dict[str, Any]:
            info: Dict[str, Any] = {"type": type(data).__name__}
            if isinstance(data, torch.Tensor):
                info.update(
                    {
                        "shape": list(data.shape),
                        "dtype": str(data.dtype),
                        "contiguous": bool(data.is_contiguous()),
                        "strides": list(data.stride()),
                        "element_size": int(data.element_size()),
                    }
                )
            return info

        def __init__(
            self,
            name: str,
            ops: List[Tuple[str, Any]],
            seed_base: int,
            seed_modulus: int,
            seed_offsets: List[int],
        ) -> None:
            super().__init__(name)
            self.ops = ops
            self.seed_base = seed_base
            self.seed_modulus = seed_modulus
            self.seed_offsets = seed_offsets
            self.next_id = 0
            self.events: List[Tuple[int, int, int, int, Dict[str, Any]]] = []
            self.record_timings = True

        def process(self, data: Any) -> Any:
            out = []
            for item in data:
                key = self.next_id
                self.next_id += 1
                value = item
                for index, (_op_name, fn) in enumerate(self.ops):
                    seed = (
                        self.seed_base + key * 1_000_003 + self.seed_offsets[index]
                    ) % self.seed_modulus
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    if self.record_timings:
                        wall_started = time.perf_counter_ns()
                        cpu_started = time.process_time_ns()
                        value = fn(value)
                        meta = self._describe(value)
                        self.events.append(
                            (
                                key,
                                index,
                                time.perf_counter_ns() - wall_started,
                                time.process_time_ns() - cpu_started,
                                meta,
                            )
                        )
                    else:
                        value = fn(value)
                out.append(value)
            return out

        def set_record_timings(self, enabled: bool):
            self.record_timings = bool(enabled)
            return True

        def take_events(self):
            return list(self.events)

        def reset_events(self):
            self.events.clear()
            self.next_id = 0
            return True

        def set_affinity(self, cpu: int):
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except (AttributeError, OSError):
                return False
            return True

    return TimedRayMapperActor, TimedRayFusedActor, get_ray_actor_options


class _TimedRayMapperVariant:
    """Factory producing the real RayMapperPipeVariant with timed actors."""

    @staticmethod
    def create(op_name: str, fn: Any, batch_size: int, actor_cls, actor_options,
               name: str, seed_args: Dict[str, Any]):
        from cedar.pipes.map import RayMapperPipeVariant

        class TimedRayMapperPipeVariantImpl(RayMapperPipeVariant):
            def _create_actor(self):
                return actor_cls.options(**actor_options).remote(
                    self.name,
                    op_name,
                    self.fn,
                    seed_args["seed_base"],
                    seed_args["seed_modulus"],
                    seed_args["seed_offset"],
                )

        ctx = RayPipeVariantContext(
            n_actors=1,
            max_inflight=1,
            max_prefetch=1,
            use_threads=True,
            submit_batch_size=batch_size,
        )
        return TimedRayMapperPipeVariantImpl(name, None, fn, ctx)


class _TimedRayFusedVariant:
    @staticmethod
    def create(ops: List[Tuple[str, Any]], batch_size: int, actor_cls,
               actor_options, name: str, seed_args: Dict[str, Any]):
        from cedar.pipes.optimize.fuse import RayFusedOptimizerPipeVariant

        class TimedRayFusedPipeVariantImpl(RayFusedOptimizerPipeVariant):
            def _create_actor(self):
                return actor_cls.options(**actor_options).remote(
                    self.name,
                    ops,
                    seed_args["seed_base"],
                    seed_args["seed_modulus"],
                    seed_args["seed_offsets"],
                )

        ctx = RayPipeVariantContext(
            n_actors=1,
            max_inflight=1,
            max_prefetch=1,
            use_threads=True,
            submit_batch_size=batch_size,
        )
        return TimedRayFusedPipeVariantImpl(
            name, None, Compose([fn for _name, fn in ops]), ctx
        )


def _run_local_batch(ops: List[TimedOp], records: Sequence[Any], keys: Sequence[int],
                     fused: bool) -> List[Any]:
    if fused:
        variant = InProcessFusedOptimizerPipeVariant(None, Compose(ops))
        samples = [DataSample(data=item) for item in records]
        variant._input_iter = _id_iter(samples, keys)
        return [x.data for x in variant._iter_impl()]
    payload = [DataSample(data=item) for item in records]
    for op in ops:
        variant = InProcessMapperPipeVariant(None, op)
        variant._input_iter = _id_iter(payload, keys)
        payload = list(variant._iter_impl())
    return [x.data for x in payload]


def _run_ray_batch(variants, records: Sequence[Any], batch_size: int,
                   timeout: float = 120.0) -> List[Any]:
    payload = list(records)
    for variant in variants:
        samples = [DataSample(data=item) for item in payload]
        for sample in samples:
            variant._submit(sample)
        out = []
        while len(out) < len(samples):
            out.append(variant._get_next_result(timeout=timeout).data)
        payload = out
    return payload


def cmd_service(args: argparse.Namespace) -> int:
    import ray

    _limit_threads()
    _pin_cpu(args.cpu)
    run_dir = Path(args.run_dir)
    blob = torch.load(run_dir / "inputs" / "block_inputs.pt")
    pool: List[Any] = blob["records"]
    if len(pool) < args.batch_size:
        raise RuntimeError("input pool smaller than one batch")
    batch_size = args.batch_size

    feature = build_feature(batch_size=batch_size)
    configs = args.configs or list(CONFIGS)
    needs_ray = any(config.startswith("R") for config in configs)
    TimedRayMapperActor = TimedRayFusedActor = None
    actor_options: Dict[str, Any] = {}
    if needs_ray:
        if "RAY_ADDRESS" not in os.environ and args.ray_ip:
            os.environ["RAY_ADDRESS"] = args.ray_ip
        from cedar.pipes.ray_variant import configure_remote_ray_experiment

        configure_remote_ray_experiment()
        ray.init(address=args.ray_ip, ignore_reinit_error=True)
        (
            TimedRayMapperActor,
            TimedRayFusedActor,
            get_ray_actor_options,
        ) = _ray_actor_classes()
        actor_options = get_ray_actor_options(0.0)

    class Runner:
        """One configuration: real variants plus their actors."""

        def __init__(self, config: str):
            self.config = config
            self.remote = config.startswith("R")
            self.fused = config.endswith("F")
            self.ops = _timed_block_ops(feature) if not self.remote else []
            self.callables = _block_callables(feature) if self.remote else []
            self.variants: List[Any] = []
            self.actors: List[Any] = []
            if not self.remote:
                return
            seed_args = {
                "seed_base": SEED_BASE,
                "seed_modulus": SEED_MODULUS,
                "seed_offset": None,
                "seed_offsets": [OP_SEED_OFFSET[name] for name, _ in self.callables],
            }
            if self.fused:
                variant = _TimedRayFusedVariant.create(
                    self.callables, batch_size, TimedRayFusedActor, actor_options,
                    "timed_fused_block", seed_args,
                )
                self.variants = [variant]
                self.actor_ops = [[name for name, _ in self.callables]]
            else:
                self.actor_ops = []
                for op_name, fn in self.callables:
                    seed_args["seed_offset"] = OP_SEED_OFFSET[op_name]
                    self.variants.append(
                        _TimedRayMapperVariant.create(
                            op_name, fn, batch_size, TimedRayMapperActor,
                            actor_options, f"timed_{op_name}",
                            dict(seed_args),
                        )
                    )
                    self.actor_ops.append([op_name])
            for variant in self.variants:
                self.actors.extend(variant.variant_ctx.service._actors)
            for index, actor in enumerate(self.actors):
                cpu = (
                    args.remote_cpu_base
                    if args.same_remote_cpu
                    else args.remote_cpu_base + index
                )
                ray.get(actor.set_affinity.remote(cpu))
            self.pinned_cpus = [
                args.remote_cpu_base
                if args.same_remote_cpu
                else args.remote_cpu_base + index
                for index in range(len(self.actors))
            ]
            self.actor_locations = ray.get(
                [actor.get_runtime_location.remote() for actor in self.actors]
            )

        def run_batch(self, records: Sequence[Any], keys: Sequence[int]):
            if self.remote:
                return _run_ray_batch(self.variants, records, batch_size)
            return _run_local_batch(self.ops, records, keys, self.fused)

        def reset(self):
            for op in self.ops:
                op.reset()
            if self.remote:
                ray.get([actor.reset_events.remote() for actor in self.actors])
                for variant in self.variants:
                    service = variant.variant_ctx.service
                    service._path_timing_samples = 0
                    service._path_timing_batches = 0
                    service._path_submit_ns = 0.0
                    service._path_get_ns = 0.0
                for variant in self.variants:
                    service = variant.variant_ctx.service
                    service._backend_compute_count = 0
                    service._backend_compute_sum_ns = 0.0
                    service._backend_compute_sum_sq_ns = 0.0

        def timing_rows(self) -> List[Dict[str, Any]]:
            """One row per (record, member): wall/cpu time plus input metadata."""
            rows: List[Dict[str, Any]] = []
            if self.remote:
                for index, actor in enumerate(self.actors):
                    names = self.actor_ops[index]
                    for key, op_index, wall_ns, cpu_ns, meta in ray.get(
                        actor.take_events.remote()
                    ):
                        rows.append(
                            {
                                "record_key": key,
                                "operator": names[op_index],
                                "wall_ms": wall_ns / 1e6,
                                "cpu_ms": cpu_ns / 1e6,
                                **meta,
                            }
                        )
                return rows
            for op in self.ops:
                for key, wall_ns, cpu_ns, meta in op.events:
                    rows.append(
                        {
                            "record_key": key,
                            "operator": op.name,
                            "wall_ms": wall_ns / 1e6,
                            "cpu_ms": cpu_ns / 1e6,
                            **meta,
                        }
                    )
            return rows

        def member_names(self) -> List[str]:
            if self.remote:
                return [name for name, _ in self.callables]
            return [op.name for op in self.ops]

        def path_stats(self):
            if not self.remote:
                return []
            return [
                variant.variant_ctx.service.get_path_timing_stats()
                for variant in self.variants
            ]

        def shutdown(self):
            for variant in self.variants:
                try:
                    variant.shutdown()
                except Exception:  # noqa: BLE001
                    pass

    def batch_inputs(index: int):
        start = index * batch_size
        records = [pool[(start + offset) % len(pool)] for offset in range(batch_size)]
        keys = [start + offset for offset in range(batch_size)]
        return records, keys

    runners = {config: Runner(config) for config in configs}

    try:
        # ---- verification: identical outputs across configurations --------
        verification: Dict[str, Any] = {}
        reference: Optional[List[Any]] = None
        for config, runner in runners.items():
            records, keys = batch_inputs(0)
            outputs = runner.run_batch(records, keys)
            if reference is None:
                reference = outputs
                verification[config] = {"matches_reference": True}
                continue
            mismatch = 0
            max_diff = 0.0
            for left, right in zip(reference, outputs):
                if left.shape != right.shape or left.dtype != right.dtype:
                    mismatch += 1
                    continue
                max_diff = max(max_diff, float((left - right).abs().max()))
            verification[config] = {
                "matches_reference": mismatch == 0 and max_diff == 0.0,
                "shape_dtype_mismatches": mismatch,
                "max_abs_diff": max_diff,
                "shape": list(outputs[0].shape),
                "dtype": str(outputs[0].dtype),
            }

        def run_batches(runner: Runner, count: int, collect: bool):
            elapsed: List[float] = []
            records_seen = 0
            for index in range(count):
                records, keys = batch_inputs(index)
                started = time.perf_counter_ns()
                out = runner.run_batch(records, keys)
                finished = time.perf_counter_ns()
                if collect:
                    elapsed.append((finished - started) / 1e6)
                records_seen += len(out)
            return elapsed, records_seen

        # ---- pilot every configuration, then size one common measurement ---
        pilot: Dict[str, float] = {}
        for config, runner in runners.items():
            runner.reset()
            # warmup also covers the 100+ batches the protocol asks for
            run_batches(runner, args.warmup_batches, collect=False)
            runner.reset()
            elapsed, _ = run_batches(runner, args.pilot_batches, collect=True)
            pilot[config] = statistics.fmean(elapsed)
            print(f"PILOT {config}: {pilot[config]:.3f} ms/batch", flush=True)
        slowest_ms = max(pilot.values())
        fastest_ms = min(pilot.values())
        measured_batches = max(
            args.min_batches,
            int((args.min_seconds * 1000.0) / max(fastest_ms, 1e-6)) + 1,
        )
        if args.max_batches:
            measured_batches = min(measured_batches, args.max_batches)
        print(
            f"MEASURE batches={measured_batches} fastest={fastest_ms:.3f} ms "
            f"slowest={slowest_ms:.3f} ms -> fastest "
            f"{measured_batches * fastest_ms / 1000.0:.1f} s",
            flush=True,
        )

        instrumented = args.instrument == "full"
        for runner in runners.values():
            for op in runner.ops:
                op.record_timings = instrumented
            if runner.remote:
                ray.get([
                    actor.set_record_timings.remote(instrumented)
                    for actor in runner.actors
                ])

        results: Dict[str, Any] = {}
        rows: List[Dict[str, Any]] = []
        timing_rows: List[Dict[str, Any]] = []
        run_order: List[Dict[str, Any]] = []
        config_names = list(runners)
        for round_index in range(args.rounds):
            order = config_names[round_index % len(config_names):] + \
                config_names[: round_index % len(config_names)]
            run_order.append({"round": round_index + 1, "order": list(order)})
            for position, config in enumerate(order):
                runner = runners[config]
                runner.reset()
                run_batches(runner, args.warmup_batches, collect=False)
                runner.reset()
                # Keep the timing window free of garbage-collection pauses;
                # the same treatment is applied to every configuration.
                gc.collect()
                gc.freeze()
                gc.disable()
                elapsed_list, total_records = run_batches(
                    runner, measured_batches, collect=True
                )
                gc.enable()
                events = runner.timing_rows()
                member_names = runner.member_names()
                # Batch-level sums first: one normalisation step only.
                per_batch_compute = {name: [0.0] * len(elapsed_list)
                                     for name in member_names}
                per_batch_cpu = {name: [0.0] * len(elapsed_list)
                                 for name in member_names}
                for event in events:
                    record_key = event["record_key"]
                    batch_index = record_key // batch_size
                    if batch_index >= len(elapsed_list):
                        continue
                    name = event["operator"]
                    per_batch_compute[name][batch_index] += event["wall_ms"]
                    per_batch_cpu[name][batch_index] += event["cpu_ms"]
                    timing_rows.append(
                        {
                            "config": config,
                            "round": round_index + 1,
                            "order_index": position,
                            "batch_id": batch_index,
                            "record_key": record_key,
                            **event,
                        }
                    )
                per_batch: List[Dict[str, Any]] = []
                for index, elapsed_ms in enumerate(elapsed_list):
                    row: Dict[str, Any] = {
                        "config": config,
                        "round": round_index + 1,
                        "order_index": position,
                        "batch_id": index,
                        "source_records": batch_size,
                        "elapsed_ms_per_batch": elapsed_ms,
                    }
                    for name in member_names:
                        row[f"compute_{name}_ms_per_batch"] = (
                            per_batch_compute[name][index]
                        )
                        row[f"cpu_{name}_ms_per_batch"] = per_batch_cpu[name][index]
                    compute_total = sum(
                        per_batch_compute[name][index] for name in member_names
                    )
                    cpu_total = sum(
                        per_batch_cpu[name][index] for name in member_names
                    )
                    row["compute_sum_ms_per_batch"] = compute_total
                    row["cpu_sum_ms_per_batch"] = cpu_total
                    row["other_overhead_ms_per_batch"] = elapsed_ms - compute_total
                    for key in (
                        "elapsed_ms_per_batch",
                        "compute_sum_ms_per_batch",
                        "cpu_sum_ms_per_batch",
                        "other_overhead_ms_per_batch",
                    ):
                        row[key.replace("_per_batch", "_per_record")] = (
                            row[key] / batch_size
                        )
                    for name in member_names:
                        row[f"compute_{name}_ms_per_record"] = (
                            row[f"compute_{name}_ms_per_batch"] / batch_size
                        )
                    # Invariant: compute + other overhead == end-to-end time.
                    assert abs(
                        row["compute_sum_ms_per_batch"]
                        + row["other_overhead_ms_per_batch"]
                        - row["elapsed_ms_per_batch"]
                    ) < 1e-9
                    per_batch.append(row)
                rows.extend(per_batch)
                key = f"{config}_r{round_index + 1}"
                results[key] = {
                    "config": config,
                    "round": round_index + 1,
                    "order_index": position,
                    "pilot_ms_per_batch": pilot.get(config),
                    "measured_batches": len(elapsed_list),
                    "measured_records": total_records,
                    "elapsed_ms_per_batch_mean": statistics.fmean(elapsed_list),
                    "elapsed_ms_per_batch_stdev": (
                        statistics.stdev(elapsed_list)
                        if len(elapsed_list) > 1 else 0.0
                    ),
                    "compute_sum_ms_per_batch_mean": statistics.fmean(
                        [r["compute_sum_ms_per_batch"] for r in per_batch]
                    ),
                    "cpu_sum_ms_per_batch_mean": statistics.fmean(
                        [r["cpu_sum_ms_per_batch"] for r in per_batch]
                    ),
                    "other_overhead_ms_per_batch_mean": statistics.fmean(
                        [r["other_overhead_ms_per_batch"] for r in per_batch]
                    ),
                    "actors": len(runner.actors),
                    "actor_cpus": getattr(runner, "pinned_cpus", []),
                    "actor_locations": getattr(runner, "actor_locations", []),
                    "path_timing": runner.path_stats(),
                }
                print(
                    f"RESULT {key}: batches={len(elapsed_list)} "
                    f"elapsed={results[key]['elapsed_ms_per_batch_mean']:.3f} "
                    f"ms/batch compute="
                    f"{results[key]['compute_sum_ms_per_batch_mean']:.3f} "
                    f"other="
                    f"{results[key]['other_overhead_ms_per_batch_mean']:.3f}",
                    flush=True,
                )
    finally:
        for runner in runners.values():
            runner.shutdown()
        ray.shutdown()

    # ---- verification across configurations -----------------------------
    _write_service_csvs(
        run_dir, rows, results, verification, args, timing_rows, run_order
    )
    return 0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _mean_of(rows: List[Dict[str, Any]], column: str) -> float:
    values = [float(row[column]) for row in rows if column in row]
    return statistics.fmean(values) if values else 0.0


def _stdev_of(rows: List[Dict[str, Any]], column: str) -> float:
    values = [float(row[column]) for row in rows if column in row]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _write_service_csvs(
    run_dir: Path, rows, results, verification, args, timing_rows, run_order
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "service_raw.csv"
    fieldnames = list(rows[0].keys()) if rows else ["config", "batch_id"]
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = run_dir / "service_summary.csv"
    member_names = list(BLOCK_NAMES.values())
    fields = (
        ["config", "round", "order_index", "repeat", "measured_batches",
         "measured_records", "source_records", "actors"]
        + [f"compute_{name}_ms_per_record_mean" for name in member_names]
        + [
            "compute_sum_ms_per_record_mean",
            "compute_sum_ms_per_record_stdev",
            "other_overhead_ms_per_record_mean",
            "other_overhead_ms_per_record_stdev",
            "elapsed_ms_per_record_mean",
            "elapsed_ms_per_record_stdev",
            "cpu_sum_ms_per_record_mean",
            "elapsed_ms_per_batch_mean",
            "elapsed_ms_per_batch_stdev",
            "elapsed_ms_per_batch_p50",
            "elapsed_ms_per_batch_p90",
            "compute_sum_ms_per_batch_mean",
            "other_overhead_ms_per_batch_mean",
            "identity_error_ms_per_batch_max",
        ]
    )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, payload in results.items():
            if "elapsed_ms_per_batch_mean" not in payload:
                continue
            config = payload["config"]
            round_index = payload["round"]
            subset = [
                r for r in rows
                if r["config"] == config and r["round"] == round_index
            ]
            compute_means = {
                f"compute_{name}_ms_per_record_mean": _mean_of(
                    subset, f"compute_{name}_ms_per_record"
                )
                for name in member_names
            }
            elapsed_batches = [r["elapsed_ms_per_batch"] for r in subset]
            compute_batches = [r["compute_sum_ms_per_batch"] for r in subset]
            other_batches = [r["other_overhead_ms_per_batch"] for r in subset]
            writer.writerow(
                {
                    "config": config,
                    "round": round_index,
                    "order_index": payload["order_index"],
                    "repeat": args.repeat,
                    "measured_batches": payload["measured_batches"],
                    "measured_records": payload["measured_records"],
                    "source_records": args.batch_size,
                    "actors": payload["actors"],
                    **compute_means,
                    "compute_sum_ms_per_record_mean": _mean_of(
                        subset, "compute_sum_ms_per_record"
                    ),
                    "compute_sum_ms_per_record_stdev": _stdev_of(
                        subset, "compute_sum_ms_per_record"
                    ),
                    "other_overhead_ms_per_record_mean": _mean_of(
                        subset, "other_overhead_ms_per_record"
                    ),
                    "other_overhead_ms_per_record_stdev": _stdev_of(
                        subset, "other_overhead_ms_per_record"
                    ),
                    "elapsed_ms_per_record_mean": _mean_of(
                        subset, "elapsed_ms_per_record"
                    ),
                    "elapsed_ms_per_record_stdev": _stdev_of(
                        subset, "elapsed_ms_per_record"
                    ),
                    "cpu_sum_ms_per_record_mean": _mean_of(
                        subset, "cpu_sum_ms_per_record"
                    ),
                    "elapsed_ms_per_batch_mean": statistics.fmean(elapsed_batches),
                    "elapsed_ms_per_batch_stdev": (
                        statistics.stdev(elapsed_batches)
                        if len(elapsed_batches) > 1 else 0.0
                    ),
                    "elapsed_ms_per_batch_p50": _quantile(elapsed_batches, 0.5),
                    "elapsed_ms_per_batch_p90": _quantile(elapsed_batches, 0.9),
                    "compute_sum_ms_per_batch_mean": statistics.fmean(
                        compute_batches
                    ),
                    "other_overhead_ms_per_batch_mean": statistics.fmean(
                        other_batches
                    ),
                    "identity_error_ms_per_batch_max": max(
                        abs(c + o - e)
                        for c, o, e in zip(
                            compute_batches, other_batches, elapsed_batches
                        )
                    ),
                }
            )

    # Cross-round summary: mean/stdev over the per-round means.
    cross_path = run_dir / "service_summary_repeats.csv"
    with cross_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["config", "rounds", "elapsed_ms_per_record_mean",
             "elapsed_ms_per_record_stdev", "compute_ms_per_record_mean",
             "compute_ms_per_record_stdev", "other_overhead_ms_per_record_mean",
             "other_overhead_ms_per_record_stdev"]
        )
        by_config: Dict[str, List[Dict[str, Any]]] = {}
        for payload in results.values():
            if "elapsed_ms_per_batch_mean" not in payload:
                continue
            by_config.setdefault(payload["config"], []).append(payload)
        for config, payloads in by_config.items():
            records = [p["measured_records"] for p in payloads]
            totals = [
                p["elapsed_ms_per_batch_mean"] * p["measured_batches"] / rec
                for p, rec in zip(payloads, records)
            ]
            computes = [
                p["compute_sum_ms_per_batch_mean"] * p["measured_batches"] / rec
                for p, rec in zip(payloads, records)
            ]
            others = [
                p["other_overhead_ms_per_batch_mean"] * p["measured_batches"] / rec
                for p, rec in zip(payloads, records)
            ]
            writer.writerow(
                [
                    config,
                    len(payloads),
                    statistics.fmean(totals),
                    statistics.stdev(totals) if len(totals) > 1 else 0.0,
                    statistics.fmean(computes),
                    statistics.stdev(computes) if len(computes) > 1 else 0.0,
                    statistics.fmean(others),
                    statistics.stdev(others) if len(others) > 1 else 0.0,
                ]
            )

    timing_path = run_dir / "operator_timing.csv"
    timing_fields = [
        "config", "round", "order_index", "batch_id", "record_key", "operator",
        "wall_ms", "cpu_ms", "type", "shape", "dtype", "contiguous", "strides",
        "element_size",
    ]
    with timing_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_fields,
                                extrasaction="ignore")
        writer.writeheader()
        for row in timing_rows:
            payload = dict(row)
            if "shape" in payload and isinstance(payload["shape"], list):
                payload["shape"] = "x".join(str(x) for x in payload["shape"])
            if "strides" in payload and isinstance(payload["strides"], list):
                payload["strides"] = "x".join(str(x) for x in payload["strides"])
            writer.writerow(payload)

    (run_dir / "service_verification.json").write_text(
        json.dumps(verification, indent=2)
    )
    (run_dir / "service_meta.json").write_text(
        json.dumps(
            {
                "configs": list(results.keys()),
                "batch_size": args.batch_size,
                "min_seconds": args.min_seconds,
                "warmup_batches": args.warmup_batches,
                "pilot_batches": args.pilot_batches,
                "actor_warmup": args.actor_warmup,
                "rounds": args.rounds,
                "run_order": run_order,
                "instrument": args.instrument,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "ray_ip": args.ray_ip,
                "remote_cpu_base": args.remote_cpu_base,
                "path_timing_enabled": os.environ.get("CEDAR_RAY_PATH_TIMING"),
                "gc_disabled_during_measurement": True,
                "elapsed_definition": (
                    "caller monotonic clock from before the first submit of the "
                    "batch until the last stage's result is materialised"
                ),
                "compute_definition": (
                    "sum over the batch's records of the wall time measured "
                    "directly around each member callable"
                ),
                "other_overhead_definition": (
                    "per-batch end-to-end time minus the batch's summed member "
                    "wall time; contains handoff/serialisation/scheduling and "
                    "framework effects, not pure network time"
                ),
                "units": "ms per batch (source_records = batch_size) and ms per record",
                "per_round_results": list(results.values()),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture")
    capture.add_argument("--run-dir", required=True)
    capture.add_argument("--records", type=int, default=400)
    capture.add_argument("--batch-size", type=int, default=4)
    capture.add_argument("--cpu", type=int, default=None)
    capture.set_defaults(func=cmd_capture)

    service = sub.add_parser("service")
    service.add_argument("--run-dir", required=True)
    service.add_argument("--batch-size", type=int, default=4)
    service.add_argument("--pilot-batches", type=int, default=40)
    service.add_argument("--warmup-batches", type=int, default=120)
    service.add_argument("--min-batches", type=int, default=120)
    service.add_argument("--max-batches", type=int, default=0)
    service.add_argument(
        "--rounds", type=int, default=1,
        help="number of measurement rounds; the config order rotates each round",
    )
    service.add_argument("--min-seconds", type=float, default=30.0)
    service.add_argument("--actor-warmup", type=int, default=5)
    service.add_argument("--record-space", type=int, default=4000)
    service.add_argument("--repeat", type=int, default=1)
    service.add_argument("--cpu", type=int, default=None)
    service.add_argument("--remote-cpu-base", type=int, default=0)
    service.add_argument(
        "--same-remote-cpu",
        action="store_true",
        help="pin every remote actor to remote_cpu_base instead of one CPU each",
    )
    service.add_argument("--ray-ip", default="172.23.166.105:6379")
    service.add_argument("--configs", nargs="*", default=[])
    service.add_argument(
        "--instrument",
        choices=("full", "none"),
        default="full",
        help="full records per-member compute; none measures raw elapsed only",
    )
    service.set_defaults(func=cmd_service)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
