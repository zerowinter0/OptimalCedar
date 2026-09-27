"""§3.3 fusion-cost experiment: overall I/O discount vs compute+boundary.

Real block ``to_float -> RandomResizedCrop -> RandomHorizontalFlip`` (the
Cedar plan's Ray block of the SimCLRv2 recipe) in three organisations:

    U  three independent Ray stages (driver in between, like the runtime)
    P  to_float+crop fused, flip independent
    F  all three fused

One batch in flight, all actors on the remote node pinned to the same CPU,
single-threaded operators, every time normalised per source record.

Three phases, always in this order so the predictions are frozen first:

  stage-profile   per-stage client-observed service time T_i, per-member
                  compute C_i, boundary bytes and configurations (U only),
                  plus payload snapshots for the boundary probe
  boundary-probe  paired no-compute round trips on the real payloads, fitted
                  independently (fixed latency + bytes/throughput)
  validate        U/P/F interleaved rounds with member instrumentation off
                  (main) and on (diagnostic)

Usage (inside the container, after ``source env/bin/activate``):
  python -u scripts/fusion_cost_split_harness.py stage-profile --run-dir <dir>
  python -u scripts/fusion_cost_split_harness.py boundary-probe --run-dir <dir>
  python -u scripts/fusion_cost_split_harness.py validate --run-dir <dir>
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import block_service_harness as bsh  # noqa: E402
from block_mechanism_common import payload_bytes  # noqa: E402
from cedar.pipes import DataSample  # noqa: E402

PIPE_TO_FLOAT = 7
PIPE_CROP = 6
PIPE_FLIP = 5
BLOCK = (
    ("to_float", PIPE_TO_FLOAT),
    ("crop", PIPE_CROP),
    ("flip", PIPE_FLIP),
)
OP_SEED = {"to_float": 71, "crop": 53, "flip": 23}
ORGANISATIONS = {
    "U": (("to_float",), ("crop",), ("flip",)),
    "P": (("to_float", "crop"), ("flip",)),
    "F": (("to_float", "crop", "flip"),),
}
DEFAULT_INPUTS = ROOT / "outputs/fusion_discount_20260923/expB"
SEED_BASE = 20260923
SEED_MODULUS = 2**31 - 1


def _limit_threads() -> None:
    bsh._limit_threads()


def _pin_cpu(cpu: Optional[int]) -> None:
    bsh._pin_cpu(cpu)


def _block_callables(batch_size: int) -> List[Tuple[str, Any]]:
    feature = bsh.build_feature(batch_size=batch_size)
    return [
        (name, feature.logical_pipes[p_id].get_fused_callable())
        for name, p_id in BLOCK
    ]


class StageRunner:
    """One organisation: variants in order plus their actors."""

    def __init__(
        self,
        organisation: str,
        callables: Dict[str, Any],
        batch_size: int,
        actor_options: Dict[str, Any],
        actor_classes,
        remote_cpu_base: int,
        same_remote_cpu: bool,
        submit_batch_by_stage: Optional[List[int]] = None,
    ) -> None:
        import ray

        TimedRayMapperActor, TimedRayFusedActor, _ = actor_classes
        self.organisation = organisation
        self.stages: List[List[str]] = [list(group) for group in ORGANISATIONS[organisation]]
        self.variants: List[Any] = []
        self.actors: List[Any] = []
        self.stage_actors: List[List[Any]] = []
        self.submit_batch_by_stage = list(submit_batch_by_stage or [])
        for stage_index, group in enumerate(self.stages):
            submit_batch = (
                self.submit_batch_by_stage[stage_index]
                if stage_index < len(self.submit_batch_by_stage)
                else None
            )
            if len(group) == 1:
                name = group[0]
                variant = bsh._TimedRayMapperVariant.create(
                    name,
                    callables[name],
                    batch_size,
                    TimedRayMapperActor,
                    actor_options,
                    f"costsplit_{organisation}_{name}",
                    {
                        "seed_base": SEED_BASE,
                        "seed_modulus": SEED_MODULUS,
                        "seed_offset": OP_SEED[name],
                    },
                    submit_batch_size=submit_batch,
                )
            else:
                ops = [(name, callables[name]) for name in group]
                variant = bsh._TimedRayFusedVariant.create(
                    ops,
                    batch_size,
                    TimedRayFusedActor,
                    actor_options,
                    f"costsplit_{organisation}_{'_'.join(group)}",
                    {
                        "seed_base": SEED_BASE,
                        "seed_modulus": SEED_MODULUS,
                        "seed_offsets": [OP_SEED[name] for name in group],
                    },
                    submit_batch_size=submit_batch,
                )
            self.variants.append(variant)
            actors = list(variant.variant_ctx.service._actors)
            self.stage_actors.append(actors)
            self.actors.extend(actors)
        for index, actor in enumerate(self.actors):
            cpu = remote_cpu_base if same_remote_cpu else remote_cpu_base + index
            ray.get(actor.set_affinity.remote(cpu))
        self.pinned_cpus = [
            remote_cpu_base if same_remote_cpu else remote_cpu_base + index
            for index in range(len(self.actors))
        ]
        self.locations = ray.get(
            [actor.get_runtime_location.remote() for actor in self.actors]
        )
        self.member_names = sorted(
            {name for group in self.stages for name in group},
            key=lambda name: [n for n, _ in BLOCK].index(name),
        )

    def run_batch(self, records: Sequence[Any], timeout: float = 240.0) -> Dict[str, Any]:
        """Run one batch, timing each stage on the client side."""
        payload = list(records)
        stage_times: List[float] = []
        stage_bytes: List[Tuple[int, int]] = []
        stage_outputs: List[List[Any]] = []
        whole_started = time.perf_counter()
        for variant in self.variants:
            in_bytes = sum(payload_bytes(item) or 0 for item in payload)
            samples = [DataSample(data=item) for item in payload]
            started = time.perf_counter()
            for sample in samples:
                variant._submit(sample)
            out = []
            while len(out) < len(samples):
                out.append(variant._get_next_result(timeout=timeout).data)
            stage_times.append(time.perf_counter() - started)
            payload = out
            stage_outputs.append(list(payload))
            out_bytes = sum(payload_bytes(item) or 0 for item in payload)
            stage_bytes.append((in_bytes, out_bytes))
        whole = time.perf_counter() - whole_started
        return {
            "stage_times": stage_times,
            "stage_bytes": stage_bytes,
            "stage_outputs": stage_outputs,
            "whole_time": whole,
            "outputs": payload,
        }

    def reset_events(self) -> None:
        import ray

        if self.actors:
            ray.get([actor.reset_events.remote() for actor in self.actors])
        for variant in self.variants:
            service = getattr(variant.variant_ctx, "service", None)
            reset = getattr(service, "reset_path_timing_stats", None)
            if reset is not None:
                reset()

    def path_stats(self):
        """Per-stage client-side submit/ray.get split (ms per sample)."""
        out = []
        for variant in self.variants:
            service = getattr(variant.variant_ctx, "service", None)
            getter = getattr(service, "get_path_timing_stats", None)
            out.append(getter() if getter is not None else None)
        return out

    def set_instrument(self, enabled: bool) -> None:
        import ray

        if self.actors:
            ray.get([actor.set_record_timings.remote(enabled) for actor in self.actors])

    def member_events(self) -> List[Dict[str, Any]]:
        import ray

        rows = []
        if not self.actors:
            return rows
        for stage_index, actors in enumerate(self.stage_actors):
            for actor in actors:
                for key, op_index, wall_ns, cpu_ns, meta in ray.get(
                    actor.take_events.remote()
                ):
                    names = self.stages[stage_index]
                    rows.append(
                        {
                            "stage_index": stage_index,
                            "operator": names[op_index],
                            "wall_ms": wall_ns / 1e6,
                            "cpu_ms": cpu_ns / 1e6,
                            "meta": meta,
                        }
                    )
        return rows

    def shutdown(self) -> None:
        for variant in self.variants:
            try:
                variant.shutdown()
            except Exception:  # noqa: BLE001
                pass


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _batch_records(pool, index: int, batch_size: int):
    start = index * batch_size
    return [pool[(start + offset) % len(pool)] for offset in range(batch_size)]


def _setup(args):
    import ray
    from cedar.pipes.ray_variant import (
        configure_remote_ray_experiment,
        get_ray_actor_options,
    )

    _limit_threads()
    _pin_cpu(args.local_cpu)
    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True, logging_level="ERROR")
    actor_classes = bsh._ray_actor_classes()
    options = get_ray_actor_options(0.0)
    return ray, actor_classes, options


# ------------------------------------------------------------ stage profile --


def cmd_stage_profile(args) -> int:
    ray, actor_classes, options = _setup(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = Path(args.inputs)
    pool = list(torch.load(inputs / "inputs" / "block_inputs.pt")["records"])
    callables = dict(_block_callables(args.batch_size))
    runner = StageRunner(
        "U", callables, args.batch_size, options, actor_classes,
        args.remote_cpu, args.same_remote_cpu,
    )
    rows: List[Dict[str, Any]] = []
    payload_samples: List[Dict[str, Any]] = []
    try:
        runner.set_instrument(True)
        for index in range(args.warmup):
            runner.run_batch(_batch_records(pool, index, args.batch_size))
        for index in range(args.batches):
            records = _batch_records(pool, index, args.batch_size)
            runner.reset_events()
            result = runner.run_batch(records)
            events = runner.member_events()
            per_stage_compute = [0.0] * len(runner.stages)
            for event in events:
                per_stage_compute[event["stage_index"]] += event["wall_ms"]
            for stage_index, stages in enumerate(runner.stages):
                stage_time = result["stage_times"][stage_index]
                compute = per_stage_compute[stage_index]
                in_bytes, out_bytes = result["stage_bytes"][stage_index]
                rows.append(
                    {
                        "config": "U",
                        "round": args.round,
                        "batch_id": index,
                        "stage_index": stage_index,
                        "stage_members": "+".join(stages),
                        "source_records": args.batch_size,
                        "T_ms": stage_time * 1000.0,
                        "T_ms_per_record": stage_time * 1000.0 / args.batch_size,
                        "C_ms": compute,
                        "C_ms_per_record": compute / args.batch_size,
                        "H_ms": stage_time * 1000.0 - compute,
                        "H_ms_per_record": (stage_time * 1000.0 - compute) / args.batch_size,
                        "in_bytes": in_bytes,
                        "out_bytes": out_bytes,
                        "in_bytes_per_record": in_bytes / args.batch_size,
                        "out_bytes_per_record": out_bytes / args.batch_size,
                        "instrument": "on",
                        "batch_whole_T_ms_per_record": (
                            result["whole_time"] * 1000.0 / args.batch_size
                        ),
                    }
                )
            if index % args.snapshot_every == 0 and len(payload_samples) < args.snapshot_batches:
                payload_samples.append(_snapshot(records, result))
    finally:
        runner.shutdown()
    _write_csv(run_dir / "baseline_stage_profile.csv", rows)
    torch.save(
        {"samples": payload_samples, "batch_size": args.batch_size},
        run_dir / "payload_samples.pt",
    )
    env = {
        "phase": "stage-profile",
        "config": "U",
        "inputs": str(inputs),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup,
        "measured_batches": args.batches,
        "round": args.round,
        "instrument": "on",
        "local_cpu": args.local_cpu,
        "remote_cpu": args.remote_cpu,
        "same_remote_cpu": args.same_remote_cpu,
        "actors": runner.locations,
        "pinned_cpus": runner.pinned_cpus,
        "stages": [list(group) for group in runner.stages],
        "seed_base": SEED_BASE,
        "timing_window": (
            "client: first _submit of the stage -> last result of the stage; "
            "C: callable-only wall time inside the actor"
        ),
    }
    (run_dir / "stage_profile_env.json").write_text(json.dumps(env, indent=1, default=str))
    print(f"stage profile rows={len(rows)}", flush=True)
    return 0


def _snapshot(records, result) -> Dict[str, Any]:
    """Payloads of every boundary in one measured batch (for the probe)."""
    stage_inputs = [list(records)]
    for index in range(len(result["stage_times"]) - 1):
        stage_inputs.append(list(result["stage_outputs"][index]))
    boundaries = []
    for index, inputs in enumerate(stage_inputs):
        boundaries.append(
            {
                "stage_index": index,
                "inputs": inputs,
                "outputs": list(result["stage_outputs"][index]),
            }
        )
    return {"boundaries": boundaries}


# ----------------------------------------------------------- boundary probe --


def _probe_actor_cls():
    import ray

    @ray.remote(num_cpus=0)
    class BoundaryProbeActor:
        """Round-trips a batch and returns preallocated response buffers.

        No operator compute runs here: the actor only echoes real request
        payloads and hands back real response objects, so the measured time is
        the request/response/scheduling path alone.
        """

        def __init__(self, responses: List[Any]) -> None:
            self.responses = responses
            self.index = 0

        def process(self, data: Any) -> Any:
            out = []
            for _item in data:
                out.append(self.responses[self.index % len(self.responses)])
                self.index += 1
            return out

        def set_record_timings(self, enabled: bool):
            return True

        def reset_events(self):
            return True

        def take_events(self):
            return []

        def set_affinity(self, cpu: int):
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except (AttributeError, OSError):
                return False
            return True

    return BoundaryProbeActor


def _probe_variant(name, actor_cls, actor_options, responses, batch_size, fused):
    from cedar.pipes.context import RayPipeVariantContext
    from cedar.pipes.map import RayMapperPipeVariant
    from cedar.pipes.optimize.fuse import Compose, RayFusedOptimizerPipeVariant

    ctx = RayPipeVariantContext(
        n_actors=1,
        max_inflight=1,
        max_prefetch=1,
        use_threads=True,
        submit_batch_size=batch_size,
    )
    if fused:
        class ProbeFusedVariant(RayFusedOptimizerPipeVariant):
            def _create_actor(self):
                return actor_cls.options(**actor_options).remote(responses)

        return ProbeFusedVariant(name, None, Compose([lambda value: value]), ctx)

    class ProbeMapperVariant(RayMapperPipeVariant):
        def _create_actor(self):
            return actor_cls.options(**actor_options).remote(responses)

    return ProbeMapperVariant(name, None, lambda value: value, ctx)


def _probe_batch(variant, inputs, timeout: float = 240.0) -> None:
    samples = [DataSample(data=item) for item in inputs]
    for sample in samples:
        variant._submit(sample)
    received = 0
    while received < len(samples):
        variant._get_next_result(timeout=timeout)
        received += 1


def cmd_boundary_probe(args) -> int:
    ray, _actor_classes, options = _setup(args)
    run_dir = Path(args.run_dir)
    blob = torch.load(run_dir / "payload_samples.pt")
    samples = blob["samples"]
    batch_size = int(blob["batch_size"])
    if not samples:
        raise RuntimeError("no payload samples: run stage-profile first")

    # (label, kind, input stage, output stage): a fused block spanning stages
    # a..b sees stage a's inputs as request and stage b's outputs as response.
    probe_specs = (
        ("U_A_mapper", "mapper", 0, 0),
        ("U_B_mapper", "mapper", 1, 1),
        ("U_C_mapper", "mapper", 2, 2),
        ("P_AB_fused", "fused", 0, 1),
        ("F_ABC_fused", "fused", 0, 2),
    )
    actor_cls = _probe_actor_cls()
    rows: List[Dict[str, Any]] = []
    for label, kind, input_stage, output_stage in probe_specs:
        batches = [
            (
                list(sample["boundaries"][input_stage]["inputs"]),
                list(sample["boundaries"][output_stage]["outputs"]),
            )
            for sample in samples
        ]
        variant = None
        try:
            variant = _probe_variant(
                f"probe_{label}", actor_cls, options,
                batches[0][1], batch_size, fused=(kind == "fused"),
            )
            actors = list(variant.variant_ctx.service._actors)
            ray.get([actor.set_affinity.remote(args.remote_cpu) for actor in actors])
            for _ in range(args.warmup):
                _probe_batch(variant, batches[0][0])
            for repeat in range(args.repeats):
                for index, (inputs, outputs) in enumerate(batches):
                    started = time.perf_counter()
                    _probe_batch(variant, inputs)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    in_bytes = sum(payload_bytes(item) or 0 for item in inputs)
                    out_bytes = sum(payload_bytes(item) or 0 for item in outputs)
                    rows.append(
                        {
                            "probe": label,
                            "kind": kind,
                            "input_stage": input_stage,
                            "output_stage": output_stage,
                            "repeat": repeat,
                            "batch_index": index,
                            "source_records": len(inputs),
                            "in_bytes": in_bytes,
                            "out_bytes": out_bytes,
                            "total_bytes": in_bytes + out_bytes,
                            "service_ms": elapsed_ms,
                            "service_ms_per_record": elapsed_ms / len(inputs),
                        }
                    )
        finally:
            if variant is not None:
                try:
                    variant.shutdown()
                except Exception:  # noqa: BLE001
                    pass
    _write_csv(run_dir / "boundary_profile.csv", rows)
    params = {
        "fits": _fit_boundary(rows),
        "protocol": (
            "paired no-compute round trip on the real stage payloads; "
            "response buffers are preallocated real outputs"
        ),
        "train_validation_split": "model fitted on even batch_index, validated on odd",
    }
    (run_dir / "boundary_params.json").write_text(json.dumps(params, indent=1))
    print(f"probe rows={len(rows)}", flush=True)
    return 0


def _nnls_fit(rows):
    from scipy.optimize import nnls

    x = np.array([row["total_bytes"] for row in rows], dtype=float)
    y = np.array([row["service_ms"] for row in rows], dtype=float)
    coefficients, _ = nnls(np.column_stack([x, np.ones_like(x)]), y)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    predicted = slope * x + intercept
    residual = predicted - y
    return {
        "slope_ms_per_byte": slope,
        "throughput_bytes_per_sec": (1000.0 / slope) if slope > 0 else float("inf"),
        "fixed_ms": intercept,
        "r2": 1.0 - float(np.sum(residual ** 2)) / max(
            float(np.sum((y - y.mean()) ** 2)), 1e-12
        ),
        "mae_ms": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual) / np.maximum(y, 1e-9))),
        "rows": len(rows),
    }


def _fit_boundary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-probe fits on even batches, validated on the odd ones."""
    out: Dict[str, Any] = {}
    for probe in sorted({row["probe"] for row in rows}):
        subset = [row for row in rows if row["probe"] == probe]
        train = [row for row in subset if int(row["batch_index"]) % 2 == 0]
        validation = [row for row in subset if int(row["batch_index"]) % 2 == 1]
        train = train or subset
        fit = _nnls_fit(train)
        if validation:
            x = np.array([row["total_bytes"] for row in validation], dtype=float)
            y = np.array([row["service_ms"] for row in validation], dtype=float)
            predicted = fit["slope_ms_per_byte"] * x + fit["fixed_ms"]
            fit["validation_mae_ms"] = float(np.mean(np.abs(predicted - y)))
            fit["validation_mape"] = float(
                np.mean(np.abs(predicted - y) / np.maximum(y, 1e-9))
            )
            fit["validation_rows"] = len(validation)
        out[probe] = fit
    return out


# ---------------------------------------------------------------- validate --


def cmd_validate(args) -> int:
    ray, actor_classes, options = _setup(args)
    run_dir = Path(args.run_dir)
    inputs = Path(args.inputs)
    pool = list(torch.load(inputs / "inputs" / "block_inputs.pt")["records"])
    callables = dict(_block_callables(args.batch_size))
    runners: Dict[str, StageRunner] = {}
    rows: List[Dict[str, Any]] = []
    try:
        for organisation in ORGANISATIONS:
            runners[organisation] = StageRunner(
                organisation, callables, args.batch_size, options, actor_classes,
                args.remote_cpu, args.same_remote_cpu,
            )
        for instrument in ("off", "on"):
            for runner in runners.values():
                runner.set_instrument(instrument == "on")
            # warm up every organisation before measuring any of them
            for runner in runners.values():
                for index in range(args.warmup):
                    runner.run_batch(_batch_records(pool, index, args.batch_size))
            for repeat in range(args.rounds):
                order = list(args.order)
                if repeat % 2 == 1:
                    order = list(reversed(order))
                for organisation in order:
                    runner = runners[organisation]
                    for index in range(args.batches):
                        records = _batch_records(pool, index, args.batch_size)
                        runner.reset_events()
                        result = runner.run_batch(records)
                        if instrument == "on":
                            events = runner.member_events()
                            stage_compute = [
                                sum(
                                    event["wall_ms"]
                                    for event in events
                                    if event["stage_index"] == stage_index
                                )
                                for stage_index in range(len(runner.stages))
                            ]
                        else:
                            stage_compute = [float("nan")] * len(runner.stages)
                        rows.append(
                            {
                                "instrument": instrument,
                                "repeat": repeat,
                                "order_index": order.index(organisation),
                                "organisation": organisation,
                                "batch_id": index,
                                "source_records": args.batch_size,
                                "whole_T_ms": result["whole_time"] * 1000.0,
                                "whole_T_ms_per_record": (
                                    result["whole_time"] * 1000.0 / args.batch_size
                                ),
                                "stage_T_ms_per_record": json.dumps(
                                    [
                                        value * 1000.0 / args.batch_size
                                        for value in result["stage_times"]
                                    ]
                                ),
                                "stage_C_ms_per_record": json.dumps(
                                    [value / args.batch_size for value in stage_compute]
                                ),
                                "stage_in_bytes_per_record": json.dumps(
                                    [
                                        value[0] / args.batch_size
                                        for value in result["stage_bytes"]
                                    ]
                                ),
                                "stage_out_bytes_per_record": json.dumps(
                                    [
                                        value[1] / args.batch_size
                                        for value in result["stage_bytes"]
                                    ]
                                ),
                                "stages": json.dumps([list(g) for g in runner.stages]),
                            }
                        )
                print(
                    f"instrument={instrument} repeat={repeat} rows={len(rows)}",
                    flush=True,
                )
    finally:
        for runner in runners.values():
            runner.shutdown()
    _write_csv(run_dir / "measurements.csv", rows)
    print(f"validation rows={len(rows)}", flush=True)
    return 0


def _quantile(values, q):
    return float(np.quantile(values, q))


def cmd_summarise(args) -> int:
    run_dir = Path(args.run_dir)
    with (run_dir / "measurements.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    summary: List[Dict[str, Any]] = []
    for instrument in ("off", "on"):
        for organisation in sorted({row["organisation"] for row in rows}):
            subset = [
                float(row["whole_T_ms_per_record"])
                for row in rows
                if row["instrument"] == instrument
                and row["organisation"] == organisation
            ]
            if not subset:
                continue
            per_repeat: Dict[str, List[float]] = {}
            for row in rows:
                if (
                    row["instrument"] == instrument
                    and row["organisation"] == organisation
                ):
                    per_repeat.setdefault(row["repeat"], []).append(
                        float(row["whole_T_ms_per_record"])
                    )
            repeat_means = [statistics.fmean(v) for v in per_repeat.values()]
            summary.append(
                {
                    "instrument": instrument,
                    "organisation": organisation,
                    "batches": len(subset),
                    "mean_ms_per_record": statistics.fmean(subset),
                    "p10_ms_per_record": _quantile(subset, 0.10),
                    "median_ms_per_record": statistics.median(subset),
                    "p90_ms_per_record": _quantile(subset, 0.90),
                    "repeat_means": json.dumps(repeat_means),
                    "repeat_mean_stdev": (
                        statistics.pstdev(repeat_means) if len(repeat_means) > 1 else 0.0
                    ),
                }
            )
    _write_csv(run_dir / "summary.csv", summary)
    check = []
    for organisation in sorted({row["organisation"] for row in rows}):
        off = [
            float(row["whole_T_ms_per_record"])
            for row in rows
            if row["instrument"] == "off" and row["organisation"] == organisation
        ]
        on = [
            float(row["whole_T_ms_per_record"])
            for row in rows
            if row["instrument"] == "on" and row["organisation"] == organisation
        ]
        if not off or not on:
            continue
        off_mean = statistics.fmean(off)
        on_mean = statistics.fmean(on)
        check.append(
            {
                "organisation": organisation,
                "instrument_off_ms": off_mean,
                "instrument_on_ms": on_mean,
                "instrument_effect_pct": 100.0 * (on_mean - off_mean) / off_mean,
            }
        )
    _write_csv(run_dir / "instrumentation_check.csv", check)
    print(f"summary rows={len(summary)}", flush=True)
    return 0


def cmd_member_check(args) -> int:
    """Paired instrumented pass: per-member compute under U, P and F."""
    ray, actor_classes, options = _setup(args)
    run_dir = Path(args.run_dir)
    inputs = Path(args.inputs)
    pool = list(torch.load(inputs / "inputs" / "block_inputs.pt")["records"])
    callables = dict(_block_callables(args.batch_size))
    runners: Dict[str, StageRunner] = {}
    rows: List[Dict[str, Any]] = []
    try:
        for organisation in ORGANISATIONS:
            runners[organisation] = StageRunner(
                organisation, callables, args.batch_size, options, actor_classes,
                args.remote_cpu, args.same_remote_cpu,
            )
            runners[organisation].set_instrument(True)
        for runner in runners.values():
            for index in range(args.warmup):
                runner.run_batch(_batch_records(pool, index, args.batch_size))
        for repeat in range(args.rounds):
            order = list(args.order)
            if repeat % 2 == 1:
                order = list(reversed(order))
            for organisation in order:
                runner = runners[organisation]
                totals: Dict[str, float] = {name: 0.0 for name in runner.member_names}
                batches = 0
                for index in range(args.batches):
                    records = _batch_records(pool, index, args.batch_size)
                    runner.reset_events()
                    runner.run_batch(records)
                    for event in runner.member_events():
                        totals[event["operator"]] += event["wall_ms"]
                    batches += 1
                for member, total in totals.items():
                    rows.append(
                        {
                            "organisation": organisation,
                            "repeat": repeat,
                            "operator": member,
                            "batches": batches,
                            "ms_per_record": total / (batches * args.batch_size),
                        }
                    )
                print(
                    f"member-check {organisation} repeat={repeat} "
                    f"{ {k: round(v/(batches*args.batch_size), 3) for k, v in totals.items()} }",
                    flush=True,
                )
    finally:
        for runner in runners.values():
            runner.shutdown()
    _write_csv(run_dir / "member_compute.csv", rows)
    print(f"member-check rows={len(rows)}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--run-dir", required=True)
        p.add_argument("--inputs", default=str(DEFAULT_INPUTS))
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--local-cpu", type=int, default=12)
        p.add_argument("--remote-cpu", type=int, default=8)
        p.add_argument("--same-remote-cpu", action="store_true", default=True)
        p.add_argument("--ray-ip", default="172.23.166.105:6379")

    profile = sub.add_parser("stage-profile")
    common(profile)
    profile.add_argument("--warmup", type=int, default=20)
    profile.add_argument("--batches", type=int, default=120)
    profile.add_argument("--round", type=int, default=0)
    profile.add_argument("--snapshot-every", type=int, default=3)
    profile.add_argument("--snapshot-batches", type=int, default=40)
    profile.set_defaults(func=cmd_stage_profile)

    probe = sub.add_parser("boundary-probe")
    common(probe)
    probe.add_argument("--warmup", type=int, default=5)
    probe.add_argument("--repeats", type=int, default=3)
    probe.set_defaults(func=cmd_boundary_probe)

    validate = sub.add_parser("validate")
    common(validate)
    validate.add_argument("--warmup", type=int, default=15)
    validate.add_argument("--batches", type=int, default=40)
    validate.add_argument("--rounds", type=int, default=3)
    validate.add_argument("--order", nargs="+", default=["U", "P", "F"])
    validate.set_defaults(func=cmd_validate)

    summarise = sub.add_parser("summarise")
    summarise.add_argument("--run-dir", required=True)
    summarise.set_defaults(func=cmd_summarise)

    member = sub.add_parser("member-check")
    common(member)
    member.add_argument("--warmup", type=int, default=10)
    member.add_argument("--batches", type=int, default=40)
    member.add_argument("--rounds", type=int, default=2)
    member.add_argument("--order", nargs="+", default=["U", "P", "F"])
    member.set_defaults(func=cmd_member_check)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
