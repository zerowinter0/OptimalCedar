"""Platform calibration for serialized parallel-stage boundaries.

The optimizer charges an input/output transport term for every Ray or SMP
stage. This module measures that term with synchronous identity round trips
instead of relying on a machine-independent bandwidth constant. Reusing one
worker pool across payload sizes isolates serialized input/output transport
without conflating the measurement with pipeline-stage overlap.
"""

import logging
import math
import multiprocessing as mp
import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import ray

from cedar.config import CedarContext
from cedar.pipes import PipeVariantType


logger = logging.getLogger(__name__)


DEFAULT_PAYLOAD_BYTES: Tuple[int, ...] = (
    64 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
)
DEFAULT_REPETITIONS = 2
DEFAULT_WARMUP_SAMPLES = 20
DEFAULT_TARGET_BYTES = 128 * 1024 * 1024
MIN_MEASURED_SAMPLES = 32
MAX_MEASURED_SAMPLES = 512
MIN_ACCEPTED_R_SQUARED = 0.75


@ray.remote
class _BoundaryRayActor:
    def round_trip(self, value: Any) -> Any:
        return value


def _smp_round_trip_worker(requests: mp.Queue, responses: mp.Queue) -> None:
    while True:
        value = requests.get()
        if value is None:
            return
        responses.put(value)


class _RoundTripPool:
    def __init__(
        self,
        variant: PipeVariantType,
        width: int,
    ) -> None:
        self.variant = variant
        self.width = width
        self.ray_actors = []
        self.smp_requests = None
        self.smp_responses = None
        self.smp_workers = []
        if variant == PipeVariantType.RAY:
            self.ray_actors = [
                _BoundaryRayActor.remote() for _ in range(width)
            ]
        elif variant == PipeVariantType.SMP:
            self.smp_requests = mp.Queue(maxsize=max(2 * width, 16))
            self.smp_responses = mp.Queue(maxsize=max(2 * width, 16))
            self.smp_workers = [
                mp.Process(
                    target=_smp_round_trip_worker,
                    args=(self.smp_requests, self.smp_responses),
                    daemon=True,
                )
                for _ in range(width)
            ]
            for worker in self.smp_workers:
                worker.start()
        else:
            raise ValueError(
                f"Unsupported boundary calibration variant: {variant}"
            )

    def round_trip(self, payload: Any) -> int:
        if self.variant == PipeVariantType.RAY:
            values = ray.get(
                [
                    actor.round_trip.remote(payload)
                    for actor in self.ray_actors
                ]
            )
            return len(values)
        for _ in range(self.width):
            self.smp_requests.put(payload)
        for _ in range(self.width):
            self.smp_responses.get()
        return self.width

    def shutdown(self) -> None:
        for actor in self.ray_actors:
            ray.kill(actor)
        self.ray_actors = []
        if self.smp_requests is not None:
            for _ in self.smp_workers:
                self.smp_requests.put(None)
        for worker in self.smp_workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)
        self.smp_workers = []
        for queue in (self.smp_requests, self.smp_responses):
            if queue is not None:
                queue.close()
                queue.join_thread()
        self.smp_requests = None
        self.smp_responses = None


def _sample_count(payload_bytes: int, target_bytes: int) -> int:
    return max(
        MIN_MEASURED_SAMPLES,
        min(MAX_MEASURED_SAMPLES, target_bytes // payload_bytes),
    )


def _measure_round_trip(
    pool: _RoundTripPool,
    payload_bytes: int,
    warmup_samples: int,
    measured_samples: int,
) -> float:
    """Return effective synchronous round-trip seconds per sample."""

    payload = np.zeros(payload_bytes, dtype=np.uint8)
    warmup_rounds = max(1, math.ceil(warmup_samples / pool.width))
    measured_rounds = max(1, math.ceil(measured_samples / pool.width))
    for _ in range(warmup_rounds):
        pool.round_trip(payload)
    started = time.perf_counter()
    count = 0
    for _ in range(measured_rounds):
        count += pool.round_trip(payload)
    return (time.perf_counter() - started) / count


def _validate_variant_ready(
    ctx: CedarContext,
    variant: PipeVariantType,
) -> None:
    if variant == PipeVariantType.RAY:
        if not ctx.use_ray() or not ray.is_initialized():
            raise RuntimeError(
                "Ray boundary calibration requires an initialized Ray context."
            )
        return
    if variant != PipeVariantType.SMP:
        raise ValueError(f"Unsupported boundary calibration variant: {variant}")


def fit_boundary_model(
    measurements: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Fit delta_sec = latency_sec + boundary_bytes / throughput."""

    points = []
    for measurement in measurements:
        payload = float(measurement["payload_bytes"])
        delta = float(measurement["boundary_sec_per_sample"])
        # Keep every finite observation in the joint slope fit; the positive
        # slope and fit-quality checks below reject unusable measurements.
        if payload > 0 and math.isfinite(delta):
            points.append((2.0 * payload, delta))
    if len(points) < 2:
        raise RuntimeError("Need at least two finite boundary measurements.")

    mean_x = statistics.mean(x for x, _ in points)
    mean_y = statistics.mean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0:
        raise RuntimeError("Boundary calibration payload sizes are degenerate.")
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in points
    ) / denominator
    if not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("Boundary calibration produced non-positive bandwidth.")

    latency_sec = max(0.0, mean_y - slope * mean_x)
    predictions = [latency_sec + slope * x for x, _ in points]
    residuals = [
        observed - predicted
        for (_, observed), predicted in zip(points, predictions)
    ]
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((observed - mean_y) ** 2 for _, observed in points)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "fixed_latency_ms": latency_sec * 1000.0,
        "throughput_bytes_per_sec": 1.0 / slope,
        "r_squared": r_squared,
    }


def profile_stage_boundary(
    ctx: CedarContext,
    variant: PipeVariantType,
    width: int,
    payload_bytes: Iterable[int] = DEFAULT_PAYLOAD_BYTES,
    repetitions: int = DEFAULT_REPETITIONS,
    warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
    target_bytes: int = DEFAULT_TARGET_BYTES,
) -> Dict[str, Any]:
    """Measure and fit one platform/backend boundary model."""

    if width < 1:
        raise ValueError(f"Boundary calibration width must be positive: {width}")
    _validate_variant_ready(ctx, variant)
    payloads = tuple(sorted(set(int(value) for value in payload_bytes)))
    if len(payloads) < 2 or any(value <= 0 for value in payloads):
        raise ValueError("Boundary calibration needs at least two payload sizes.")
    if repetitions < 1:
        raise ValueError("Boundary calibration repetitions must be positive.")

    raw_runs: List[Dict[str, Any]] = []
    measurements: List[Dict[str, float]] = []
    pool = _RoundTripPool(variant=variant, width=width)
    try:
        for payload in payloads:
            samples = _sample_count(payload, target_bytes)
            values = []
            for repeat in range(repetitions):
                value = _measure_round_trip(
                    pool=pool,
                    payload_bytes=payload,
                    warmup_samples=warmup_samples,
                    measured_samples=samples,
                )
                values.append(value)
                raw_runs.append(
                    {
                        "payload_bytes": payload,
                        "repeat": repeat + 1,
                        "samples": samples,
                        "boundary_sec_per_sample": value,
                    }
                )
            measurements.append(
                {
                    "payload_bytes": float(payload),
                    "boundary_sec_per_sample": statistics.median(values),
                }
            )
    finally:
        pool.shutdown()

    model = fit_boundary_model(measurements)
    if model["r_squared"] < MIN_ACCEPTED_R_SQUARED:
        raise RuntimeError(
            "Boundary calibration fit is too noisy: "
            f"r_squared={model['r_squared']:.4f}, "
            f"required={MIN_ACCEPTED_R_SQUARED:.2f}"
        )
    model.update(
        {
            "method": "synchronous_round_trip_linear_fit",
            "variant": variant.name,
            "stage_width": width,
            "payload_bytes": list(payloads),
            "repetitions": repetitions,
            "warmup_samples": warmup_samples,
            "measurements": measurements,
            "runs": raw_runs,
        }
    )
    logger.info("Profiled %s boundary model: %s", variant.name, model)
    return model
