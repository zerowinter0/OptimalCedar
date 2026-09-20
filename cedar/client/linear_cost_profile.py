"""Natural-input statistics plus controlled local affine latency sweeps."""
import math
import os
import random
import time

import numpy as np

from cedar.client.profiler import FeatureProfiler
from cedar.pipes import MapperPipe, FilterPipe, BatcherPipe, ImageReaderPipe
from cedar.client.affine_profile_sweep import SnapshotPool, image_of, area, sweep_callable
from cedar.pipes.common import get_sizeof_data


def fit_affine(observations, minimum_observations=12, binning=True):
    """Fit nonnegative k,b in ms/record = k * bytes/record + b.

    Equal-width size bins prevent dense size regions dominating the fit.
    No slope is inferred from constant-size or insufficient observations.
    """
    values = np.asarray([(x, ns / 1e6) for x, ns in observations
                         if math.isfinite(x) and math.isfinite(ns)
                         and x > 0 and ns >= 0], dtype=float).reshape(-1, 2)
    if not len(values):
        return {"status": "unavailable", "reason": "no_observations",
                "n_observations": 0}
    x, y = values.T
    result = {"n_observations": len(values), "input_min_bytes": float(x.min()),
              "input_max_bytes": float(x.max()), "mean_ms": float(y.mean()),
              "input_mean_bytes": float(x.mean())}
    if len(values) < minimum_observations or x.max() <= x.min() * 1.05 or len(np.unique(x)) < 3:
        return dict(result, status="constant_fallback", k=0.0, b=float(y.mean()),
                    reason="insufficient_observations_or_size_variation")
    edges = np.linspace(x.min(), x.max(), 13)
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 11)
    points = np.array([(x[bins == i].mean(), y[bins == i].mean())
                       for i in range(12) if (bins == i).any()])
    if not binning:
        points = values
    bx, by = points.T
    # Solve in scaled coordinates for numerical stability, including NNLS
    # boundary solutions instead of independently clipping OLS coefficients.
    scale = float(bx.max())
    z = bx / scale
    slope, intercept = np.linalg.lstsq(
        np.column_stack((z, np.ones(len(z)))), by, rcond=None)[0]
    candidates = [(0.0, max(0.0, float(by.mean()))),
                  (max(0.0, float(z @ by / (z @ z))), 0.0)]
    if slope >= 0 and intercept >= 0:
        candidates.append((float(slope), float(intercept)))
    slope, intercept = min(candidates,
        key=lambda pair: float(np.sum((pair[0] * z + pair[1] - by) ** 2)))
    prediction = slope * z + intercept
    residual = float(np.sum((prediction - by) ** 2))
    total = float(np.sum((by - by.mean()) ** 2))
    return dict(result, status="fitted", k=slope / scale, b=intercept,
                r_squared=1.0-residual/total if total > 0 else 1.0,
                points=[{"input_bytes": float(a), "mean_ms": float(b)}
                        for a, b in points], reason="nonnegative_affine_least_squares")


class TimedCallable:
    def __init__(self, fn, is_filter=False, capacity=4096, shared=None, capture_output=False):
        self.samples = SnapshotPool(shared=shared)
        self.outputs = SnapshotPool(shared=shared) if capture_output else None
        self.field = getattr(fn, "field", None)
        self.fn, self.is_filter, self.capacity = fn, is_filter, capacity
        self.observations = []
        self.seen = self.passed = 0
        self.input_sum = self.output_sum = 0.0
        self.rng = random.Random(731)

    def __call__(self, *args, **kwargs):
        value = args[0] if len(args) == 1 else args
        size = float(get_sizeof_data(value))
        if image_of(value, self.field) is not None:
            self.samples.capture(value)
        start = time.perf_counter_ns()
        result = self.fn(*args, **kwargs)
        elapsed = time.perf_counter_ns() - start
        if self.outputs is not None and image_of(result) is not None:
            self.outputs.capture(result)
        self.seen += 1
        self.input_sum += size
        passed = bool(result) if self.is_filter else True
        if passed:
            self.passed += 1
            self.output_sum += float(get_sizeof_data(value if self.is_filter else result))
        observation = (size, float(elapsed))
        if len(self.observations) < self.capacity:
            self.observations.append(observation)
        else:
            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.observations[index] = observation
        return result

    def model(self):
        result = fit_affine(self.observations)
        result.update(method="local_callable_wall_clock", input_count=self.seen,
                      output_count=self.passed,
                      selectivity=self.passed/self.seen if self.seen else 1.0)
        if self.seen:
            result["input_mean_bytes"] = self.input_sum/self.seen
        if self.passed:
            result["output_mean_bytes"] = self.output_sum/self.passed
        return result


class NativeBatchCall:
    def __init__(self, batch_size, drop_last):
        self.batch_size, self.drop_last = batch_size, drop_last

    def __call__(self, value):
        from cedar.pipes.batch import InProcessBatcherPipeVariant
        from cedar.pipes.common import DataSample
        variant = InProcessBatcherPipeVariant(None, self.batch_size, self.drop_last)
        variant._input_iter = iter(DataSample(value) for _ in range(self.batch_size))
        return next(variant._iter_impl()).data


def profile_linear_feature(feature, ctx, duration_sec=10.0, n_samples=None):
    """Separate pass: sizing/timing instrumentation never changes baseline throughput."""
    original, wrappers = {}, {}
    shared = [0, 64 * 1024**2]
    batch_inputs = {p.input_pipes[0].id for p in feature.logical_pipes.values()
                    if isinstance(p, BatcherPipe)}
    try:
        for pid, pipe in feature.logical_pipes.items():
            if isinstance(pipe, (MapperPipe, FilterPipe)):
                original[pid] = pipe.fn
                wrappers[pid] = TimedCallable(pipe.fn, isinstance(pipe, FilterPipe),
                    shared=shared, capture_output=pid in batch_inputs)
                pipe.fn = wrappers[pid]
        loaded = feature.profile(ctx, None)
        # Materialized variants retain wrappers; keep logical callables clean.
        for pid, fn in original.items():
            feature.logical_pipes[pid].fn = fn
        profiler = FeatureProfiler(feature, profile_mode=True)
        start, count = time.monotonic(), 0
        for sample in loaded:
            if sample.dummy:
                continue
            profiler.update_ds(sample)
            count += feature.get_batch_size()
            if (n_samples is not None and count >= n_samples) or (
                    n_samples is None and time.monotonic()-start >= duration_sec):
                break
        natural_elapsed = time.monotonic()-start
        sweep_budget = float(os.environ.get("CEDAR_CM_SWEEP_TIME_SEC", str(duration_sec)))
        if not math.isfinite(sweep_budget) or sweep_budget <= 0:
            raise ValueError("CEDAR_CM_SWEEP_TIME_SEC must be finite and positive")
        reference_areas = [float(np.quantile([area(image_of(v, w.field))
                           for v in w.samples.values], .95)) for w in wrappers.values()
                           if w.samples.values]
        models = {}
        for pid, pipe in feature.logical_pipes.items():
            if pid in wrappers:
                wrapper = wrappers[pid]
                models[pid] = sweep_callable(wrapper.fn, wrapper.samples.values,
                    wrapper.model(), fit_affine, reference_areas,
                    budget_sec=sweep_budget, field=wrapper.field)
            else:
                models[pid] = fit_affine(
                    profiler.natural_observations.get(pid, []))
                models[pid]["method"] = "local_trace_wall_clock"
            if isinstance(pipe, ImageReaderPipe) and "mean_ms" in models[pid]:
                # A pathname's in-memory bytes do not measure decoder work.
                models[pid].update(k=0.0, b=models[pid]["mean_ms"],
                    status="constant_fallback", reason="file_bytes_not_path_object_required")
                models[pid]["sweep"] = {"status": "unsupported_input",
                    "reason": "Reader needs file-format and decoded-pixel axes"}
            if isinstance(pipe, BatcherPipe):
                upstream = wrappers.get(pipe.input_pipes[0].id)
                samples = upstream.outputs.values if upstream and upstream.outputs else []
                if pipe.batch_size > 1 and samples:
                    models[pid] = sweep_callable(
                        NativeBatchCall(pipe.batch_size, pipe.drop_last), samples,
                        models[pid], fit_affine, reference_areas,
                        budget_sec=sweep_budget, records_per_call=pipe.batch_size)
                else:
                    models[pid]["sweep"] = {"status": "constant_batch_passthrough"
                        if pipe.batch_size == 1 else "no_input_snapshot"}
                models[pid]["batch_size"] = pipe.batch_size
            models[pid]["pipe_name"] = pipe.get_logical_name()
            models[pid]["tag"] = pipe.tag
        return {"schema_version": 1, "equation": "ms_per_record = k * input_bytes + b",
                "input_scope": "whole_record", "input_policy": "natural_capture_plus_controlled_sweep",
                "natural_duration_sec": natural_elapsed, "snapshot_bytes": shared[0],
                "profile_backend": "INPROCESS", "duration_sec": time.monotonic()-start,
                "source_samples": count, "operators": models}
    finally:
        for pid, fn in original.items():
            feature.logical_pipes[pid].fn = fn
        try:
            if feature.loaded:
                feature.reset()
        finally:
            feature.release_profile_resources()
