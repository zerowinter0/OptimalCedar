"""Bounded local image-size sweeps over snapshots of legal intermediate data."""
import copy
from contextlib import contextmanager
import math
import random
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from cedar.pipes.common import get_sizeof_data


@contextmanager
def preserved_rng():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(731)
        np.random.seed(731)
        torch.random.default_generator.manual_seed(731)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)


def image_of(value, field=None):
    value = value[field] if field is not None and isinstance(value, dict) else value
    if isinstance(value, Image.Image):
        return value
    if (isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.ndim == 3 and value.shape[0] in (1, 3, 4)
            and value.dtype in (torch.uint8, torch.float32, torch.float64)):
        return value
    return None


def area(value):
    if isinstance(value, Image.Image):
        return value.width * value.height
    return int(value.shape[-1] * value.shape[-2])


def resized_record(value, pixels, field=None):
    image = image_of(value, field)
    if image is None:
        raise TypeError("No supported CPU image input")
    h, w = (image.height, image.width) if isinstance(image, Image.Image) else image.shape[-2:]
    ratio = math.sqrt(pixels / (h * w))
    h, w = max(1, round(h * ratio)), max(1, round(w * ratio))
    resized = (image.resize((w, h), Image.Resampling.BILINEAR)
               if isinstance(image, Image.Image) else TF.resize(image, [h, w], antialias=True))
    if field is None:
        return resized
    result = copy.deepcopy(value)
    result[field] = resized
    return result


class SnapshotPool:
    """Uniform reservoir within explicit record and aggregate memory limits."""
    def __init__(self, count=6, byte_limit=32 * 1024**2, shared=None):
        self.shared = shared if shared is not None else [0, 64 * 1024**2]
        self.count, self.byte_limit = count, byte_limit
        self.values, self.sizes = [], []
        self.seen = 0
        self.rng = random.Random(929)
        self.skipped = 0

    def capture(self, value):
        self.seen += 1
        slot = len(self.values) if len(self.values) < self.count else self.rng.randrange(self.seen)
        if slot >= self.count:
            return
        size = get_sizeof_data(value)
        previous = self.sizes[slot] if slot < len(self.values) else 0
        if (size > self.byte_limit or sum(self.sizes) - previous + size > self.byte_limit
                or self.shared[0] - previous + size > self.shared[1]):
            self.skipped += 1
            return
        try:
            snapshot = copy.deepcopy(value)
        except Exception:
            self.skipped += 1
            return
        self.shared[0] += size - previous
        if slot < len(self.values):
            self.values[slot], self.sizes[slot] = snapshot, size
        else:
            self.values.append(snapshot)
            self.sizes.append(size)


def size_targets(images, reference_areas):
    observed = [area(x) for x in images]
    typical = float(np.median(observed))
    # Cover natural shapes and upstream image sizes without allowing outliers
    # to dictate the entire fit; caps are recorded in the returned metadata.
    refs = reference_areas or observed
    upper = min(2048**2, max(typical * 4, float(max(refs))))
    lower = min(64**2, typical)
    train = sorted(set(round(x) for x in (
        list(np.linspace(lower, upper, 10)) + [typical, max(lower, typical / 2),
                                              min(upper, lower * 2)])))
    validate = [(train[i] + train[i + 1]) // 2 for i in (1, len(train)//2, len(train)-2)]
    return train, validate


def validated_fit(rows, natural, fit_affine):
    train = [r for r in rows if r["split"] == "train"]
    # Equal weight per size, not per timing batch; extra rounds reduce noise.
    points = {}
    for row in train:
        points.setdefault(row["target_pixels"], []).append(row)
    observations = [(np.mean([r["input_bytes"] for r in group]),
                     np.mean([r["mean_ns"] for r in group]))
                    for group in points.values()]
    fit = fit_affine(observations, minimum_observations=3, binning=False)
    validation = [r for r in rows if r["split"] == "validation"]
    if "k" not in fit or not validation:
        return None
    grouped = {}
    for r in validation:
        grouped.setdefault(r["target_pixels"], []).append(r)
    errors = []
    for group in grouped.values():
        x = float(np.mean([r["input_bytes"] for r in group]))
        actual = float(np.mean([r["mean_ns"] for r in group])) / 1e6
        errors.append(abs(fit["k"]*x + fit["b"] - actual) / max(actual, 1e-9))
    x = np.array([a for a, _ in observations], dtype=float)
    y = np.array([b / 1e6 for _, b in observations], dtype=float)
    slope, intercept = np.linalg.lstsq(np.column_stack([x/x.max(), np.ones(len(x))]), y, rcond=None)[0]
    fit["unconstrained_fit"] = {"k": float(slope/x.max()), "b": float(intercept)}
    fit["validation"] = dict(mean_relative_error=float(np.mean(errors)),
                             max_relative_error=float(max(errors)),
                             held_out_sizes=len(errors))
    fit["quality"] = "validated" if max(errors) <= .30 else "low_confidence"
    fit["observations"] = rows
    fit["natural_fit"] = {k: v for k, v in natural.items() if k not in ("observations",)}
    # Propagation uses untouched pipeline statistics, never synthetic outputs.
    for key in ("input_count", "output_count", "selectivity", "input_mean_bytes", "output_mean_bytes"):
        if key in natural:
            fit[key] = natural[key]
    fit["method"] = "local_controlled_image_sweep"
    fit["input_policy"] = "structure_preserving_counterfactual"
    fit["extrapolation"] = "clamp_to_measured_range"
    fit["weighting"] = "equal_per_size_preserving_small_input_points"
    return fit


def sweep_callable(fn, samples, natural, fit_affine, reference_areas=(),
                   budget_sec=10.0, records_per_call=1, field=None, timer=None):
    timer = timer or (lambda fn, value: _time_call(fn, value))
    images = [image_of(v, field) for v in samples]
    if not images or any(v is None for v in images):
        return dict(natural, sweep={"status": "unsupported_input"})
    try:
        fn = copy.deepcopy(fn)
    except Exception as exc:
        return dict(natural, sweep={"status": "uncopyable_callable", "reason": str(exc)})
    train, validation = size_targets(images, reference_areas)
    targets = [("train", p) for p in train] + [("validation", p) for p in validation]
    rows = []
    start = time.monotonic()
    planner = random.Random(739)
    with preserved_rng():
        try:
            for round_idx in range(6):
                shuffled = list(targets)
                planner.shuffle(shuffled)
                for split, pixels in shuffled:
                    if time.monotonic() - start > budget_sec:
                        raise TimeoutError("sweep_time_budget")
                    # Rotate original image content; never mutate the captured input.
                    prepared = [resized_record(v, pixels, field) for v in samples]
                    fn(copy.deepcopy(prepared[round_idx % len(prepared)]))  # warmup
                    total, sizes, calls = 0, 0, 0
                    while calls < 4 or (total < 20e6 and calls < 256):
                        value = copy.deepcopy(prepared[(calls+round_idx) % len(prepared)])
                        sizes += get_sizeof_data(value)
                        total += timer(fn, value)
                        calls += 1
                        if time.monotonic() - start > budget_sec:
                            raise TimeoutError("sweep_time_budget")
                    rows.append(dict(split=split, target_pixels=pixels, round=round_idx+1,
                                     input_bytes=sizes/calls, calls=calls,
                                     records_per_call=records_per_call,
                                     mean_ns=total/calls/records_per_call))
                if round_idx >= 2:
                    cvs = []
                    for split, pixels in targets:
                        ys = [r["mean_ns"] for r in rows
                              if r["split"] == split and r["target_pixels"] == pixels]
                        cvs.append(float(np.std(ys, ddof=1)/max(np.mean(ys), 1)))
                    if max(cvs) <= .15:
                        break
        except (TimeoutError, RuntimeError, ValueError, TypeError) as exc:
            counts = [sum(r["split"] == s and r["target_pixels"] == p for r in rows)
                      for s, p in targets]
            if min(counts, default=0) < 3:
                return dict(natural, sweep=dict(status="incomplete", reason=str(exc),
                            observations=rows, elapsed_sec=time.monotonic()-start))
            # Full three-round evidence remains usable if extra noise-reduction
            # rounds hit the budget; it is retained with its validation score.
    fitted = validated_fit(rows, natural, fit_affine)
    if fitted is None:
        return dict(natural, sweep={"status": "insufficient_data", "observations": rows})
    cvs = []
    for split, pixels in targets:
        ys = [r["mean_ns"] for r in rows
              if r["split"] == split and r["target_pixels"] == pixels]
        cvs.append(float(np.std(ys, ddof=1)/max(np.mean(ys), 1)))
    fitted["max_timing_cv"] = max(cvs)
    if max(cvs) > .15:
        fitted["quality"] = "low_confidence"
    fitted["sweep"] = dict(status="completed", elapsed_sec=time.monotonic()-start,
                          train_pixels=train, validation_pixels=validation,
                          size_cap_pixels=2048**2, budget_sec=budget_sec)
    return fitted


def _time_call(fn, value):
    start = time.perf_counter_ns()
    result = fn(value)
    elapsed = time.perf_counter_ns() - start
    del result
    return elapsed
