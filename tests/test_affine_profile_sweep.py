import copy
import random

import numpy as np
import pytest
import torch

from cedar.client.affine_profile_sweep import SnapshotPool, resized_record, sweep_callable
from cedar.client.linear_cost_profile import fit_affine, NativeBatchCall
from cedar.pipes.common import get_sizeof_data


def test_resize_preserves_record_dtype_channels_and_aspect():
    value = {"image": torch.ones(3, 16, 32, dtype=torch.uint8), "caption": "keep me"}
    result = resized_record(value, 2048, "image")
    assert result["image"].shape == (3, 32, 64)
    assert result["image"].dtype == torch.uint8
    assert result["caption"] == value["caption"]
    assert value["image"].shape == (3, 16, 32)


def test_snapshot_is_immutable_and_globally_bounded():
    shared = [0, 1024]
    a, b = SnapshotPool(shared=shared), SnapshotPool(shared=shared)
    x = torch.zeros(3, 8, 8)
    a.capture(x)
    x.add_(1)
    b.capture(x)
    assert a.values[0].sum() == 0
    assert shared[0] <= shared[1]
    assert b.skipped == 1


def test_sweep_recovers_known_curve_and_restores_rng():
    torch.set_num_threads(1)
    value = torch.ones(3, 16, 32)
    natural = {"k": 0, "b": 3, "selectivity": .3,
               "input_mean_bytes": 123, "output_mean_bytes": 45}
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    def operation(x):
        random.random()
        np.random.random()
        torch.rand(1)
        return x.add_(1)
    def timer(fn, x):
        size = get_sizeof_data(x)
        fn(x)
        return (size * .002 + 2) * 1e6
    result = sweep_callable(operation, [value], natural, fit_affine,
                            budget_sec=10, timer=timer)
    assert result["k"] == pytest.approx(.002, rel=.01)
    assert result["b"] == pytest.approx(2, rel=.01)
    assert result["validation"]["held_out_sizes"] == 3
    assert result["quality"] == "validated"
    assert result["selectivity"] == .3
    assert result["input_mean_bytes"] == 123
    assert result["output_mean_bytes"] == 45
    assert torch.all(value == 1)
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)


def test_unsupported_input_keeps_natural_model():
    result = sweep_callable(lambda x: x, ["text"], {"k": 0, "b": 2}, fit_affine)
    assert result["k"] == 0 and result["b"] == 2
    assert result["sweep"]["status"] == "unsupported_input"


def test_native_batch_uses_configured_batch_size():
    value = torch.ones(3, 8, 8)
    assert NativeBatchCall(1, False)(value).shape == (3, 8, 8)
    assert NativeBatchCall(7, False)(value).shape == (7, 3, 8, 8)


def test_incomplete_sweep_retains_natural_estimate():
    result = sweep_callable(lambda x: x, [torch.ones(3, 8, 8)],
                            {"k": 0, "b": 2}, fit_affine, budget_sec=1e-12)
    assert result["b"] == 2
    assert result["sweep"]["status"] == "incomplete"

def test_new_models_clamp_extrapolation_but_old_models_do_not():
    from cedar.compose.affine_cost_utils import affine_value
    model = {"k": 2, "b": 3, "input_min_bytes": 10, "input_max_bytes": 100}
    assert affine_value(model, 1000) == 2003
    model["extrapolation"] = "clamp_to_measured_range"
    assert affine_value(model, 1000) == 203
    assert affine_value(model, 1) == 23
    assert affine_value(model, 50) == 103


def test_validation_marks_nonlinear_curve_low_confidence():
    from cedar.client.affine_profile_sweep import validated_fit
    rows = []
    for split, points in (("train", range(1, 14)), ("validation", (.5, 6.5, 13.5))):
        for x in points:
            for repeat in range(3):
                rows.append(dict(split=split, target_pixels=x, input_bytes=x,
                                 mean_ns=x*x*1e6, round=repeat+1))
    model = validated_fit(rows, {"selectivity": .5}, fit_affine)
    assert model["quality"] == "low_confidence"
    assert model["validation"]["max_relative_error"] > .3
    assert model["selectivity"] == .5
