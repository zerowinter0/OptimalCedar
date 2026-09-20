import math

import pytest

from cedar.compose.my_optimizer import MyOptimizer, fit_width_curve
from cedar.pipes import PipeVariantType


def _entry(measured):
    return {"widths": {width: _converged(width, mean) for width, mean in measured.items()}}


def _converged(width, mean_ms):
    return {
        "width": width,
        "mean_ms_per_sample": mean_ms,
        "adaptive_profile": {"width": width, "converged": True},
    }


def test_fit_needs_at_least_two_widths():
    assert fit_width_curve({1: 10.0}) is None
    assert fit_width_curve({}) is None
    assert fit_width_curve({2: 5.0, 4: 2.5}) is not None


def test_fit_recovers_a_power_law():
    scale, exponent = 20.0, 0.7
    measured = {
        width: scale / width ** exponent for width in (1, 2, 4, 8)
    }

    fit = fit_width_curve(measured)

    assert fit["exponent"] == pytest.approx(exponent, rel=1e-6)
    assert fit["scale_ms"] == pytest.approx(scale, rel=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert fit["monotone_adjusted"] is False


def test_fit_makes_non_monotone_measurements_order_preserving():
    fit = fit_width_curve({1: 10.0, 2: 9.0, 4: 13.0, 8: 8.0})

    assert fit["monotone_adjusted"] is True
    values = [value for _, value in fit["points"]]
    assert values == sorted(values, reverse=True)
    assert fit["exponent"] >= 0.0


def test_measured_points_win_and_unmeasured_widths_use_the_fit():
    optimizer = MyOptimizer()
    entry = _entry({1: 20.0, 2: 12.29, 4: 7.56, 8: 4.65})

    measured = optimizer._dp_scaling_cost(
        entry, 7, PipeVariantType.RAY, 2, "mean_ms_per_sample"
    )
    assert measured == pytest.approx(12.29)

    fitted = optimizer._dp_scaling_cost(
        entry, 7, PipeVariantType.RAY, 6, "mean_ms_per_sample"
    )
    expected = 20.0 / 6 ** 0.7
    assert fitted == pytest.approx(expected, rel=0.05)


def test_extrapolation_beyond_the_measured_range_keeps_the_measured_anchor():
    optimizer = MyOptimizer()
    entry = _entry({1: 20.0, 2: 12.29, 4: 7.56, 8: 4.65})

    far = optimizer._dp_scaling_cost(
        entry, 7, PipeVariantType.RAY, 64, "mean_ms_per_sample"
    )
    anchor = optimizer._dp_scaling_cost(
        entry, 7, PipeVariantType.RAY, 8, "mean_ms_per_sample"
    )

    assert far == pytest.approx(anchor)
    assert far > 20.0 / 64 ** 0.7
