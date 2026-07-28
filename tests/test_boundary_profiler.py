import pytest

from cedar.client import boundary_profiler
from cedar.client.boundary_profiler import (
    fit_boundary_model,
    profile_stage_boundary_cached,
)
from cedar.pipes import PipeVariantType


def test_fit_boundary_model_recovers_latency_and_bandwidth():
    latency_sec = 0.00025
    throughput = 2_000_000_000.0
    measurements = [
        {
            "payload_bytes": float(payload),
            "boundary_sec_per_sample": (
                latency_sec + 2.0 * payload / throughput
            ),
        }
        for payload in (64 * 1024, 1024 * 1024, 4 * 1024 * 1024)
    ]

    model = fit_boundary_model(measurements)

    assert model["fixed_latency_ms"] == pytest.approx(0.25)
    assert model["throughput_bytes_per_sec"] == pytest.approx(throughput)
    assert model["r_squared"] == pytest.approx(1.0)


def test_fit_boundary_model_rejects_degenerate_payload_sizes():
    with pytest.raises(RuntimeError, match="payload sizes are degenerate"):
        fit_boundary_model(
            [
                {
                    "payload_bytes": 1024.0,
                    "boundary_sec_per_sample": 0.001,
                },
                {
                    "payload_bytes": 1024.0,
                    "boundary_sec_per_sample": 0.002,
                },
            ]
        )


def test_boundary_calibration_reuses_exact_signature(tmp_path, monkeypatch):
    calls = []

    def fake_profile(**kwargs):
        calls.append(kwargs)
        return {
            "throughput_bytes_per_sec": 123.0,
            "fixed_latency_ms": 0.5,
        }

    monkeypatch.setattr(
        boundary_profiler, "profile_stage_boundary", fake_profile
    )
    cache = tmp_path / "boundary.json"
    first = profile_stage_boundary_cached(
        ctx=object(),
        variant=PipeVariantType.SMP,
        width=3,
        cache_path=str(cache),
    )
    second = profile_stage_boundary_cached(
        ctx=object(),
        variant=PipeVariantType.SMP,
        width=3,
        cache_path=str(cache),
    )

    assert len(calls) == 1
    assert first["calibration_source"] == "measured"
    assert second["calibration_source"] == "cache"
    assert second["calibration_key"] == first["calibration_key"]
