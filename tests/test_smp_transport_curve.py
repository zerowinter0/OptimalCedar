import pytest
from cedar.client.boundary_profiler import smp_aggregate_throughput

def curve():
    return {"max_inflight": 10, "points": [
        {"workers": 1, "throughput_bytes_per_sec": 100},
        {"workers": 16, "throughput_bytes_per_sec": 1200},
        {"workers": 32, "throughput_bytes_per_sec": 1000}]}

def test_interpolation_preserves_saturation_and_decline():
    assert smp_aggregate_throughput(curve(), 24, 10) == 1100
    assert smp_aggregate_throughput(curve(), 32, 10) == 1000

def test_rejects_extrapolation_and_queue_mismatch():
    with pytest.raises(ValueError):
        smp_aggregate_throughput(curve(), 33, 10)
    with pytest.raises(ValueError):
        smp_aggregate_throughput(curve(), 8, 1)
