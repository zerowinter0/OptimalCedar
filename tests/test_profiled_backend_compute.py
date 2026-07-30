import math

import pytest

from cedar.compose.my_optimizer import MyOptimizer
from cedar.pipes import PipeVariantType
from cedar.service.smp import SMPService


class _Response:
    def __init__(self, elapsed_ns):
        self.backend_compute_ns = elapsed_ns


class _Queue:
    def __init__(self, values):
        self.values = iter(values)

    def get(self, **kwargs):
        return next(self.values)


def test_smp_service_reports_worker_compute_uncertainty():
    service = SMPService()
    service.resp_q = _Queue([_Response(1_000_000), _Response(3_000_000)])
    service.next()
    service.next()

    stats = service.get_backend_compute_stats()
    assert stats["count"] == 2
    assert stats["mean_ms_per_sample"] == pytest.approx(2.0)
    assert stats["stddev_ms_per_sample"] == pytest.approx(math.sqrt(2.0))
    assert stats["stderr_ms_per_sample"] == pytest.approx(1.0)


def test_boundary_cost_amortizes_fixed_latency_by_profile_inflight():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "resource_config": {"smp_max_inflight": 100},
        "physical_model": {
            "boundary": {
                "SMP": {
                    "fixed_latency_ms": 2.0,
                    "throughput_bytes_per_sec": 1_000_000_000.0,
                }
            }
        }
    }

    assert optimizer._dp_boundary_cost_ms(
        PipeVariantType.SMP, input_size=100.0, output_size=100.0
    ) == pytest.approx(0.0202)
    assert optimizer._dp_boundary_cost_ms(
        PipeVariantType.SMP,
        input_size=2_000_000.0,
        output_size=2_000_000.0,
    ) == pytest.approx(4.02)


def test_boundary_cost_uses_profile_inflight_default():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "physical_model": {
            "boundary": {
                "SMP": {
                    "fixed_latency_ms": 2.0,
                    "throughput_bytes_per_sec": 1_000_000_000.0,
                }
            }
        }
    }

    assert optimizer._dp_boundary_cost_ms(
        PipeVariantType.SMP, input_size=100.0, output_size=100.0
    ) == pytest.approx(0.0202)
