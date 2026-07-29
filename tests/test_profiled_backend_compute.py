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


def test_optimizer_preserves_profiled_singleton_total_cost():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {
            "input_sizes": {7: 100.0},
            "output_sizes": {7: 100.0},
        },
        "physical_model": {
            "boundary": {
                "SMP": {
                    "fixed_latency_ms": 2.0,
                    "throughput_bytes_per_sec": 1_000_000_000.0,
                }
            }
        },
        "offloads": {
            "SMP": {
                7: {
                    "backend_compute": {
                        "count": 100,
                        "mean_ms_per_sample": 2.0,
                        "stderr_ms_per_sample": 0.1,
                    }
                }
            }
        }
    }

    cost = optimizer._dp_profiled_operator_compute_cost(
        7, PipeVariantType.SMP, profiled_total_cost=3.0
    )
    boundary = optimizer._dp_boundary_cost_ms(
        PipeVariantType.SMP, input_size=100.0, output_size=100.0
    )
    assert cost + boundary == pytest.approx(3.0)


def test_optimizer_uses_direct_worker_bound_without_boundary_model():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "offloads": {
            "SMP": {
                7: {
                    "backend_compute": {
                        "count": 100,
                        "mean_ms_per_sample": 2.0,
                        "stderr_ms_per_sample": 0.1,
                    }
                }
            }
        }
    }

    cost = optimizer._dp_profiled_operator_compute_cost(
        7, PipeVariantType.SMP, profiled_total_cost=999.0
    )
    assert cost == pytest.approx(2.0 + 1.645 * 0.1)


def test_optimizer_uses_wall_clock_cost_domain_for_new_profiles():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {
            "throughput": 250.0,
            "latencies": {7: 99.0, 8: 1.0},
            "wall_latencies": {7: 1_000_000.0, 8: 3_000_000.0},
            "input_sizes": {7: 100.0, 8: 100.0},
            "output_sizes": {7: 100.0, 8: 100.0},
        },
        "disk_info": {"read_latency": 0.0, "write_latency": 0.0},
        "offloads": {},
    }

    optimizer._init_stats()

    assert optimizer._dp_wall_latency_scale == pytest.approx(1.0)
    assert optimizer._base_cost_map == pytest.approx({7: 1.0, 8: 3.0})


def test_optimizer_rejects_wall_clock_costs_inconsistent_with_throughput():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {
            "throughput": 250.0,
            "latencies": {7: 99.0, 8: 1.0},
            "wall_latencies": {7: 2_000_000.0, 8: 6_000_000.0},
            "input_sizes": {7: 100.0, 8: 100.0},
            "output_sizes": {7: 100.0, 8: 100.0},
        },
        "disk_info": {"read_latency": 0.0, "write_latency": 0.0},
        "offloads": {},
    }

    optimizer._init_stats()

    assert optimizer._dp_wall_latency_scale is None
    assert optimizer._base_cost_map == pytest.approx({7: 3.96, 8: 0.04})


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
