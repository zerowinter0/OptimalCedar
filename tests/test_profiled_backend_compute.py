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


def test_optimizer_prefers_direct_worker_compute_upper_bound():
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
