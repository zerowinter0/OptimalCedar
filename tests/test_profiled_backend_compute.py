import math
from types import SimpleNamespace

import pytest

from cedar.client.dataset import _accept_profile_value
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


def test_boundary_pass_through_predicate_accepts_tuple_values_unchanged():
    value = ("waveform", {"sample_rate": 16_000})
    assert _accept_profile_value(value) is True


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


def test_ray_boundary_cost_amortizes_fixed_latency_by_submit_batch():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "physical_model": {
            "boundary": {
                "RAY": {
                    "fixed_latency_ms": 2.0,
                    "throughput_bytes_per_sec": 1_000_000_000.0,
                }
            }
        }
    }

    assert optimizer._dp_boundary_cost_ms(
        PipeVariantType.RAY, input_size=100.0, output_size=100.0
    ) == pytest.approx(0.0042)
    assert optimizer._dp_boundary_cost_ms(
        PipeVariantType.RAY,
        input_size=2_000_000.0,
        output_size=2_000_000.0,
    ) == pytest.approx(6.0)


def test_smp_boundary_cost_does_not_amortize_by_queue_capacity():
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
    ) == pytest.approx(2.0002)


def test_stage_boundary_separates_item_size_cardinality_and_service_lane():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {"output_sizes": {99: 100.0}},
        "physical_model": {
            "boundary": {
                "RAY": {
                    "fixed_latency_ms": 2.0,
                    "throughput_bytes_per_sec": 1_000.0,
                }
            }
        },
    }
    optimizer._get_source_p_id = lambda: 99
    optimizer._dp_r_prod = [1.0, 2.0]
    optimizer._dp_cardinality_prod = [1.0, 0.25]
    block = SimpleNamespace(mask=1, variant=PipeVariantType.RAY)

    local_ms, parallel_ms = optimizer._dp_stage_boundary_components(0, block)

    # Batch selection sees one 100-byte input and one 200-byte output (batch
    # 500), while transport sees 100 input bytes and only 0.25 * 200 output
    # bytes per source record.
    assert local_ms == 0.0
    assert parallel_ms == pytest.approx(150.0 + 2.0 / 500.0)
    assert optimizer._dp_stage_boundary_cost(0, block) == pytest.approx(
        local_ms + parallel_ms
    )


def test_real_object_boundary_charges_marshalling_to_local_lane():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {"output_sizes": {99: 100.0}},
        "physical_model": {
            "boundary": {
                "RAY": {
                    "fixed_latency_ms": 2.0,
                    "throughput_bytes_per_sec": 1_000.0,
                }
            },
            "object_boundary": {
                "RAY": {
                    "operators": {
                        10: {
                            "input_serialize_ms_per_sample": 3.0,
                            "output_deserialize_ms_per_sample": 99.0,
                        },
                        11: {
                            "input_serialize_ms_per_sample": 99.0,
                            "output_deserialize_ms_per_sample": 5.0,
                        },
                    }
                }
            },
        },
    }
    optimizer._get_source_p_id = lambda: 99
    optimizer._dp_inner_ops = [10, 11]
    optimizer._dp_r_prod = [1.0, 2.0, 3.0, 6.0]
    optimizer._dp_cardinality_prod = [1.0, 0.5, 0.25, 0.125]
    block = SimpleNamespace(
        mask=3, order=(0, 1), variant=PipeVariantType.RAY
    )

    local_ms, parallel_ms = optimizer._dp_stage_boundary_components(0, block)

    assert local_ms == pytest.approx(2.0 / 500.0 + 3.0 + 5.0 * 0.125)
    assert parallel_ms == pytest.approx(100.0 + 600.0 * 0.125)


def test_real_identity_stage_boundary_supersedes_synthetic_model():
    optimizer = MyOptimizer()
    optimizer.profiled_stats = {
        "baseline": {"output_sizes": {99: 100.0}},
        "physical_model": {
            "boundary": {
                "RAY": {
                    # Deliberately enormous fallback values. A schema-v2
                    # real-object identity measurement must take precedence.
                    "fixed_latency_ms": 10_000.0,
                    "throughput_bytes_per_sec": 1.0,
                }
            },
            "object_boundary": {
                "RAY": {
                    "operators": {
                        10: {
                            "input_identity_stage": {
                                "mean_ms_per_sample": 4.0,
                            },
                            "output_identity_stage": {
                                "mean_ms_per_sample": 99.0,
                            },
                        },
                        11: {
                            "input_identity_stage": {
                                "mean_ms_per_sample": 99.0,
                            },
                            "output_identity_stage": {
                                "mean_ms_per_sample": 8.0,
                            },
                        },
                    }
                }
            },
        },
    }
    optimizer._get_source_p_id = lambda: 99
    optimizer._dp_inner_ops = [10, 11]
    optimizer._dp_r_prod = [1.0, 2.0, 3.0, 6.0]
    optimizer._dp_cardinality_prod = [1.0, 0.5, 0.25, 0.125]
    block = SimpleNamespace(
        mask=3, order=(0, 1), variant=PipeVariantType.RAY
    )

    local_ms, parallel_ms = optimizer._dp_stage_boundary_components(0, block)

    # A complete identity stage contains both sides of one same-type
    # boundary. The mixed-type block uses half its measured input stage and
    # half its measured output stage, weighted by the actual cardinalities.
    assert local_ms == pytest.approx(0.5 * 4.0 + 0.5 * 8.0 * 0.125)
    assert parallel_ms == 0.0


@pytest.mark.parametrize(
    ("input_size", "output_size", "expected"),
    [
        (100.0, 100.0, 500),
        (500_000.0, 500_000.0, 2),
        (1_000_000.0, 1_000_000.0, 1),
        (2_000_000.0, 2_000_000.0, 1),
    ],
)
def test_ray_boundary_batch_rule_matches_final_plan_tuning(
    input_size, output_size, expected
):
    assert MyOptimizer._dp_ray_submit_batch_size(
        input_size, output_size
    ) == expected


def test_finite_workload_caps_multistage_ray_batch_for_overlap():
    optimizer = MyOptimizer()
    optimizer.options = SimpleNamespace(num_samples=20_000)
    optimizer.physical_plan = SimpleNamespace(n_local_workers=8)

    assert optimizer._dp_finite_workload_ray_batch_cap(1) == 500
    assert optimizer._dp_finite_workload_ray_batch_cap(3) == 125
