import copy
import math
from types import SimpleNamespace

import pytest

from cedar.compose.optimizer import Optimizer, OptimizerOptions, PipeDesc
from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer
from cedar.client.dataset import DataSet
from cedar.config import CedarContext
from cedar.pipes import PipeVariantType
from cedar.sources import IterSource
from tests.test_dp_cache_fusion_optimizer import (
    TwoMapFeature,
    _profile_for,
    _ray_profile_for,
)


def _run(profile, *, offload: bool, fusion: bool, caching: bool):
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    optimizer = SimpleDpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    plan = optimizer.run(
        profile(feature),
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=offload,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=fusion,
            enable_caching=caching,
        ),
    )
    return optimizer, plan, feature


@pytest.mark.parametrize(
    "profile,offload,fusion,caching",
    [
        (_profile_for, False, False, False),
        (_profile_for, False, True, False),
        (_profile_for, False, True, True),
        (_ray_profile_for, True, True, False),
    ],
)
def test_simple_dp_search_cost_equals_materialized_cedar_cost(
    profile, offload, fusion, caching
):
    optimizer, plan, _ = _run(
        profile,
        offload=offload,
        fusion=fusion,
        caching=caching,
    )
    cedar_cost = optimizer._calculate_materialized_cedar_cost(plan)
    assert math.isclose(
        optimizer._last_dp_state_cost,
        cedar_cost,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert plan.validate()


def test_simple_dp_pipe_cost_ignores_extended_profile_layers():
    optimizer, _, feature = _run(
        _ray_profile_for,
        offload=True,
        fusion=False,
        caching=False,
    )
    p_id = next(
        p_id
        for p_id, pipe in feature.logical_pipes.items()
        if not pipe.is_source()
    )
    input_size = optimizer.profiled_stats["baseline"]["input_sizes"][p_id]
    desc = PipeDesc(name=None, variant_type=PipeVariantType.RAY)
    cedar_cost = Optimizer._calculate_pipe_cost(
        optimizer, p_id, input_size, desc
    )

    extended = copy.deepcopy(optimizer.profiled_stats)
    extended["physical_model"] = {
        "boundary": {"RAY": {"serialization_ns_per_byte": 1e12}},
        "scaling": {"RAY": {p_id: {"speedup": 1000.0}}},
    }
    extended["offloads"]["RAY"][p_id]["backend_compute"] = {
        "mean_ms_per_sample": 1e-12,
        "count": 1000000,
    }
    optimizer.profiled_stats = extended
    assert optimizer._calculate_pipe_cost(p_id, input_size, desc) == cedar_cost


def test_simple_dp_joint_search_ignores_multiwidth_scaling_curves():
    baseline_optimizer, baseline_plan, _ = _run(
        _ray_profile_for,
        offload=True,
        fusion=True,
        caching=False,
    )

    def extended_profile(feature):
        profile = copy.deepcopy(_ray_profile_for(feature))
        ray_entries = profile["offloads"]["RAY"]
        profile["physical_model"] = {
            "scaling": {
                "RAY": {
                    p_id: {
                        "widths": {
                            1: {
                                "mean_ms_per_sample": 1e9,
                                "adaptive_profile": {"converged": True},
                            }
                        }
                    }
                    for p_id in ray_entries
                }
            }
        }
        return profile

    optimizer, plan, _ = _run(
        extended_profile,
        offload=True,
        fusion=True,
        caching=False,
    )
    assert optimizer._last_dp_search_result == (
        baseline_optimizer._last_dp_search_result
    )
    assert plan.to_dict() == baseline_plan.to_dict()


def test_simple_dp_profiling_emits_original_cedar_schema(
    monkeypatch, tmp_path
):
    dataset = object.__new__(DataSet)
    feature = object()
    dataset.features = {"feature": feature}
    dataset.ctx = SimpleNamespace(use_ray=lambda: True)

    extended_measurement = {
        "latencies": {0: 1.0},
        "input_sizes": {0: 2.0},
        "output_sizes": {0: 2.0},
        "throughput": 3.0,
        "wall_latencies": {0: 4.0},
        "backend_compute": {"mean_ms_per_sample": 5.0},
        "selectivities": {0: 0.5},
    }
    monkeypatch.setattr(
        dataset,
        "_profile_feature",
        lambda *args, **kwargs: copy.deepcopy(extended_measurement),
    )

    def profile_ray(result, *args, **kwargs):
        assert kwargs["profile_backend_compute"] is False
        result.setdefault("offloads", {})["RAY"] = {
            0: copy.deepcopy(extended_measurement)
        }

    def profile_smp(result, *args, **kwargs):
        assert kwargs["profile_backend_compute"] is False
        result.setdefault("offloads", {})["SMP"] = {
            0: copy.deepcopy(extended_measurement)
        }

    monkeypatch.setattr(dataset, "_profile_ray", profile_ray)
    monkeypatch.setattr(dataset, "_profile_smp", profile_smp)
    monkeypatch.setattr(
        dataset,
        "_profile_tf",
        lambda result, *args: result.update(
            {"tf_fuse": {"throughput": 7.0}}
        ),
    )
    monkeypatch.setattr(dataset, "_profile_io", lambda: (0.1, 0.2))
    monkeypatch.setenv("CEDAR_PROFILE_INFER_COMPUTE_SCALING", "1")
    monkeypatch.setenv("CEDAR_PROFILE_FILTER_SELECTIVITY", "1")

    result = dataset._profile_legacy_cedar(
        "feature", output_file=str(tmp_path / "profile.yml")
    )
    assert set(result) == {"baseline", "offloads", "tf_fuse", "disk_info"}
    expected_measurement_keys = {
        "latencies",
        "input_sizes",
        "output_sizes",
        "throughput",
    }
    assert set(result["baseline"]) == expected_measurement_keys
    assert set(result["offloads"]["RAY"][0]) == expected_measurement_keys
    assert set(result["offloads"]["SMP"][0]) == expected_measurement_keys


def test_dataset_selector_uses_simple_dp_and_legacy_profile_mode():
    feature = TwoMapFeature()
    feature.apply(IterSource([1, 2, 3]))
    dataset = DataSet(
        CedarContext(),
        {"feature": feature},
        prefetch=False,
        enable_controller=False,
        enable_optimizer=False,
        optimizer_options=OptimizerOptions(use_my_optimizer=11),
    )
    assert isinstance(feature.optimizer, SimpleDpOptimizer)
    assert dataset._legacy_cedar_profile is True
