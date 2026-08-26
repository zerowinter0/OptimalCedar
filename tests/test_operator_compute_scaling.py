import threading
from collections import deque
from types import SimpleNamespace

import pytest
import yaml

from cedar.client.dataset import DataSet
from cedar.client.profiler import FeatureProfiler
from cedar.compose.dp_optimizer import BlockCandidateProvider, DpOptimizer
from cedar.compose.optimizer import (
    Optimizer,
    OptimizerOptions,
    PipeDesc,
    PipeVariantType,
)
from cedar.pipes import PipeComputeScaling
from cedar.sources import IterSource
from evaluation.verify_dp_optimizer_optimality import (
    GeneratedFeature,
    build_profile,
    generate_case,
)


def test_pipe_scaling_defaults_to_unannotated_per_data():
    feature = GeneratedFeature(generate_case(3))
    feature.apply(IterSource([0]))
    pipe = next(
        pipe
        for pipe in feature.logical_pipes.values()
        if pipe.input_pipes
    )

    assert pipe.compute_scaling == PipeComputeScaling.PER_DATA
    assert pipe.compute_scaling_explicit is False

    pipe.set_compute_scaling(PipeComputeScaling.PER_RECORD)
    assert pipe.compute_scaling == PipeComputeScaling.PER_RECORD
    assert pipe.compute_scaling_explicit is True


def test_profiler_distinguishes_data_and_record_scaling():
    profiler = FeatureProfiler.__new__(FeatureProfiler)
    profiler._lock = threading.Lock()
    profiler.compute_scaling_observations = {
        1: deque((float(size), float(size * 10)) for size in range(10, 40)),
        2: deque((float(size), 1000.0) for size in range(10, 40)),
    }

    inferred = profiler.infer_compute_scaling()

    assert inferred[1]["scaling"] == "per_data"
    assert inferred[1]["reason"] == "classified"
    assert inferred[2]["scaling"] == "per_record"
    assert inferred[2]["reason"] == "classified"


def test_profiler_abstains_to_per_data_on_low_confidence():
    profiler = FeatureProfiler.__new__(FeatureProfiler)
    profiler._lock = threading.Lock()
    profiler.compute_scaling_observations = {
        1: deque(
            (float(size), float(size ** 0.45))
            for size in range(10, 40)
        )
    }

    inferred = profiler.infer_compute_scaling()[1]

    assert inferred["predicted_scaling"] == "per_data"
    assert inferred["scaling"] == "per_data"
    assert inferred["reason"] == "low_confidence"


def test_dp_allows_fusing_mixed_scaling_operators():
    case = generate_case(11)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    inner_ops = [p_id_by_tag[operator.tag] for operator in case.operators]
    feature.logical_pipes[inner_ops[0]].set_compute_scaling(
        PipeComputeScaling.PER_DATA
    )
    feature.logical_pipes[inner_ops[1]].set_compute_scaling(
        PipeComputeScaling.PER_RECORD
    )

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(
        enable_offload=True,
        enable_fusion=True,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(inner_ops)
    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()

    mixed_mask = 0b11
    assert list(provider.candidates_for(mixed_mask))


def test_dp_block_candidates_are_cached_by_prefix_and_mask():
    case = generate_case(11)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    inner_ops = [p_id_by_tag[operator.tag] for operator in case.operators]

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(
        enable_offload=True,
        enable_fusion=True,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(inner_ops)
    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()

    cache_key = (0, 1)
    first = provider.candidates_for_prefix(*cache_key)
    second = provider.candidates_for_prefix(*cache_key)

    assert provider._candidates_by_prefix_and_mask[cache_key] is first
    assert second is first


def test_profile_inference_only_overrides_unannotated_operator():
    case = generate_case(13)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    inner_ops = [p_id_by_tag[operator.tag] for operator in case.operators]
    explicit_p_id, inferred_p_id = inner_ops[:2]
    feature.logical_pipes[explicit_p_id].set_compute_scaling(
        PipeComputeScaling.PER_RECORD
    )
    profile["operator_compute_scaling"] = {
        explicit_p_id: {"scaling": "per_data", "mode": "inferred"},
        inferred_p_id: {"scaling": "per_record", "mode": "inferred"},
    }

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions()
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(inner_ops)

    assert optimizer._dp_compute_scalings[0] == PipeComputeScaling.PER_RECORD
    assert optimizer._dp_compute_scalings[1] == PipeComputeScaling.PER_RECORD


def test_incremental_scaling_profile_preserves_existing_measurements(tmp_path):
    case = generate_case(17)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    inner_ops = [p_id_by_tag[operator.tag] for operator in case.operators]
    feature.logical_pipes[inner_ops[0]].set_compute_scaling(
        PipeComputeScaling.PER_RECORD
    )
    original_baseline = profile["baseline"].copy()
    source_path = tmp_path / "source.yaml"
    output_path = tmp_path / "output.yaml"
    source_path.write_text(yaml.safe_dump(profile))

    dataset = DataSet.__new__(DataSet)
    dataset._profile_feature = lambda *args, **kwargs: {
        "compute_scaling_inference": {
            inner_ops[0]: {"scaling": "per_data", "reason": "classified"},
            inner_ops[1]: {"scaling": "per_record", "reason": "classified"},
        }
    }
    updated = dataset._profile_compute_scaling_incremental(
        "feature",
        feature,
        None,
        str(output_path),
        str(source_path),
    )

    assert updated["baseline"] == original_baseline
    assert updated["operator_compute_scaling"][inner_ops[0]] == {
        "scaling": "per_record",
        "mode": "explicit",
        "inference": {"scaling": "per_data", "reason": "classified"},
    }
    assert updated["operator_compute_scaling"][inner_ops[1]]["mode"] == "inferred"


def test_dp_uses_direct_backend_compute_mean_with_end_to_end_floor():
    case = generate_case(23)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    inner_ops = [p_id_by_tag[operator.tag] for operator in case.operators]
    target = inner_ops[1]
    feature.logical_pipes[target].set_compute_scaling(
        PipeComputeScaling.PER_RECORD
    )
    profile["offloads"]["RAY"][target]["backend_compute"] = {
        "count": 16,
        "mean_ms_per_sample": 4.0,
        "stderr_ms_per_sample": 2.0,
    }

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(inner_ops)

    expected_per_record = 4.0
    expected = expected_per_record * (
        optimizer._dp_profiled_input_cardinality[target]
    )
    actual = optimizer._calculate_pipe_cost(
        target,
        profile["baseline"]["input_sizes"][target],
        PipeDesc(name=None, variant_type=PipeVariantType.RAY),
    )

    inferred = Optimizer._calculate_pipe_cost(
        optimizer,
        target,
        profile["baseline"]["input_sizes"][target],
        PipeDesc(name=None, variant_type=PipeVariantType.RAY),
    )
    assert actual == max(expected, inferred)


def test_dp_uses_converged_width_compute_for_backend_and_local_costs():
    case = generate_case(29)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    inner_ops = [p_id_by_tag[operator.tag] for operator in case.operators]
    target = inner_ops[0]
    profile["operator_compute_scaling"] = {
        target: {"scaling": "per_data"}
    }
    width_mean = 1_000.0
    profile["physical_model"] = {
        "scaling": {
            backend: {
                target: {
                    "mean_ms_per_sample": width_mean,
                    "adaptive_profile": {
                        "width": 8,
                        "converged": True,
                    },
                }
            }
            for backend in ("RAY", "SMP")
        }
    }

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(inner_ops)
    input_size = profile["baseline"]["input_sizes"][target]

    assert optimizer._calculate_pipe_cost(
        target,
        input_size,
        PipeDesc(name=None, variant_type=PipeVariantType.RAY),
    ) == pytest.approx(width_mean)
    assert optimizer._calculate_pipe_cost(
        target, input_size, None
    ) == pytest.approx(width_mean)


def test_dp_ignores_unconverged_width_compute():
    optimizer = DpOptimizer()
    optimizer.profiled_stats = {
        "baseline": {"input_sizes": {3: 10.0}},
        "physical_model": {
            "scaling": {
                "RAY": {
                    3: {
                        "mean_ms_per_sample": 1_000.0,
                        "adaptive_profile": {
                            "width": 8,
                            "converged": False,
                        },
                    }
                }
            }
        },
    }
    assert (
        optimizer._dp_profiled_width_compute_cost(
            3, PipeVariantType.RAY, 10.0
        )
        is None
    )


def test_dp_recovers_missing_selectivity_from_offload_counts():
    case = generate_case(31)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    target = p_id_by_tag[case.operators[0].tag]
    profile["baseline"].pop("selectivities")
    observations = [0.2, 0.4, 0.8]
    for entry, value in zip(
        profile["offloads"]["RAY"].values(), observations
    ):
        entry["selectivities"] = {target: value}

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile

    assert optimizer._dp_observed_selectivities()[target] == 0.4


def test_multiwidth_actor_curve_interpolates_per_actor_latency():
    entry = {
        "widths": {
            2: {
                "mean_ms_per_sample": 10.0,
                "adaptive_profile": {"width": 2, "converged": True},
            },
            4: {
                "mean_ms_per_sample": 14.0,
                "adaptive_profile": {"width": 4, "converged": True},
            },
        }
    }

    assert DpOptimizer._dp_scaling_mean(entry, 2) == 10.0
    assert DpOptimizer._dp_scaling_mean(entry, 3) == 12.0
    assert DpOptimizer._dp_scaling_mean(entry, 6) == 14.0


def test_actor_curve_uses_global_formal_concurrency(monkeypatch):
    optimizer = DpOptimizer()
    optimizer.physical_plan = SimpleNamespace(n_local_workers=1)
    optimizer.profiled_stats = {
        "baseline": {"input_sizes": {3: 10.0}},
        "physical_model": {
            "scaling": {
                "RAY": {
                    3: {
                        "widths": {
                            2: {
                                "mean_ms_per_sample": 2.0,
                                "adaptive_profile": {"converged": True},
                            },
                            16: {
                                "mean_ms_per_sample": 16.0,
                                "adaptive_profile": {"converged": True},
                            },
                        }
                    }
                }
            }
        },
    }
    optimizer._dp_compute_scaling_for_pipe = lambda _: PipeComputeScaling.PER_DATA
    monkeypatch.setenv("CEDAR_MATCH_PROFILE_RESOURCES", "1")
    monkeypatch.setenv("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", "8")

    assert optimizer._dp_pipe_cost_at_parallelism(
        3, PipeVariantType.RAY, 2, 1.0
    ) == pytest.approx(16.0)

    optimizer._dp_assumed_total_parallel_stage_cpus = 2
    assert optimizer._dp_pipe_cost_at_parallelism(
        3, PipeVariantType.RAY, 1, 1.0
    ) == pytest.approx(16.0)

    optimizer._dp_assumed_total_parallel_stage_cpus = 1
    assert optimizer._dp_pipe_cost_at_parallelism(
        3, PipeVariantType.RAY, 1, 1.0
    ) == pytest.approx(8.0)
