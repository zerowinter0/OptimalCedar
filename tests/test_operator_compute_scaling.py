import threading
from collections import deque

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
