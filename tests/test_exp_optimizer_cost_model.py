import itertools
import math
from pathlib import Path
from typing import List

import yaml

from cedar.compose import Feature
from cedar.compose.exp_optimizer import ExpBlockCandidateProvider, ExpOptimizer
from cedar.compose.optimizer import OptimizerOptions, PipeDesc, PipeVariantType
from cedar.pipes import MapperPipe, Pipe
from cedar.sources import IterSource
from evaluation.pipelines.coco.cedar_dataset import COCOFeature


COCO_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "evaluation/chapter6_experiments/formal_results/"
    "paper_artifacts/optimizer/profiles/coco.yaml"
)


def _identity(value):
    return value


class ThreeOpFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        pipe = source_pipes[0]
        for index in range(3):
            pipe = MapperPipe(pipe, _identity, tag=f"op_{index}")
        return pipe


def _load_coco_optimizer():
    with COCO_PROFILE.open() as profile_file:
        profile = yaml.safe_load(profile_file)
    feature = COCOFeature(1)
    feature.apply(IterSource([0]))
    optimizer = ExpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(
        enable_offload=True,
        enable_reorder=True,
        enable_fusion=True,
        enable_caching=False,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    return optimizer, feature, profile


def _decoded_order(plan, feature):
    source = next(
        p_id for p_id, pipe in feature.logical_pipes.items() if pipe.is_source()
    )
    current = source
    order = []
    while plan.graph[current]:
        current = next(iter(plan.graph[current]))
        desc = plan.pipe_descs[current]
        if current in feature.logical_pipes:
            order.append(current)
        elif desc.fused_pipes:
            order.extend(desc.fused_pipes)
    return order


def test_invalid_amdahl_inversion_uses_nonzero_baseline_fallback():
    optimizer, _, profile = _load_coco_optimizer()
    distort_p_id = 1
    # Force a throughput above Amdahl's identifiable range; the recorded COCO
    # value itself can legitimately move as the formal profile is refreshed.
    profile["offloads"]["RAY"][distort_p_id]["throughput"] = 1e12
    baseline_input = profile["baseline"]["input_sizes"][distort_p_id]
    ray_cost = optimizer._calculate_pipe_cost(
        distort_p_id,
        baseline_input,
        PipeDesc(None, PipeVariantType.RAY, None),
    )

    assert ray_cost > 0
    assert math.isclose(
        ray_cost,
        optimizer._base_cost_map[distort_p_id],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_coco_fixed_ray_block_matches_compute_only_exhaustive_oracle():
    optimizer, _, _ = _load_coco_optimizer()
    inner_ops = [5, 4, 3, 2, 1, 0]
    optimizer._prepare_dp_metadata(inner_ops)
    provider = ExpBlockCandidateProvider(optimizer, inner_ops)
    mask = 31
    ray_costs = [
        optimizer._exp_compute_cost_per_byte(p_id, PipeVariantType.RAY)
        for p_id in inner_ops
    ]
    dp_order, dp_cost = provider._best_compute_order(mask, ray_costs)

    best_cost = float("inf")
    best_order = None
    for order in itertools.permutations(range(5)):
        position = {index: rank for rank, index in enumerate(order)}
        if not (position[0] < position[1] < position[2] < position[3]):
            continue
        ratio = 1.0
        cost = 0.0
        for index in order:
            cost += ratio * ray_costs[index]
            ratio *= optimizer._dp_ratios[index]
        if cost < best_cost:
            best_cost = cost
            best_order = order

    assert tuple(dp_order) == best_order
    assert math.isclose(dp_cost, best_cost, rel_tol=1e-12, abs_tol=1e-12)
    assert dp_order.index(4) < dp_order.index(0)  # distort before zoom


def test_exp_optimizer_coco_plan_places_distort_before_zoom():
    optimizer, feature, profile = _load_coco_optimizer()
    plan = optimizer.run(
        profile,
        OptimizerOptions(
            enable_prefetch=True,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=True,
            available_local_cpus=16,
            enable_fusion=True,
            enable_caching=False,
        ),
    )
    order = _decoded_order(plan, feature)

    assert order.index(1) < order.index(5)  # distort before zoom
    assert order.index(5) < order.index(4) < order.index(3) < order.index(2)
    assert plan.validate()


def test_explicit_stage_boundary_cost_is_paid_once_per_fused_block():
    feature = ThreeOpFeature()
    feature.apply(IterSource([0]))
    source = next(
        p_id for p_id, pipe in feature.logical_pipes.items() if pipe.is_source()
    )
    operator_ids = [
        next(p_id for p_id, pipe in feature.logical_pipes.items() if pipe.tag == tag)
        for tag in ("op_0", "op_1", "op_2")
    ]
    input_sizes = {p_id: 1.0 for p_id in feature.logical_pipes}
    output_sizes = {p_id: 1.0 for p_id in feature.logical_pipes}
    latencies = {source: 0.0, **{p_id: 10.0 for p_id in operator_ids}}
    profile = {
        "baseline": {
            "throughput": 1000.0 / 30.0,
            "latencies": latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
        },
        "disk_info": {"read_latency": 0.0, "write_latency": 0.0},
        "offloads": {
            "RAY": {p_id: {"throughput": 100.0} for p_id in operator_ids}
        },
        "exp_cost_model": {
            "operator_compute": {
                "RAY": {
                    p_id: {"cost_per_byte_ms": 1.0} for p_id in operator_ids
                }
            },
            "stage_boundaries": {
                "RAY": {
                    "input_cost_per_byte_ms": 1.0,
                    "output_cost_per_byte_ms": 1.0,
                }
            },
        },
    }
    optimizer = ExpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    plan = optimizer.run(
        profile,
        OptimizerOptions(
            enable_prefetch=False,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=False,
            enable_fusion=True,
            enable_caching=False,
        ),
    )
    fused_blocks = [
        desc.fused_pipes
        for desc in plan.pipe_descs.values()
        if desc.fused_pipes and len(desc.fused_pipes) > 1
    ]

    assert len(fused_blocks) == 1
    assert set(fused_blocks[0]) == set(operator_ids)
    assert math.isclose(
        optimizer.calculate_cost(plan.graph, plan.pipe_descs, plan=plan),
        5.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
