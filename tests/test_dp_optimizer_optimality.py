import math
import itertools
import random

from evaluation.verify_dp_optimizer_optimality import (
    BACKENDS,
    unconstrained_plan_count,
    GeneratedFeature,
    build_profile,
    generate_case,
    verify_case,
)
from cedar.compose.dp_optimizer import (
    BlockCandidate,
    DpObjectiveCost,
    DpOptimizer,
    DpResourceUsage,
    DpStateSummary,
    ExtensibleDpSearch,
    TransitionChoice,
    _BlockCostIndex,
    _is_infeasible_conditioned_search_error,
)
from cedar.compose import constants
from cedar.compose.my_optimizer import MyOptimizer
from cedar.compose.optimizer import OptimizerOptions, PipeDesc
from cedar.pipes import PipeComputeScaling, PipeVariantType
from cedar.sources import IterSource


def test_unconstrained_six_operator_plan_count():
    assert unconstrained_plan_count() == 2_211_840


def test_only_expected_infeasible_conditioned_search_errors_are_skipped():
    assert _is_infeasible_conditioned_search_error(
        RuntimeError("Extensible DP failed: no feasible final state.")
    )
    assert _is_infeasible_conditioned_search_error(
        RuntimeError("No plan reaches required parallel CPU total 6.")
    )
    assert not _is_infeasible_conditioned_search_error(
        RuntimeError("profile data is malformed")
    )


def test_generated_profile_preserves_direct_backend_costs():
    case = generate_case(7)
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)

    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile
    optimizer.options = OptimizerOptions(enable_offload=True)
    optimizer._validate_stats()
    optimizer._init_stats()

    for operator in case.operators:
        p_id = p_id_by_tag[operator.tag]
        for backend in BACKENDS:
            desc = PipeDesc(None, PipeVariantType[backend], None)
            actual = optimizer._calculate_pipe_cost(p_id, 1.0, desc)
            expected = operator.costs[backend]
            if backend != "INPROCESS":
                expected = max(
                    expected,
                    operator.costs["INPROCESS"]
                    / constants.MAX_UNIDENTIFIABLE_OPERATOR_SPEEDUP,
                )
            assert math.isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-12
            )


def test_dp_optimizer_matches_exhaustive_oracle():
    # Fixed seeds keep this regression test deterministic.  The CLI is used for
    # larger randomized campaigns.
    for seed in (11, 29):
        result = verify_case(generate_case(seed))
        assert math.isclose(
            result.dp_cost,
            result.oracle.cost,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        assert result.oracle.enumerated_plans > 0


def test_width_aware_dp_matches_integer_width_exhaustive_oracle():
    result = verify_case(generate_case(31), parallel_stage_limit=2)
    assert math.isclose(
        result.dp_cost,
        result.oracle.cost,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


def test_external_service_coordinates_are_losslessly_collapsed():
    class OptimizerStub:
        collapse_external_service_coordinates = True
        _dp_pred_indices = [[]]

        def _dp_parallel_stage_cpu_limit(self):
            return DpResourceUsage(ray_cpus=1, smp_cpus=1)

    class ProviderStub:
        pass

    class PolicyStub:
        enabled = False

    search = ExtensibleDpSearch(
        OptimizerStub(), [0], ProviderStub(), PolicyStub()
    )
    skyline = search._exact_skyline(
        {
            DpObjectiveCost(ray_serial=10.0): object(),
            DpObjectiveCost(smp_serial=10.0): object(),
        }
    )

    assert len(skyline) == 1
    assert skyline[0][0] == DpObjectiveCost(
        ray_serial=10.0, smp_serial=10.0
    )


def test_selectivity_aware_block_cost_moves_cheap_rejector_first():
    optimizer = DpOptimizer()
    optimizer._dp_pred_indices = [[], []]
    optimizer._dp_r_prod = [1.0, 1.0, 1.0, 1.0]
    optimizer._dp_volume_prod = [1.0, 0.1, 1.0, 0.1]
    optimizer.profiled_stats = {
        "baseline": {
            "input_sizes": {0: 1.0, 1: 1.0},
            "output_sizes": {99: 1.0},
        }
    }
    optimizer._get_source_p_id = lambda: 99

    index = _BlockCostIndex(
        optimizer,
        inner_ops=[0, 1],
        costs=[1.0, 10.0],
    )
    cost, order = index.get(0b11)

    assert order == (0, 1)
    assert math.isclose(cost, 2.0)


def test_block_index_retains_exact_minimum_for_each_endpoint_pair():
    optimizer = DpOptimizer()
    optimizer._dp_pred_indices = [[], [], []]
    optimizer._dp_compute_scalings = [PipeComputeScaling.PER_BYTE] * 3
    ratios = [0.25, 0.5, 0.8]
    optimizer._dp_r_prod = [1.0] * 8
    optimizer._dp_volume_prod = [1.0] * 8
    for mask in range(1, 8):
        product = 1.0
        for idx, ratio in enumerate(ratios):
            if mask & (1 << idx):
                product *= ratio
        optimizer._dp_r_prod[mask] = product
        optimizer._dp_volume_prod[mask] = product
    optimizer.profiled_stats = {
        "baseline": {
            "input_sizes": {0: 1.0, 1: 1.0, 2: 1.0},
            "output_sizes": {99: 1.0},
        }
    }
    optimizer._get_source_p_id = lambda: 99
    costs = [7.0, 2.0, 5.0]
    index = _BlockCostIndex(optimizer, [0, 1, 2], costs)

    expected = {}
    for order in itertools.permutations(range(3)):
        prefix = 0
        cost = 0.0
        for idx in order:
            cost += optimizer._dp_work_prod(prefix) * costs[idx]
            prefix |= 1 << idx
        endpoints = (order[0], order[-1])
        expected[endpoints] = min(expected.get(endpoints, math.inf), cost)

    actual = {
        (order[0], order[-1]): cost
        for cost, order in index.get_endpoint_minima(0b111)
    }
    assert actual.keys() == expected.keys()
    for endpoints in expected:
        assert math.isclose(
            actual[endpoints], expected[endpoints],
            rel_tol=1e-12, abs_tol=1e-12,
        )


def test_old_profile_keeps_exact_size_only_work_product():
    optimizer = DpOptimizer()
    optimizer._dp_r_prod = [1.0, 0.5, 2.0, 1.0]
    optimizer._dp_volume_prod = []

    assert [
        optimizer._dp_work_prod(mask) for mask in range(4)
    ] == optimizer._dp_r_prod


def test_per_record_compute_keeps_boundaries_byte_scaled():
    optimizer = DpOptimizer()
    optimizer._dp_compute_scaling = PipeComputeScaling.PER_RECORD
    optimizer._dp_r_prod = [1.0, 4.0]
    optimizer._dp_volume_prod = [1.0, 2.0]
    optimizer._dp_cardinality_prod = [1.0, 0.5]

    assert optimizer._dp_compute_work_prod(1) == 0.5
    assert optimizer._dp_work_prod(1) == 2.0


def test_per_record_cost_is_anchored_at_its_profiled_position():
    optimizer = DpOptimizer()
    optimizer._dp_pred_indices = [[], []]
    optimizer._dp_compute_scalings = [
        PipeComputeScaling.PER_BYTE,
        PipeComputeScaling.PER_RECORD,
    ]
    optimizer._dp_r_prod = [1.0, 4.0, 1.0, 4.0]
    optimizer._dp_volume_prod = [1.0, 2.0, 1.0, 2.0]
    optimizer._dp_cardinality_prod = [1.0, 0.5, 1.0, 0.5]
    optimizer.profiled_stats = {
        "baseline": {
            "input_sizes": {0: 100.0, 1: 400.0},
            "output_sizes": {99: 100.0},
        }
    }
    optimizer._get_source_p_id = lambda: 99

    index = _BlockCostIndex(
        optimizer,
        inner_ops=[0, 1],
        costs=[0.1, 10.0],
    )
    cost, order = index.get(0b11)

    assert order == (0, 1)
    assert math.isclose(cost, 10.1)


def test_per_record_normalization_survives_two_stage_reorder():
    optimizer = DpOptimizer()
    optimizer._dp_inner_ops = [20, 10]
    optimizer._dp_compute_scalings = [
        PipeComputeScaling.PER_RECORD,
        PipeComputeScaling.PER_RECORD,
    ]
    optimizer._dp_profiled_input_cardinality = {10: 1.0, 20: 0.25}
    optimizer._dp_cardinality_prod = [1.0, 1.0, 0.25, 0.25]

    assert math.isclose(
        optimizer._dp_compute_cost_denominator(0, 999.0, 100.0),
        25.0,
    )
    assert math.isclose(
        optimizer._dp_compute_cost_denominator(1, 999.0, 100.0),
        100.0,
    )


def test_search_drops_resource_coordinate_without_strict_limit():
    """Resource counts must not split states when they cannot affect plans."""

    observed_parallel_cpus = []

    class OptimizerStub:
        _dp_pred_indices = [[]]
        _dp_r_prod = [1.0, 1.0]

        def _dp_parallel_stage_cpu_limit(self):
            return None

        def _dp_valid_single_last(self, prev_mask, idx):
            return prev_mask == 0 and idx == 0

        def _dp_stage_boundary_cost(self, prev_mask, block):
            return 0.0

        def _dp_regular_transition_cost(self, prev_mask, block):
            return block.cost + self._dp_stage_boundary_cost(prev_mask, block)

        def _dp_parallel_stage_cpu_cost(self, variant):
            return 8

    class ProviderStub:
        def candidates_for(self, mask):
            if mask != 1:
                return []
            return [
                BlockCandidate(
                    mask=1,
                    order=(0,),
                    variant=PipeVariantType.RAY,
                    cost=1.0,
                    materializes_fusion=False,
                )
            ]

    class PolicyStub:
        def initial_state(self):
            return DpStateSummary(
                parallel_stage_cpus=DpResourceUsage(ray_cpus=7)
            )

        def transitions(
            self,
            prev_mask,
            next_mask,
            prev_state,
            regular_cost,
            block,
            next_parallel_stage_cpus=None,
        ):
            observed_parallel_cpus.append(next_parallel_stage_cpus)
            yield TransitionChoice(
                state=DpStateSummary(
                    parallel_stage_cpus=next_parallel_stage_cpus
                ),
                extra_cost=regular_cost,
            )

    optimizer = OptimizerStub()
    result = ExtensibleDpSearch(
        optimizer, [0], ProviderStub(), PolicyStub()
    ).run()

    assert result.cost == 1.0
    assert observed_parallel_cpus == [DpResourceUsage()]
    assert optimizer._dp_last_search_stats["retained_states"] == 2


def test_pareto_frontier_keeps_initially_slower_parallel_choice():
    """A scalar partial score would discard the globally faster pipeline."""

    class OptimizerStub:
        _dp_pred_indices = [[], [0]]

        def _dp_parallel_stage_cpu_limit(self):
            return None

        def _dp_valid_single_last(self, prev_mask, idx):
            return idx == 0 or bool(prev_mask & 1)

        def _dp_regular_transition_cost(self, prev_mask, block):
            return block.cost

        def _dp_parallel_stage_cpu_cost(self, variant):
            return 1

        def _dp_accumulate_objective_cost(
            self, previous, extra, block, prev_mask
        ):
            if block.variant == PipeVariantType.RAY:
                return DpObjectiveCost(
                    local_serial=previous.local_serial,
                    ray_serial=previous.ray_serial + extra,
                    smp_serial=previous.smp_serial,
                )
            return DpObjectiveCost(
                local_serial=previous.local_serial + extra,
                ray_serial=previous.ray_serial,
                smp_serial=previous.smp_serial,
            )

    class ProviderStub:
        def candidates_for(self, mask):
            if mask == 1:
                return [
                    BlockCandidate(
                        1, (0,), PipeVariantType.INPROCESS, 10.0, False
                    ),
                    BlockCandidate(1, (0,), PipeVariantType.RAY, 11.0, False),
                ]
            if mask == 2:
                return [
                    BlockCandidate(
                        2, (1,), PipeVariantType.INPROCESS, 100.0, False
                    )
                ]
            return []

    class PolicyStub:
        def initial_state(self):
            return DpStateSummary()

        def transitions(
            self, prev_mask, next_mask, prev_state, regular_cost, block,
            next_parallel_stage_cpus=None,
        ):
            yield TransitionChoice(prev_state, regular_cost)

    result = ExtensibleDpSearch(
        OptimizerStub(), [0, 1], ProviderStub(), PolicyStub()
    ).run()

    assert result.variants_by_idx[0] == PipeVariantType.RAY
    assert result.objective == DpObjectiveCost(100.0, 11.0)
    assert result.cost == 100.0


def test_resource_bottleneck_objective_pipelines_two_ray_stages():
    class OptimizerStub:
        _dp_pred_indices = [[], [0], [1], [2]]

        def _dp_parallel_stage_cpu_limit(self):
            return DpResourceUsage(ray_cpus=2)

        def _dp_valid_single_last(self, prev_mask, idx):
            return idx == 0 or bool(prev_mask & (1 << (idx - 1)))

        def _dp_fusion_block_valid(self, prev_mask, block_mask):
            indices = [i for i in range(4) if block_mask & (1 << i)]
            return indices == list(range(indices[0], indices[-1] + 1)) and (
                indices[0] == 0 or bool(prev_mask & (1 << (indices[0] - 1)))
            )

        def _dp_fusion_transport_feasible(self, prev_mask, block):
            return True

        def _dp_regular_transition_cost(self, prev_mask, block):
            return block.cost

        def _dp_parallel_stage_cpu_cost(self, variant):
            return DpResourceUsage(ray_cpus=1)

        def _dp_accumulate_objective_cost(
            self, previous, extra, block, prev_mask
        ):
            return DpObjectiveCost(
                local_serial=previous.local_serial,
                ray_serial=max(previous.ray_serial, extra),
                smp_serial=previous.smp_serial,
            )

    class ProviderStub:
        def candidates_for(self, mask):
            indices = [i for i in range(4) if mask & (1 << i)]
            if not indices:
                return []
            return [
                BlockCandidate(
                    mask,
                    tuple(indices),
                    PipeVariantType.RAY,
                    float(len(indices)),
                    len(indices) > 1,
                )
            ]

    class PolicyStub:
        def initial_state(self):
            return DpStateSummary()

        def transitions(
            self, prev_mask, next_mask, prev_state, regular_cost, block,
            next_parallel_stage_cpus=None,
        ):
            yield TransitionChoice(
                DpStateSummary(
                    parallel_stage_cpus=next_parallel_stage_cpus
                ),
                regular_cost,
            )

    result = ExtensibleDpSearch(
        OptimizerStub(), list(range(4)), ProviderStub(), PolicyStub()
    ).run()

    assert result.blocks == [[0, 1], [2, 3]]
    assert result.objective == DpObjectiveCost(ray_serial=2.0)


def _block_objective(order, ratios, costs, input_sizes):
    selected_mask = 0
    compute_cost = 0.0
    io_base_partial = 1.0
    ratio_product = 1.0
    for position, idx in enumerate(order):
        compute_cost += ratio_product * costs[idx] / input_sizes[idx]
        if position > 0:
            io_base_partial += 2.0 * ratio_product
        selected_mask |= 1 << idx
        ratio_product *= ratios[idx]
    fused_io = 1.0 + ratio_product
    baseline_io = io_base_partial + ratio_product
    return compute_cost * fused_io / baseline_io, selected_mask


def test_convex_frontier_matches_exhaustive_reorder_for_every_subset():
    """The compact hull must preserve the exact optimum for every DP block."""
    for seed in (3, 17, 41, 89):
        rng = random.Random(seed)
        n = 7
        inner_ops = list(range(n))
        ratios = [rng.uniform(0.2, 1.4) for _ in range(n)]
        costs = [rng.uniform(0.1, 20.0) for _ in range(n)]
        input_sizes = [rng.uniform(0.2, 5.0) for _ in range(n)]

        full_mask = 1 << n
        ratio_products = [1.0] * full_mask
        for mask in range(1, full_mask):
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            ratio_products[mask] = ratio_products[mask ^ lsb] * ratios[idx]

        optimizer = MyOptimizer()
        optimizer._dp_pred_indices = [[] for _ in range(n)]
        optimizer._dp_r_prod = ratio_products
        optimizer.profiled_stats = {
            "baseline": {
                "input_sizes": dict(enumerate(input_sizes)),
            }
        }

        actual = optimizer._dp_naive_reorder_cost_per_variant(
            inner_ops,
            n,
            [PipeVariantType.INPROCESS],
            {PipeVariantType.INPROCESS: costs},
        )[PipeVariantType.INPROCESS]

        for mask in range(1, full_mask):
            indices = [idx for idx in range(n) if mask & (1 << idx)]
            expected = min(
                _block_objective(order, ratios, costs, input_sizes)[0]
                for order in itertools.permutations(indices)
            )
            assert math.isclose(
                actual[mask], expected, rel_tol=1e-12, abs_tol=1e-12
            )

            best_order = optimizer._reconstruct_naive_reorder_order(
                PipeVariantType.INPROCESS, mask
            )
            reconstructed, reconstructed_mask = _block_objective(
                best_order, ratios, costs, input_sizes
            )
            assert reconstructed_mask == mask
            assert math.isclose(
                reconstructed, expected, rel_tol=1e-12, abs_tol=1e-12
            )
