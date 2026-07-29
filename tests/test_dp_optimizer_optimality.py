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
from cedar.compose.dp_optimizer import DpOptimizer, _BlockCostIndex
from cedar.compose import constants
from cedar.compose.my_optimizer import MyOptimizer
from cedar.compose.optimizer import OptimizerOptions, PipeDesc
from cedar.pipes import PipeVariantType
from cedar.sources import IterSource


def test_unconstrained_six_operator_plan_count():
    assert unconstrained_plan_count() == 2_211_840


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


def test_old_profile_keeps_exact_size_only_work_product():
    optimizer = DpOptimizer()
    optimizer._dp_r_prod = [1.0, 0.5, 2.0, 1.0]
    optimizer._dp_volume_prod = []

    assert [
        optimizer._dp_work_prod(mask) for mask in range(4)
    ] == optimizer._dp_r_prod


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
