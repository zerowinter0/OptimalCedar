from itertools import permutations

from cedar.compose.optimizer import PipeVariantType
from cedar.pipes import PipeExecutionResource
from cedar.compose.sequential_exhaustive_optimizer import (
    MinimalParallelDpOptimizer,
    SingleWorkerCudaDpOptimizer,
    SequentialExhaustiveOptimizer,
    SequentialPlan,
    enumerate_fusion_candidates,
    enumerate_reorder_candidates,
    select_stage_winner,
    variant_allowed_for_execution_resource,
)


LOCAL = PipeVariantType.INPROCESS
RAY = PipeVariantType.RAY


def test_reorder_stage_moves_existing_blocks_without_splitting_them() -> None:
    plan = SequentialPlan(blocks=((0, 1), (2,), (3,)), variants=(RAY, LOCAL, RAY))
    # 0 < 1 is internal to the first block, and 2 < 3 constrains two blocks.
    candidates = enumerate_reorder_candidates(
        plan,
        predecessor_masks=(0, 1 << 0, 0, 1 << 2),
    )

    assert candidates
    assert all((0, 1) in candidate.blocks for candidate in candidates)
    assert all(candidate.blocks.index((2,)) < candidate.blocks.index((3,)) for candidate in candidates)
    assert {candidate.variants[candidate.blocks.index((0, 1))] for candidate in candidates} == {RAY}


def test_fusion_stage_only_merges_adjacent_blocks_on_the_same_backend() -> None:
    plan = SequentialPlan(blocks=((0,), (1,), (2,), (3,)), variants=(RAY, RAY, LOCAL, RAY))

    candidates = enumerate_fusion_candidates(plan)

    assert SequentialPlan(
        blocks=((0, 1), (2,), (3,)),
        variants=(RAY, LOCAL, RAY),
    ) in candidates
    assert all((1, 2) not in candidate.blocks for candidate in candidates)
    assert all((2, 3) not in candidate.blocks for candidate in candidates)


def test_each_stage_commits_its_winner_before_the_next_stage() -> None:
    initial = SequentialPlan(blocks=((0,), (1,)), variants=(LOCAL, LOCAL))
    reordered = SequentialPlan(blocks=((1,), (0,)), variants=(LOCAL, LOCAL))
    fused_initial = SequentialPlan(blocks=((0, 1),), variants=(RAY,))
    fused_reordered = SequentialPlan(blocks=((1, 0),), variants=(RAY,))
    costs = {
        initial: 10.0,
        reordered: 8.0,
        fused_initial: 2.0,
        fused_reordered: 6.0,
    }

    reorder_winner = select_stage_winner((initial, reordered), costs.__getitem__)
    fusion_winner = select_stage_winner((fused_reordered,), costs.__getitem__)

    assert reorder_winner == reordered
    assert fusion_winner == fused_reordered
    assert costs[fusion_winner] > costs[fused_initial]


def test_all_six_stage_orders_are_declared() -> None:
    declared = {
        "".join(order)
        for order in permutations(("r", "f", "o"))
    }

    assert declared == {"rfo", "rof", "fro", "for", "orf", "ofr"}


def test_minimum_width_optimizers_preserve_their_selected_widths() -> None:
    for optimizer_type in (
        MinimalParallelDpOptimizer,
        SequentialExhaustiveOptimizer,
    ):
        assert optimizer_type.joint_actor_allocation is False
        assert optimizer_type.preserve_optimizer_widths is True


def test_cuda_blocks_can_only_use_ray_backends() -> None:
    assert variant_allowed_for_execution_resource(RAY, PipeExecutionResource.CUDA)
    assert not variant_allowed_for_execution_resource(
        LOCAL, PipeExecutionResource.CUDA
    )
    assert not variant_allowed_for_execution_resource(
        PipeVariantType.SMP, PipeExecutionResource.CUDA
    )


def test_single_worker_dp_also_considers_inprocess_cuda() -> None:
    optimizer = SingleWorkerCudaDpOptimizer()
    assert optimizer._dp_variant_allowed_for_execution_resource(
        LOCAL, PipeExecutionResource.CUDA
    )
    assert optimizer._dp_variant_allowed_for_execution_resource(
        RAY, PipeExecutionResource.CUDA
    )
    assert not optimizer._dp_variant_allowed_for_execution_resource(
        PipeVariantType.SMP, PipeExecutionResource.CUDA
    )
