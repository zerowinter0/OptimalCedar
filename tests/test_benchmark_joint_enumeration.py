import math

from evaluation.benchmark_joint_enumeration import (
    JointEnumerationConfig,
    OperatorProfile,
    exhaustive_joint_search,
)


def test_exhaustive_search_visits_every_budget_feasible_joint_plan():
    """Catches omitted orders, partitions, implementations, or cache choices."""
    profiles = (
        OperatorProfile(1.0, 0.8, {"LOCAL": 4.0, "RAY": 2.0, "SMP": 3.0}),
        OperatorProfile(0.5, 1.0, {"LOCAL": 5.0, "RAY": 2.5, "SMP": 3.5}),
    )
    config = JointEnumerationConfig(
        ray_width_budget=2,
        smp_width_budget=2,
        max_width=2,
    )

    result = exhaustive_joint_search(profiles, config)

    # Per order: 5 one-stage implementations x 2 cache choices, plus
    # 19 budget-feasible two-stage assignments x 3 cache choices.  There are
    # two logical orders.
    assert result.visited_plans == 2 * (5 * 2 + 19 * 3)
    assert result.best_cost > 0
    assert sorted(result.best_order) == [0, 1]
    assert result.elapsed_seconds >= 0


def test_one_operator_count_includes_backend_width_and_cache_decisions():
    """Catches treating physical variants or the cache boundary as metadata."""
    profiles = (
        OperatorProfile(1.0, 1.0, {"LOCAL": 4.0, "RAY": 2.0, "SMP": 3.0}),
    )
    config = JointEnumerationConfig(
        ray_width_budget=2,
        smp_width_budget=2,
        max_width=2,
    )

    result = exhaustive_joint_search(profiles, config)

    # LOCAL@1, RAY@{1,2}, and SMP@{1,2}; each admits no-cache or cache-after.
    assert result.visited_plans == 10
    assert math.isclose(result.best_cost, 0.05, rel_tol=1e-12)
