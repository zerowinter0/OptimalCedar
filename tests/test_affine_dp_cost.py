import itertools

import pytest

from cedar.compose.dp_optimizer import DpOptimizer, _BlockCostIndex
from cedar.compose.optimizer import PipeDesc, PipeVariantType
from tests.test_cm_optimizer import setup


def affine_dp():
    feature, _, profile, ids = setup()
    opt = DpOptimizer()
    opt.init(feature.logical_pipes, feature.logical_adj_list)
    opt.profiled_stats = profile
    opt._init_stats()
    opt._prepare_dp_metadata(list(ids[1:]))
    return opt, ids


def test_affine_dp_index_matches_all_orders_with_size_and_selectivity():
    opt, (source, shrink, filt, consumer) = affine_dp()
    ops = [shrink, filt, consumer]
    costs = [opt._calculate_pipe_cost(pid, opt.profiled_stats["baseline"]["input_sizes"][pid], None)
             for pid in ops]
    index = _BlockCostIndex(opt, ops, costs)
    for order in itertools.permutations(range(3)):
        mask, size, reach, expected, actual = 0, 100, 1, 0, 0
        models = [(0.1, 2), (0, 5), (0.2, 10)]
        for i in order:
            k, b = models[i]
            expected += reach * (k * size + b)
            cost, _ = index.get(1 << i, mask)
            actual += cost
            size *= [0.5, 1, 1][i]
            reach *= [1, 0.25, 1][i]
            mask |= 1 << i
        assert actual == pytest.approx(expected)
    assert index.get(7)[0] == pytest.approx(13)


def test_affine_dp_worker_anchor_and_intercept():
    opt, (_, _, _, consumer) = affine_dp()
    # The worker mean is measured at the profiled 50-byte input; the fitted
    # kx+b shape (30 / 20) rescales it to 12 ms per processed record at 100 B.
    assert opt._dp_affine_worker_cost(consumer, 8, 100) == pytest.approx(12)
    # An operator is priced by its own kx+b value; the DP turns that per-record
    # price into a per-source-record total through the surviving-record work
    # product, so no cardinality is folded in here.
    assert opt._calculate_pipe_cost(consumer, 100, None) == pytest.approx(30)


def test_affine_dp_does_not_consult_old_classification():
    opt, ids = affine_dp()
    opt._dp_compute_scaling_for_pipe = lambda *_: pytest.fail("old classification used")
    opt._dp_compute_scaling_for_idx = lambda *_: pytest.fail("old classification used")
    opt._prepare_dp_metadata(list(ids[1:]))
    opt._calculate_pipe_cost(ids[-1], 100, PipeDesc(None, PipeVariantType.RAY, None))
    assert opt._dp_compute_work_prod(3, 2) == pytest.approx(5)


def test_affine_dp_zero_selectivity_removes_intercept():
    opt, ids = affine_dp()
    opt.profiled_stats["baseline"]["selectivities"][ids[2]] = 0
    opt._init_stats()
    opt._prepare_dp_metadata(list(ids[1:]))
    assert opt._dp_compute_work_prod(2, 2) == 0

    costs = [opt._calculate_pipe_cost(pid, opt.profiled_stats["baseline"]["input_sizes"][pid], None)
             for pid in ids[1:]]
    index = _BlockCostIndex(opt, list(ids[1:]), costs)
    assert index.get(4, 0)[0] == pytest.approx(30)
    assert index.get(4, 2)[0] == 0


def test_selector_two_enables_affine_profiling():
    from cedar.client import DataSet
    from cedar.config import CedarContext
    from cedar.compose.optimizer import OptimizerOptions
    from cedar.sources import IterSource
    from tests.test_cm_optimizer import Example
    feature = Example()
    feature.apply(IterSource(["abcdefgh"]))
    ds = DataSet(CedarContext(), {"feature": feature},
                 enable_optimizer=False, enable_controller=False, prefetch=False,
                 optimizer_options=OptimizerOptions(use_my_optimizer=2))
    try:
        assert ds._cm_profile
        assert isinstance(feature.optimizer, DpOptimizer)
        assert list(ds) == ["abcd"]
    finally:
        ds._exit()
