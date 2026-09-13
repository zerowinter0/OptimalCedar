import copy

import pytest

from cedar.compose.dp_optimizer import DpOptimizer, _BlockCostIndex
from cedar.compose.old_dp_optimizer import OldDpOptimizer
from cedar.compose.optimizer import OptimizerOptions, PipeDesc, PipeVariantType
from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.sources import IterSource
from tests.test_cm_optimizer import Example, setup


def test_old_dp_preserves_legacy_costs_with_shared_affine_profile():
    feature, _, profile, ids = setup()
    before = copy.deepcopy(profile)
    legacy_profile = copy.deepcopy(profile)
    legacy_profile.pop("cm_model")
    optimizers = []
    for cls, stats in ((OldDpOptimizer, profile), (DpOptimizer, legacy_profile)):
        opt = cls()
        opt.init(feature.logical_pipes, feature.logical_adj_list)
        opt.profiled_stats = stats
        opt._init_stats()
        opt._prepare_dp_metadata(list(ids[1:]))
        optimizers.append(opt)
    old, reference = optimizers
    assert not old._dp_affine_enabled
    for vt in (PipeVariantType.INPROCESS, PipeVariantType.RAY):
        desc = PipeDesc(None, vt, None)
        for pid in ids[1:]:
            if vt != PipeVariantType.INPROCESS and pid not in profile["offloads"].get(vt.name, {}):
                continue
            for size in (25, 50, 100, 200):
                assert old._calculate_pipe_cost(pid, size, desc) == pytest.approx(
                    reference._calculate_pipe_cost(pid, size, desc))
    indexes = []
    for opt in optimizers:
        costs = [opt._calculate_pipe_cost(pid, profile["baseline"]["input_sizes"][pid], None)
                 for pid in ids[1:]]
        indexes.append(_BlockCostIndex(opt, list(ids[1:]), costs))
    assert indexes[0].get(7) == indexes[1].get(7)
    assert profile == before


def test_selector_seventeen_installs_old_dp():
    feature = Example()
    feature.apply(IterSource(["abcdefgh"]))
    ds = DataSet(CedarContext(), {"feature": feature},
                 enable_optimizer=False, enable_controller=False, prefetch=False,
                 optimizer_options=OptimizerOptions(use_my_optimizer=17))
    try:
        assert isinstance(feature.optimizer, OldDpOptimizer)
        assert list(ds) == ["abcd"]
    finally:
        ds._exit()
