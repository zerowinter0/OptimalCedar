import copy

import pytest

from cedar.compose.dp_optimizer import DpOptimizer, _BlockCostIndex
from cedar.compose.old_dp_optimizer import OldDpOptimizer
from cedar.compose.optimizer import (
    Optimizer,
    OptimizerOptions,
    PipeDesc,
    PipeVariantType,
)
from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.sources import IterSource
from tests.test_cm_optimizer import Example, setup


def test_old_dp_keeps_cedar_costs_with_shared_affine_profile():
    feature, _, profile, ids = setup()
    before = copy.deepcopy(profile)
    old = OldDpOptimizer()
    old.init(feature.logical_pipes, feature.logical_adj_list)
    old.profiled_stats = profile
    old._init_stats()
    old._prepare_dp_metadata(list(ids[1:]))

    # The control prices every variant exactly like Cedar's own cost model,
    # even though the shared profile carries fitted kx+b coefficients.
    for vt in (PipeVariantType.INPROCESS, PipeVariantType.RAY):
        desc = PipeDesc(None, vt, None)
        for pid in ids[1:]:
            if (
                vt != PipeVariantType.INPROCESS
                and pid not in profile["offloads"].get(vt.name, {})
            ):
                continue
            for size in (25, 50, 100, 200):
                assert old._calculate_pipe_cost(
                    pid, size, desc
                ) == pytest.approx(
                    Optimizer._calculate_pipe_cost(old, pid, size, desc)
                )

    # And the fitted affine layer really is ignored by the control: the joint
    # DP prices the same operator differently once its input shrinks.
    affine = DpOptimizer()
    affine.init(feature.logical_pipes, feature.logical_adj_list)
    affine.profiled_stats = profile
    affine._init_stats()
    affine._prepare_dp_metadata(list(ids[1:]))
    shrink = ids[3]
    assert old._calculate_pipe_cost(
        shrink, 25.0, None
    ) != pytest.approx(affine._calculate_pipe_cost(shrink, 25.0, None))
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
