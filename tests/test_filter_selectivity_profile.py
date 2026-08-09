from typing import List

import pytest

from cedar.client import DataSet
from cedar.client import dataset as dataset_module
from cedar.client.dataset import (
    _ProfiledFilterCallable,
    _consolidate_filter_selectivity,
)
from cedar.compose import Feature
from cedar.config import CedarContext
from cedar.pipes import (
    FilterPipe,
    InProcessPipeVariantContext,
    MapperPipe,
    Pipe,
)
from cedar.sources import IterSource


def test_profiled_filter_callable_counts_seen_and_kept():
    counter = _ProfiledFilterCallable(lambda value: value % 2 == 0)

    assert counter(2)
    assert not counter(3)
    assert counter(4)
    assert counter.input_count == 3
    assert counter.output_count == 2


def test_profiled_filter_callable_preserves_operator_exception():
    def fail(_value):
        raise RuntimeError("operator failure")

    counter = _ProfiledFilterCallable(fail)
    try:
        counter("sample")
    except RuntimeError as exc:
        assert str(exc) == "operator failure"
    else:
        raise AssertionError("operator exception was swallowed")
    assert counter.input_count == 1
    assert counter.output_count == 0


def test_selectivity_consolidation_uses_highest_coverage_pass():
    profile = {
        "baseline": {
            "input_counts": {1: 10, 2: 8},
            "output_counts": {1: 5, 2: 4},
            "selectivities": {1: 0.5, 2: 0.5},
        },
        "offloads": {
            "RAY": {
                1: {
                    # A mutated filter's remote counter is not visible.
                    "input_counts": {1: 0, 2: 80},
                    "output_counts": {1: 0, 2: 20},
                },
                3: {
                    "input_counts": {1: 100, 2: 60},
                    "output_counts": {1: 20, 2: 30},
                },
            },
            "SMP": {
                3: {
                    "input_counts": {1: 50, 2: 100},
                    "output_counts": {1: 20, 2: 10},
                },
            },
        },
    }

    _consolidate_filter_selectivity(profile)

    baseline = profile["baseline"]
    assert baseline["input_counts"] == {1: 100, 2: 100}
    assert baseline["output_counts"] == {1: 20, 2: 10}
    assert baseline["selectivities"] == {1: 0.2, 2: 0.1}
    assert baseline["selectivity_observation_sources"] == {
        1: "RAY:3",
        2: "SMP:3",
    }


def test_selectivity_consolidation_keeps_baseline_on_coverage_tie():
    profile = {
        "baseline": {
            "input_counts": {1: 10},
            "output_counts": {1: 4},
            "selectivities": {1: 0.4},
        },
        "offloads": {
            "RAY": {
                2: {
                    "input_counts": {1: 10},
                    "output_counts": {1: 1},
                },
            },
        },
    }

    _consolidate_filter_selectivity(profile)

    baseline = profile["baseline"]
    assert baseline["selectivities"] == {1: 0.4}
    assert baseline["selectivity_observation_sources"] == {1: "baseline"}


def test_baseline_profile_records_conditional_filter_selectivity(
    monkeypatch,
):
    first_filter = lambda value: value % 2 == 0
    second_filter = lambda value: value > 2

    class FilterFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            pipe = FilterPipe(
                source_pipes[0], first_filter, tag="first"
            )
            return FilterPipe(pipe, second_filter, tag="second")

    feature = FilterFeature()
    feature.apply(IterSource([1, 2, 3, 4, 5]))
    tagged = {
        pipe.tag: pipe
        for pipe in feature.logical_pipes.values()
        if pipe.tag is not None
    }
    first_id = tagged["first"].id
    second_id = tagged["second"].id
    dataset = object.__new__(DataSet)
    dataset.ctx = CedarContext()

    monkeypatch.setenv("CEDAR_PROFILE_FILTER_SELECTIVITY", "1")
    monkeypatch.setattr(dataset_module.time, "sleep", lambda _seconds: None)
    result = dataset._profile_feature(
        "feature",
        feature,
        n_samples=1,
        mutation_dict=None,
    )

    assert result["input_counts"][first_id] == 4
    assert result["output_counts"][first_id] == 2
    assert result["selectivities"][first_id] == 0.5
    assert result["input_counts"][second_id] == 2
    assert result["output_counts"][second_id] == 1
    assert result["selectivities"][second_id] == 0.5
    assert tagged["first"].fn is first_filter
    assert tagged["second"].fn is second_filter


def test_mutated_profile_pass_also_records_filter_selectivity(
    monkeypatch,
):
    first_filter = lambda value: value % 2 == 0
    second_filter = lambda value: value > 2

    class FilterFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            pipe = FilterPipe(
                source_pipes[0], first_filter, tag="first"
            )
            return FilterPipe(pipe, second_filter, tag="second")

    feature = FilterFeature()
    feature.apply(IterSource([1, 2, 3, 4, 5]))
    tagged = {
        pipe.tag: pipe
        for pipe in feature.logical_pipes.values()
        if pipe.tag is not None
    }
    first_id = tagged["first"].id
    second_id = tagged["second"].id
    dataset = object.__new__(DataSet)
    dataset.ctx = CedarContext()

    monkeypatch.setenv("CEDAR_PROFILE_FILTER_SELECTIVITY", "1")
    monkeypatch.setattr(dataset_module.time, "sleep", lambda _seconds: None)
    result = dataset._profile_feature(
        "feature",
        feature,
        n_samples=1,
        mutation_dict={
            first_id: InProcessPipeVariantContext(),
        },
    )

    assert result["input_counts"][first_id] == 4
    assert result["output_counts"][first_id] == 2
    assert result["selectivities"][first_id] == 0.5
    assert result["input_counts"][second_id] == 2
    assert result["output_counts"][second_id] == 1
    assert result["selectivities"][second_id] == 0.5
    assert tagged["first"].fn is first_filter
    assert tagged["second"].fn is second_filter


def test_profile_releases_workload_resources_after_operator_failure(
    monkeypatch,
):
    def fail(_value):
        raise RuntimeError("operator failure")

    class ReleasingFeature(Feature):
        def __init__(self):
            super().__init__()
            self.release_count = 0

        def _compose(self, source_pipes: List[Pipe]):
            return MapperPipe(source_pipes[0], fail)

        def release_profile_resources(self):
            self.release_count += 1

    feature = ReleasingFeature()
    feature.apply(IterSource([1]))
    dataset = object.__new__(DataSet)
    dataset.ctx = CedarContext()
    monkeypatch.setattr(dataset_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="operator failure"):
        dataset._profile_feature(
            "feature",
            feature,
            n_samples=1,
            mutation_dict=None,
        )

    assert feature.release_count == 1
    assert not feature.loaded
