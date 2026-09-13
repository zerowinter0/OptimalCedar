import copy
import math
from types import SimpleNamespace

import pytest
import yaml

from cedar.client import DataSet
from cedar.client.linear_cost_profile import fit_affine, profile_linear_feature
from cedar.compose import Feature
from cedar.compose.cm_optimizer import CmOptimizer
from cedar.compose.optimizer import Optimizer, OptimizerOptions, PipeDesc
from cedar.config import CedarContext
from cedar.pipes import MapperPipe, FilterPipe
from cedar.pipes.context import PipeVariantType
from cedar.sources import IterSource


class Example(Feature):
    def _compose(self, sources):
        pipe = MapperPipe(sources[0], lambda x: x[:len(x)//2], tag="shrink")
        pipe = FilterPipe(pipe, lambda x: len(x) % 2 == 0, tag="filter")
        return MapperPipe(pipe, lambda x: x, tag="consumer")


def setup():
    feature = Example()
    feature.apply(IterSource(["x"*i for i in range(10, 100)]))
    ids = {p.tag: i for i, p in feature.logical_pipes.items()}
    source = next(i for i, p in feature.logical_pipes.items() if p.is_source())
    a, f, b = (ids[k] for k in ("shrink", "filter", "consumer"))
    profile = {
        "baseline": {
            "throughput": 100,
            "latencies": {source: 0, a: 1, f: 1, b: 1},
            "input_sizes": {source: 100, a: 100, f: 50, b: 50},
            "output_sizes": {source: 100, a: 50, f: 50, b: 50},
            "selectivities": {f: .25}},
        "offloads": {"RAY": {b: {"throughput": 110,
            "backend_compute": {"count": 100, "mean_ms_per_sample": 8}}}},
        "disk_info": {"read_latency": 0, "write_latency": 0},
        "cm_model": {"schema_version": 1, "operators": {
            a: {"k": .1, "b": 2}, f: {"k": 0, "b": 5}, b: {"k": .2, "b": 10}}}}
    opt = CmOptimizer()
    opt.init(feature.logical_pipes, feature.logical_adj_list)
    opt.profiled_stats = profile
    opt.options = OptimizerOptions(enable_offload=False)
    opt._init_stats()
    return feature, opt, profile, (source, a, f, b)


def chain(order):
    return {pid: {order[i+1]} if i+1 < len(order) else set()
            for i, pid in enumerate(order)}


def test_affine_fit_has_correct_units_and_coefficients():
    pairs = [(float(i), (2*i+3)*1e6) for i in range(1, 101)]
    model = fit_affine(pairs)
    assert model["status"] == "fitted"
    assert model["k"] == pytest.approx(2)
    assert model["b"] == pytest.approx(3)
    assert model["r_squared"] == pytest.approx(1)


def test_fixed_size_abstains_from_inventing_slope():
    result = fit_affine([(100, 5e6)] * 20)
    assert result["status"] == "constant_fallback"
    assert result["k"] == 0 and result["b"] == 5


def test_nonnegative_fit_does_not_predict_negative_time():
    model = fit_affine([(x, (101-x)*1e6) for x in range(1, 100)])
    assert model["k"] >= 0 and model["b"] >= 0


def test_reorder_separates_selectivity_from_per_record_bytes():
    _, opt, _, (source, a, f, b) = setup()
    assert opt.calculate_cost(chain([source, a, f, b])) == pytest.approx(22)
    assert opt.calculate_cost(chain([source, f, a, b])) == pytest.approx(13)
    # Moving a constant-cost filter does not reduce the input bytes to shrink.
    assert opt._cm_reach == {}


def test_backend_uses_local_curve_and_direct_worker_anchor():
    _, opt, _, (source, a, f, b) = setup()
    desc = PipeDesc(None, PipeVariantType.RAY, None)
    assert opt._calculate_pipe_cost(b, 100, desc) == pytest.approx(12)
    assert opt.calculate_cost(chain([source, a, f, b]), {b: desc}) == pytest.approx(19)


def test_zero_selectivity_zeroes_all_downstream_intercepts():
    _, opt, profile, (source, a, f, b) = setup()
    profile["baseline"]["selectivities"][f] = 0
    opt._init_stats()
    assert opt.calculate_cost(chain([source, f, a, b])) == pytest.approx(5)


def test_search_passes_are_exactly_inherited_from_cedar():
    for name in ("_logical_opt", "_physical_opt", "_pass_reordering",
                 "_find_optimal_reordering", "_calculate_cost_fused",
                 "_calculate_local_parallelism"):
        assert getattr(CmOptimizer, name) is getattr(Optimizer, name)


def test_missing_model_is_rejected_instead_of_silent_old_model():
    _, opt, profile, _ = setup()
    profile.pop("cm_model")
    with pytest.raises(ValueError, match="cm_model"):
        opt._init_stats()


def test_local_profile_uses_real_prefix_inputs_and_counts_rejections():
    feature = Example()
    feature.apply(IterSource(["x"*i for i in range(10, 100)]))
    originals = {i: p.fn for i, p in feature.logical_pipes.items() if hasattr(p, "fn")}
    section = profile_linear_feature(feature, CedarContext(), n_samples=1000)
    models = {m["tag"]: m for m in section["operators"].values()}
    assert models["shrink"]["input_count"] == 90
    assert models["filter"]["input_count"] == 90
    assert 0 < models["filter"]["output_count"] < 90
    assert models["consumer"]["input_count"] == models["filter"]["output_count"]
    assert models["filter"]["input_mean_bytes"] < models["shrink"]["input_mean_bytes"]
    assert section["input_policy"] == "natural_capture_plus_controlled_sweep"
    from cedar.client.linear_cost_profile import TimedCallable
    for i, fn in originals.items():
        restored = feature.logical_pipes[i].fn
        assert not isinstance(restored, TimedCallable)
        assert restored("abcdefgh") == fn("abcdefgh")
    assert not feature.loaded
    yaml.safe_dump(section)


def test_selector_installs_cm_without_changing_execution():
    feature = Example()
    feature.apply(IterSource(["abcdefgh"]))
    dataset = DataSet(CedarContext(), {"feature": feature},
        enable_optimizer=False, enable_controller=False, prefetch=False,
        optimizer_options=OptimizerOptions(use_my_optimizer=16))
    try:
        assert isinstance(feature.optimizer, CmOptimizer)
        assert list(dataset) == ["abcd"]
    finally:
        dataset._exit()


def test_profile_wrapper_reuses_enhanced_path_and_adds_cm(monkeypatch, tmp_path):
    from cedar.client import dataset as module
    feature = Example()
    feature.apply(IterSource(["x"*i for i in range(10, 40)]))
    obj = object.__new__(DataSet)
    obj._cm_profile = True
    obj.features = {"feature": feature}
    obj.ctx = CedarContext()
    original = {"baseline": {"throughput": 123}, "offloads": {"SMP": {"direct": 1}}}
    obj._profile_standard = lambda *args: copy.deepcopy(original)
    destination = str(tmp_path/"profile.yaml")
    result = obj._profile("feature", n_samples=100, output_file=destination)
    assert result["baseline"] == original["baseline"]
    assert result["offloads"] == original["offloads"]
    assert result["cm_model"]["operators"]
    assert yaml.safe_load(open(destination)) == result


def test_complete_cedar_reorder_run_prefers_selective_prefix():
    from cedar.compose.utils import topological_sort
    feature, opt, profile, (source, a, f, b) = setup()
    plan = opt.run(profile, OptimizerOptions(
        enable_offload=False, enable_fusion=False, enable_caching=False,
        enable_local_parallelism=False, enable_prefetch=False,
        enable_reorder=True))
    assert topological_sort(plan.graph) == [source, f, a, b]
    assert opt.calculate_cost(plan.graph) == pytest.approx(13)
    opt.calculate_final_plan_cost_breakdown(plan)


def test_profile_failure_restores_callable_and_feature():
    class Broken(Feature):
        def _compose(self, source):
            def fail(value):
                raise RuntimeError("deliberate")
            return MapperPipe(source[0], fail)
    feature = Broken()
    feature.apply(IterSource(["x"]))
    pipe = next(p for p in feature.logical_pipes.values() if isinstance(p, MapperPipe))
    original = pipe.fn
    with pytest.raises(RuntimeError, match="deliberate"):
        profile_linear_feature(feature, CedarContext(), n_samples=1)
    assert pipe.fn is original and not feature.loaded
