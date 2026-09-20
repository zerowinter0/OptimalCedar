import pickle
from types import SimpleNamespace

import pytest

from cedar.client.cuda_work_profiler import metadata_counterfactual, assess_invariance
from cedar.pipes import PipeExecutionResource


def test_metadata_counterfactual_preserves_payload_and_changes_only_record_size():
    source={"text":"<image> a cat","images":["x.jpg"],"context":{"image_root":"/data"}}
    raw=pickle.dumps(source)
    padded=metadata_counterfactual([raw], padding_bytes=2048)[0]
    result=pickle.loads(padded)
    assert result["text"]==source["text"]
    assert result["images"]==source["images"]
    assert result["context"]==source["context"]
    assert len(padded)>len(raw)+1900


def test_invariance_requires_byte_contrast_and_precise_equal_timings():
    base=[{"mean_ms_per_sample":30.0,"stderr_ms_per_sample":0.05,"adaptive_profile":{"converged":True}}]*2
    padded=[{"mean_ms_per_sample":30.2,"stderr_ms_per_sample":0.05,"adaptive_profile":{"converged":True}}]*2
    assert assess_invariance(base,padded,1000.0,2000.0)["accepted"]
    assert not assess_invariance(base,padded,1000.0,1010.0)["accepted"]
    slow=[{"mean_ms_per_sample":35.0,"stderr_ms_per_sample":0.05,"adaptive_profile":{"converged":True}}]*2
    assert not assess_invariance(base,slow,1000.0,2000.0)["accepted"]


def test_unconverged_backend_timings_cannot_prove_metadata_invariance():
    unconverged = {"mean_ms_per_sample": 30.0,
                   "stderr_ms_per_sample": 0.01,
                   "adaptive_profile": {"converged": False}}
    result = assess_invariance(
        [unconverged] * 2, [unconverged] * 2, 1000.0, 2000.0
    )
    assert not result["accepted"]


def test_zero_slope_cuda_coefficients_ignore_metadata_growth(monkeypatch):
    """Accepted metadata invariance is expressed as k = 0 coefficients."""
    from cedar.compose.simple_dp_ablation_optimizer import SimpleDpBoundaryOptimizer
    from cedar.pipes import PipeVariantType
    from tests.test_simple_dp_ablations import make
    opt, _ = make(SimpleDpBoundaryOptimizer, monkeypatch)
    p_id = next(iter(opt.profiled_stats["offloads"]["RAY"]))
    reference = opt.profiled_stats["baseline"]["input_sizes"][p_id]
    model = opt.profiled_stats["physical_model"]["operator_affine"][
        "operators"
    ][str(p_id)]
    model["k_ms_per_byte"] = 0.0
    model["b_ms"] = 2.0
    model["x_reference_bytes"] = reference
    desc = SimpleNamespace(variant_type=PipeVariantType.RAY)
    mean = opt.profiled_stats["offloads"]["RAY"][p_id][
        "backend_compute"
    ]["mean_ms_per_sample"]
    anchor = opt._calculate_pipe_cost(p_id, reference, desc)
    grown = opt._calculate_pipe_cost(p_id, 2 * reference, desc)
    assert anchor == pytest.approx(mean)
    assert grown == pytest.approx(anchor)


def test_affine_work_product_uses_surviving_record_cardinality(monkeypatch):
    from cedar.compose.simple_dp_ablation_optimizer import SimpleDpBoundaryOptimizer
    from tests.test_simple_dp_ablations import make
    opt, _ = make(SimpleDpBoundaryOptimizer, monkeypatch)
    p_id = opt._dp_inner_ops[0]
    opt._dp_cardinality_prod = [1.0, 0.5]
    opt._dp_r_prod = [1.0, 3.0]
    source_size = opt.profiled_stats["baseline"]["output_sizes"][
        opt._get_source_p_id()
    ]
    model = opt.profiled_stats["physical_model"]["operator_affine"][
        "operators"
    ][str(p_id)]
    k = model["k_ms_per_byte"]
    b = model["b_ms"]
    item_size = source_size * 3.0
    # Half the records survive and each one pays k * bytes + b.
    assert opt._dp_compute_work_prod(1, 0) == pytest.approx(
        0.5 * (k * item_size + b)
    )
    # The denominator is the profiled per-record cost at the profiled size;
    # reached-record weighting lives in the work product.
    assert opt._dp_compute_cost_denominator(0, 900.0, 300.0) == pytest.approx(
        300.0 * (k * 900.0 + b)
    )


def test_zero_slope_coefficients_remove_byte_dependence():
    from tests.test_affine_dp_cost import affine_dp
    opt, ids = affine_dp()
    p_id = ids[-1]
    baseline_size = opt.profiled_stats["baseline"]["input_sizes"][p_id]
    assert opt._dp_affine_value(p_id, baseline_size * 2) != opt._dp_affine_value(p_id, baseline_size)
    model = opt.profiled_stats["physical_model"]["operator_affine"][
        "operators"
    ][str(p_id)]
    anchor = opt._dp_affine_value(p_id, baseline_size)
    model["k_ms_per_byte"] = 0.0
    model["b_ms"] = anchor
    assert opt._dp_affine_value(p_id, baseline_size * 2) == anchor
    assert opt._dp_affine_value(p_id, baseline_size) == anchor


def test_metadata_invariance_fit_reports_zero_slope_coefficients():
    base = [
        {"mean_ms_per_sample": 30.0, "stderr_ms_per_sample": 0.05,
         "adaptive_profile": {"converged": True}}
    ] * 2
    padded = [
        {"mean_ms_per_sample": 30.2, "stderr_ms_per_sample": 0.05,
         "adaptive_profile": {"converged": True}}
    ] * 2
    result = assess_invariance(base, padded, 1000.0, 5000.0)
    assert result["accepted"]
    coefficients = result["affine_coefficients"]
    assert coefficients["k_ms_per_byte"] == pytest.approx(0.00005)
    assert coefficients["b_ms"] == pytest.approx(29.95)
    assert coefficients["x_reference_bytes"] == 1000.0
    assert coefficients["fixed_fraction"] == pytest.approx(29.95 / 30.0)


def test_llava_model_work_screen_rejects_noop_records():
    from cedar.client.cuda_work_profiler import has_llava_model_work
    assert not has_llava_model_work(pickle.dumps({"raw_content": "cat", "images": []}))
    assert not has_llava_model_work(pickle.dumps({"raw_content": "cat", "images": ["cat.jpg"]}))
    assert has_llava_model_work(pickle.dumps({"raw_content": "<image> cat", "images": ["cat.jpg"]}))


def test_layered_boundary_charges_only_surviving_records(monkeypatch):
    from cedar.compose.simple_dp_ablation_optimizer import SimpleDpBoundaryOptimizer
    from cedar.pipes import PipeVariantType
    from tests.test_simple_dp_ablations import make

    opt, _ = make(SimpleDpBoundaryOptimizer, monkeypatch)
    opt._dp_cardinality_prod = [1.0, 0.25, 1.0, 0.125]
    opt._dp_r_prod = [1.0, 2.0, 1.0, 3.0]
    monkeypatch.setattr(opt, "_dp_boundary_fixed_latency_ms", lambda *_: 4.0)
    monkeypatch.setattr(opt, "_dp_boundary_throughput", lambda *_: 1000.0)
    monkeypatch.setattr(opt, "_dp_ray_submit_batch_size", lambda *_: 2)
    block = SimpleNamespace(cost=0.0, variant=PipeVariantType.RAY, mask=2)
    source_size = opt.profiled_stats["baseline"]["output_sizes"][opt._get_source_p_id()]
    expected = 0.25 * 4.0 / 2 + (
        source_size * 2.0 * 0.25 + source_size * 3.0 * 0.125
    ) / 1000.0 * 1000.0
    assert opt._dp_regular_transition_cost(1, block) == expected


def test_dual_profile_preserves_legacy_tf_measurement_order(monkeypatch, tmp_path):
    from cedar.client import DataSet
    dataset = DataSet.__new__(DataSet)
    dataset.ctx = SimpleNamespace(use_ray=lambda: False)
    dataset.features = {"feature": SimpleNamespace(logical_pipes={})}
    calls = []
    monkeypatch.setenv("CEDAR_LAYERED_ADAPTIVE_PROFILE", "1")
    monkeypatch.setenv("CEDAR_PROFILE_BOUNDARY_MODEL", "0")
    monkeypatch.setattr(dataset, "_init_ctx", lambda: None)
    monkeypatch.setattr(dataset, "_profile_feature", lambda *args: calls.append("baseline") or {})
    monkeypatch.setattr(dataset, "_profile_smp", lambda *args, **kwargs: calls.append("smp"))
    monkeypatch.setattr(dataset, "_profile_tf", lambda *args: calls.append("tf"))
    monkeypatch.setattr(dataset, "_collect_profile_input_reservoir",
                        lambda *args: calls.append("capture"))
    monkeypatch.setattr(dataset, "_profile_layered_backends",
                        lambda *args: calls.append("layered"))
    monkeypatch.setattr(dataset, "_profile_io", lambda: (0.0, 0.0))
    dataset._profile_standard("feature", 1, str(tmp_path / "profile.yaml"))
    assert calls == ["baseline", "smp", "tf", "capture", "layered"]


def test_cuda_actor_trials_require_one_remote_gpu_location():
    from cedar.client.cuda_work_profiler import assess_actor_placement
    rows = [{"actor_locations": [{"node_ip": "10.0.0.2", "gpu_ids": [0]}]}] * 4
    assert assess_actor_placement(rows, "10.0.0.1")["accepted"]
    changed = rows[:3] + [{"actor_locations": [{"node_ip": "10.0.0.3", "gpu_ids": [0]}]}]
    assert not assess_actor_placement(changed, "10.0.0.1")["accepted"]
    assert not assess_actor_placement(rows, "10.0.0.2")["accepted"]


def test_adaptive_series_reuses_one_warmed_actor_for_counterfactuals():
    from cedar.client import DataSet
    from cedar.pipes import PipeVariantType

    class Service:
        def reset_backend_compute_stats(self):
            self.values = []

        def get_backend_compute_stats(self):
            if not self.values:
                return None
            return {"count": len(self.values), "mean_ms_per_sample":
                    sum(self.values) / len(self.values),
                    "stderr_ms_per_sample": 0.0}

    class Variant:
        def __init__(self, replay):
            self.replay = replay
            self.service = Service()
            self.service.reset_backend_compute_stats()
            self.stopped = False

        def __iter__(self):
            value = pickle.loads(self.replay.snapshots[0])
            for _ in range(self.replay.record_count):
                self.service.values.append(float(value))
                yield value

        def shutdown(self):
            self.stopped = True

    predecessor = SimpleNamespace(id=7, pipe_variant=None)
    created = []

    def create_variant(*_):
        variant = Variant(predecessor.pipe_variant)
        created.append(variant)
        return variant

    pipe = SimpleNamespace(id=8, input_pipes=[predecessor], pipe_spec=None,
                           execution_resource=PipeExecutionResource.CPU,
                           _create_pipe_variant=create_variant)
    trial_inputs = [("original", [pickle.dumps(1)]),
                    ("padded", [pickle.dumps(2)])]
    result = DataSet.__new__(DataSet)._adaptive_operator_benchmark(
        pipe, trial_inputs[0][1], PipeVariantType.RAY, 1,
        min_duration=0, max_duration=1, target_rse=0.1,
        min_observations=2, snapshot_sequence=trial_inputs)
    assert len(created) == 1
    assert created[0].stopped
    assert [row["mean_ms_per_sample"] for row in result] == [1.0, 2.0]


def test_cuda_profiler_replays_abba_inside_one_actor(monkeypatch):
    """All four counterfactual conditions share one warmed remote actor."""
    import ray

    from cedar.client import cuda_work_profiler as module

    class ImageTextMatchingFilter:
        pass

    ImageTextMatchingFilter.__module__ = (
        "evaluation.pipelines.llava_pretrain.dj_operators"
    )
    pipe = SimpleNamespace(
        id=1,
        fn=ImageTextMatchingFilter(),
        input_pipes=[SimpleNamespace(id=0)],
    )
    feature = SimpleNamespace(logical_pipes={1: pipe})
    record = {
        "raw_content": "<image> a cat",
        "images": ["cat.jpg"],
        "context": {"image_root": "/data"},
    }
    reservoir = SimpleNamespace(
        values_for=lambda p_id: [pickle.dumps(record) for _ in range(4)]
    )

    calls = []
    location = [{"node_ip": "10.0.0.2", "gpu_ids": [0]}]

    def benchmark(pipe_arg, snapshots, variant_type, width, **kwargs):
        calls.append({"snapshots": snapshots, "kwargs": kwargs})
        rows = []
        for label, _ in kwargs["snapshot_sequence"]:
            rows.append({
                "mean_ms_per_sample": 30.0 if label == "original" else 30.2,
                "stderr_ms_per_sample": 0.01,
                "adaptive_profile": {"converged": True},
                "actor_locations": location,
            })
        return rows

    dataset = SimpleNamespace(_adaptive_operator_benchmark=benchmark)
    monkeypatch.setattr(
        ray.util, "get_node_ip_address", lambda: "10.0.0.1"
    )

    result = module.profile_cuda_metadata_invariance(
        dataset, feature, reservoir
    )

    assert len(calls) == 1, "the four conditions must share one actor"
    sequence = calls[0]["kwargs"]["snapshot_sequence"]
    assert [label for label, _ in sequence] == [
        "original", "padded", "padded", "original"
    ]
    assert calls[0]["kwargs"]["record_actor_locations"] is True

    entry = result["operators"]["1"]
    assert entry["method"] == "same_actor_abba_metadata_counterfactual"
    assert entry["accepted"] is True
    assert entry["actor_placement"]["accepted"] is True
    assert len(entry["runs"]["original"]) == 2
    assert len(entry["runs"]["padded"]) == 2
