import pickle

import pytest

from cedar.client import boundary_profiler
from cedar.client.boundary_profiler import (
    DEFAULT_PAYLOAD_BYTES,
    fit_boundary_model,
    profile_object_marshalling,
    profile_stage_boundary_cached,
)
from cedar.pipes import PipeVariantType


def test_synthetic_boundary_calibration_covers_small_objects():
    assert min(DEFAULT_PAYLOAD_BYTES) <= 512
    assert 4 * 1024 in DEFAULT_PAYLOAD_BYTES


def test_object_marshalling_uses_real_nested_values():
    snapshots = [
        pickle.dumps({"text": "x" * 100, "ids": list(range(16))})
        for _ in range(3)
    ]
    result = profile_object_marshalling(
        snapshots, PipeVariantType.SMP, max_samples=2
    )
    assert result["samples"] == 2
    assert result["serialized_bytes_per_sample"] > 100
    assert result["serialize_ms_per_sample"] >= 0
    assert result["deserialize_ms_per_sample"] >= 0


def test_fit_boundary_model_recovers_latency_and_bandwidth():
    latency_sec = 0.00025
    throughput = 2_000_000_000.0
    measurements = [
        {
            "payload_bytes": float(payload),
            "boundary_sec_per_sample": (
                latency_sec + 2.0 * payload / throughput
            ),
        }
        for payload in (64 * 1024, 1024 * 1024, 4 * 1024 * 1024)
    ]

    model = fit_boundary_model(measurements)

    assert model["fixed_latency_ms"] == pytest.approx(0.25)
    assert model["throughput_bytes_per_sec"] == pytest.approx(throughput)
    assert model["r_squared"] == pytest.approx(1.0)


def test_fit_boundary_model_rejects_degenerate_payload_sizes():
    with pytest.raises(RuntimeError, match="payload sizes are degenerate"):
        fit_boundary_model(
            [
                {
                    "payload_bytes": 1024.0,
                    "boundary_sec_per_sample": 0.001,
                },
                {
                    "payload_bytes": 1024.0,
                    "boundary_sec_per_sample": 0.002,
                },
            ]
        )


def test_boundary_calibration_reuses_exact_signature(tmp_path, monkeypatch):
    calls = []

    def fake_profile(**kwargs):
        calls.append(kwargs)
        return {
            "throughput_bytes_per_sec": 123.0,
            "fixed_latency_ms": 0.5,
        }

    monkeypatch.setattr(
        boundary_profiler, "profile_stage_boundary", fake_profile
    )
    cache = tmp_path / "boundary.json"
    first = profile_stage_boundary_cached(
        ctx=object(),
        variant=PipeVariantType.SMP,
        width=3,
        cache_path=str(cache),
    )
    second = profile_stage_boundary_cached(
        ctx=object(),
        variant=PipeVariantType.SMP,
        width=3,
        cache_path=str(cache),
    )

    assert len(calls) == 1
    assert first["calibration_source"] == "measured"
    assert second["calibration_source"] == "cache"
    assert second["calibration_key"] == first["calibration_key"]

def test_ray_calibration_includes_large_payloads():
    assert max(DEFAULT_PAYLOAD_BYTES) >= 16 * 1024 * 1024


def test_boundary_ray_pool_uses_actor_placement_options(monkeypatch):
    options = []
    class FakeActor:
        def location(self):
            pass
    class Method:
        def remote(self):
            return "location_ref"
    handle = FakeActor()
    handle.location = Method()
    class Factory:
        def options(self, **kwargs):
            options.append(kwargs)
            return self
        def remote(self):
            return handle
    monkeypatch.setattr(boundary_profiler, "_BoundaryRayActor", Factory())
    monkeypatch.setattr(boundary_profiler, "get_ray_actor_options",
                        lambda: {"num_cpus": 1, "resources": {"cedar_remote": 0.001}})
    monkeypatch.setattr(boundary_profiler.ray, "get",
                        lambda refs, **kwargs: [{"ip": "remote"}])
    monkeypatch.setattr(boundary_profiler.ray, "kill", lambda actor: None)
    pool = boundary_profiler._RoundTripPool(PipeVariantType.RAY, 1)
    assert options == [{"num_cpus": 1, "resources": {"cedar_remote": 0.001}}]
    assert pool.actor_locations == [{"ip": "remote"}]
    pool.shutdown()


def test_remote_boundary_validator_rejects_legacy_or_failed_profiles():
    with pytest.raises(RuntimeError, match="remote.*boundary"):
        boundary_profiler.validate_remote_ray_boundary({
            "physical_model": {"calibration_errors": {"RAY": "noisy"}}})
    with pytest.raises(RuntimeError, match="remote.*boundary"):
        boundary_profiler.validate_remote_ray_boundary({
            "physical_model": {"boundary": {"RAY": {
                "throughput_bytes_per_sec": 1e10, "fixed_latency_ms": 0}}}})


def test_remote_boundary_validator_accepts_proven_remote_measurement():
    profile = {"physical_model": {"boundary": {"RAY": {
        "throughput_bytes_per_sec": 110e6, "fixed_latency_ms": 0.5,
        "r_squared": 0.99,
        "calibration_schema_version": boundary_profiler.CALIBRATION_SCHEMA_VERSION,
        "placement_resource": "cedar_remote", "driver_ip": "local",
        "actor_locations": [{"ip": "remote"}]}}}}
    boundary_profiler.validate_remote_ray_boundary(profile)
    profile["physical_model"]["boundary"]["RAY"]["actor_locations"] = [{"ip": "local"}]
    with pytest.raises(RuntimeError, match="remote.*boundary"):
        boundary_profiler.validate_remote_ray_boundary(profile)


def test_ray_calibration_failure_preserves_raw_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("CEDAR_BOUNDARY_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setattr(boundary_profiler, "_validate_variant_ready", lambda *a: None)
    class Pool:
        actor_locations = [{"ip": "remote"}]
        def __init__(self, **kwargs): pass
        def shutdown(self): pass
    monkeypatch.setattr(boundary_profiler, "_RoundTripPool", Pool)
    monkeypatch.setattr(boundary_profiler, "_measure_round_trip", lambda **kw: 0.01)
    with pytest.raises(RuntimeError):
        boundary_profiler.profile_stage_boundary(
            object(), PipeVariantType.RAY, 1, payload_bytes=[512, 4096])
    reports = list(tmp_path.glob("*.json"))
    assert reports
    import json
    result = json.loads(reports[0].read_text())
    assert len(result["runs"]) == 4
    assert result["actor_locations"] == [{"ip": "remote"}]

def test_ray_boundary_failure_aborts_dataset_profiling(monkeypatch):
    from cedar.client.dataset import DataSet
    import cedar.client.dataset as dataset_module
    dataset = object.__new__(DataSet)
    dataset.ctx = object()
    def fail(**kwargs):
        raise RuntimeError("noisy measurement")
    monkeypatch.setattr(dataset_module, "profile_stage_boundary_cached", fail)
    profile = {}
    with pytest.raises(RuntimeError, match="refusing unmeasured"):
        dataset._profile_boundary_model(profile, PipeVariantType.RAY, 1)
    assert "RAY" in profile["physical_model"]["calibration_errors"]


def test_optimizer_cannot_use_constant_in_remote_experiment(monkeypatch):
    from cedar.compose.my_optimizer import MyOptimizer
    monkeypatch.setenv("CEDAR_RAY_REQUIRE_REMOTE", "1")
    opt = MyOptimizer()
    opt.profiled_stats = {}
    with pytest.raises(RuntimeError, match="remote.*boundary"):
        opt._dp_boundary_throughput(PipeVariantType.RAY)


def test_ray_cache_signature_tracks_placement_nodes(monkeypatch):
    monkeypatch.setenv("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
    monkeypatch.setattr(boundary_profiler.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(boundary_profiler.ray.util, "get_node_ip_address", lambda: "local")
    monkeypatch.setattr(boundary_profiler.ray, "cluster_resources",
                        lambda: {"CPU": 128, "cedar_remote": 1})
    nodes = [{"NodeID": "node1", "NodeManagerAddress": "remote1",
              "Alive": True, "Resources": {"cedar_remote": 1}}]
    monkeypatch.setattr(boundary_profiler.ray, "nodes", lambda: nodes)
    first = boundary_profiler.boundary_calibration_signature(PipeVariantType.RAY, 1)
    import json
    assert json.loads(json.dumps(first)) == first
    nodes[0]["NodeManagerAddress"] = "remote2"
    second = boundary_profiler.boundary_calibration_signature(PipeVariantType.RAY, 1)
    assert first != second
