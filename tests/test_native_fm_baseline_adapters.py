from types import SimpleNamespace

from evaluation.pipelines import native_fm_adapters
from evaluation.pipelines.datajuicer_workloads import PipelineStage


def _uppercase(value):
    return value.upper()


def _keep(value):
    return "KEEP" in value


def _stub_stages(workload, image_root=""):
    return [
        PipelineStage("map", "uppercase", _uppercase),
        PipelineStage("filter", "keep", _keep),
    ]


def _spec(path, **kwargs):
    values = {
        "batch_size": 1,
        "num_workers": 0,
        "num_parallel_calls": 1,
        "kwargs": {"dataset_path": str(path), **kwargs},
    }
    return SimpleNamespace(**values)


def test_torch_fm_adapter_executes_shared_stages(tmp_path, monkeypatch):
    source = tmp_path / "samples.jsonl"
    source.write_text("keep first\ndrop second\n", encoding="utf-8")
    monkeypatch.setattr(native_fm_adapters, "build_stages", _stub_stages)

    dataset = native_fm_adapters.get_torch_dataset(
        _spec(source),
        "stackexchange",
    )

    assert [[value.strip() for value in batch] for batch in dataset] == [
        ["KEEP FIRST"]
    ]


def test_tf_fm_adapter_executes_shared_stages(tmp_path, monkeypatch):
    source = tmp_path / "samples.jsonl"
    source.write_text("keep café 世界\ndrop second\n", encoding="utf-8")
    monkeypatch.setattr(native_fm_adapters, "build_stages", _stub_stages)

    dataset = native_fm_adapters.get_tf_dataset(
        _spec(source),
        "stackexchange",
    )

    assert [batch.numpy().tolist() for batch in dataset] == [
        ["KEEP CAFÉ 世界".encode("utf-8")]
    ]


def test_ray_fm_adapter_builds_native_map_filter_chain(
    tmp_path,
    monkeypatch,
):
    import ray

    source = tmp_path / "samples.jsonl"
    source.write_text("keep first\n", encoding="utf-8")
    monkeypatch.setattr(native_fm_adapters, "build_stages", _stub_stages)

    class FakeDataset:
        def __init__(self):
            self.calls = []

        def map(self, operator):
            self.calls.append(("map", operator))
            return self

        def filter(self, operator):
            self.calls.append(("filter", operator))
            return self

    fake = FakeDataset()
    monkeypatch.setattr(ray.data, "read_text", lambda path: fake)

    result = native_fm_adapters.get_ray_dataset(
        _spec(source),
        "stackexchange",
    )

    assert result is fake
    assert [kind for kind, _ in fake.calls] == ["map", "filter"]


def test_ray_fm_adapter_wraps_scalar_final_output_as_row(
    tmp_path,
    monkeypatch,
):
    import ray

    source = tmp_path / "samples.jsonl"
    source.write_text("keep first\n", encoding="utf-8")
    monkeypatch.setattr(
        native_fm_adapters,
        "build_stages",
        lambda workload, image_root="": [
            PipelineStage("map", "uppercase", _uppercase),
        ],
    )

    class FakeDataset:
        def __init__(self):
            self.operator = None

        def map(self, operator):
            self.operator = operator
            return self

    fake = FakeDataset()
    monkeypatch.setattr(ray.data, "read_text", lambda path: fake)

    native_fm_adapters.get_ray_dataset(_spec(source), "redpajama_c4")

    assert fake.operator("keep first") == {"item": "KEEP FIRST"}
