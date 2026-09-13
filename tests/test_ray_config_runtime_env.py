from cedar.config import CedarContext, RayConfig
from evaluation.motivation_multimodal.runner import (
    _ray_runtime_env,
    _stage_ray_image_package,
)


def test_native_ray_connection_forwards_runtime_environment(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("cedar.config.ray.is_initialized", lambda: False)
    monkeypatch.setattr(
        "cedar.config.ray.init",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    runtime_env = {"py_modules": ["/workspace/OptimalCedar/cedar"]}

    CedarContext(
        RayConfig(ip="10.0.0.1:6379", runtime_env=runtime_env)
    ).init_ray()

    assert calls == [
        ((), {"address": "10.0.0.1:6379", "runtime_env": runtime_env})
    ]


def test_multimodal_runtime_uses_collision_free_actor_package(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("CEDAR_RAY_PY_MODULE_ROOT", raising=False)
    image_root = tmp_path / "source"
    image_root.mkdir()
    (image_root / "sample.jpg").write_bytes(b"same-image-bytes")
    dataset = tmp_path / "records.jsonl"
    dataset.write_text('{"image_path":"sample.jpg"}\n', encoding="utf-8")

    runtime_env = _ray_runtime_env(dataset, image_root, tmp_path / "packages")

    module_names = [path.rsplit("/", 1)[-1] for path in runtime_env["py_modules"]]
    assert module_names == ["cedar", "pico_multimodal", "pico_multimodal_data"]


def test_stage_ray_image_package_copies_only_referenced_images(tmp_path) -> None:
    image_root = tmp_path / "source"
    image_root.mkdir()
    (image_root / "used.jpg").write_bytes(b"used")
    (image_root / "unused.jpg").write_bytes(b"unused")
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        '{"image_path":"used.jpg"}\n{"image_path":"used.jpg"}\n',
        encoding="utf-8",
    )

    package = _stage_ray_image_package(dataset, image_root, tmp_path / "packages")

    assert (package / "used.jpg").read_bytes() == b"used"
    assert not (package / "unused.jpg").exists()
