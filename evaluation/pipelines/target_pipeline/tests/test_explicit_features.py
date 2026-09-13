"""Native equivalence, reference order, and executable sparse orders."""
import ast
import inspect
import random
import numpy as np
import pytest
import torch
from PIL import Image
from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.sources import IterSource
from evaluation.pipelines.target_pipeline.catalog import build_workload
from evaluation.pipelines.target_pipeline.simclr import cedar_dataset as simclr
from evaluation.pipelines.simclrv2 import cedar_dataset as native
from evaluation.pipelines.target_pipeline.dino.cedar_dataset import DINOFeature
from evaluation.pipelines.target_pipeline.swav.cedar_dataset import SwAVFeature
from evaluation.pipelines.target_pipeline.clip.cedar_dataset import CLIPFeature
from evaluation.pipelines.target_pipeline.blip.cedar_dataset import BLIPFeature

CLASSES = {"dino": DINOFeature, "swav": SwAVFeature,
           "clip": CLIPFeature, "blip": BLIPFeature}

def seed():
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)

@pytest.fixture
def record(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray(np.random.default_rng(7).integers(
        0, 256, (320, 480, 3), dtype=np.uint8)).save(path)
    return {"image": str(path), "caption": "Hello! World. " * 40}

def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b

def execute(feature, record):
    feature.apply(IterSource([record]))
    dataset = DataSet(CedarContext(), {"feature": feature},
                      enable_optimizer=False, enable_controller=False,
                      prefetch=False)
    try:
        return list(dataset)[0]
    finally:
        dataset._exit()

def test_simclr_copies_native_feature_and_operators(record):
    for name in ("SimCLRV2Feature", "to_float"):
        assert ast.dump(ast.parse(inspect.getsource(getattr(simclr, name)))) == (
            ast.dump(ast.parse(inspect.getsource(getattr(native, name)))))
    seed()
    expected = execute(native.SimCLRV2Feature(1), record["image"])
    seed()
    equal(execute(simclr.SimCLRV2Feature(1), record["image"]), expected)

@pytest.mark.parametrize("name", CLASSES)
def test_default_order_matches_previous_reference_recipe(name, record):
    workload = build_workload(name)
    state = workload.prepare(record)
    seed()
    for stage in workload.stages:
        state[stage.target] = stage.operation(state[stage.source])
    expected = workload.finalize(state)
    seed()
    equal(execute(CLASSES[name](), record), expected)

def reorder_class(base, order_seed):
    class Reordered(base):
        def _compose(self, sources):
            sink = super()._compose(sources)
            chain = []
            pipe = sink
            while pipe is not sources[0]:
                chain.append(pipe)
                pipe = pipe.input_pipes[0]
            chain.reverse()
            assert chain[0]._fix_order and chain[-1]._fix_order
            remaining = list(chain[1:-1])
            done, ordered = set(), []
            rng = random.Random(order_seed)
            while remaining:
                ready = [p for p in remaining
                         if set(p._depends_on_tags or []) <= done]
                assert ready
                pipe = rng.choice(ready)
                remaining.remove(pipe)
                ordered.append(pipe)
                done.add(pipe.tag)
            self.test_order = [p.tag for p in ordered]
            pipe = chain[0]
            for next_pipe in ordered + [chain[-1]]:
                next_pipe.set_input_pipes([pipe])
                pipe = next_pipe
            return pipe
    return Reordered

@pytest.mark.parametrize("name", CLASSES)
@pytest.mark.parametrize("order_seed", range(4))
def test_sparse_orders_execute_with_valid_outputs(name, order_seed, record):
    feature = reorder_class(CLASSES[name], order_seed)()
    seed()
    result = execute(feature, record)
    if name in ("dino", "swav"):
        sides = [224, 224] + [96] * (8 if name == "dino" else 6)
        assert len(result["views"]) == len(sides)
        for view, side in zip(result["views"], sides):
            assert view.shape == (3, side, side)
            assert view.dtype == torch.float32 and torch.isfinite(view).all()
        assert any(feature.test_order.index(f"view{i}_jitter") <
                   feature.test_order.index(f"view{i}_crop")
                   for i in range(len(sides)))
    else:
        assert result["pixel_values"].shape == (3, 224, 224)
        assert torch.isfinite(result["pixel_values"]).all()
        if name == "blip":
            assert result["caption"] == " ".join(["hello", "world"] * 15)
        else:
            assert result["input_ids"].shape == (77,)
            assert result["attention_mask"].shape == (77,)

def test_clip_float_can_precede_resize():
    feature = CLIPFeature()
    feature.apply(IterSource([]))
    by_tag = {p.tag: p for p in feature.logical_pipes.values()}
    assert not by_tag["float"]._depends_on_tags
    assert by_tag["crop"]._depends_on_tags == ["resize"]
    assert set(by_tag["normalize"]._depends_on_tags) == {"float", "crop"}

def test_blip_aug_can_precede_crop():
    feature = BLIPFeature()
    feature.apply(IterSource([]))
    by_tag = {p.tag: p for p in feature.logical_pipes.values()}
    assert not by_tag["augment"]._depends_on_tags
    assert by_tag["flip"]._depends_on_tags == ["crop"]
    assert by_tag["truncate"]._depends_on_tags == ["clean"]


def test_blip_pil_adapter_preserves_upstream_pixels(record):
    from evaluation.pipelines.target_pipeline.blip.cedar_dataset import PILRandomAugment
    from evaluation.pipelines.target_pipeline.references.blip.transform.randaugment import RandomAugment
    image = Image.open(record["image"]).convert("RGB")
    seed()
    expected = RandomAugment(2, 5, isPIL=True)(image)
    seed()
    actual = PILRandomAugment(2, 5, isPIL=True)(image)
    assert isinstance(actual, Image.Image)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_dispatch_uses_explicit_modules(monkeypatch):
    from types import SimpleNamespace
    from importlib import import_module
    from evaluation.pipelines.target_pipeline.cedar_dataset import get_dataset_for
    for name in ("simclr", "dino", "swav", "clip", "blip"):
        module = import_module(
            f"evaluation.pipelines.target_pipeline.{name}.cedar_dataset")
        sentinel = object()
        monkeypatch.setattr(module, "get_dataset", lambda spec: sentinel)
        assert get_dataset_for(name, SimpleNamespace()) is sentinel


@pytest.mark.parametrize("name", CLASSES)
def test_manifest_entrypoint_and_batching(name, record, tmp_path):
    import json
    from evaluation.cedar_utils import CedarEvalSpec
    from evaluation.eval_cedar import import_module_from_path
    from pathlib import Path
    manifest = tmp_path / "records.jsonl"
    manifest.write_text((json.dumps(record) + "\n") * 2)
    path = Path(__file__).resolve().parents[1] / name / "cedar_dataset.py"
    module = import_module_from_path(str(path))
    spec = CedarEvalSpec(
        batch_size=2, num_total_samples=2, num_epochs=1,
        kwargs={"dataset_path": str(manifest)},
        disable_optimizer=True, disable_controller=True, disable_prefetch=True)
    dataset = module.get_dataset(spec)
    try:
        batches = list(dataset)
        assert len(batches) == 1
        assert len(batches[0]) == 2
    finally:
        dataset._exit()
