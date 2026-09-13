import copy
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def test_suite_is_implemented():
    assert (ROOT / "catalog.py").is_file(), "target pipeline catalog is missing"

def suite():
    from evaluation.pipelines.target_pipeline.catalog import build_workload
    return build_workload

def example(tmp_path):
    path = tmp_path / "sample.png"
    Image.fromarray(np.random.default_rng(7).integers(
        0, 256, (320, 480, 3), dtype=np.uint8)).save(path)
    return {"id": "sample", "image": str(path),
            "caption": "A photo of a dog. " * 40}

def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b

@pytest.mark.parametrize("name", ["simclr", "dino", "swav", "clip", "blip"])
def test_legal_interleavings_preserve_output(name, tmp_path):
    workload = suite()(name, seed=23)
    record = example(tmp_path)
    reference = workload.run(record)
    for order in workload.sample_orders(4):
        equal(reference, workload.run(record, order=order))
    assert workload.order_count() >= 10
    assert any(stage.scaling != "none" for stage in workload.stages)

@pytest.mark.parametrize("name,count,sides", [
    ("simclr", 2, [244,244]),
    ("dino", 10, [224,224]+[96]*8),
    ("swav", 8, [224,224]+[96]*6),
])
def test_multicrop_dimensions(name, count, sides, tmp_path):
    workload = suite()(name)
    result = workload.run(example(tmp_path))
    assert len(result["views"]) == count
    assert [v.shape[-1] for v in result["views"]] == sides

def test_invalid_order_is_rejected(tmp_path):
    workload = suite()("dino")
    with pytest.raises(ValueError, match="depend"):
        workload.run(example(tmp_path), order=list(reversed(workload.tags)))

def test_blip_caption_matches_reference_and_truncates(tmp_path):
    result = suite()("blip").run(example(tmp_path))
    assert result["caption"] == " ".join(("a photo of a dog " * 40).split()[:30])
    assert result["pixel_values"].shape == (3,224,224)

def test_stage_pickle_roundtrip(tmp_path):
    import pickle
    workload = pickle.loads(pickle.dumps(suite()("dino")))
    result = workload.run(example(tmp_path))
    assert len(result["views"]) == 10

@pytest.mark.parametrize("name", ["simclr", "dino", "swav", "clip", "blip"])
def test_cedar_execution_and_dependency_contract(name, tmp_path):
    from cedar.client import DataSet
    from cedar.config import CedarContext
    from cedar.sources import IterSource
    from evaluation.pipelines.target_pipeline.cedar_dataset import TargetFeature
    workload = suite()(name, seed=13)
    record = example(tmp_path)
    feature = TargetFeature(workload)
    feature.apply(IterSource([record]))
    by_tag = {pipe.tag: pipe for pipe in feature.logical_pipes.values()
              if pipe.tag}
    for stage in workload.stages:
        assert tuple(by_tag[stage.tag]._depends_on_tags or ()) == stage.dependencies
    dataset = DataSet(CedarContext(), {"feature": feature},
                      enable_optimizer=False, enable_controller=False)
    results = list(dataset)
    equal(results[0], workload.run(record))
    dataset._exit()

@pytest.mark.parametrize("name", ["simclr", "dino", "swav", "clip", "blip"])
def test_reordered_cedar_execution(name, tmp_path):
    from cedar.client import DataSet
    from cedar.config import CedarContext
    from cedar.sources import IterSource
    from evaluation.pipelines.target_pipeline.cedar_dataset import TargetFeature
    workload = suite()(name, seed=15)
    record = example(tmp_path)
    order = next(workload.sample_orders(1, seed=31))
    feature = TargetFeature(workload, order=order)
    feature.apply(IterSource([record]))
    dataset = DataSet(CedarContext(), {"feature": feature},
                      enable_optimizer=False, enable_controller=False)
    equal(list(dataset)[0], workload.run(record))
    dataset._exit()

def test_manifest_conversion_preserves_llava_pairs(tmp_path):
    import json
    from evaluation.pipelines.target_pipeline.prepare_manifest import convert_llava
    source=tmp_path/"source.jsonl"
    source.write_text(json.dumps({
        "id":"one","image":"000/image.jpg",
        "conversations":[{"value":"question"},{"value":"original caption"}]
    })+"\n")
    destination=tmp_path/"manifest.jsonl"
    assert convert_llava(source,destination,tmp_path)==1
    row=json.loads(destination.read_text())
    assert row == {"id":"one","image":str(tmp_path/"000/image.jpg"),
                   "caption":"original caption"}
    with pytest.raises(FileExistsError):
        convert_llava(source,destination,tmp_path)

def test_random_state_is_restored(tmp_path):
    import random
    from evaluation.pipelines.target_pipeline.core import random_stream
    random.seed(7)
    torch.manual_seed(7)
    np.random.seed(7)
    py_state=random.getstate()
    torch_state=torch.get_rng_state().clone()
    np_state=np.random.get_state()
    with pytest.raises(RuntimeError):
        with random_stream(42):
            random.random(); torch.rand(2); np.random.rand()
            raise RuntimeError("test exception")
    assert random.getstate()==py_state
    assert torch.equal(torch.get_rng_state(),torch_state)
    assert np.array_equal(np.random.get_state()[1],np_state[1])


def test_data_juicer_target_catalog_has_five_audited_scaler_workloads():
    from evaluation.pipelines.target_pipeline.hub_catalog import HUB_WORKLOADS
    from evaluation.pipelines.pile_recipe_registry import RECIPES

    assert len(HUB_WORKLOADS) == 5
    assert {
        item.name for item in HUB_WORKLOADS
    } <= set(RECIPES) | {"pile_europarl"}
    for item in HUB_WORKLOADS:
        assert item.reorderable_scalers
        if item.name in RECIPES:
            available = {spec.tag for spec in RECIPES[item.name].filters}
        else:
            available = {"text_length", "words_num", "alphanumeric", "perplexity"}
        assert set(item.reorderable_scalers) <= available
        assert item.official_recipe.startswith("refined_recipes/pretrain/")


@pytest.mark.parametrize(
    "name",
    [
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_europarl",
        "pile_uspto_backgrounds",
    ],
)
def test_target_hub_adapter_uses_frozen_recipe(name):
    from evaluation.pipelines.target_pipeline.hub_dataset import is_hub_workload

    assert is_hub_workload(name)
