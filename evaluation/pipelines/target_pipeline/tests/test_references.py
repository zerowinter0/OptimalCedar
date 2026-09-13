"""Independent comparisons with the pinned upstream preprocessing code."""
import ast
import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from evaluation.pipelines.target_pipeline.catalog import build_workload
from evaluation.pipelines.target_pipeline.core import random_stream

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def seeded_reference(record, tag, operation, value):
    seed = int.from_bytes(hashlib.blake2b(
        f"{record['_seed']}:{tag}".encode(), digest_size=8
    ).digest(), "big") % (2**63)
    with random_stream(seed):
        return operation(value)


@pytest.fixture
def record(tmp_path):
    image = tmp_path / "reference.png"
    Image.fromarray(np.random.default_rng(4).integers(
        0,256,(330,510,3),dtype=np.uint8)).save(image)
    return {"id":"reference", "image":str(image), "caption":"Hello!  World. " * 40}


def test_source_snapshots_match_hashes():
    manifest = json.loads((ROOT/"provenance.json").read_text())
    assert set(manifest) == {"simclr","dino","swav","clip","blip"}
    for recipe in manifest.values():
        for entry in recipe["files"]:
            assert hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_dino_matches_upstream(record):
    ns = dict(random=random, ImageFilter=ImageFilter, ImageOps=ImageOps)
    definitions(ROOT/"references/dino/utils.py", ["GaussianBlur","Solarization"], ns)
    ns = dict(transforms=T, Image=Image, utils=SimpleNamespace(**ns))
    definitions(ROOT/"references/dino/main_dino.py", ["DataAugmentationDINO"], ns)
    native = ns["DataAugmentationDINO"]((0.4,1.0),(0.05,0.4),8)
    workload = build_workload("dino",seed=31)
    prepared = workload.prepare(record)
    references = [native.global_transfo1,native.global_transfo2] + [native.local_transfo]*8
    expected = []
    for i, transform in enumerate(references):
        value = seeded_reference(prepared,f"view{i}_crop",
                                 transform.transforms[0],prepared["original"])
        value = seeded_reference(prepared,f"view{i}_augment",
                                 T.Compose(transform.transforms[1:]),value)
        expected.append(value)
    for actual, reference in zip(workload.run(record)["views"], expected):
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)


def test_swav_matches_upstream(record):
    ns = dict(np=np,random=random,ImageFilter=ImageFilter,transforms=T)
    definitions(ROOT/"references/swav/src/multicropdataset.py",
                ["PILRandomGaussianBlur","get_color_distortion"],ns)
    workload = build_workload("swav",seed=31)
    prepared = workload.prepare(record)
    for i, actual in enumerate(workload.run(record)["views"]):
        size, scale = (224,(0.14,1.0)) if i<2 else (96,(0.05,0.14))
        crop = T.RandomResizedCrop(size,scale=scale)
        rest = T.Compose([
            T.RandomHorizontalFlip(p=0.5),ns["get_color_distortion"](),
            ns["PILRandomGaussianBlur"](),T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.228,0.224,0.225]),
        ])
        value = seeded_reference(prepared,f"view{i}_crop",crop,prepared["original"])
        value = seeded_reference(prepared,f"view{i}_augment",rest,value)
        torch.testing.assert_close(actual,value,rtol=0,atol=0)


def test_simclr_preserves_existing_cedar_per_view(record):
    ns = dict(torch=torch,transforms=T)
    tree=ast.parse((ROOT/"references/simclr/cedar_dataset.py").read_text())
    # Get the actual callable sequence by constructing the existing Feature.
    from evaluation.pipelines.simclrv2.cedar_dataset import SimCLRV2Feature
    from cedar.sources import IterSource
    from cedar.pipes import MapperPipe
    feature=SimCLRV2Feature(1)
    feature.apply(IterSource([record["image"]]))
    maps=[p for _,p in sorted(feature.logical_pipes.items(),reverse=True)
          if isinstance(p,MapperPipe)]
    workload=build_workload("simclr",seed=31)
    prepared=workload.prepare(record)
    labels=["float","crop","flip","jitter","gray","blur","normalize"]
    for i,actual in enumerate(workload.run(record)["views"]):
        value=prepared["original"]
        for label,pipe in zip(labels,maps):
            if label in ("crop","flip","jitter","blur"):
                value=seeded_reference(prepared,f"view{i}_{label}",pipe.fn,value)
            else:
                value=pipe.fn(value)
        torch.testing.assert_close(actual,value,rtol=0,atol=0)


def test_clip_matches_upstream_transform_and_tokenizer(record):
    ns=dict(torch=torch,Resize=T.Resize,CenterCrop=T.CenterCrop,
            ConvertImageDtype=T.ConvertImageDtype,Normalize=T.Normalize,
            InterpolationMode=InterpolationMode)
    definitions(ROOT/"references/clip/examples/pytorch/contrastive-image-text/run_clip.py",
                ["Transform"],ns)
    from torchvision.io import read_image,ImageReadMode
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True)
    native=ns["Transform"](224,[0.48145466,0.4578275,0.40821073],
                          [0.26862954,0.26130258,0.27577711])
    result=build_workload("clip").run(record)
    torch.testing.assert_close(result["pixel_values"],
        native(read_image(record["image"],mode=ImageReadMode.RGB)),rtol=0,atol=0)
    tokens=tokenizer(record["caption"],max_length=77,padding="max_length",truncation=True)
    assert result["input_ids"].tolist()==tokens["input_ids"]
    assert result["attention_mask"].tolist()==tokens["attention_mask"]


def test_blip_matches_pinned_pre_caption(record):
    ns=definitions(ROOT/"references/blip/data/utils.py",["pre_caption"],dict(re=re))
    assert build_workload("blip").run(record)["caption"] == ns["pre_caption"](record["caption"],30)
