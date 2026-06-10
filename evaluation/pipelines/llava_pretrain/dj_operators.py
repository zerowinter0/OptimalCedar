"""
Per-sample Cedar operators for Data-Juicer Hub's
``refined_recipes/image/llava-pretrain-refine.yaml`` recipe.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import threading
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from evaluation.pipelines.redpajama_c4 import dj_operators as text_ops

TEXT_KEY = text_ops.TEXT_KEY
FIELDS_STATS = text_ops.FIELDS_STATS
FIELDS_CONTEXT = text_ops.FIELDS_CONTEXT
FIELDS_META = text_ops.FIELDS_META
IMAGE_KEY = "images"

FixUnicodeMapper = text_ops.FixUnicodeMapper
PunctuationNormalizationMapper = text_ops.PunctuationNormalizationMapper
AlphanumericFilter = text_ops.AlphanumericFilter
CharacterRepetitionFilter = text_ops.CharacterRepetitionFilter
SpecialCharactersFilter = text_ops.SpecialCharactersFilter
WordRepetitionFilter = text_ops.WordRepetitionFilter

_DEFAULT_FLAGGED_WORDS = {
    "porn",
    "porno",
    "sex",
    "nude",
    "naked",
    "violence",
    "gore",
    "hate",
}
_SIZE_RE = re.compile(r"^\s*([0-9.]+)\s*([kmgt]?b)?\s*$", re.IGNORECASE)
_MODEL_LOCK = threading.Lock()
_CLIP_MODELS: Dict[str, tuple[CLIPProcessor, CLIPModel]] = {}
_BLIP_MODELS: Dict[str, tuple[BlipProcessor, BlipForImageTextRetrieval]] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_json_line(line: str) -> Dict[str, Any]:
    sample = json.loads(line)
    if TEXT_KEY not in sample:
        sample[TEXT_KEY] = sample.get("text", "")
    if IMAGE_KEY not in sample:
        image = sample.get("image")
        sample[IMAGE_KEY] = [image] if image else []
    elif isinstance(sample[IMAGE_KEY], str):
        sample[IMAGE_KEY] = [sample[IMAGE_KEY]]
    sample.setdefault(FIELDS_STATS, {})
    sample.setdefault(FIELDS_CONTEXT, {})
    sample.setdefault(FIELDS_META, {})
    return sample


def sync_text_key(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample["text"] = sample.get(TEXT_KEY, "")
    return sample


def _get_text(sample: Dict[str, Any]) -> str:
    value = sample.get(TEXT_KEY, "")
    return value if isinstance(value, str) else str(value)


def _image_paths(sample: Dict[str, Any]) -> List[pathlib.Path]:
    paths = sample.get(IMAGE_KEY, [])
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    root = sample.get(FIELDS_CONTEXT, {}).get("image_root", "")
    root_path = pathlib.Path(root) if root else None
    resolved = []
    for path in paths:
        p = pathlib.Path(path)
        if not p.is_absolute() and root_path is not None:
            p = root_path / p
        resolved.append(p)
    return resolved


def _reduce(results: Sequence[bool], any_or_all: str) -> bool:
    if not results:
        return False
    if any_or_all == "all":
        return all(results)
    if any_or_all == "any":
        return any(results)
    raise ValueError(f"Unsupported any_or_all={any_or_all!r}")


def _parse_size(value: int | float | str) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(f"Invalid size value: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    scale = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    }[unit]
    return int(amount * scale)


def _open_rgb(path: pathlib.Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _resolve_hf_model_path(model_name: str) -> str:
    cache_roots = []
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        cache_roots.append(pathlib.Path(transformers_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(pathlib.Path(hf_home) / "hub")
    cache_roots.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")

    model_dir_name = "models--" + model_name.replace("/", "--")
    for cache_root in cache_roots:
        model_dir = cache_root / model_dir_name
        ref_path = model_dir / "refs" / "main"
        if not ref_path.exists():
            continue
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = model_dir / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    return model_name


def _get_clip_model(model_name: str) -> tuple[CLIPProcessor, CLIPModel]:
    with _MODEL_LOCK:
        if model_name not in _CLIP_MODELS:
            model_path = _resolve_hf_model_path(model_name)
            processor = CLIPProcessor.from_pretrained(model_path)
            model = CLIPModel.from_pretrained(model_path, use_safetensors=False)
            model.to(_DEVICE)
            model.eval()
            _CLIP_MODELS[model_name] = (processor, model)
        return _CLIP_MODELS[model_name]


def _get_blip_model(
    model_name: str,
) -> tuple[BlipProcessor, BlipForImageTextRetrieval]:
    with _MODEL_LOCK:
        if model_name not in _BLIP_MODELS:
            model_path = _resolve_hf_model_path(model_name)
            processor = BlipProcessor.from_pretrained(model_path)
            model = BlipForImageTextRetrieval.from_pretrained(
                model_path,
                use_safetensors=False,
            )
            model.to(_DEVICE)
            model.eval()
            _BLIP_MODELS[model_name] = (processor, model)
        return _BLIP_MODELS[model_name]


class SetImageRootMapper:
    def __init__(self, image_root: str | pathlib.Path | None):
        self.image_root = str(image_root) if image_root else ""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.image_root:
            sample.setdefault(FIELDS_CONTEXT, {})["image_root"] = self.image_root
        return sample


class FlaggedWordsFilter:
    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        max_ratio: float = 0.0,
        flagged_words: Iterable[str] | None = None,
    ):
        if tokenization:
            raise NotImplementedError("The LLaVA Cedar workload uses whitespace words.")
        self.lang = lang
        self.max_ratio = max_ratio
        self.flagged_words = set(flagged_words or _DEFAULT_FLAGGED_WORDS)

    def __call__(self, sample: Dict[str, Any]) -> bool:
        words = _words(_get_text(sample))
        ratio = (
            sum(1 for word in words if word in self.flagged_words) / len(words)
            if words
            else 0.0
        )
        ratio = min(ratio, 1.0)
        sample[FIELDS_STATS]["flagged_words_ratio"] = ratio
        return ratio <= self.max_ratio


class PerplexityFilter:
    def __init__(self, lang: str = "en", max_ppl: float = 14435.5806):
        self.inner = text_ops.PerplexityFilter(lang=lang, max_ppl=max_ppl)

    def __call__(self, sample: Dict[str, Any]) -> bool:
        return self.inner(sample)


class ImageAspectRatioFilter:
    def __init__(
        self,
        min_ratio: float = 0.0,
        max_ratio: float = float("inf"),
        any_or_all: str = "any",
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.any_or_all = any_or_all

    def __call__(self, sample: Dict[str, Any]) -> bool:
        ratios = []
        results = []
        for path in _image_paths(sample):
            with Image.open(path) as image:
                width, height = image.size
            ratio = width / height if height else 0.0
            ratios.append(ratio)
            results.append(self.min_ratio <= ratio <= self.max_ratio)
        sample[FIELDS_STATS]["image_aspect_ratios"] = ratios
        return _reduce(results, self.any_or_all)


class ImageShapeFilter:
    def __init__(
        self,
        min_width: int = 0,
        max_width: int = 2**31 - 1,
        min_height: int = 0,
        max_height: int = 2**31 - 1,
        any_or_all: str = "any",
    ):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.any_or_all = any_or_all

    def __call__(self, sample: Dict[str, Any]) -> bool:
        shapes = []
        results = []
        for path in _image_paths(sample):
            with Image.open(path) as image:
                width, height = image.size
            shapes.append([width, height])
            results.append(
                self.min_width <= width <= self.max_width
                and self.min_height <= height <= self.max_height
            )
        sample[FIELDS_STATS]["image_shapes"] = shapes
        return _reduce(results, self.any_or_all)


class ImageSizeFilter:
    def __init__(
        self,
        min_size: int | str = 0,
        max_size: int | str = 2**63 - 1,
        any_or_all: str = "any",
    ):
        self.min_size = _parse_size(min_size)
        self.max_size = _parse_size(max_size)
        self.any_or_all = any_or_all

    def __call__(self, sample: Dict[str, Any]) -> bool:
        sizes = []
        results = []
        for path in _image_paths(sample):
            size = path.stat().st_size
            sizes.append(size)
            results.append(self.min_size <= size <= self.max_size)
        sample[FIELDS_STATS]["image_sizes"] = sizes
        return _reduce(results, self.any_or_all)


class ImageTextSimilarityFilter:
    def __init__(
        self,
        hf_clip: str = "openai/clip-vit-base-patch32",
        min_score: float = 0.0,
        max_score: float = 1.0,
    ):
        self.hf_clip = hf_clip
        self.min_score = min_score
        self.max_score = max_score

    @torch.inference_mode()
    def __call__(self, sample: Dict[str, Any]) -> bool:
        processor, model = _get_clip_model(self.hf_clip)
        text = _get_text(sample)
        scores = []
        for path in _image_paths(sample):
            image = _open_rgb(path)
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            score = (image_features * text_features).sum(dim=-1).item()
            scores.append(score)
        score = max(scores) if scores else 0.0
        sample[FIELDS_STATS]["image_text_similarity"] = score
        return self.min_score <= score <= self.max_score


class ImageTextMatchingFilter:
    def __init__(
        self,
        hf_blip: str = "Salesforce/blip-itm-base-coco",
        min_score: float = 0.0,
        max_score: float = 1.0,
    ):
        self.hf_blip = hf_blip
        self.min_score = min_score
        self.max_score = max_score

    @torch.inference_mode()
    def __call__(self, sample: Dict[str, Any]) -> bool:
        processor, model = _get_blip_model(self.hf_blip)
        text = _get_text(sample)
        scores = []
        for path in _image_paths(sample):
            image = _open_rgb(path)
            inputs = processor(
                images=image,
                text=text,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
            output = model(**inputs, use_itm_head=True)
            logits = output.itm_score if hasattr(output, "itm_score") else output[0]
            score = logits.softmax(dim=-1)[0, 1].item()
            scores.append(score)
        score = max(scores) if scores else 0.0
        sample[FIELDS_STATS]["image_text_matching"] = score
        return self.min_score <= score <= self.max_score
