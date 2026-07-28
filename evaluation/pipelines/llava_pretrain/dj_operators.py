"""
Per-sample Cedar operators for Data-Juicer Hub's
``refined_recipes/image/llava-pretrain-refine.yaml`` recipe.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import threading
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image, ImageOps
from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from evaluation.pipelines.redpajama_c4 import dj_operators as text_ops
from evaluation.pipelines.bloom_oscar.dj_operators import _load_words_dict

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

_SIZE_RE = re.compile(r"^\s*([0-9.]+)\s*([kmgt]?b)?\s*$", re.IGNORECASE)
IMAGE_TOKEN = "<image>"
EOC_TOKEN = "<|__dj__eoc|>"
_SPECIAL_TOKENS = (IMAGE_TOKEN, "<audio>", "<video>", EOC_TOKEN)
_MODEL_LOCK = threading.Lock()
_CLIP_MODELS: Dict[str, tuple[CLIPProcessor, CLIPModel]] = {}
_BLIP_MODELS: Dict[str, tuple[BlipProcessor, BlipForImageTextRetrieval]] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_json_line(line: Any) -> Dict[str, Any]:
    # Ray Data 2.7 represents read_text rows as {"text": raw_line};
    # Cedar/PyTorch/tf.py_function pass the raw string directly.
    if isinstance(line, dict):
        line = line.get("text", "")
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
        return True
    if any_or_all == "all":
        return all(results)
    if any_or_all == "any":
        return any(results)
    raise ValueError(f"Unsupported any_or_all={any_or_all!r}")


def _validate_multi_image_options(any_or_all: str, reduce_mode: str | None = None):
    if any_or_all not in {"any", "all"}:
        raise ValueError(f"Unsupported any_or_all={any_or_all!r}")
    if reduce_mode is not None and reduce_mode not in {"avg", "max", "min"}:
        raise ValueError(f"Unsupported reduce_mode={reduce_mode!r}")


def _reduce_scores(scores: Sequence[float], reduce_mode: str) -> float:
    if reduce_mode == "avg":
        return sum(scores) / len(scores)
    if reduce_mode == "max":
        return max(scores)
    if reduce_mode == "min":
        return min(scores)
    raise ValueError(f"Unsupported reduce_mode={reduce_mode!r}")


def _remove_special_tokens(text: str) -> str:
    for token in _SPECIAL_TOKENS:
        text = text.replace(token, "").strip()
    return text


def _iter_image_text_chunks(sample: Dict[str, Any]):
    paths = _image_paths(sample)
    offset = 0
    for chunk in _get_text(sample).split(EOC_TOKEN):
        count = chunk.count(IMAGE_TOKEN)
        if count and chunk:
            yield _remove_special_tokens(chunk), paths[offset : offset + count]
        offset += count


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
        min_ratio: float = 0.0,
        max_ratio: float = 0.0,
        flagged_words: Iterable[str] | None = None,
    ):
        self.lang = lang
        self.tokenization = tokenization
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.flagged_words = set(
            flagged_words
            if flagged_words is not None
            else _load_words_dict("flagged_words").get(lang, [])
        )

    def __call__(self, sample: Dict[str, Any]) -> bool:
        token_func = None
        if self.tokenization:
            token_func = text_ops._get_sentencepiece_model(
                self.lang
            ).encode_as_pieces
        words = text_ops._get_words_from_document(
            _get_text(sample), token_func=token_func
        )
        words = text_ops._words_refinement(
            words,
            lower_case=True,
            strip_chars=text_ops.SPECIAL_CHARACTERS,
        )
        ratio = (
            sum(1 for word in words if word in self.flagged_words) / len(words)
            if words
            else 0.0
        )
        ratio = min(ratio, 1.0)
        sample[FIELDS_STATS]["flagged_words_ratio"] = ratio
        return self.min_ratio <= ratio <= self.max_ratio


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
        _validate_multi_image_options(any_or_all)

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
        min_width: int = 1,
        max_width: int = sys.maxsize,
        min_height: int = 1,
        max_height: int = sys.maxsize,
        any_or_all: str = "any",
    ):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.any_or_all = any_or_all
        _validate_multi_image_options(any_or_all)

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
        min_size: int | str = "0",
        max_size: int | str = "1TB",
        any_or_all: str = "any",
    ):
        self.min_size = _parse_size(min_size)
        self.max_size = _parse_size(max_size)
        self.any_or_all = any_or_all
        _validate_multi_image_options(any_or_all)

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
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        any_or_all: str = "any",
        reduce_mode: str = "avg",
    ):
        self.hf_clip = hf_clip
        self.min_score = min_score
        self.max_score = max_score
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.any_or_all = any_or_all
        self.reduce_mode = reduce_mode
        _validate_multi_image_options(any_or_all, reduce_mode)

    @torch.inference_mode()
    def __call__(self, sample: Dict[str, Any]) -> bool:
        if not _image_paths(sample):
            sample[FIELDS_STATS]["image_text_similarity"] = []
            return True
        processor, model = _get_clip_model(self.hf_clip)
        scores = []
        for text_chunk, paths in _iter_image_text_chunks(sample):
            images = []
            for path in paths:
                image = _open_rgb(path)
                if self.horizontal_flip:
                    image = ImageOps.mirror(image)
                if self.vertical_flip:
                    image = ImageOps.flip(image)
                images.append(image)
            if not images:
                continue
            inputs = processor(
                text=text_chunk,
                images=images,
                return_tensors="pt",
                truncation=True,
                max_length=model.config.text_config.max_position_embeddings,
                padding=True,
            )
            inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
            logits = model(**inputs).logits_per_text / 100.0
            scores.append(
                _reduce_scores(
                    [float(value) for value in logits.detach().cpu().reshape(-1)],
                    self.reduce_mode,
                )
            )
        sample[FIELDS_STATS]["image_text_similarity"] = scores
        return _reduce(
            [self.min_score <= score <= self.max_score for score in scores],
            self.any_or_all,
        )


class ImageTextMatchingFilter:
    def __init__(
        self,
        hf_blip: str = "Salesforce/blip-itm-base-coco",
        min_score: float = 0.0,
        max_score: float = 1.0,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        any_or_all: str = "any",
        reduce_mode: str = "avg",
    ):
        self.hf_blip = hf_blip
        self.min_score = min_score
        self.max_score = max_score
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.any_or_all = any_or_all
        self.reduce_mode = reduce_mode
        _validate_multi_image_options(any_or_all, reduce_mode)

    @torch.inference_mode()
    def __call__(self, sample: Dict[str, Any]) -> bool:
        if not _image_paths(sample):
            sample[FIELDS_STATS]["image_text_matching_score"] = []
            return True
        processor, model = _get_blip_model(self.hf_blip)
        scores = []
        for text_chunk, paths in _iter_image_text_chunks(sample):
            chunk_scores = []
            for path in paths:
                image = _open_rgb(path)
                if self.horizontal_flip:
                    image = ImageOps.mirror(image)
                if self.vertical_flip:
                    image = ImageOps.flip(image)
                inputs = processor(
                    images=image,
                    text=text_chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=model.config.text_config.max_position_embeddings,
                    padding=True,
                )
                inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
                output = model(**inputs)
                logits = output.itm_score if hasattr(output, "itm_score") else output[0]
                chunk_scores.append(float(logits.softmax(dim=-1)[0, 1].detach().cpu()))
            if chunk_scores:
                scores.append(_reduce_scores(chunk_scores, self.reduce_mode))
        sample[FIELDS_STATS]["image_text_matching_score"] = scores
        return _reduce(
            [self.min_score <= score <= self.max_score for score in scores],
            self.any_or_all,
        )
