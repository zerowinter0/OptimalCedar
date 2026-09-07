"""Per-record operators for the multimodal paper running example."""

from __future__ import annotations

import pathlib
import threading
from typing import Any, Mapping

import cv2
import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModel

from evaluation.pipelines.llava_pretrain import dj_operators as llava_ops


Record = Mapping[str, Any]

MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "blip": "Salesforce/blip-itm-base-coco",
    "aesthetic": (
        "shunk031/"
        "aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
    ),
}
MODEL_REVISIONS = {
    "clip": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "blip": "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c",
    "aesthetic": "684098de3856fa4678bf800efc05635de5b6cde5",
}

_MODEL_LOCK = threading.Lock()
_AESTHETIC_MODEL: tuple[Any, Any] | None = None
_SNAPSHOTS: dict[str, str] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _image_path(record: Record, image_root: str | pathlib.Path | None) -> pathlib.Path:
    path = pathlib.Path(str(record["image_path"]))
    if not path.is_absolute() and image_root:
        path = pathlib.Path(image_root) / path
    return path


def _text_sample(record: Record) -> dict[str, Any]:
    return {
        llava_ops.TEXT_KEY: str(record.get("caption", "")),
        llava_ops.FIELDS_STATS: {},
        llava_ops.FIELDS_CONTEXT: {},
        llava_ops.FIELDS_META: {},
    }


def _multimodal_sample(
    record: Record,
    image_root: str | pathlib.Path | None,
) -> dict[str, Any]:
    return {
        llava_ops.TEXT_KEY: llava_ops.IMAGE_TOKEN + str(record.get("caption", "")),
        llava_ops.IMAGE_KEY: [str(_image_path(record, image_root))],
        llava_ops.FIELDS_STATS: {},
        llava_ops.FIELDS_CONTEXT: {},
        llava_ops.FIELDS_META: {},
    }


def _snapshot_path(kind: str) -> str:
    """Resolve a model from its immutable revision without network access."""

    if kind not in _SNAPSHOTS:
        _SNAPSHOTS[kind] = snapshot_download(
            repo_id=MODEL_NAMES[kind],
            revision=MODEL_REVISIONS[kind],
            local_files_only=True,
        )
    return _SNAPSHOTS[kind]


def _get_aesthetic_model() -> tuple[Any, Any]:
    global _AESTHETIC_MODEL
    with _MODEL_LOCK:
        if _AESTHETIC_MODEL is None:
            model_path = _snapshot_path("aesthetic")
            processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            model.to(_DEVICE)
            model.eval()
            _AESTHETIC_MODEL = (processor, model)
        return _AESTHETIC_MODEL


class TextNormalizer:
    """Apply the same Unicode and punctuation normalization as Data-Juicer."""

    def __init__(self) -> None:
        self._unicode = llava_ops.FixUnicodeMapper()
        self._punctuation = llava_ops.PunctuationNormalizationMapper()

    def __call__(self, record: Record) -> dict[str, Any]:
        sample = _text_sample(record)
        self._punctuation(self._unicode(sample))
        output = dict(record)
        output["caption"] = sample[llava_ops.TEXT_KEY]
        return output


class PerplexityPredicate:
    """Retain captions whose KenLM perplexity is at most ``max_score``."""

    def __init__(self, max_score: float) -> None:
        self.max_score = float(max_score)
        self._operator = llava_ops.PerplexityFilter(
            lang="en",
            max_ppl=self.max_score,
        )

    def score(self, record: Record) -> float:
        sample = _text_sample(record)
        self._operator(sample)
        return float(sample[llava_ops.FIELDS_STATS]["perplexity"])

    def __call__(self, record: Record) -> bool:
        return self.score(record) <= self.max_score


class SharpnessPredicate:
    """Retain images whose luminance Laplacian variance exceeds a threshold."""

    def __init__(
        self,
        min_score: float,
        image_root: str | pathlib.Path | None = None,
    ) -> None:
        self.min_score = float(min_score)
        self.image_root = str(image_root) if image_root else ""

    def score(self, record: Record) -> float:
        path = _image_path(record, self.image_root)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to decode image for sharpness: {path}")
        return float(cv2.Laplacian(image, cv2.CV_64F).var())

    def __call__(self, record: Record) -> bool:
        return self.score(record) >= self.min_score


class AestheticPredicate:
    def __init__(
        self,
        min_score: float,
        image_root: str | pathlib.Path | None = None,
    ) -> None:
        self.min_score = float(min_score)
        self.image_root = str(image_root) if image_root else ""

    @torch.inference_mode()
    def score(self, record: Record) -> float:
        processor, model = _get_aesthetic_model()
        image = llava_ops._open_rgb(_image_path(record, self.image_root))
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(_DEVICE) for key, value in inputs.items()}
        output = model(**inputs)
        return float(output.logits.detach().cpu().reshape(-1)[0] / 10.0)

    def __call__(self, record: Record) -> bool:
        return self.score(record) >= self.min_score


class ClipPredicate:
    def __init__(
        self,
        min_score: float,
        image_root: str | pathlib.Path | None = None,
    ) -> None:
        self.min_score = float(min_score)
        self.image_root = str(image_root) if image_root else ""

    def score(self, record: Record) -> float:
        sample = _multimodal_sample(record, self.image_root)
        operator = llava_ops.ImageTextSimilarityFilter(
            hf_clip=_snapshot_path("clip"),
            min_score=float("-inf"),
            max_score=float("inf"),
        )
        operator(sample)
        scores = sample[llava_ops.FIELDS_STATS]["image_text_similarity"]
        if len(scores) != 1:
            raise ValueError(f"Expected one CLIP score, observed {len(scores)}")
        return float(scores[0])

    def __call__(self, record: Record) -> bool:
        return self.score(record) >= self.min_score


class BlipPredicate:
    def __init__(
        self,
        min_score: float,
        image_root: str | pathlib.Path | None = None,
    ) -> None:
        self.min_score = float(min_score)
        self.image_root = str(image_root) if image_root else ""

    def score(self, record: Record) -> float:
        sample = _multimodal_sample(record, self.image_root)
        operator = llava_ops.ImageTextMatchingFilter(
            hf_blip=_snapshot_path("blip"),
            min_score=float("-inf"),
            max_score=float("inf"),
        )
        operator(sample)
        scores = sample[llava_ops.FIELDS_STATS]["image_text_matching_score"]
        if len(scores) != 1:
            raise ValueError(f"Expected one BLIP score, observed {len(scores)}")
        return float(scores[0])

    def __call__(self, record: Record) -> bool:
        return self.score(record) >= self.min_score
