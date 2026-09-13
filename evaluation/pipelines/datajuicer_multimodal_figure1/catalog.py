"""Curated Cedar migrations of Data-Juicer Hub multimodal recipes.

The catalog separates recipe selection from heavyweight model construction.
Every entry contains at most ten processing operators and retains the data
modalities and accelerator operators that determine its physical plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MultimodalPipeline:
    name: str
    source_recipes: Tuple[str, ...]
    modalities: Tuple[str, ...]
    operators: Tuple[str, ...]
    gpu_operators: Tuple[str, ...]


MULTIMODAL_PIPELINES = (
    MultimodalPipeline(
        "llava_quality",
        ("image/llava-pretrain-refine.yaml",),
        ("image", "text"),
        (
            "FixUnicode",
            "NormalizePunctuation",
            "LanguageQuality",
            "Perplexity",
            "ImageShape",
            "ImageSize",
            "ImageTextSimilarity",
            "ImageTextMatching",
        ),
        ("ImageTextSimilarity", "ImageTextMatching"),
    ),
    MultimodalPipeline(
        "llava_safety",
        ("image/llava-pretrain-refine.yaml",),
        ("image", "text"),
        (
            "FixUnicode",
            "CharacterRepetition",
            "FlaggedWords",
            "SpecialCharacters",
            "ImageAspectRatio",
            "ImageSize",
            "ImageTextSimilarity",
            "ImageTextMatching",
        ),
        ("ImageTextSimilarity", "ImageTextMatching"),
    ),
    MultimodalPipeline(
        "image_edit_generation",
        ("image/img-diff-recipe.yaml",),
        ("source_image", "target_image", "text"),
        (
            "SentenceAugmentation",
            "PromptToPrompt",
            "DifferenceArea",
            "DifferenceCaption",
        ),
        ("PromptToPrompt", "DifferenceArea", "DifferenceCaption"),
    ),
    MultimodalPipeline(
        "video_sandbox",
        ("video/data-juicer-sandbox-optimal.yaml",),
        ("video", "text"),
        ("VideoNSFW", "VideoTextSimilarity"),
        ("VideoNSFW", "VideoTextSimilarity"),
    ),
    MultimodalPipeline(
        "video_self_evolution",
        ("video/data-juicer-sandbox-self-evolution.yaml",),
        ("video", "text"),
        (
            "VideoDuration",
            "VideoMotion",
            "VideoAesthetics",
            "VideoNSFW",
            "VideoTextSimilarity",
        ),
        ("VideoAesthetics", "VideoNSFW", "VideoTextSimilarity"),
    ),
    MultimodalPipeline(
        "general_video_refine",
        ("video/general-video-refine-example.yaml",),
        ("video", "text"),
        (
            "LanguageIdentification",
            "Perplexity",
            "VideoAesthetics",
            "VideoTextSimilarity",
            "VideoMotion",
            "VideoNSFW",
            "VideoWatermark",
        ),
        (
            "VideoAesthetics",
            "VideoTextSimilarity",
            "VideoNSFW",
            "VideoWatermark",
        ),
    ),
    MultimodalPipeline(
        "video_caption_refine",
        (
            "video/general-video-refine-example.yaml",
            "video/data-juicer-sandbox-self-evolution.yaml",
        ),
        ("video", "text"),
        (
            "ParseRecord",
            "ResolveVideoPath",
            "LanguageIdentification",
            "Perplexity",
            "VideoAesthetics",
            "VideoTextSimilarity",
            "VideoMotion",
            "VideoNSFW",
            "VideoWatermark",
            "VideoDuration",
        ),
        (
            "VideoAesthetics",
            "VideoTextSimilarity",
            "VideoNSFW",
            "VideoWatermark",
        ),
    ),
)

SELECTED_PIPELINE = next(
    pipeline
    for pipeline in MULTIMODAL_PIPELINES
    if pipeline.name == "video_caption_refine"
)

# Target layouts for the first profiling/plan-generation gate. They name
# resource stages rather than implementation details, so the same contract can
# drive the paper figure and the eventual plan validator.
CEDAR_PLAN_TARGET = (
    ("Local", ("ParseRecord", "ResolveVideoPath")),
    (
        "Ray-GPU",
        (
            "LanguageIdentification",
            "Perplexity",
            "VideoAesthetics",
            "VideoTextSimilarity",
        ),
    ),
    ("Local-CPU", ("VideoMotion",)),
    ("Ray-GPU", ("VideoNSFW", "VideoWatermark")),
    ("Local-CPU", ("VideoDuration",)),
)

JOINT_DP_PLAN_TARGET = (
    ("Local", ("ParseRecord", "ResolveVideoPath")),
    (
        "SMP-CPU",
        (
            "LanguageIdentification",
            "Perplexity",
            "VideoDuration",
            "VideoMotion",
        ),
    ),
    (
        "Ray-GPU",
        (
            "VideoAesthetics",
            "VideoNSFW",
            "VideoWatermark",
            "VideoTextSimilarity",
        ),
    ),
)
