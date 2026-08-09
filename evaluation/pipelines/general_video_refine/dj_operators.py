"""Per-record adapters for Data-Juicer's general video refinement recipe."""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict, Mapping

from evaluation.pipelines.redpajama_c4 import dj_operators as text_ops
from evaluation.pipelines.general_video_refine.data_juicer_bootstrap import (
    ensure_data_juicer_path,
)


ensure_data_juicer_path()

TEXT_KEY = text_ops.TEXT_KEY
FIELDS_STATS = text_ops.FIELDS_STATS
FIELDS_CONTEXT = text_ops.FIELDS_CONTEXT
FIELDS_META = text_ops.FIELDS_META
VIDEO_KEY = "videos"
VIDEO_TOKEN = "<__dj__video>"

LanguageIDScoreFilter = text_ops.LanguageIDScoreFilter
PerplexityFilter = text_ops.PerplexityFilter


def parse_json_line(line: Any) -> Dict[str, Any]:
    if isinstance(line, dict):
        line = line.get("text", "")
    sample = json.loads(line)
    text = sample.get("text", "")
    sample["text"] = text if isinstance(text, str) else str(text)
    sample[TEXT_KEY] = sample["text"]
    videos = sample.get(VIDEO_KEY, [])
    if isinstance(videos, (str, pathlib.Path)):
        videos = [str(videos)]
    sample[VIDEO_KEY] = list(videos)
    sample.setdefault(FIELDS_STATS, {})
    sample.setdefault(FIELDS_CONTEXT, {})
    sample.setdefault(FIELDS_META, {})
    return sample


class SetVideoRootMapper:
    def __init__(self, video_root: str | pathlib.Path):
        self.video_root = pathlib.Path(video_root).resolve()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        resolved = []
        for value in sample.get(VIDEO_KEY, []):
            path = pathlib.Path(value)
            if not path.is_absolute():
                path = self.video_root / path
            resolved.append(str(path))
        sample[VIDEO_KEY] = resolved
        return sample


def project_output(sample: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "video_id": sample.get("video_id"),
        "sentence_id": sample.get("sentence_id"),
        "videos": list(sample.get(VIDEO_KEY, [])),
        "text": sample.get("text", ""),
    }


def _operator_class(name: str):
    if name == "video_aesthetics_filter":
        from data_juicer.ops.filter.video_aesthetics_filter import (
            VideoAestheticsFilter,
        )

        return VideoAestheticsFilter
    if name == "video_frames_text_similarity_filter":
        from data_juicer.ops.filter.video_frames_text_similarity_filter import (
            VideoFramesTextSimilarityFilter,
        )

        return VideoFramesTextSimilarityFilter
    if name == "video_motion_score_filter":
        from data_juicer.ops.filter.video_motion_score_filter import (
            VideoMotionScoreFilter,
        )

        return VideoMotionScoreFilter
    if name == "video_nsfw_filter":
        from data_juicer.ops.filter.video_nsfw_filter import VideoNSFWFilter

        return VideoNSFWFilter
    if name == "video_watermark_filter":
        from data_juicer.ops.filter.video_watermark_filter import (
            VideoWatermarkFilter,
        )

        return VideoWatermarkFilter
    raise ValueError(f"Unsupported Data-Juicer video operator: {name}")


class DataJuicerVideoFilter:
    """Make a pinned Data-Juicer compute-stats/filter pair a Cedar predicate.

    The wrapped operator is constructed lazily in the process that executes the
    predicate. This keeps CUDA models out of the parent process and makes the
    callable safe for Cedar SMP/Ray serialization. Open video objects are never
    stored in the sample context or transferred across Cedar stage boundaries.
    """

    def __init__(self, operator_name: str, **operator_kwargs: Any):
        self.operator_name = operator_name
        self.operator_kwargs = dict(operator_kwargs)
        self._operator = None

    def _get_operator(self):
        if self._operator is None:
            self._operator = _operator_class(self.operator_name)(
                **self.operator_kwargs
            )
        return self._operator

    def __call__(self, sample: Dict[str, Any]) -> bool:
        sample.setdefault(FIELDS_STATS, {})
        sample.setdefault(FIELDS_CONTEXT, {})
        operator = self._get_operator()
        use_cuda = getattr(operator, "use_cuda", lambda: False)
        if not use_cuda():
            computed = operator.compute_stats_single(
                sample, rank=0, context=False
            )
            return bool(operator.process_single(computed))

        # Data-Juicer normally invokes accelerator operators from an outer
        # inference execution context. Cedar calls compute_stats_single
        # directly, so supply that context here: several pinned video filters
        # do not contain their own no_grad guard. Without it, eight formal
        # workers retain autograd workspaces and fill the shared A6000 even
        # though every returned statistic is a scalar.
        import torch

        with torch.inference_mode():
            computed = operator.compute_stats_single(
                sample, rank=0, context=False
            )
            return bool(operator.process_single(computed))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_operator"] = None
        return state

    def __deepcopy__(self, memo):
        result = type(self)(
            self.operator_name,
            **copy.deepcopy(self.operator_kwargs, memo),
        )
        memo[id(self)] = result
        return result


def video_aesthetics_filter() -> DataJuicerVideoFilter:
    return DataJuicerVideoFilter(
        "video_aesthetics_filter",
        hf_scorer_model=(
            "shunk031/"
            "aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        ),
        min_score=0.31767486,
        max_score=1.0,
        frame_sampling_method="uniform",
        frame_num=3,
        reduce_mode="avg",
        any_or_all="any",
    )


def video_text_similarity_filter() -> DataJuicerVideoFilter:
    return DataJuicerVideoFilter(
        "video_frames_text_similarity_filter",
        hf_clip="openai/clip-vit-base-patch32",
        min_score=0.16571071,
        max_score=1.0,
        frame_sampling_method="all_keyframes",
        frame_num=3,
        horizontal_flip=False,
        vertical_flip=False,
        reduce_mode="avg",
        any_or_all="any",
    )


def video_motion_filter() -> DataJuicerVideoFilter:
    return DataJuicerVideoFilter(
        "video_motion_score_filter",
        min_score=0.25,
        max_score=10000.0,
        sampling_fps=2,
        any_or_all="any",
    )


def video_nsfw_filter() -> DataJuicerVideoFilter:
    return DataJuicerVideoFilter(
        "video_nsfw_filter",
        hf_nsfw_model="Falconsai/nsfw_image_detection",
        max_score=0.34847191,
        frame_sampling_method="all_keyframes",
        frame_num=3,
        reduce_mode="avg",
        any_or_all="any",
    )


def video_watermark_filter() -> DataJuicerVideoFilter:
    return DataJuicerVideoFilter(
        "video_watermark_filter",
        hf_watermark_model="amrul-hzz/watermark_detector",
        prob_threshold=0.96510297,
        frame_sampling_method="all_keyframes",
        frame_num=3,
        reduce_mode="avg",
        any_or_all="any",
    )


__all__ = [
    "DataJuicerVideoFilter",
    "LanguageIDScoreFilter",
    "PerplexityFilter",
    "SetVideoRootMapper",
    "parse_json_line",
    "project_output",
    "video_aesthetics_filter",
    "video_motion_filter",
    "video_nsfw_filter",
    "video_text_similarity_filter",
    "video_watermark_filter",
]
