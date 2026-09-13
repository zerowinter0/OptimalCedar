"""Figure-1 video--caption pipeline migrated from Data-Juicer Hub."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from pathlib import Path
from typing import List

from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import (
    FilterPipe,
    MapperPipe,
    Pipe,
    PipeComputeScaling,
    PipeExecutionResource,
    PipeVariantType,
)
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.general_video_refine import dj_operators as ops


DEFAULT_DATASET_PATH = Path(
    "datasets/general_video_refine/msrvtt-video-text-200000.jsonl"
)
DEFAULT_VIDEO_ROOT = Path("datasets/general_video_refine/videos")

FIGURE1_TAGS = (
    "parse_record",
    "resolve_video_path",
    "language_identification",
    "perplexity",
    "video_aesthetics",
    "video_text_similarity",
    "video_motion",
    "video_nsfw",
    "video_watermark",
    "video_duration",
)

mp.set_start_method(
    os.environ.get("DATAJUICER_FIGURE1_MP_START_METHOD", "spawn"),
    force=True,
)


def _configure(
    pipe: Pipe,
    *,
    variants: List[PipeVariantType] | None = None,
    fusable: bool | None = None,
    cuda: bool = False,
) -> Pipe:
    pipe.set_compute_scaling(PipeComputeScaling.PER_RECORD)
    if cuda:
        pipe.set_execution_resource(PipeExecutionResource.CUDA)
    if variants is not None or fusable is not None:
        pipe.pipe_spec = copy.copy(pipe.pipe_spec)
        if variants is not None:
            pipe.pipe_spec.mutable_variants = list(variants)
        if fusable is not None:
            pipe.pipe_spec.is_fusable = fusable
    return pipe


class VideoCaptionRefineFeature(Feature):
    """Ten-operator video--text refinement pipeline for the running example.

    The operator set is the union of Data-Juicer Hub's general-video-refine
    recipe and its self-evolution duration filter. Motion and duration use
    CPU implementations because they are metadata/OpenCV gates; accelerator
    semantic filters may execute locally or on Ray.
    """

    def __init__(self, video_root: str | Path):
        super().__init__()
        self.video_root = Path(video_root)

    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        pipe = _configure(
            MapperPipe(
                source_pipes[0],
                ops.parse_json_line,
                tag="parse_record",
            )
        ).fix()
        pipe = _configure(
            MapperPipe(
                pipe,
                ops.SetVideoRootMapper(self.video_root),
                tag="resolve_video_path",
            )
        ).fix()

        pipe = _configure(
            FilterPipe(
                pipe,
                ops.LanguageIDScoreFilter(lang="en", min_score=0.26311219),
                tag="language_identification",
            )
        )
        pipe = _configure(
            FilterPipe(
                pipe,
                ops.PerplexityFilter(lang="en", max_ppl=7376.81378),
                tag="perplexity",
            )
        )
        pipe = _configure(
            FilterPipe(
                pipe,
                ops.video_aesthetics_filter(),
                tag="video_aesthetics",
            ),
            cuda=True,
        )
        pipe = _configure(
            FilterPipe(
                pipe,
                ops.video_text_similarity_filter(),
                tag="video_text_similarity",
            ),
            cuda=True,
        )

        # These stages remain reorderable but cannot be hidden in a mixed
        # CPU/GPU fusion. They represent distinct OpenCV/metadata kernels.
        pipe = _configure(
            FilterPipe(
                pipe, ops.sandbox_video_motion_filter(), tag="video_motion"
            ),
            variants=[PipeVariantType.INPROCESS, PipeVariantType.SMP],
            fusable=False,
        )
        pipe = _configure(
            FilterPipe(pipe, ops.video_nsfw_filter(), tag="video_nsfw"),
            cuda=True,
        )
        pipe = _configure(
            FilterPipe(
                pipe,
                ops.video_watermark_filter(),
                tag="video_watermark",
            ),
            cuda=True,
        )
        return _configure(
            FilterPipe(
                pipe,
                ops.DataJuicerVideoFilter(
                    "video_duration_filter",
                    min_duration=2.0,
                    max_duration=15.0,
                    any_or_all="any",
                ),
                tag="video_duration",
            ),
            variants=[PipeVariantType.INPROCESS, PipeVariantType.SMP],
            fusable=False,
        )


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    kwargs = spec.kwargs or {}
    dataset_path = Path(kwargs.get("dataset_path", DEFAULT_DATASET_PATH))
    video_root = Path(kwargs.get("video_root", DEFAULT_VIDEO_ROOT))
    feature = VideoCaptionRefineFeature(video_root)
    feature.apply(LocalLineSource(str(dataset_path)))
    context = CedarContext(ray_config=spec.to_ray_config())
    if spec.config:
        return DataSet(
            context,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    return DataSet(
        context,
        {"feature": feature},
        enable_controller=not spec.disable_controller,
        enable_optimizer=not spec.disable_optimizer,
        profiled_data=spec.profiled_stats,
        run_profiling=spec.run_profiling,
        optimizer_options=OptimizerOptions(
            enable_prefetch=not spec.disable_prefetch,
            est_throughput=None,
            available_local_cpus=mp.cpu_count(),
            enable_offload=not spec.disable_offload,
            enable_reorder=not spec.disable_reorder,
            enable_local_parallelism=not spec.disable_parallelism,
            enable_fusion=not spec.disable_fusion,
            enable_caching=False,
            num_samples=spec.num_total_samples,
            use_my_optimizer=spec.use_my_optimizer,
            reorder_timeout_sec=spec.reorder_timeout_sec,
        ),
        generate_plan=spec.generate_plan,
    )


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_VIDEO_ROOT",
    "FIGURE1_TAGS",
    "VideoCaptionRefineFeature",
    "get_dataset",
]
