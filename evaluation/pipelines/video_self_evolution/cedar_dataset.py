"""Cedar implementation of Data-Juicer's video self-evolution recipe."""

from __future__ import annotations

import gc
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
)
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.general_video_refine import dj_operators as ops


DEFAULT_DATASET_PATH = Path(
    "datasets/general_video_refine/msrvtt-video-text-200000.jsonl"
)
DEFAULT_VIDEO_ROOT = Path("datasets/general_video_refine/videos")

mp.set_start_method(
    os.environ.get("VIDEO_SELF_EVOLUTION_MP_START_METHOD", "spawn"),
    force=True,
)


class VideoSelfEvolutionFeature(Feature):
    """Eight Cedar pipes preserving the official five-filter recipe."""

    def __init__(self, video_root: str | Path):
        super().__init__()
        self.video_root = Path(video_root)

    @staticmethod
    def _per_record(pipe: Pipe, *, cuda: bool = False) -> Pipe:
        pipe.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        if cuda:
            pipe.set_execution_resource(PipeExecutionResource.CUDA)
        return pipe

    def _compose(self, source_pipes: List[Pipe]):
        fp = self._per_record(
            MapperPipe(source_pipes[0], ops.parse_json_line, tag="parse")
        ).fix()
        fp = self._per_record(
            MapperPipe(
                fp,
                ops.SetVideoRootMapper(self.video_root),
                tag="video_root",
            )
        ).fix()
        fp = self._per_record(
            FilterPipe(fp, ops.sandbox_video_nsfw_filter(), tag="video_nsfw"),
            cuda=True,
        )
        fp = self._per_record(
            FilterPipe(
                fp,
                ops.sandbox_video_text_similarity_filter(),
                tag="video_text_similarity",
            ),
            cuda=True,
        )
        fp = self._per_record(
            FilterPipe(
                fp, ops.sandbox_video_motion_filter(), tag="video_motion"
            )
        )
        fp = self._per_record(
            FilterPipe(
                fp,
                ops.sandbox_video_aesthetics_filter(),
                tag="video_aesthetics",
            ),
            cuda=True,
        )
        fp = self._per_record(
            FilterPipe(
                fp,
                ops.sandbox_video_duration_filter(),
                tag="video_duration",
            )
        )
        return self._per_record(
            MapperPipe(fp, ops.project_output, tag="project_output")
        ).fix()

    def release_profile_resources(self) -> None:
        from data_juicer.utils.model_utils import free_models

        free_models(clear_model_zoo=True)
        gc.collect()


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = Path(
        spec.kwargs.get("dataset_path", DEFAULT_DATASET_PATH)
        if spec.kwargs
        else DEFAULT_DATASET_PATH
    )
    video_root = Path(
        spec.kwargs.get("video_root", DEFAULT_VIDEO_ROOT)
        if spec.kwargs
        else DEFAULT_VIDEO_ROOT
    )
    feature = VideoSelfEvolutionFeature(video_root)
    feature.apply(LocalLineSource(str(dataset_path)))
    ctx = CedarContext(ray_config=spec.to_ray_config())
    if spec.config:
        return DataSet(
            ctx,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    return DataSet(
        ctx,
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
            enable_caching=not spec.disable_caching,
            num_samples=getattr(spec, "num_total_samples", None),
            enable_local_parallelism=not spec.disable_parallelism,
            enable_fusion=not spec.disable_fusion,
            use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
            reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
        ),
        generate_plan=spec.generate_plan,
    )


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_VIDEO_ROOT",
    "VideoSelfEvolutionFeature",
    "get_dataset",
]
