"""Caption-filtered SimCLR-v2 pipeline over paired COCO images and text."""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import pathlib
from dataclasses import dataclass
from typing import Any, List, Mapping

from torchvision import transforms
from torchvision.io import ImageReadMode
from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import (
    BatcherPipe,
    FilterPipe,
    ImageReaderPipe,
    MapperPipe,
    Pipe,
    PipeExecutionResource,
)
from cedar.pipes.common import CedarPipeSpec
from cedar.pipes.context import PipeVariantType
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from pico_multimodal.operators import (
    ExtractImagePath,
    ClipPredicate,
    PerImageStandardize,
    TextNormalizer,
    ToFloatImage,
    parse_json_record,
)


IMAGE_SIDE = 224
DISPLAYED_TAGS = (
    "caption_normalize",
    "image_text_filter",
    "random_crop",
    "random_flip",
    "color_jitter",
    "gaussian_blur",
)

# The local CUDA filter executes before any optional multiprocessing suffix.
# Spawn prevents SMP workers from inheriting an initialized CUDA context.
mp.set_start_method(
    os.environ.get("SIMCLRV2_MULTIMODAL_MP_START_METHOD", "spawn"),
    force=True,
)


def _fixed_mapper(input_pipe: Pipe, fn: Any, tag: str) -> MapperPipe:
    """Create an in-process semantic boundary outside the plan search."""

    pipe = MapperPipe(input_pipe, fn, tag=tag)
    pipe.pipe_spec = CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[PipeVariantType.INPROCESS],
        is_fusable=False,
    )
    return pipe.fix()


@dataclass(frozen=True)
class CaptionThreshold:
    clip_min: float

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "CaptionThreshold":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "thresholds" in payload:
            payload = payload["thresholds"]
        if not isinstance(payload, Mapping) or "clip_min" not in payload:
            raise ValueError("Threshold file must contain clip_min")
        return cls(clip_min=float(payload["clip_min"]))


class MultimodalSimCLRV2Feature(Feature):
    """Use caption quality to select inputs for SimCLR-style augmentation."""

    def __init__(
        self,
        threshold: CaptionThreshold,
        image_root: str | pathlib.Path,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.image_root = str(image_root)
        self.batch_size = int(batch_size)

    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        pipe = _fixed_mapper(source_pipes[0], parse_json_record, tag="parse")
        pipe = _fixed_mapper(
            pipe,
            TextNormalizer(),
            tag="caption_normalize",
        )
        pipe = FilterPipe(
            pipe,
            ClipPredicate(self.threshold.clip_min, self.image_root),
            tag="image_text_filter",
        ).depends_on(["caption_normalize"]).fix()
        # The CLIP filter is a fixed physical boundary: every optimizer must
        # execute it as an independent local GPU stage, while the SimCLR suffix
        # remains available to reorder, fuse, and place independently.
        pipe.pipe_spec = copy.copy(pipe.pipe_spec)
        pipe.pipe_spec.mutable_variants = [PipeVariantType.INPROCESS]
        pipe.pipe_spec.is_fusable = False
        pipe.set_execution_resource(PipeExecutionResource.CUDA)

        # The boundary marks the point where caption metadata is intentionally
        # discarded. Image decoding and the SimCLR augmentation search remain
        # unchanged from the original workload after this point.
        pipe = _fixed_mapper(
            pipe,
            ExtractImagePath(self.image_root),
            tag="extract_image",
        )
        pipe = ImageReaderPipe(
            pipe,
            mode=ImageReadMode.RGB,
            tag="decode",
        ).fix()
        pipe = MapperPipe(
            pipe,
            ToFloatImage(),
            tag="to_float",
        )
        pipe = MapperPipe(
            pipe,
            transforms.RandomResizedCrop(IMAGE_SIDE),
            tag="random_crop",
        ).depends_on(["image_text_filter"])
        pipe = MapperPipe(
            pipe,
            transforms.RandomHorizontalFlip(),
            tag="random_flip",
        ).depends_on(["random_crop"])
        pipe = MapperPipe(
            pipe,
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            tag="color_jitter",
        )
        pipe = MapperPipe(
            pipe,
            transforms.Grayscale(num_output_channels=1),
            tag="grayscale",
        )
        pipe = MapperPipe(
            pipe,
            transforms.GaussianBlur(kernel_size=23),
            tag="gaussian_blur",
        )
        pipe = MapperPipe(
            pipe,
            PerImageStandardize(),
            tag="image_normalize",
        ).depends_on(["to_float"])
        return BatcherPipe(pipe, batch_size=self.batch_size).fix()


def displayed_signature(feature: MultimodalSimCLRV2Feature) -> dict[str, Any]:
    by_tag = {pipe.tag: pipe for pipe in feature.logical_pipes.values()}
    missing = set(DISPLAYED_TAGS) - set(by_tag)
    if missing:
        raise ValueError(f"Incomplete multimodal SimCLR-v2 feature: {sorted(missing)}")
    return {
        "tags": list(DISPLAYED_TAGS),
        "text_tags": ["caption_normalize"],
        "multimodal_tags": ["image_text_filter"],
        "image_tags": ["random_crop", "random_flip", "color_jitter", "gaussian_blur"],
        "dependencies": [
            (dependency, tag)
            for tag in DISPLAYED_TAGS
            for dependency in (by_tag[tag]._depends_on_tags or ())
            if dependency in DISPLAYED_TAGS
        ],
        "fixed_tags": ["caption_normalize", "image_text_filter"],
        "cuda_tags": ["image_text_filter"],
    }


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    kwargs = spec.kwargs or {}
    required = ("dataset_path", "image_root", "threshold_path")
    missing = [name for name in required if not kwargs.get(name)]
    if missing:
        raise ValueError(f"Missing dataset kwargs: {missing}")
    feature = MultimodalSimCLRV2Feature(
        CaptionThreshold.from_json(kwargs["threshold_path"]),
        kwargs["image_root"],
        spec.batch_size,
    )
    feature.apply(LocalLineSource(str(kwargs["dataset_path"])))
    if spec.use_ray and spec.ray_runtime_env is None:
        # Ray workers may run on a node without this checkout mounted. Ship the
        # exact actor code and fixture images through Ray's job runtime.
        from evaluation.motivation_multimodal.runner import _ray_runtime_env

        spec.ray_runtime_env = _ray_runtime_env(
            pathlib.Path(kwargs["dataset_path"]),
            pathlib.Path(kwargs["image_root"]),
        )
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
            enable_offload=not spec.disable_offload,
            enable_local_parallelism=not spec.disable_parallelism,
            enable_reorder=not spec.disable_reorder,
            enable_fusion=not spec.disable_fusion,
            enable_caching=False,
            num_samples=spec.num_total_samples,
            use_my_optimizer=spec.use_my_optimizer,
            reorder_timeout_sec=spec.reorder_timeout_sec,
        ),
        generate_plan=spec.generate_plan,
    )
