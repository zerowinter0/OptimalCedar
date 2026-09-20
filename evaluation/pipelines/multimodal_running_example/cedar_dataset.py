"""Cedar workload used by the paper's multimodal running example."""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import pathlib
from dataclasses import dataclass, fields
from typing import Any, List, Mapping

from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import (
    FilterPipe,
    MapperPipe,
    Pipe,
    PipeExecutionResource,
    PipeVariantType,
)
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from pico_multimodal.operators import (
    AestheticPredicate,
    BlipPredicate,
    ClipPredicate,
    PerplexityPredicate,
    SafetyPredicate,
    TextNormalizer,
    parse_json_record as ray_parse_json_record,
)


DEFAULT_CPU_BUDGET = 64
RUNNING_EXAMPLE_TAGS = (
    "normalize",
    "perplexity",
    "safety",
    "aesthetic",
    "clip",
    "blip",
)

# Profiling first exercises CUDA operators in process and then profiles SMP.
# Spawn prevents those workers from inheriting the initialized CUDA runtime.
mp.set_start_method(
    os.environ.get("MULTIMODAL_EXAMPLE_MP_START_METHOD", "spawn"),
    force=True,
)


@dataclass(frozen=True)
class Thresholds:
    perplexity_max: float
    safety_max: float
    aesthetic_min: float
    clip_min: float
    blip_min: float

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "Thresholds":
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if "thresholds" in payload:
            payload = payload["thresholds"]
        names = {field.name for field in fields(cls)}
        if not isinstance(payload, Mapping) or set(payload) != names:
            missing = sorted(names - set(payload if isinstance(payload, Mapping) else ()))
            extra = sorted(set(payload if isinstance(payload, Mapping) else ()) - names)
            raise ValueError(
                "Invalid threshold file; "
                f"missing={missing}, extra={extra}"
            )
        try:
            return cls(**{name: float(payload[name]) for name in names})
        except (TypeError, ValueError) as exc:
            raise ValueError("All threshold values must be numeric") from exc


def parse_json_record(line: Any) -> dict[str, Any]:
    if isinstance(line, Mapping):
        line = line.get("text", line)
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    if isinstance(line, str):
        record = json.loads(line)
    else:
        record = dict(line)
    if not isinstance(record, dict):
        raise ValueError("Each input line must contain one JSON object")
    return record


def _per_record(pipe: Pipe, *, cuda: bool = False) -> Pipe:
    if cuda:
        pipe.set_execution_resource(PipeExecutionResource.CUDA)
    if os.environ.get("PICO_MULTIMODAL_DISABLE_SMP") == "1":
        # Decorated Pipe classes share their default PipeSpec. Copy it before
        # narrowing this experiment's variants to avoid changing class state.
        pipe.pipe_spec = copy.copy(pipe.pipe_spec)
        pipe.pipe_spec.mutable_variants = [
            variant
            for variant in pipe.pipe_spec.mutable_variants
            if variant != PipeVariantType.SMP
        ]
    return pipe


class MultimodalRunningExampleFeature(Feature):
    """Six optimizable operators over paired COCO images and captions."""

    def __init__(
        self,
        thresholds: Thresholds,
        image_root: str | pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.image_root = str(image_root) if image_root else ""

    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        fp = _per_record(
            MapperPipe(source_pipes[0], ray_parse_json_record, tag="parse")
        ).fix()
        fp = _per_record(MapperPipe(fp, TextNormalizer(), tag="normalize"))
        fp = _per_record(
            FilterPipe(
                fp,
                PerplexityPredicate(self.thresholds.perplexity_max),
                tag="perplexity",
            )
        ).depends_on(["normalize"])
        fp = _per_record(
            FilterPipe(
                fp,
                SafetyPredicate(
                    self.thresholds.safety_max,
                    self.image_root,
                ),
                tag="safety",
            )
        )
        fp = _per_record(
            FilterPipe(
                fp,
                AestheticPredicate(
                    self.thresholds.aesthetic_min,
                    self.image_root,
                ),
                tag="aesthetic",
            ),
            cuda=True,
        )
        fp = _per_record(
            FilterPipe(
                fp,
                ClipPredicate(self.thresholds.clip_min, self.image_root),
                tag="clip",
            ),
            cuda=True,
        ).depends_on(["perplexity", "safety", "aesthetic"])
        return _per_record(
            FilterPipe(
                fp,
                BlipPredicate(self.thresholds.blip_min, self.image_root),
                tag="blip",
            ),
            cuda=True,
        ).depends_on(["clip"])


def logical_signature(feature: MultimodalRunningExampleFeature) -> dict[str, object]:
    by_tag = {
        pipe.tag: pipe
        for pipe in feature.logical_pipes.values()
        if pipe.tag in RUNNING_EXAMPLE_TAGS
    }
    missing = set(RUNNING_EXAMPLE_TAGS) - set(by_tag)
    if missing:
        raise ValueError(f"Feature has not been composed correctly; missing={missing}")
    dependencies = []
    for tag in RUNNING_EXAMPLE_TAGS:
        for dependency in by_tag[tag]._depends_on_tags or ():
            if dependency in by_tag:
                dependencies.append((dependency, tag))
    return {
        "tags": list(RUNNING_EXAMPLE_TAGS),
        "dependencies": dependencies,
        "cuda_tags": [
            tag
            for tag in RUNNING_EXAMPLE_TAGS
            if by_tag[tag].execution_resource == PipeExecutionResource.CUDA
        ],
    }


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    kwargs = spec.kwargs or {}
    try:
        dataset_path = pathlib.Path(kwargs["dataset_path"])
        threshold_path = pathlib.Path(kwargs["threshold_path"])
    except KeyError as exc:
        raise ValueError(
            "dataset_path and threshold_path are required dataset kwargs"
        ) from exc
    image_root = kwargs.get("image_root")

    feature = MultimodalRunningExampleFeature(
        Thresholds.from_json(threshold_path),
        image_root=image_root,
    )
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
            available_local_cpus=DEFAULT_CPU_BUDGET,
            enable_offload=not spec.disable_offload,
            enable_reorder=not spec.disable_reorder,
            enable_caching=False,
            num_samples=spec.num_total_samples,
            enable_local_parallelism=not spec.disable_parallelism,
            enable_fusion=not spec.disable_fusion,
            use_my_optimizer=spec.use_my_optimizer,
            reorder_timeout_sec=spec.reorder_timeout_sec,
        ),
        generate_plan=spec.generate_plan,
    )


__all__ = [
    "DEFAULT_CPU_BUDGET",
    "MultimodalRunningExampleFeature",
    "Thresholds",
    "get_dataset",
    "logical_signature",
]
