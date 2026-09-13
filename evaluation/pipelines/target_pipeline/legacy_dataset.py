"""Previous strict-chain adapter, retained only for reference validation."""
from __future__ import annotations

from pathlib import Path
from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import MapperPipe, BatcherPipe
from cedar.sources import LocalLineSource

from .catalog import build_workload
from .hub_dataset import get_dataset as get_hub_dataset, is_hub_workload


class TargetFeature(Feature):
    def __init__(self, workload, batch_size=1, order=None):
        super().__init__()
        self.workload = workload
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.order = workload.tags if order is None else list(order)
        workload.validate_order(self.order)

    def _compose(self, source_pipes):
        pipe = MapperPipe(source_pipes[0], self.workload.prepare,
                          tag="prepare").fix()
        by_tag = {s.tag: s for s in self.workload.stages}
        for tag in self.order:
            stage = by_tag[tag]
            pipe = MapperPipe(pipe, stage, tag=tag)
            if stage.dependencies:
                pipe.depends_on(list(stage.dependencies))
        pipe = MapperPipe(pipe, self.workload.finalize, tag="finalize").fix()
        if self.batch_size > 1:
            pipe = BatcherPipe(pipe, batch_size=self.batch_size).fix()
        return pipe


def get_dataset_for(name, spec):
    kwargs = dict(spec.kwargs or {})
    dataset_path = kwargs.get("dataset_path")
    if not dataset_path:
        raise ValueError("target_pipeline requires dataset_path=<JSONL manifest>")
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(dataset_path)
    workload = build_workload(
        name, seed=int(kwargs.get("seed", 0)), epoch=int(kwargs.get("epoch", 0)),
        image_root=kwargs.get("image_root", ""),
        tokenizer_path=kwargs.get("tokenizer_path", "openai/clip-vit-base-patch32"),
    )
    feature = TargetFeature(workload, spec.batch_size)
    feature.apply(LocalLineSource(str(dataset_path)))
    context = CedarContext(ray_config=spec.to_ray_config())
    if spec.config:
        return DataSet(context, {"feature": feature}, spec.config,
                       enable_controller=False, enable_optimizer=False,
                       prefetch=not spec.disable_prefetch)
    return DataSet(
        context, {"feature": feature},
        enable_controller=not spec.disable_controller,
        enable_optimizer=not spec.disable_optimizer,
        prefetch=not spec.disable_prefetch,
        profiled_data=spec.profiled_stats,
        run_profiling=spec.run_profiling,
        optimizer_options=OptimizerOptions(
            enable_prefetch=not spec.disable_prefetch,
            available_local_cpus=64,
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


def get_dataset(spec):
    name = (spec.kwargs or {}).get("workload")
    if not name:
        raise ValueError(
            "Specify workload=simclr|dino|swav|clip|blip or a Hub target name"
        )
    if is_hub_workload(name):
        return get_hub_dataset(name, spec)
    return get_dataset_for(name, spec)
