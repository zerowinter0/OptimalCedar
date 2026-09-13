"""Shared dataset setup; operator graphs live in each workload module."""
from pathlib import Path
from cedar.client import DataSet
from cedar.compose import OptimizerOptions
from cedar.config import CedarContext
from cedar.sources import LocalLineSource


def create_dataset(feature, spec):
    dataset_path = (spec.kwargs or {}).get("dataset_path")
    if not dataset_path or not Path(dataset_path).is_file():
        raise FileNotFoundError("dataset_path must name an existing JSONL manifest")
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
