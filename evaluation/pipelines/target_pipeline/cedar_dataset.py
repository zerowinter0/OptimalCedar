"""Dispatch to the five explicit workload implementations."""
from importlib import import_module
from evaluation.pipelines.target_pipeline.hub_dataset import get_dataset as get_hub_dataset, is_hub_workload
# Compatibility for old strict-chain reference tests; not used by entrypoints.
from evaluation.pipelines.target_pipeline.legacy_dataset import TargetFeature


def get_dataset_for(name, spec):
    if name == "simclrv2":
        name = "simclr"
    if name not in ("simclr", "dino", "swav", "clip", "blip"):
        raise ValueError(f"Unknown target workload: {name!r}")
    module = import_module(f"evaluation.pipelines.target_pipeline.{name}.cedar_dataset")
    return module.get_dataset(spec)


def get_dataset(spec):
    name = (spec.kwargs or {}).get("workload")
    if is_hub_workload(name):
        return get_hub_dataset(name, spec)
    return get_dataset_for(name, spec)
