"""Target-pipeline adapters for frozen Data-Juicer Hub recipes."""

from __future__ import annotations

from .hub_catalog import HUB_BY_NAME, get_hub_workload


def get_dataset(name, spec):
    """Run the already audited Cedar migration through the target namespace."""

    get_hub_workload(name)
    if name == "pile_europarl":
        from evaluation.pipelines.pile_europarl.cedar_dataset import get_dataset as run

        return run(spec)
    from evaluation.pipelines.pile_recipe_registry import get_pile_recipe_dataset

    return get_pile_recipe_dataset(spec, name)


def is_hub_workload(name: str) -> bool:
    return name in HUB_BY_NAME


__all__ = ["get_dataset", "is_hub_workload"]
