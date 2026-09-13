"""Target-pipeline entry point for the frozen EuroParl recipe."""

from evaluation.pipelines.target_pipeline.hub_dataset import get_dataset


def get_target_dataset(spec):
    return get_dataset("pile_europarl", spec)
