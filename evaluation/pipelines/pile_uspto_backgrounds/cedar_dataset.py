"""Cedar entry point for the frozen Data-Juicer USPTO recipe."""

from evaluation.pipelines.pile_recipe_registry import get_pile_recipe_dataset


def get_dataset(spec):
    return get_pile_recipe_dataset(spec, "pile_uspto_backgrounds")
