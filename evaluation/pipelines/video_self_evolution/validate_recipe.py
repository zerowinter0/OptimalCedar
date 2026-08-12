#!/usr/bin/env python3
"""Verify the Cedar adapters against the pinned Hub recipe arguments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from evaluation.pipelines.general_video_refine import dj_operators as ops


FACTORIES = (
    ops.sandbox_video_nsfw_filter,
    ops.sandbox_video_text_similarity_filter,
    ops.sandbox_video_motion_filter,
    ops.sandbox_video_aesthetics_filter,
    ops.sandbox_video_duration_filter,
)


def validate(recipe: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    process = payload.get("process")
    if not isinstance(process, list) or len(process) != len(FACTORIES):
        raise RuntimeError(f"unexpected process list in {recipe}")
    observed = []
    for recipe_item, factory in zip(process, FACTORIES):
        if not isinstance(recipe_item, dict) or len(recipe_item) != 1:
            raise RuntimeError(f"invalid recipe entry: {recipe_item!r}")
        recipe_name, recipe_kwargs = next(iter(recipe_item.items()))
        recipe_kwargs = dict(recipe_kwargs or {})
        # Data-Juicer's scheduler consumes this resource hint; it does not
        # change the per-record predicate. Cedar represents the same resource
        # distinction with PipeExecutionResource.CUDA.
        recipe_kwargs.pop("mem_required", None)
        adapter = factory()
        if adapter.operator_name != recipe_name:
            raise RuntimeError(
                f"operator mismatch: {adapter.operator_name} != {recipe_name}"
            )
        if adapter.operator_kwargs != recipe_kwargs:
            raise RuntimeError(
                f"argument mismatch for {recipe_name}: "
                f"Cedar={adapter.operator_kwargs!r}, Hub={recipe_kwargs!r}"
            )
        observed.append(
            {"operator": recipe_name, "arguments": recipe_kwargs}
        )
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(
            "data-juicer-hub/refined_recipes/video/"
            "data-juicer-sandbox-self-evolution.yaml"
        ),
    )
    args = parser.parse_args()
    result = validate(args.recipe)
    print(f"validated {len(result)} operators against {args.recipe}")


if __name__ == "__main__":
    main()
