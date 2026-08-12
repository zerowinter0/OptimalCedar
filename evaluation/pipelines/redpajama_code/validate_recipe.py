#!/usr/bin/env python3
"""Verify RP-Code Cedar operators against the pinned Hub recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.pipelines.redpajama_code.cedar_dataset import (
    RedPajamaCodeFeature,
)
from evaluation.pipelines.validate_text_hub_recipe import validate_text_recipe


def validate(recipe: Path):
    return validate_text_recipe(
        recipe,
        RedPajamaCodeFeature(),
        adapter_tags={"parse", "sync_text", "extract_text"},
        omitted_operators={"document_simhash_deduplicator"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(
            "data-juicer-hub/refined_recipes/github_code/"
            "redpajama-code-refine.yaml"
        ),
    )
    args = parser.parse_args()
    result = validate(args.recipe)
    print(f"validated {len(result)} operators against {args.recipe}")


if __name__ == "__main__":
    main()
