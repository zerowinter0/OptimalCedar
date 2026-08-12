#!/usr/bin/env python3
"""Verify Alpaca-CoT Cedar operators against the pinned Hub recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.pipelines.alpaca_cot.cedar_dataset import AlpacaCotFeature
from evaluation.pipelines.validate_text_hub_recipe import validate_text_recipe


def validate(recipe: Path):
    return validate_text_recipe(
        recipe,
        AlpacaCotFeature(),
        adapter_tags={"parse_and_format", "sync_text", "extract_text"},
        omitted_operators={
            "document_deduplicator",
            "document_simhash_deduplicator",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(
            "data-juicer-hub/refined_recipes/alpaca_cot/"
            "alpaca-cot-en-refine.yaml"
        ),
    )
    args = parser.parse_args()
    result = validate(args.recipe)
    print(f"validated {len(result)} operators against {args.recipe}")


if __name__ == "__main__":
    main()
