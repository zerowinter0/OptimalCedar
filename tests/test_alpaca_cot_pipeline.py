import json
from pathlib import Path

import pytest

from cedar.sources import IterSource

from evaluation.pipelines.alpaca_cot.cedar_dataset import (
    AlpacaCotFeature,
    parse_and_format,
)
from evaluation.pipelines.alpaca_cot.validate_recipe import validate


HUB_RECIPE = Path(
    "data-juicer-hub/refined_recipes/alpaca_cot/"
    "alpaca-cot-en-refine.yaml"
)


def test_alpaca_formatter_preserves_instruction_context_and_answer():
    sample = parse_and_format(
        json.dumps(
            {
                "instruction": "Solve the problem.",
                "input": "2 + 2",
                "output": "4",
            }
        )
    )
    assert sample["raw_content"] == "Solve the problem.\n\n2 + 2\n\n4"


def test_alpaca_cot_has_eight_reported_cedar_operators():
    feature = AlpacaCotFeature()
    feature.apply(IterSource(["{}"]))
    non_source = [
        pipe for pipe in feature.logical_pipes.values() if not pipe.is_source()
    ]
    assert len(non_source) == 8
    assert {pipe.tag for pipe in non_source} == {
        "parse_and_format",
        "alphanumeric",
        "char_repeat",
        "flagged_words",
        "max_line_len",
        "text_length",
        "sync_text",
        "extract_text",
    }


@pytest.mark.skipif(not HUB_RECIPE.is_file(), reason="Hub checkout not present")
def test_alpaca_cot_matches_pinned_hub_recipe():
    observed = validate(HUB_RECIPE)

    assert [item["operator"] for item in observed] == [
        "alphanumeric_filter",
        "character_repetition_filter",
        "flagged_words_filter",
        "maximum_line_length_filter",
        "text_length_filter",
    ]
