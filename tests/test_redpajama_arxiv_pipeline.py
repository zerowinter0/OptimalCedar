from pathlib import Path

import pytest

from cedar.sources import IterSource

from evaluation.pipelines.redpajama_arxiv.cedar_dataset import (
    RedPajamaArxivFeature,
)
from evaluation.pipelines.redpajama_arxiv.validate_recipe import validate


HUB_RECIPE = Path(
    "data-juicer-hub/refined_recipes/pretrain/redpajama-arxiv-refine.yaml"
)


def test_redpajama_arxiv_has_eighteen_cedar_operators():
    feature = RedPajamaArxivFeature()
    feature.apply(IterSource(["{}"]),)
    tags = {
        pipe.tag for pipe in feature.logical_pipes.values() if not pipe.is_source()
    }
    assert len(tags) == 18
    assert tags == {
        "parse",
        "clean_email",
        "clean_links",
        "fix_unicode",
        "normalize_punct",
        "normalize_space",
        "alphanumeric",
        "avg_line_len",
        "char_repeat",
        "flagged_words",
        "max_line_len",
        "perplexity",
        "special_chars",
        "text_length",
        "words_num",
        "word_repeat",
        "sync_text",
        "extract_text",
    }


@pytest.mark.skipif(not HUB_RECIPE.is_file(), reason="Hub checkout not present")
def test_redpajama_arxiv_matches_pinned_hub_recipe():
    observed = validate(HUB_RECIPE)

    assert len(observed) == 15
    assert observed[0]["operator"] == "clean_email_mapper"
    assert observed[-1]["operator"] == "word_repetition_filter"
