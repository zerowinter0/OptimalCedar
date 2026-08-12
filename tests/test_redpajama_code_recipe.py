from pathlib import Path

import pytest

from evaluation.pipelines.redpajama_code.validate_recipe import validate


RECIPE = Path(
    "data-juicer-hub/refined_recipes/github_code/"
    "redpajama-code-refine.yaml"
)


@pytest.mark.skipif(not RECIPE.is_file(), reason="Data-Juicer Hub not cloned")
def test_redpajama_code_matches_pinned_hub_recipe():
    operators = validate(RECIPE)

    assert len(operators) == 14
    assert operators[0]["operator"] == "clean_email_mapper"
    assert operators[-1]["operator"] == "word_repetition_filter"
