from cedar.pipes import FilterPipe
from cedar.sources import LocalLineSource

from evaluation.pipelines.pile_recipe_registry import (
    PileRecipeFeature,
    RECIPES,
    make_filter,
)
from evaluation.pipelines.datajuicer_workloads import _PILE_NATIVE_RECIPES


def _tags(feature):
    return {
        pipe.tag: pipe
        for pipe in feature.logical_pipes.values()
        if pipe.tag is not None
    }


def _linear_tags(feature):
    source_id = feature.source_pipes[0].id
    ordered = []
    pipe_id = source_id
    while feature.logical_adj_list[pipe_id]:
        successors = feature.logical_adj_list[pipe_id]
        assert len(successors) == 1
        pipe_id = next(iter(successors))
        tag = feature.logical_pipes[pipe_id].tag
        if tag is not None:
            ordered.append(tag)
    return ordered


def test_registered_sources_are_complete_bounded_100k_splits():
    assert set(RECIPES) == {
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    }
    for workload, recipe in RECIPES.items():
        assert recipe.workload == workload
        assert recipe.dataset_id.startswith("timaeus/pile-")
        assert len(recipe.revision) == 40
        assert recipe.logical_bytes < 3 * 1024**3
        assert len(recipe.filters) >= 9
        assert recipe.official_recipe.endswith("-refine.yaml")


def test_registry_materializes_fixed_mappers_and_reorderable_filters():
    for workload, recipe in RECIPES.items():
        feature = PileRecipeFeature(workload)
        feature.apply(LocalLineSource("/tmp/nonexistent-pile-recipe.jsonl"))
        tagged = _tags(feature)

        expected_fixed = {
            "parse",
            "clean_email",
            "fix_unicode",
            "normalize_punct",
            "normalize_space",
            "sync_text",
            "extract_text",
        }
        if recipe.clean_links:
            expected_fixed.add("clean_links")
        else:
            assert "clean_links" not in tagged
        for tag in expected_fixed:
            assert tagged[tag]._fix_order

        linear_filter_tags = [
            tag
            for tag in _linear_tags(feature)
            if isinstance(tagged[tag], FilterPipe)
        ]
        assert [spec.tag for spec in recipe.filters] == linear_filter_tags
        for filter_spec in recipe.filters:
            pipe = tagged[filter_spec.tag]
            assert isinstance(pipe, FilterPipe)
            assert not pipe._fix_order

        copied = feature.create_copy()
        assert copied.workload == workload
        assert set(_tags(copied)) == set(tagged)


def test_every_registered_filter_constructor_is_supported():
    for recipe in RECIPES.values():
        operators = [make_filter(spec) for spec in recipe.filters]
        assert len(operators) == len(recipe.filters)

    free_law_stopwords = next(
        spec
        for spec in RECIPES["pile_freelaw"].filters
        if spec.tag == "stopwords"
    )
    operator = make_filter(free_law_stopwords)
    assert operator.tokenization is True
    assert operator.min_ratio == 0.1


def test_native_framework_recipe_metadata_exactly_matches_cedar_registry():
    assert set(_PILE_NATIVE_RECIPES) == set(RECIPES)
    for workload, recipe in RECIPES.items():
        native = _PILE_NATIVE_RECIPES[workload]
        assert native["clean_links"] == recipe.clean_links
        assert native["filters"] == tuple(
            (spec.tag, spec.operator, dict(spec.kwargs))
            for spec in recipe.filters
        )
