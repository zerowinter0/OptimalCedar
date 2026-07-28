import json

from cedar.sources import IterSource
from cedar.compose.utils import topological_sort

from evaluation.pipelines.stackexchange import dj_operators as ops
from evaluation.pipelines.stackexchange.cedar_dataset import StackExchangeFeature


EXPECTED_TAGS = [
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
    "language_id",
    "max_line_len",
    "perplexity",
    "special_chars",
    "text_length",
    "words_num",
    "word_repeat",
    "sync_text",
    "extract_text",
]


def test_stackexchange_recipe_order_and_semantic_boundaries():
    feature = StackExchangeFeature()
    feature.apply(IterSource([json.dumps({"text": "example"})]))
    tags = [
        feature.logical_pipes[pipe_id].tag
        for pipe_id in topological_sort(feature.logical_adj_list)
        if not feature.logical_pipes[pipe_id].is_source()
    ]
    assert tags == EXPECTED_TAGS
    assert not any("dedup" in tag for tag in tags)
    fixed_tags = {
        pipe.tag
        for pipe in feature.logical_pipes.values()
        if not pipe.is_source() and pipe._fix_order
    }
    assert fixed_tags == {
        "parse",
        "clean_email",
        "clean_links",
        "fix_unicode",
        "normalize_punct",
        "normalize_space",
        "sync_text",
        "extract_text",
    }


def test_stackexchange_added_operator_semantics():
    sample = ops.parse_json_line(json.dumps({"text": "abc def"}))
    assert not ops.TextLengthFilter(min_len=8)(sample)
    assert ops.TextLengthFilter(min_len=7)(sample)

    flagged = ops.FlaggedWordsFilter.__new__(ops.FlaggedWordsFilter)
    flagged.lang = "en"
    flagged.tokenization = False
    flagged.min_ratio = 0.0
    flagged.max_ratio = 0.2
    flagged._flagged = {"bad"}
    sample = ops.parse_json_line(json.dumps({"text": "good bad text"}))
    assert not flagged(sample)
    assert sample[ops.FIELDS_STATS]["flagged_words_ratio"] == 1 / 3


def test_clean_links_uses_data_juicer_regex_engine_for_nested_urls():
    text = (
        "javascript:void(document.location="
        "'http://trans.hiragana.jp/ruby/'+escape(document.location))"
    )
    sample = ops.parse_json_line(json.dumps({"text": text}))
    result = ops.CleanLinksMapper()(sample)
    assert "http://trans.hiragana.jp/ruby/" not in result[ops.TEXT_KEY]
    assert ops.LINK_PATTERN.__class__.__module__.startswith("_regex")
