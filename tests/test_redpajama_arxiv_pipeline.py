from cedar.sources import IterSource

from evaluation.pipelines.redpajama_arxiv.cedar_dataset import (
    RedPajamaArxivFeature,
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
