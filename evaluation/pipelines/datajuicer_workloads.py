"""Shared native-framework definitions for Data-Juicer-derived workloads.

The operator parameters intentionally mirror the corresponding Cedar Features.
Framework adapters consume these stages through their own execution APIs; this
module does not import or execute Cedar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional


DEFAULT_PATHS = {
    "llava_pretrain": Path("/tmp/llava_pretrain_cedar_fixture.jsonl"),
    "redpajama_c4": Path(
        "datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl"
    ),
    "stackexchange": Path(
        "datasets/stackexchange/redpajama-stackexchange-35000.jsonl"
    ),
    "pile_europarl": Path(
        "datasets/pile_europarl/pile-europarl-raw.jsonl"
    ),
    "redpajama_code": Path(
        "datasets/redpajama_code/redpajama-github-raw-50000.jsonl"
    ),
    "pile_hackernews": Path(
        "datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
    ),
    "pile_pubmed_abstracts": Path(
        "datasets/pile_pubmed_abstracts/"
        "pile-pubmed-abstracts-raw-100000.jsonl"
    ),
    "pile_freelaw": Path(
        "datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl"
    ),
    "pile_uspto_backgrounds": Path(
        "datasets/pile_uspto_backgrounds/"
        "pile-uspto-backgrounds-raw-100000.jsonl"
    ),
}

# Native-framework copy of the four frozen recipe parameter sets.  A test
# compares every stage and constructor field against pile_recipe_registry so
# Cedar and the external systems cannot silently diverge.  Keeping this small
# metadata table here avoids importing Cedar (and therefore TensorFlow) in
# every PyTorch DataLoader or Ray worker.
_PILE_NATIVE_RECIPES = {
    "pile_hackernews": {
        "clean_links": False,
        "filters": (
            ("alphanumeric", "AlphanumericFilter", dict(tokenization=False, min_ratio=0.2, max_ratio=2**63 - 1)),
            ("avg_line_len", "AverageLineLengthFilter", dict(min_len=15, max_len=2**63 - 1)),
            ("char_repeat", "CharacterRepetitionFilter", dict(rep_len=10, max_ratio=0.3)),
            ("flagged_words", "FlaggedWordsFilter", dict(lang="en", tokenization=True, max_ratio=0.05)),
            ("language_id", "LanguageIDScoreFilter", dict(min_score=0.2)),
            ("max_line_len", "MaximumLineLengthFilter", dict(min_len=20, max_len=2**63 - 1)),
            ("perplexity", "PerplexityFilter", dict(lang="en", max_ppl=10_000)),
            ("special_chars", "SpecialCharactersFilter", dict(max_ratio=0.7)),
            ("text_length", "TextLengthFilter", dict(min_len=100)),
            ("words_num", "WordsNumFilter", dict(lang="en", tokenization=True, min_num=30, max_num=2**63 - 1)),
            ("word_repeat", "WordRepetitionFilter", dict(lang="en", tokenization=True, rep_len=10, max_ratio=0.8)),
        ),
    },
    "pile_pubmed_abstracts": {
        "clean_links": True,
        "filters": (
            ("alphanumeric", "AlphanumericFilter", dict(tokenization=False, min_ratio=0.7, max_ratio=0.881)),
            ("avg_line_len", "AverageLineLengthFilter", dict(max_len=2100)),
            ("char_repeat", "CharacterRepetitionFilter", dict(rep_len=10, max_ratio=0.2)),
            ("flagged_words", "FlaggedWordsFilter", dict(lang="en", tokenization=True, max_ratio=0.00232)),
            ("language_id", "LanguageIDScoreFilter", dict(min_score=0.5)),
            ("max_line_len", "MaximumLineLengthFilter", dict(max_len=4000)),
            ("perplexity", "PerplexityFilter", dict(lang="en", max_ppl=4000)),
            ("special_chars", "SpecialCharactersFilter", dict(max_ratio=0.38)),
            ("text_length", "TextLengthFilter", dict(max_len=4000)),
            ("words_num", "WordsNumFilter", dict(lang="en", tokenization=True, min_num=20, max_num=700)),
            ("word_repeat", "WordRepetitionFilter", dict(lang="en", tokenization=True, rep_len=10, max_ratio=0.0887)),
        ),
    },
    "pile_freelaw": {
        "clean_links": True,
        "filters": (
            ("alphanumeric", "AlphanumericFilter", dict(tokenization=False, min_ratio=0.3, max_ratio=2**63 - 1)),
            ("avg_line_len", "AverageLineLengthFilter", dict(max_len=697)),
            ("char_repeat", "CharacterRepetitionFilter", dict(rep_len=10, max_ratio=0.4)),
            ("flagged_words", "FlaggedWordsFilter", dict(lang="en", tokenization=True, max_ratio=0.0053)),
            ("language_id", "LanguageIDScoreFilter", dict(min_score=0.5)),
            ("max_line_len", "MaximumLineLengthFilter", dict(max_len=4229)),
            ("perplexity", "PerplexityFilter", dict(lang="en", max_ppl=5322)),
            ("special_chars", "SpecialCharactersFilter", dict(max_ratio=0.7)),
            ("stopwords", "StopwordsFilter", dict(lang="en", tokenization=True, min_ratio=0.1)),
            ("text_length", "TextLengthFilter", dict(max_len=84_026)),
            ("words_num", "WordsNumFilter", dict(lang="en", tokenization=True, min_num=100, max_num=15_208)),
            ("word_repeat", "WordRepetitionFilter", dict(lang="en", tokenization=True, rep_len=10, max_ratio=0.155)),
        ),
    },
    "pile_uspto_backgrounds": {
        "clean_links": True,
        "filters": (
            ("alphanumeric", "AlphanumericFilter", dict(tokenization=False, min_ratio=0.7, max_ratio=2**63 - 1)),
            ("avg_line_len", "AverageLineLengthFilter", dict(max_len=2000)),
            ("char_repeat", "CharacterRepetitionFilter", dict(rep_len=10, max_ratio=0.2)),
            ("flagged_words", "FlaggedWordsFilter", dict(lang="en", tokenization=True, max_ratio=0.0016)),
            ("language_id", "LanguageIDScoreFilter", dict(min_score=0.6)),
            ("max_line_len", "MaximumLineLengthFilter", dict(max_len=3061)),
            ("perplexity", "PerplexityFilter", dict(lang="en", max_ppl=4000)),
            ("special_chars", "SpecialCharactersFilter", dict(max_ratio=0.3)),
            ("text_length", "TextLengthFilter", dict(max_len=21_556)),
            ("words_num", "WordsNumFilter", dict(lang="en", tokenization=True, min_num=100, max_num=6000)),
            ("word_repeat", "WordRepetitionFilter", dict(lang="en", tokenization=True, rep_len=10, max_ratio=0.169)),
        ),
    },
}


@dataclass(frozen=True)
class PipelineStage:
    kind: str
    name: str
    operator: Callable[[Any], Any]


def _map(name: str, operator: Callable[[Any], Any]) -> PipelineStage:
    return PipelineStage("map", name, operator)


def _filter(name: str, operator: Callable[[Any], bool]) -> PipelineStage:
    return PipelineStage("filter", name, operator)


def build_stages(
    workload: str,
    *,
    image_root: str = "",
) -> List[PipelineStage]:
    if workload == "llava_pretrain":
        return _llava_stages(image_root)
    if workload == "redpajama_c4":
        return _redpajama_stages()
    if workload == "stackexchange":
        return _stackexchange_stages()
    if workload == "pile_europarl":
        return _pile_europarl_stages()
    if workload == "redpajama_code":
        return _redpajama_code_stages()
    if workload in {
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    }:
        return _registered_pile_stages(workload)
    raise ValueError(f"Unknown Data-Juicer-derived workload: {workload}")


def execute_stages(
    line: str,
    stages: Iterable[PipelineStage],
) -> Optional[Any]:
    value: Any = line
    for stage in stages:
        if stage.kind == "map":
            value = stage.operator(value)
        elif stage.kind == "filter":
            if not stage.operator(value):
                return None
        else:
            raise ValueError(f"Unknown pipeline stage kind: {stage.kind}")
    return value


def output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "raw_content"):
            if key in value:
                return str(value[key])
    return str(value)


def _llava_stages(image_root: str) -> List[PipelineStage]:
    from evaluation.pipelines.llava_pretrain import dj_operators as ops

    return [
        _map("parse", ops.parse_json_line),
        _map("image_root", ops.SetImageRootMapper(image_root)),
        _map("fix_unicode", ops.FixUnicodeMapper()),
        _map("punctuation_norm", ops.PunctuationNormalizationMapper()),
        _filter(
            "alphanumeric",
            ops.AlphanumericFilter(min_ratio=0.60, max_ratio=1.0),
        ),
        _filter(
            "char_repeat",
            ops.CharacterRepetitionFilter(
                rep_len=10,
                max_ratio=0.09373663,
            ),
        ),
        _filter(
            "flagged_words",
            ops.FlaggedWordsFilter(
                lang="en",
                tokenization=False,
                max_ratio=0.0,
            ),
        ),
        _filter(
            "perplexity",
            ops.PerplexityFilter(lang="en", max_ppl=14435.5806),
        ),
        _filter(
            "special_chars",
            ops.SpecialCharactersFilter(
                min_ratio=0.16534802,
                max_ratio=0.42023757,
            ),
        ),
        _filter(
            "word_repeat",
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=False,
                rep_len=10,
                max_ratio=0.03085751,
            ),
        ),
        _filter(
            "image_aspect_ratio",
            ops.ImageAspectRatioFilter(
                min_ratio=0.333,
                max_ratio=3.0,
                any_or_all="any",
            ),
        ),
        _filter(
            "image_shape",
            ops.ImageShapeFilter(
                max_width=727,
                max_height=606,
                any_or_all="any",
            ),
        ),
        _filter(
            "image_size",
            ops.ImageSizeFilter(max_size="124KB", any_or_all="any"),
        ),
        _filter(
            "image_text_similarity",
            ops.ImageTextSimilarityFilter(
                hf_clip="openai/clip-vit-base-patch32",
                min_score=0.20315419,
            ),
        ),
        _filter(
            "image_text_matching",
            ops.ImageTextMatchingFilter(
                hf_blip="Salesforce/blip-itm-base-coco",
                min_score=0.44930778,
            ),
        ),
        _map("sync_text", ops.sync_text_key),
    ]


def _redpajama_stages() -> List[PipelineStage]:
    from evaluation.pipelines.redpajama_c4 import dj_operators as ops

    return [
        _map("parse", ops.parse_json_line),
        _map("clean_email", ops.CleanEmailMapper()),
        _map("clean_links", ops.CleanLinksMapper()),
        _map("fix_unicode", ops.FixUnicodeMapper()),
        _map("normalize_punct", ops.PunctuationNormalizationMapper()),
        _map("normalize_space", ops.WhitespaceNormalizationMapper()),
        _filter(
            "alphanumeric",
            ops.AlphanumericFilter(min_ratio=0.65, max_ratio=0.9),
        ),
        _filter(
            "avg_line_len",
            ops.AverageLineLengthFilter(max_len=3000),
        ),
        _filter(
            "char_repeat",
            ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.3),
        ),
        _filter(
            "lang_id",
            ops.LanguageIDScoreFilter(min_score=0.6),
        ),
        _filter(
            "max_line_len",
            ops.MaximumLineLengthFilter(max_len=4000),
        ),
        _filter(
            "perplexity",
            ops.PerplexityFilter(lang="en", max_ppl=6000),
        ),
        _filter(
            "special_chars",
            ops.SpecialCharactersFilter(max_ratio=0.4),
        ),
        _filter(
            "words_num",
            ops.WordsNumFilter(
                lang="en",
                tokenization=True,
                min_num=20,
                max_num=10000,
            ),
        ),
        _filter(
            "word_repeat",
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.231,
            ),
        ),
        _map("sync_text", ops.sync_text_key),
        _map("extract_text", ops.extract_output_text),
    ]


def _stackexchange_stages() -> List[PipelineStage]:
    from evaluation.pipelines.stackexchange import dj_operators as ops

    return [
        _map("parse", ops.parse_json_line),
        _map("clean_email", ops.CleanEmailMapper()),
        _map("clean_links", ops.CleanLinksMapper()),
        _map("fix_unicode", ops.FixUnicodeMapper()),
        _map("normalize_punct", ops.PunctuationNormalizationMapper()),
        _map("normalize_space", ops.WhitespaceNormalizationMapper()),
        _filter(
            "alphanumeric",
            ops.AlphanumericFilter(min_ratio=0.35, max_ratio=0.943),
        ),
        _filter(
            "avg_line_len",
            ops.AverageLineLengthFilter(min_len=20, max_len=400),
        ),
        _filter(
            "char_repeat",
            ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.4),
        ),
        _filter(
            "flagged_words",
            ops.FlaggedWordsFilter(
                lang="en",
                tokenization=True,
                max_ratio=0.01,
            ),
        ),
        _filter(
            "language_id",
            ops.LanguageIDScoreFilter(min_score=0.1),
        ),
        _filter(
            "max_line_len",
            ops.MaximumLineLengthFilter(min_len=80),
        ),
        _filter(
            "perplexity",
            ops.PerplexityFilter(lang="en", max_ppl=10_000),
        ),
        _filter(
            "special_chars",
            ops.SpecialCharactersFilter(
                min_ratio=0.232,
                max_ratio=0.7,
            ),
        ),
        _filter(
            "text_length",
            ops.TextLengthFilter(min_len=200),
        ),
        _filter(
            "words_num",
            ops.WordsNumFilter(
                lang="en",
                tokenization=True,
                min_num=100,
                max_num=2**63 - 1,
            ),
        ),
        _filter(
            "word_repeat",
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.8,
            ),
        ),
        _map("sync_text", ops.sync_text_key),
        _map("extract_text", ops.extract_output_text),
    ]


def _pile_europarl_stages() -> List[PipelineStage]:
    from evaluation.pipelines.stackexchange import dj_operators as ops

    return [
        _map("parse", ops.parse_json_line),
        _map("clean_email", ops.CleanEmailMapper()),
        _map("clean_links", ops.CleanLinksMapper()),
        _map("fix_unicode", ops.FixUnicodeMapper()),
        _map("normalize_punct", ops.PunctuationNormalizationMapper()),
        _map("normalize_space", ops.WhitespaceNormalizationMapper()),
        _filter(
            "alphanumeric",
            ops.AlphanumericFilter(
                min_ratio=0.75, max_ratio=0.90, tokenization=False
            ),
        ),
        _filter(
            "avg_line_len", ops.AverageLineLengthFilter(max_len=588)
        ),
        _filter(
            "char_repeat",
            ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.16),
        ),
        _filter(
            "flagged_words",
            ops.FlaggedWordsFilter(
                lang="en", tokenization=True, max_ratio=0.0007
            ),
        ),
        _filter(
            "language_id", ops.LanguageIDScoreFilter(min_score=0.7)
        ),
        _filter(
            "max_line_len", ops.MaximumLineLengthFilter(max_len=4000)
        ),
        _filter(
            "perplexity", ops.PerplexityFilter(lang="en", max_ppl=7596)
        ),
        _filter(
            "special_chars", ops.SpecialCharactersFilter(max_ratio=0.3)
        ),
        _filter("text_length", ops.TextLengthFilter(max_len=200_000)),
        _filter(
            "words_num",
            ops.WordsNumFilter(
                lang="en",
                tokenization=True,
                min_num=20,
                max_num=100_000,
            ),
        ),
        _filter(
            "word_repeat",
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.2,
            ),
        ),
        _map("sync_text", ops.sync_text_key),
        _map("extract_text", ops.extract_output_text),
    ]


def _redpajama_code_stages() -> List[PipelineStage]:
    from evaluation.pipelines.redpajama_c4 import dj_operators as ops

    return [
        _map("parse", ops.parse_json_line),
        _map("clean_email", ops.CleanEmailMapper()),
        _map("clean_links", ops.CleanLinksMapper()),
        _map("fix_unicode", ops.FixUnicodeMapper()),
        _map("normalize_punct", ops.PunctuationNormalizationMapper()),
        _map("normalize_space", ops.WhitespaceNormalizationMapper()),
        _map("clean_copyright", ops.CleanCopyrightMapper()),
        _filter(
            "alphanumeric_chars",
            ops.AlphanumericFilter(
                min_ratio=0.4, max_ratio=0.8, tokenization=False
            ),
        ),
        _filter(
            "alphanumeric_tokens",
            ops.AlphanumericFilter(
                min_ratio=1.5, max_ratio=3.0, tokenization=True
            ),
        ),
        _filter(
            "avg_line_len",
            ops.AverageLineLengthFilter(min_len=15, max_len=100),
        ),
        _filter(
            "char_repeat",
            ops.CharacterRepetitionFilter(
                rep_len=10, min_ratio=0.05, max_ratio=0.3
            ),
        ),
        _filter(
            "max_line_len",
            ops.MaximumLineLengthFilter(min_len=50, max_len=500),
        ),
        _filter("text_length", ops.TextLengthFilter(min_len=300)),
        _filter(
            "words_num",
            ops.WordsNumFilter(
                lang="en",
                tokenization=False,
                min_num=30,
                max_num=5000,
            ),
        ),
        _filter(
            "word_repeat",
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=False,
                rep_len=10,
                max_ratio=0.1,
            ),
        ),
        _map("sync_text", ops.sync_text_key),
        _map("extract_text", ops.extract_output_text),
    ]


def _registered_pile_stages(workload: str) -> List[PipelineStage]:
    """Build a native-framework chain from the frozen Pile recipe."""
    from evaluation.pipelines.stackexchange import dj_operators as ops

    recipe = _PILE_NATIVE_RECIPES[workload]
    stages = [
        _map("parse", ops.parse_json_line),
        _map("clean_email", ops.CleanEmailMapper()),
    ]
    if recipe["clean_links"]:
        stages.append(_map("clean_links", ops.CleanLinksMapper()))
    stages.extend(
        [
            _map("fix_unicode", ops.FixUnicodeMapper()),
            _map("normalize_punct", ops.PunctuationNormalizationMapper()),
            _map("normalize_space", ops.WhitespaceNormalizationMapper()),
        ]
    )
    stages.extend(
        _filter(tag, getattr(ops, operator)(**kwargs))
        for tag, operator, kwargs in recipe["filters"]
    )
    stages.extend(
        [
            _map("sync_text", ops.sync_text_key),
            _map("extract_text", ops.extract_output_text),
        ]
    )
    return stages


__all__ = [
    "DEFAULT_PATHS",
    "PipelineStage",
    "build_stages",
    "execute_stages",
    "output_text",
]
