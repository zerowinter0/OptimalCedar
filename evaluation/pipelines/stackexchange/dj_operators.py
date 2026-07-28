"""Per-sample equivalents of the Data-Juicer StackExchange recipe operators.

The recipe's global ``document_simhash_deduplicator`` is intentionally omitted:
it requires cross-sample state and is outside Cedar's per-sample DP search space.
"""

from __future__ import annotations

from typing import Any, Dict, Set

from evaluation.pipelines.bloom_oscar.dj_operators import _load_words_dict
from evaluation.pipelines.redpajama_c4.dj_operators import (
    FIELDS_STATS,
    LINK_PATTERN,
    SPECIAL_CHARACTERS,
    TEXT_KEY,
    AlphanumericFilter,
    AverageLineLengthFilter,
    CharacterRepetitionFilter,
    CleanEmailMapper,
    CleanLinksMapper,
    FixUnicodeMapper,
    LanguageIDScoreFilter,
    MaximumLineLengthFilter,
    PerplexityFilter,
    PunctuationNormalizationMapper,
    SpecialCharactersFilter,
    WhitespaceNormalizationMapper,
    WordRepetitionFilter,
    WordsNumFilter,
    _get_sentencepiece_model,
    _get_text,
    _get_words_from_document,
    _words_refinement,
    extract_output_text,
    parse_json_line,
    sync_text_key,
)


class FlaggedWordsFilter:
    """Data-Juicer ``flagged_words_filter`` with recipe tokenization."""

    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        min_ratio: float = 0.0,
        max_ratio: float = 0.045,
    ):
        self.lang = lang
        self.tokenization = tokenization
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self._flagged: Set[str] = set(
            _load_words_dict("flagged_words").get(lang, [])
        )

    def __call__(self, sample: Dict[str, Any]) -> bool:
        token_func = None
        if self.tokenization:
            token_func = _get_sentencepiece_model(self.lang).encode_as_pieces
        words = _get_words_from_document(_get_text(sample), token_func=token_func)
        words = _words_refinement(
            words, lower_case=True, strip_chars=SPECIAL_CHARACTERS
        )
        ratio = (
            sum(1 for word in words if word in self._flagged) / len(words)
            if words
            else 0.0
        )
        ratio = min(ratio, 1.0)
        sample[FIELDS_STATS]["flagged_words_ratio"] = ratio
        return self.min_ratio <= ratio <= self.max_ratio


class TextLengthFilter:
    """Data-Juicer ``text_length_filter`` for character length."""

    def __init__(self, min_len: int = 10, max_len: int = 2**63 - 1):
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, sample: Dict[str, Any]) -> bool:
        text_len = len(_get_text(sample))
        sample[FIELDS_STATS]["text_len"] = text_len
        return self.min_len <= text_len <= self.max_len


class StopwordsFilter:
    """Data-Juicer ``stopwords_filter`` including SentencePiece mode."""

    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        min_ratio: float = 0.3,
    ):
        self.lang = lang
        self.tokenization = tokenization
        self.min_ratio = min_ratio
        self._stopwords: Set[str] = set(
            _load_words_dict("stopwords").get(lang, [])
        )

    def __call__(self, sample: Dict[str, Any]) -> bool:
        token_func = None
        if self.tokenization:
            token_func = _get_sentencepiece_model(self.lang).encode_as_pieces
        words = _get_words_from_document(
            _get_text(sample), token_func=token_func
        )
        words = _words_refinement(
            words, lower_case=True, strip_chars=SPECIAL_CHARACTERS
        )
        ratio = (
            sum(1 for word in words if word in self._stopwords) / len(words)
            if words
            else 0.0
        )
        ratio = min(ratio, 1.0)
        sample[FIELDS_STATS]["stopwords_ratio"] = ratio
        return ratio >= self.min_ratio


__all__ = [
    "AlphanumericFilter",
    "AverageLineLengthFilter",
    "CharacterRepetitionFilter",
    "CleanEmailMapper",
    "CleanLinksMapper",
    "FixUnicodeMapper",
    "FlaggedWordsFilter",
    "LanguageIDScoreFilter",
    "LINK_PATTERN",
    "MaximumLineLengthFilter",
    "PerplexityFilter",
    "PunctuationNormalizationMapper",
    "SpecialCharactersFilter",
    "StopwordsFilter",
    "TextLengthFilter",
    "TEXT_KEY",
    "WhitespaceNormalizationMapper",
    "WordRepetitionFilter",
    "WordsNumFilter",
    "extract_output_text",
    "parse_json_line",
    "sync_text_key",
]
