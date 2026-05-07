"""
DataJuicer BLOOM Oscar 流水线在 Cedar 中的逐条等价算子（见
``/data-juicer/configs/reproduced_bloom/bloom-oscar.yaml``）。

未实现 ``document_simhash_deduplicator``：该算子在 DataJuicer 中依赖
``simhash-pybind`` 并对全量样本建索引后再过滤，与 Cedar 的逐样本 Pipe
模型不同；若需与 DataJuicer 完全一致的去重，请在 Cedar 流水线之后单独跑
DataJuicer 去重或自行批处理。
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import urllib.request
from typing import Any, Dict, List, Optional, Set

from evaluation.pipelines.redpajama_c4.dj_operators import (
    TEXT_KEY,
    FIELDS_STATS,
    CharacterRepetitionFilter,
    FixUnicodeMapper,
    LanguageIDScoreFilter,
    PerplexityFilter,
    PunctuationNormalizationMapper,
    SpecialCharactersFilter,
    WhitespaceNormalizationMapper,
    WordRepetitionFilter,
    WordsNumFilter,
    _get_text,
    _get_words_from_document,
    _strip_chars,
    _words_refinement,
    extract_output_text,
    parse_json_line,
    sync_text_key,
    SPECIAL_CHARACTERS,
)

logger = logging.getLogger(__name__)

ASSETS_CACHE_DIR = pathlib.Path.home() / ".cache" / "data_juicer" / "assets"
ASSET_URLS = {
    "stopwords": (
        "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/"
        "data_juicer/stopwords.json"
    ),
    "flagged_words": (
        "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/"
        "data_juicer/flagged_words.json"
    ),
}


def _split_on_whitespace(document: str, new_line: bool = False, tab: bool = False) -> List[str]:
    sep = [" "] + new_line * ["\n"] + tab * ["\t"]
    pattern = "|".join(map(re.escape, sep))
    split_document = re.split(pattern, document)
    return [word for word in split_document if word]


def split_on_newline_tab_whitespace(document: str):
    sentences = document.split("\n")
    sentences = [sentence.split("\t") for sentence in sentences]
    return [
        [_split_on_whitespace(subsentence) for subsentence in sentence]
        for sentence in sentences
    ]


def merge_on_whitespace_tab_newline(sentences) -> str:
    sentences = [
        [" ".join(subsentence) for subsentence in sentence if subsentence]
        for sentence in sentences
    ]
    sentences = ["\t".join(sentence) for sentence in sentences if sentence]
    if not sentences:
        return ""
    return "\n".join(sentences)


def _load_words_dict(words_type: str) -> Dict[str, List[str]]:
    """对齐 data_juicer.utils.asset_utils.load_words_asset（简化版）。"""
    ASSETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    words_dict: Dict[str, List[str]] = {}
    for path in ASSETS_CACHE_DIR.iterdir():
        if path.suffix == ".json" and words_type in path.name:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for key, vals in loaded.items():
                words_dict.setdefault(key, [])
                words_dict[key] += vals if isinstance(vals, list) else list(vals)

    if not words_dict:
        url = ASSET_URLS.get(words_type)
        if not url:
            raise ValueError(f"Unknown words_type: {words_type}")
        target = ASSETS_CACHE_DIR / f"{words_type}.json"
        logger.info("Downloading %s asset to %s", words_type, target)
        urllib.request.urlretrieve(url, target)
        with open(target, "r", encoding="utf-8") as f:
            words_dict = json.load(f)

    if "all" not in words_dict:
        words_dict["all"] = [w for vals in words_dict.values() for w in vals]
    return words_dict


class RemoveWordsWithIncorrectSubstringsMapper:
    """对应 ``remove_words_with_incorrect_substrings_mapper``（无 tokenization）。"""

    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        substrings: Optional[List[str]] = None,
    ):
        self.lang = lang
        self.tokenization = tokenization
        self.substrings = substrings or ["http", "www", ".com", "href", "//"]
        if tokenization:
            raise NotImplementedError(
                "bloom_oscar Cedar 流水线仅移植 yaml 中的默认（无 tokenization）"
            )

    @staticmethod
    def _should_keep_word(word: str, substrings: List[str]) -> bool:
        w = _strip_chars(word, SPECIAL_CHARACTERS)
        return all(sub not in w for sub in substrings)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = _get_text(sample)
        sentences = split_on_newline_tab_whitespace(text)
        sentences = [
            [
                [
                    word
                    for word in subsentence
                    if self._should_keep_word(word, self.substrings)
                ]
                for subsentence in sentence
            ]
            for sentence in sentences
        ]
        sample[TEXT_KEY] = merge_on_whitespace_tab_newline(sentences)
        return sample


class RemoveLongWordsMapper:
    """对应 ``remove_long_words_mapper``（默认 min_len=1）。"""

    def __init__(self, min_len: int = 1, max_len: int = 2**63 - 1):
        self.min_len = min_len
        self.max_len = max_len

    def _should_keep_long_word(self, word: str) -> bool:
        if self.min_len <= len(word) <= self.max_len:
            return True
        stripped = _strip_chars(word, SPECIAL_CHARACTERS)
        return self.min_len <= len(stripped) <= self.max_len

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = _get_text(sample)
        sentences = split_on_newline_tab_whitespace(text)
        sentences = [
            [
                [word for word in subsentence if self._should_keep_long_word(word)]
                for subsentence in sentence
            ]
            for sentence in sentences
        ]
        sample[TEXT_KEY] = merge_on_whitespace_tab_newline(sentences)
        return sample


class StopwordsFilter:
    """对应 ``stopwords_filter``（默认无 tokenization）。"""

    def __init__(self, lang: str = "en", tokenization: bool = False, min_ratio: float = 0.3):
        if tokenization:
            raise NotImplementedError(
                "bloom_oscar yaml 未启用 tokenization；如需请扩展本模块"
            )
        self.lang = lang
        self.min_ratio = min_ratio
        self._stopwords: Set[str] = set(_load_words_dict("stopwords").get(lang, []))

    def __call__(self, sample: Dict[str, Any]) -> bool:
        words = _get_words_from_document(_get_text(sample), token_func=None)
        words = _words_refinement(words, lower_case=True, strip_chars=SPECIAL_CHARACTERS)
        if not words:
            ratio = 0.0
        else:
            ratio = sum(1 for w in words if w in self._stopwords) / len(words)
        ratio = min(ratio, 1.0)
        sample[FIELDS_STATS]["stopwords_ratio"] = ratio
        return ratio >= self.min_ratio


class FlaggedWordsFilter:
    """对应 ``flagged_words_filter``（默认无 tokenization）。"""

    def __init__(self, lang: str = "en", tokenization: bool = False, max_ratio: float = 0.045):
        if tokenization:
            raise NotImplementedError(
                "bloom_oscar yaml 未启用 tokenization；如需请扩展本模块"
            )
        self.lang = lang
        self.max_ratio = max_ratio
        self._flagged: Set[str] = set(_load_words_dict("flagged_words").get(lang, []))

    def __call__(self, sample: Dict[str, Any]) -> bool:
        words = _get_words_from_document(_get_text(sample), token_func=None)
        words = _words_refinement(words, lower_case=True, strip_chars=SPECIAL_CHARACTERS)
        if not words:
            ratio = 0.0
        else:
            ratio = sum(1 for w in words if w in self._flagged) / len(words)
        ratio = min(ratio, 1.0)
        sample[FIELDS_STATS]["flagged_words_ratio"] = ratio
        return ratio <= self.max_ratio
