import argparse
import json
import logging
import math
import multiprocessing as mp
import pathlib
import re
import string
import threading
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import fasttext
import ftfy
import kenlm
import sentencepiece as spm

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import FilterPipe, MapperPipe, Pipe
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.redpajama_c4 import dj_operators as dj_ops


DEFAULT_DATASET_PATH = pathlib.Path(
    "datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl"
)
DEFAULT_PROFILE_PATH = "/tmp/redpajama_c4_feature_profile.yml"
TEXT_KEY = "raw_content"
FIELDS_STATS = "__dj__stats__"
FIELDS_CONTEXT = "__dj__context__"
FIELDS_META = "__dj__meta__"

MODEL_CACHE_DIR = pathlib.Path.home() / ".cache" / "data_juicer" / "models"
FASTTEXT_LID_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
KENLM_MODEL_URL_TEMPLATE = (
    "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/{lang}.arpa.bin"
)
SENTENCEPIECE_MODEL_URL_TEMPLATE = (
    "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/{lang}.sp.model"
)

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+", re.DOTALL)
LINK_PATTERN = re.compile(
    r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|"""
    r"""[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|"""
    r"""(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"""
    r"""[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""",
    re.DOTALL,
)
PUNCTUATION_UNICODE_MAP = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}
VARIOUS_WHITESPACES = {
    " ",
    "\t",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u00a0",
    "\u202f",
    "\u205f",
    "\u3000",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufffc",
    "\u0084",
}
SPECIAL_CHARACTERS = set(
    string.punctuation
    + string.digits
    + string.whitespace
    + (
        "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
        "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
        "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
        "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
        "」﴾》"
    )
)

_MODEL_LOCK = threading.Lock()
_FASTTEXT_MODEL = None
_KENLM_MODELS: Dict[str, kenlm.Model] = {}
_SENTENCEPIECE_MODELS: Dict[str, spm.SentencePieceProcessor] = {}


def _ensure_model_file(filename: str, url: str) -> pathlib.Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_CACHE_DIR / filename
    if target.exists():
        return target

    logging.info("Downloading model %s to %s", url, target)
    urllib.request.urlretrieve(url, target)
    return target


def _get_fasttext_model():
    global _FASTTEXT_MODEL
    with _MODEL_LOCK:
        if _FASTTEXT_MODEL is None:
            model_path = _ensure_model_file("lid.176.bin", FASTTEXT_LID_URL)
            _FASTTEXT_MODEL = fasttext.load_model(str(model_path))
        return _FASTTEXT_MODEL


def _get_sentencepiece_model(lang: str):
    with _MODEL_LOCK:
        if lang not in _SENTENCEPIECE_MODELS:
            model_path = _ensure_model_file(
                f"{lang}.sp.model",
                SENTENCEPIECE_MODEL_URL_TEMPLATE.format(lang=lang),
            )
            processor = spm.SentencePieceProcessor()
            processor.load(str(model_path))
            _SENTENCEPIECE_MODELS[lang] = processor
        return _SENTENCEPIECE_MODELS[lang]


def _get_kenlm_model(lang: str):
    with _MODEL_LOCK:
        if lang not in _KENLM_MODELS:
            model_path = _ensure_model_file(
                f"{lang}.arpa.bin",
                KENLM_MODEL_URL_TEMPLATE.format(lang=lang),
            )
            _KENLM_MODELS[lang] = kenlm.Model(str(model_path))
        return _KENLM_MODELS[lang]


def _split_on_whitespace(document: str, new_line: bool = False, tab: bool = False):
    separators = [" "]
    if new_line:
        separators.append("\n")
    if tab:
        separators.append("\t")
    pattern = "|".join(map(re.escape, separators))
    return [word for word in re.split(pattern, document) if word]


def _get_words_from_document(
    document: str,
    token_func=None,
    new_line: bool = True,
    tab: bool = True,
) -> List[str]:
    if token_func is not None:
        return token_func(document)
    return _split_on_whitespace(document, new_line=new_line, tab=tab)


def _strip_chars(document: str, strip_characters: Sequence[str]) -> str:
    if not document:
        return document
    strip_characters = set(strip_characters)
    begin_idx = 0
    end_idx = len(document)
    for idx, char in enumerate(document):
        if char in strip_characters:
            begin_idx = idx + 1
        else:
            break
    for idx in range(len(document) - 1, -1, -1):
        if document[idx] in strip_characters:
            end_idx = idx
        else:
            break
    return document[begin_idx:end_idx]


def _words_refinement(
    words: List[str],
    lower_case: bool = False,
    strip_chars: Optional[Sequence[str]] = None,
) -> List[str]:
    if lower_case:
        words = [word.lower() for word in words]
    if strip_chars:
        words = [_strip_chars(word, strip_chars) for word in words]
        words = [word for word in words if word]
    return words


def _get_text(sample: Dict[str, Any]) -> str:
    value = sample.get(TEXT_KEY, "")
    return value if isinstance(value, str) else str(value)


def _char_repetition_ratio(text: str, rep_len: int) -> float:
    char_ngrams = [text[idx : idx + rep_len] for idx in range(len(text) - rep_len + 1)]
    freq_char_ngrams = Counter(char_ngrams)
    if not freq_char_ngrams:
        return 0.0

    freqs = sorted(freq_char_ngrams.values(), reverse=True)
    num_no_rep = sum(1 for freq in freqs if freq == 1)
    num_rep = min(int(math.sqrt(len(freqs))), len(freqs) - num_no_rep)
    if num_rep <= 0:
        return 0.0
    return sum(freqs[:num_rep]) / sum(freqs)


def _word_repetition_ratio(words: List[str], rep_len: int) -> float:
    word_ngrams = [
        " ".join(words[idx : idx + rep_len]) for idx in range(len(words) - rep_len + 1)
    ]
    freq_word_ngrams = Counter(word_ngrams)
    if not freq_word_ngrams:
        return 0.0

    freq_values = list(freq_word_ngrams.values())
    rep_more_than_one = [freq for freq in freq_values if freq > 1]
    return sum(rep_more_than_one) / sum(freq_values) if sum(freq_values) else 0.0


class CleanEmailMapper:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[TEXT_KEY] = EMAIL_PATTERN.sub("", _get_text(sample))
        return sample


class CleanLinksMapper:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[TEXT_KEY] = LINK_PATTERN.sub("", _get_text(sample))
        return sample


class FixUnicodeMapper:
    def __init__(self, normalization: str = "NFC"):
        self.normalization = normalization

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[TEXT_KEY] = ftfy.fix_text(_get_text(sample), normalization=self.normalization)
        return sample


class PunctuationNormalizationMapper:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[TEXT_KEY] = "".join(
            PUNCTUATION_UNICODE_MAP.get(char, char) for char in _get_text(sample)
        )
        return sample


class WhitespaceNormalizationMapper:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = _get_text(sample).strip()
        sample[TEXT_KEY] = "".join(
            char if char not in VARIOUS_WHITESPACES else " " for char in text
        )
        return sample


class AlphanumericFilter:
    def __init__(self, min_ratio: float, max_ratio: float):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample: Dict[str, Any]) -> bool:
        text = _get_text(sample)
        ratio = sum(char.isalnum() for char in text) / len(text) if text else 0.0
        sample[FIELDS_STATS]["alnum_ratio"] = ratio
        return self.min_ratio <= ratio <= self.max_ratio


class AverageLineLengthFilter:
    def __init__(self, min_len: int = 10, max_len: int = 3000):
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, sample: Dict[str, Any]) -> bool:
        text = _get_text(sample)
        lines = text.splitlines()
        avg_line_len = len(text) / len(lines) if lines else 0.0
        sample[FIELDS_STATS]["avg_line_length"] = avg_line_len
        return self.min_len <= avg_line_len <= self.max_len


class CharacterRepetitionFilter:
    def __init__(self, rep_len: int, min_ratio: float = 0.0, max_ratio: float = 0.3):
        self.rep_len = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample: Dict[str, Any]) -> bool:
        ratio = _char_repetition_ratio(_get_text(sample), self.rep_len)
        sample[FIELDS_STATS]["char_rep_ratio"] = ratio
        return self.min_ratio <= ratio <= self.max_ratio


class LanguageIDScoreFilter:
    def __init__(self, min_score: float, lang: str = ""):
        self.lang = [lang] if lang else None
        self.min_score = min_score

    def __call__(self, sample: Dict[str, Any]) -> bool:
        text = _get_text(sample).lower().replace("\n", " ")
        labels, scores = _get_fasttext_model().predict(text)
        lang_id = labels[0].replace("__label__", "")
        lang_score = float(scores[0])
        sample[FIELDS_STATS]["lang"] = lang_id
        sample[FIELDS_STATS]["lang_score"] = lang_score
        if self.lang:
            return lang_id in self.lang and lang_score >= self.min_score
        return lang_score >= self.min_score


class MaximumLineLengthFilter:
    def __init__(self, min_len: int = 10, max_len: int = 4000):
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, sample: Dict[str, Any]) -> bool:
        lines = _get_text(sample).splitlines()
        max_line_len = max(map(len, lines)) if lines else 0
        sample[FIELDS_STATS]["max_line_length"] = max_line_len
        return self.min_len <= max_line_len <= self.max_len


class PerplexityFilter:
    def __init__(self, lang: str, max_ppl: float):
        self.lang = lang
        self.max_ppl = max_ppl

    def __call__(self, sample: Dict[str, Any]) -> bool:
        tokenizer = _get_sentencepiece_model(self.lang)
        words = _get_words_from_document(
            _get_text(sample), token_func=tokenizer.encode_as_pieces
        )
        text = " ".join(words)
        logits = 0.0
        length = 0
        kenlm_model = _get_kenlm_model(self.lang)
        for line in text.splitlines():
            logits += kenlm_model.score(line)
            length += len(line.split()) + 1
        ppl = 10.0 ** (-logits / length) if length else 0.0
        ppl = round(ppl, 1)
        sample[FIELDS_STATS]["perplexity"] = ppl
        return ppl <= self.max_ppl


class SpecialCharactersFilter:
    def __init__(self, min_ratio: float = 0.0, max_ratio: float = 0.4):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample: Dict[str, Any]) -> bool:
        text = _get_text(sample)
        ratio = (
            len([char for char in text if char in SPECIAL_CHARACTERS]) / len(text)
            if text
            else 0.0
        )
        sample[FIELDS_STATS]["special_char_ratio"] = ratio
        return self.min_ratio <= ratio <= self.max_ratio


class WordsNumFilter:
    def __init__(self, lang: str, tokenization: bool, min_num: int, max_num: int):
        self.lang = lang
        self.tokenization = tokenization
        self.min_num = min_num
        self.max_num = max_num

    def __call__(self, sample: Dict[str, Any]) -> bool:
        token_func = None
        if self.tokenization:
            token_func = _get_sentencepiece_model(self.lang).encode_as_pieces
        words = _get_words_from_document(_get_text(sample), token_func=token_func)
        words = _words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
        num_words = len(words)
        sample[FIELDS_STATS]["num_words"] = num_words
        return self.min_num <= num_words <= self.max_num


class WordRepetitionFilter:
    def __init__(
        self,
        lang: str,
        tokenization: bool,
        rep_len: int,
        min_ratio: float = 0.0,
        max_ratio: float = 0.231,
    ):
        self.lang = lang
        self.tokenization = tokenization
        self.rep_len = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample: Dict[str, Any]) -> bool:
        token_func = None
        if self.tokenization:
            token_func = _get_sentencepiece_model(self.lang).encode_as_pieces
        words = _get_words_from_document(_get_text(sample), token_func=token_func)
        words = _words_refinement(
            words, lower_case=True, strip_chars=SPECIAL_CHARACTERS
        )
        ratio = _word_repetition_ratio(words, self.rep_len)
        sample[FIELDS_STATS]["word_rep_ratio"] = ratio
        return self.min_ratio <= ratio <= self.max_ratio


def _parse_json_line(line: str) -> Dict[str, Any]:
    sample = json.loads(line)
    if TEXT_KEY not in sample:
        if "text" in sample:
            sample[TEXT_KEY] = sample["text"]
        else:
            sample[TEXT_KEY] = ""
    sample.setdefault(FIELDS_STATS, {})
    sample.setdefault(FIELDS_CONTEXT, {})
    sample.setdefault(FIELDS_META, {})
    return sample


def _sync_text_key(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample["text"] = sample.get(TEXT_KEY, "")
    return sample


def _extract_output_text(sample: Dict[str, Any]) -> str:
    return sample.get(TEXT_KEY, "")


class RedPajamaC4Feature(Feature):
    def __init__(self):
        super().__init__()

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, dj_ops.parse_json_line, tag="parse").fix()

        fp = MapperPipe(fp, dj_ops.CleanEmailMapper(), tag="clean_email").fix()
        fp = MapperPipe(fp, dj_ops.CleanLinksMapper(), tag="clean_links").fix()
        fp = MapperPipe(fp, dj_ops.FixUnicodeMapper(), tag="fix_unicode").fix()
        fp = MapperPipe(
            fp,
            dj_ops.PunctuationNormalizationMapper(),
            tag="normalize_punct",
        ).fix()
        fp = MapperPipe(
            fp,
            dj_ops.WhitespaceNormalizationMapper(),
            tag="normalize_space",
        ).fix()

        fp = FilterPipe(
            fp,
            dj_ops.AlphanumericFilter(min_ratio=0.65, max_ratio=0.9),
            tag="alphanumeric",
        )
        fp = FilterPipe(
            fp,
            dj_ops.AverageLineLengthFilter(max_len=3000),
            tag="avg_line_len",
        )
        fp = FilterPipe(
            fp,
            dj_ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.3),
            tag="char_repeat",
        )
        fp = FilterPipe(
            fp,
            dj_ops.LanguageIDScoreFilter(min_score=0.6),
            tag="lang_id",
        )
        fp = FilterPipe(
            fp,
            dj_ops.MaximumLineLengthFilter(max_len=4000),
            tag="max_line_len",
        )
        fp = FilterPipe(
            fp,
            dj_ops.PerplexityFilter(lang="en", max_ppl=6000),
            tag="perplexity",
        )
        fp = FilterPipe(
            fp,
            dj_ops.SpecialCharactersFilter(max_ratio=0.4),
            tag="special_chars",
        )
        fp = FilterPipe(
            fp,
            dj_ops.WordsNumFilter(
                lang="en", tokenization=True, min_num=20, max_num=10000
            ),
            tag="words_num",
        )
        fp = FilterPipe(
            fp,
            dj_ops.WordRepetitionFilter(
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.231,
            ),
            tag="word_repeat",
        )

        fp = MapperPipe(fp, dj_ops.sync_text_key, tag="sync_text").fix()
        fp = MapperPipe(fp, dj_ops.extract_output_text, tag="extract_text").fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = DEFAULT_DATASET_PATH
    if spec.kwargs and spec.kwargs.get("dataset_path"):
        dataset_path = pathlib.Path(spec.kwargs["dataset_path"])

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = LocalLineSource(str(dataset_path))
    feature = RedPajamaC4Feature()
    feature.apply(source)

    if spec.config:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    else:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            enable_controller=not spec.disable_controller,
            enable_optimizer=not spec.disable_optimizer,
            profiled_data=spec.profiled_stats,
            run_profiling=spec.run_profiling,
            optimizer_options=OptimizerOptions(
                enable_prefetch=not spec.disable_prefetch,
                est_throughput=None,
                available_local_cpus=mp.cpu_count(),
                enable_offload=not spec.disable_offload,
                enable_reorder=not spec.disable_reorder,
                enable_caching=not spec.disable_caching,
                num_samples=getattr(spec, "num_total_samples", None),
                enable_local_parallelism=not spec.disable_parallelism,
                enable_fusion=not spec.disable_fusion,
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
                reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
            ),
            generate_plan=spec.generate_plan,
        )

    return dataset


def _build_default_spec(args: argparse.Namespace) -> CedarEvalSpec:
    use_ray = args.use_ray
    if not args.run_profiling and not args.disable_offload:
        use_ray = True

    return CedarEvalSpec(
        batch_size=1,
        num_total_samples=args.num_total_samples,
        num_epochs=1,
        config=args.config,
        kwargs={"dataset_path": args.dataset_path},
        use_ray=use_ray,
        profiled_stats=args.profiled_stats,
        run_profiling=args.run_profiling,
        disable_prefetch=args.disable_prefetch,
        disable_offload=args.disable_offload,
        disable_parallelism=args.disable_parallelism,
        disable_reorder=args.disable_reorder,
        disable_fusion=args.disable_fusion,
        disable_caching=args.disable_caching,
        use_my_optimizer=args.use_my_optimizer,
        generate_plan=args.generate_plan,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--profiled_stats", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_profiling", action="store_true")
    parser.add_argument("--generate_plan", action="store_true")
    parser.add_argument("--num_total_samples", type=int, default=20)
    parser.add_argument("--num_preview_samples", type=int, default=5)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument(
        "--use_my_optimizer",
        type=int,
        choices=list(range(9)),
        default=0,
    )
    parser.add_argument("--disable_prefetch", action="store_true")
    parser.add_argument("--disable_offload", action="store_true")
    parser.add_argument("--disable_parallelism", action="store_true")
    parser.add_argument("--disable_reorder", action="store_true")
    parser.add_argument("--disable_fusion", action="store_true")
    parser.add_argument("--disable_caching", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    spec = _build_default_spec(args)
    ds = get_dataset(spec)

    if args.run_profiling:
        return

    for idx, sample in enumerate(ds):
        print(sample)
        if idx + 1 >= args.num_preview_samples:
            break


if __name__ == "__main__":
    main()
