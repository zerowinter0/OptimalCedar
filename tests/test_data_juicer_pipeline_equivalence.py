import json
import sys
from types import SimpleNamespace

import pytest

from cedar.compose.utils import topological_sort
from cedar.sources import IterSource
from evaluation.pipelines.llava_pretrain import dj_operators as llava_ops
from evaluation.pipelines.llava_pretrain.cedar_dataset import LlavaPretrainFeature
from evaluation.pipelines.llava_pretrain.convert_caption_subset import (
    convert_subset,
)
from evaluation.pipelines.redpajama_c4 import dj_operators as text_ops
from evaluation.pipelines.redpajama_c4.cedar_dataset import RedPajamaC4Feature
from evaluation.pipelines.pile_europarl.cedar_dataset import (
    PileEuroparlFeature,
)
from evaluation.pipelines.redpajama_code.cedar_dataset import (
    RedPajamaCodeFeature,
)


def _sample(text):
    return text_ops.parse_json_line(json.dumps({"text": text}))


def _fixed_tags(feature):
    feature.apply(IterSource([json.dumps({"text": "example"})]))
    return {
        pipe.tag
        for pipe in feature.logical_pipes.values()
        if not pipe.is_source() and pipe._fix_order
    }


def test_latest_special_characters_include_emoji():
    assert "😀" in text_ops.SPECIAL_CHARACTERS


def test_maximum_line_length_default_is_unbounded():
    sample = _sample("x" * 5000)
    assert text_ops.MaximumLineLengthFilter()(sample)
    assert not text_ops.MaximumLineLengthFilter(max_len=4000)(sample)
    assert text_ops.MaximumLineLengthFilter().max_len == sys.maxsize


def test_language_id_does_not_accept_flat_low_confidence(monkeypatch):
    model = SimpleNamespace(
        predict=lambda text: (["__label__en"], [0.25001001358032227])
    )
    monkeypatch.setattr(text_ops, "_get_fasttext_model", lambda: model)
    assert not text_ops.LanguageIDScoreFilter(min_score=0.8, lang="en")(
        _sample("English text")
    )


def test_perplexity_model_errors_propagate(monkeypatch):
    tokenizer = SimpleNamespace(encode_as_pieces=lambda text: text.split())
    monkeypatch.setattr(text_ops, "_get_sentencepiece_model", lambda lang: tokenizer)

    def fail(lang):
        raise RuntimeError("missing kenlm")

    monkeypatch.setattr(text_ops, "_get_kenlm_model", fail)
    with pytest.raises(RuntimeError, match="missing kenlm"):
        text_ops.PerplexityFilter(lang="en", max_ppl=100)(_sample("some text"))


def test_llava_flagged_words_uses_complete_asset():
    op = llava_ops.FlaggedWordsFilter(lang="en", max_ratio=0.0)
    assert len(op.flagged_words) > 400
    sample = llava_ops.parse_json_line(
        json.dumps({"text": "<image> an arsehole caption <|__dj__eoc|>"})
    )
    assert not op(sample)


@pytest.mark.parametrize(
    "operator",
    [
        llava_ops.ImageAspectRatioFilter(),
        llava_ops.ImageShapeFilter(),
        llava_ops.ImageSizeFilter(),
        llava_ops.ImageTextSimilarityFilter(),
        llava_ops.ImageTextMatchingFilter(),
    ],
)
def test_llava_image_filters_keep_samples_without_images(operator):
    sample = llava_ops.parse_json_line(json.dumps({"text": "caption", "images": []}))
    assert operator(sample)


def test_llava_chunks_remove_special_tokens_and_match_images():
    sample = llava_ops.parse_json_line(
        json.dumps(
            {
                "text": "<image> first caption <|__dj__eoc|>"
                "<image><image> second caption <|__dj__eoc|>",
                "images": ["one.jpg", "two.jpg", "three.jpg"],
            }
        )
    )
    chunks = list(llava_ops._iter_image_text_chunks(sample))
    assert [text for text, _ in chunks] == ["first caption", "second caption"]
    assert [[path.name for path in paths] for _, paths in chunks] == [
        ["one.jpg"],
        ["two.jpg", "three.jpg"],
    ]


def test_caption_subset_conversion(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "1",
                "image": "00000/image.jpg",
                "conversations": [
                    {"from": "human", "value": "prompt"},
                    {"from": "gpt", "value": "the caption"},
                ],
            }
        )
        + "\n"
    )
    assert convert_subset(source, output, 1) == 1
    assert json.loads(output.read_text()) == {
        "id": "1",
        "images": ["00000/image.jpg"],
        "text": "<image>\nthe caption <|__dj__eoc|>",
    }


def test_caption_subset_conversion_resolves_shared_image_root(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    image_root = tmp_path / "images"
    source.write_text(
        json.dumps(
            {
                "id": "1",
                "image": "00000/image.jpg",
                "conversations": [
                    {"from": "human", "value": "prompt"},
                    {"from": "gpt", "value": "the caption"},
                ],
            }
        )
        + "\n"
    )

    assert convert_subset(source, output, 1, image_root=image_root) == 1
    converted = json.loads(output.read_text())
    assert converted["images"] == [
        str((image_root / "00000/image.jpg").resolve())
    ]


def test_mapper_boundaries_preserve_recipe_order():
    assert _fixed_tags(RedPajamaC4Feature()) == {
        "parse",
        "clean_email",
        "clean_links",
        "fix_unicode",
        "normalize_punct",
        "normalize_space",
        "sync_text",
        "extract_text",
    }
    assert _fixed_tags(LlavaPretrainFeature()) == {
        "parse",
        "image_root",
        "fix_unicode",
        "punctuation_norm",
        "sync_text",
    }
    assert _fixed_tags(PileEuroparlFeature()) == {
        "parse",
        "clean_email",
        "clean_links",
        "fix_unicode",
        "normalize_punct",
        "normalize_space",
        "sync_text",
        "extract_text",
    }
    assert _fixed_tags(RedPajamaCodeFeature()) == {
        "parse",
        "clean_email",
        "clean_links",
        "fix_unicode",
        "normalize_punct",
        "normalize_space",
        "clean_copyright",
        "sync_text",
        "extract_text",
    }


def test_clean_copyright_matches_data_juicer_header_behavior():
    operator = text_ops.CleanCopyrightMapper()
    sample = _sample("/* Copyright 2024 Example */\nprint('kept')")
    assert operator(sample)[text_ops.TEXT_KEY] == "\nprint('kept')"

    sample = _sample("# license\n// generated\n\nprint('kept')")
    assert operator(sample)[text_ops.TEXT_KEY] == "print('kept')"


def test_tokenized_alphanumeric_uses_alpha_characters_per_token(monkeypatch):
    tokenizer = SimpleNamespace(tokenize=lambda text: ["one", "two"])
    monkeypatch.setattr(
        text_ops, "_get_alphanumeric_tokenizer", lambda: tokenizer
    )
    sample = _sample("ab12")
    operator = text_ops.AlphanumericFilter(
        min_ratio=0.9,
        max_ratio=1.1,
        tokenization=True,
    )
    assert operator(sample)
    assert sample[text_ops.FIELDS_STATS]["alpha_token_ratio"] == 1.0
