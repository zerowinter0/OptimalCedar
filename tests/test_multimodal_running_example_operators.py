import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation.pipelines.multimodal_running_example.operators import (
    MODEL_REVISIONS,
    AestheticPredicate,
    BlipPredicate,
    ClipPredicate,
    PerplexityPredicate,
    SharpnessPredicate,
    TextNormalizer,
)


@pytest.fixture
def sample_record(tmp_path: Path) -> dict[str, object]:
    rows, columns = np.indices((48, 64))
    pixels = np.stack(
        (
            (rows * 5 + columns * 3) % 256,
            (rows * 2 + columns * 7) % 256,
            (rows * 11 + columns) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(pixels, mode="RGB").save(image_path, quality=95)
    return {
        "record_id": "1:10",
        "caption": "A café — with two tables.",
        "image_path": str(image_path),
    }


def test_text_operators_ignore_image_changes(sample_record: dict[str, object]) -> None:
    changed = {**sample_record, "image_path": "missing-but-never-opened.jpg"}

    assert TextNormalizer()(sample_record)["caption"] == TextNormalizer()(changed)[
        "caption"
    ]
    assert PerplexityPredicate(max_score=float("inf")).score(
        sample_record
    ) == PerplexityPredicate(max_score=float("inf")).score(changed)


def test_text_normalizer_copies_input(sample_record: dict[str, object]) -> None:
    original = dict(sample_record)
    normalized = TextNormalizer()(sample_record)

    assert sample_record == original
    assert normalized is not sample_record
    assert normalized["caption"] == "A café  -  with two tables."


def test_sharpness_ignores_caption(sample_record: dict[str, object]) -> None:
    changed = {**sample_record, "caption": "unrelated text"}
    predicate = SharpnessPredicate(min_score=0.0)

    assert predicate.score(sample_record) == predicate.score(changed)
    assert predicate(sample_record)


def test_predicate_thresholds_use_score_direction(
    sample_record: dict[str, object],
) -> None:
    perplexity = PerplexityPredicate(max_score=-1.0)
    sharpness = SharpnessPredicate(min_score=float("inf"))

    assert not perplexity(sample_record)
    assert not sharpness(sample_record)


def test_model_revisions_are_frozen() -> None:
    assert MODEL_REVISIONS == {
        "clip": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "blip": "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c",
        "aesthetic": "684098de3856fa4678bf800efc05635de5b6cde5",
    }


def test_gpu_scores_are_finite(sample_record: dict[str, object]) -> None:
    for predicate in (
        AestheticPredicate(min_score=0.0),
        ClipPredicate(min_score=0.0),
        BlipPredicate(min_score=0.0),
    ):
        assert math.isfinite(predicate.score(sample_record))
