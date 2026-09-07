import json
from pathlib import Path

import pytest

from cedar.sources import LocalLineSource
from evaluation.pipelines.multimodal_running_example.cedar_dataset import (
    MultimodalRunningExampleFeature,
    Thresholds,
    logical_signature,
)


@pytest.fixture
def feature(tmp_path: Path) -> MultimodalRunningExampleFeature:
    source_path = tmp_path / "records.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "record_id": "1:1",
                "caption": "a caption",
                "image_path": "image.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = MultimodalRunningExampleFeature(
        thresholds=Thresholds(
            perplexity_max=100.0,
            sharpness_min=10.0,
            aesthetic_min=0.4,
            clip_min=0.2,
            blip_min=0.5,
        )
    )
    result.apply(LocalLineSource(str(source_path)))
    return result


def test_feature_has_six_operators_and_declared_constraints(
    feature: MultimodalRunningExampleFeature,
) -> None:
    signature = logical_signature(feature)

    assert signature["tags"] == [
        "normalize",
        "perplexity",
        "sharpness",
        "aesthetic",
        "clip",
        "blip",
    ]
    assert set(signature["dependencies"]) == {
        ("normalize", "perplexity"),
        ("sharpness", "aesthetic"),
        ("perplexity", "clip"),
        ("aesthetic", "clip"),
        ("clip", "blip"),
    }
    assert signature["cuda_tags"] == ["aesthetic", "clip", "blip"]


def test_representative_legal_orders_retain_identical_records() -> None:
    thresholds = {
        "perplexity": ("max", 100.0),
        "sharpness": ("min", 10.0),
        "aesthetic": ("min", 0.4),
        "clip": ("min", 0.2),
        "blip": ("min", 0.5),
    }
    scores = {
        "keep": {
            "perplexity": 50.0,
            "sharpness": 20.0,
            "aesthetic": 0.8,
            "clip": 0.4,
            "blip": 0.9,
        },
        "drop-text": {
            "perplexity": 150.0,
            "sharpness": 20.0,
            "aesthetic": 0.8,
            "clip": 0.4,
            "blip": 0.9,
        },
        "drop-image": {
            "perplexity": 50.0,
            "sharpness": 5.0,
            "aesthetic": 0.8,
            "clip": 0.4,
            "blip": 0.9,
        },
    }

    def execute(order: tuple[str, ...]) -> set[str]:
        retained = set(scores)
        for tag in order:
            if tag == "normalize":
                continue
            direction, threshold = thresholds[tag]
            retained = {
                record_id
                for record_id in retained
                if (
                    scores[record_id][tag] <= threshold
                    if direction == "max"
                    else scores[record_id][tag] >= threshold
                )
            }
        return retained

    assert execute(("normalize", "perplexity", "sharpness", "aesthetic", "clip", "blip")) == execute(
        ("sharpness", "aesthetic", "normalize", "perplexity", "clip", "blip")
    ) == {"keep"}


def test_threshold_file_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text('{"perplexity_max": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="threshold"):
        Thresholds.from_json(path)
