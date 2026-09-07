import hashlib
import json
from pathlib import Path

import pytest

from evaluation.pipelines.multimodal_running_example.fixture import (
    build_fixture,
    read_record_ids,
)


@pytest.fixture
def mini_coco(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "val2017"
    image_root.mkdir()
    images = []
    annotations = []
    for index in range(8):
        image_id = 100 + index
        file_name = f"{image_id:012d}.jpg"
        (image_root / file_name).write_bytes(f"image-{image_id}".encode())
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 32 + index,
                "height": 24 + index,
            }
        )
        annotations.append(
            {
                "id": 1000 + index,
                "image_id": image_id,
                "caption": f"caption {index}",
            }
        )
    annotation_path = tmp_path / "captions.json"
    annotation_path.write_text(
        json.dumps({"images": images, "annotations": annotations}),
        encoding="utf-8",
    )
    return annotation_path, image_root


def test_partition_boundaries_are_disjoint(
    tmp_path: Path,
    mini_coco: tuple[Path, Path],
) -> None:
    manifest = build_fixture(
        *mini_coco,
        output_dir=tmp_path / "fixture",
        split_sizes=(2, 2, 3, 1),
    )

    ids = [
        set(read_record_ids(tmp_path / "fixture" / f"{name}.jsonl"))
        for name in manifest.split_names
    ]
    assert [len(items) for items in ids] == [2, 2, 3, 1]
    assert all(
        not (left & right)
        for index, left in enumerate(ids)
        for right in ids[index + 1 :]
    )


def test_fixture_is_byte_reproducible(
    tmp_path: Path,
    mini_coco: tuple[Path, Path],
) -> None:
    first = build_fixture(
        *mini_coco,
        output_dir=tmp_path / "a",
        split_sizes=(2, 2, 3, 1),
    )
    second = build_fixture(
        *mini_coco,
        output_dir=tmp_path / "b",
        split_sizes=(2, 2, 3, 1),
    )

    assert first.output_sha256 == second.output_sha256
    assert first.output_sha256 == hashlib.sha256(
        b"".join(
            (tmp_path / "a" / f"{name}.jsonl").read_bytes()
            for name in first.split_names
        )
    ).hexdigest()


def test_fixture_rejects_missing_image(
    tmp_path: Path,
    mini_coco: tuple[Path, Path],
) -> None:
    annotation_path, image_root = mini_coco
    next(image_root.iterdir()).unlink()

    with pytest.raises(FileNotFoundError, match="COCO image is missing"):
        build_fixture(
            annotation_path,
            image_root,
            output_dir=tmp_path / "fixture",
            split_sizes=(2, 2, 3, 1),
        )
