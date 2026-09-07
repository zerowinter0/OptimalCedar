"""Build deterministic, disjoint COCO image--caption experiment splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_SPLIT_SIZES = (500, 500, 3000, 1000)
SPLIT_NAMES = ("calibration", "pilot", "formal", "scaling")


@dataclass(frozen=True)
class FixtureManifest:
    annotation_sha256: str
    output_sha256: str
    splits: dict[str, int]
    split_names: tuple[str, ...] = SPLIT_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _jsonl_bytes(records: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for record in records
    )


def _partition(
    records: Sequence[dict[str, object]],
    sizes: tuple[int, int, int, int],
) -> Iterator[tuple[str, Sequence[dict[str, object]]]]:
    offset = 0
    for name, size in zip(SPLIT_NAMES, sizes):
        yield name, records[offset : offset + size]
        offset += size


def build_fixture(
    annotation_path: Path,
    image_root: Path,
    output_dir: Path,
    split_sizes: tuple[int, int, int, int] = DEFAULT_SPLIT_SIZES,
) -> FixtureManifest:
    """Materialize stable JSONL splits from a COCO captions annotation."""

    annotation_path = Path(annotation_path)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    if len(split_sizes) != len(SPLIT_NAMES) or any(size < 0 for size in split_sizes):
        raise ValueError("split_sizes must contain four non-negative integers")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    annotations = sorted(
        payload["annotations"],
        key=lambda item: (int(item["image_id"]), int(item["id"])),
    )
    required = sum(split_sizes)
    if len(annotations) < required:
        raise ValueError(
            f"COCO annotation contains {len(annotations)} captions; need {required}"
        )

    records: list[dict[str, object]] = []
    for annotation in annotations[:required]:
        image_id = int(annotation["image_id"])
        image = images.get(image_id)
        if image is None:
            raise KeyError(f"COCO image metadata is missing for image_id={image_id}")
        file_name = str(image["file_name"])
        image_path = image_root / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO image is missing: {image_path}")
        caption_id = int(annotation["id"])
        records.append(
            {
                "caption": str(annotation["caption"]),
                "caption_id": caption_id,
                "height": int(image["height"]),
                "image_id": image_id,
                "image_path": file_name,
                "record_id": f"{image_id}:{caption_id}",
                "width": int(image["width"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = hashlib.sha256()
    split_counts: dict[str, int] = {}
    for name, split_records in _partition(records, split_sizes):
        encoded = _jsonl_bytes(split_records)
        _atomic_write(output_dir / f"{name}.jsonl", encoded)
        combined.update(encoded)
        split_counts[name] = len(split_records)

    manifest = FixtureManifest(
        annotation_sha256=_sha256(annotation_path),
        output_sha256=combined.hexdigest(),
        splits=split_counts,
    )
    manifest_payload = asdict(manifest)
    manifest_payload["split_names"] = list(manifest.split_names)
    _atomic_write(
        output_dir / "manifest.json",
        (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return manifest


def read_record_ids(path: Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield str(json.loads(line)["record_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_fixture(args.annotations, args.image_root, args.output)
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
