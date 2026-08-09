#!/usr/bin/env python3
"""Validate and materialize the frozen MSR-VTT video-text source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


DATASET_REVISION = "a9c822473969ee469e224da2187fda193c62e960"
EXPECTED_VIDEOS = 10_000
EXPECTED_SENTENCES = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (output / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(output)


def build(args: argparse.Namespace) -> None:
    archive = args.archive.resolve()
    annotation = args.annotation.resolve()
    video_root = args.video_root.resolve()
    output = args.output.resolve()
    metadata_path = args.metadata.resolve()
    if not archive.is_file() or not annotation.is_file():
        raise FileNotFoundError("MSR-VTT archive or annotation is missing")

    if args.reextract and video_root.exists():
        shutil.rmtree(video_root)
    if not video_root.exists() or not any(video_root.rglob("*.mp4")):
        safe_extract(archive, video_root)

    payload = json.loads(annotation.read_text(encoding="utf-8"))
    videos = payload.get("videos", [])
    sentences = payload.get("sentences", [])
    if len(videos) != EXPECTED_VIDEOS:
        raise RuntimeError(f"Expected 10000 videos, found {len(videos)}")
    if len(sentences) != EXPECTED_SENTENCES:
        raise RuntimeError(
            f"Expected 200000 sentences, found {len(sentences)}"
        )

    video_paths = {}
    for path in sorted(video_root.rglob("*.mp4")):
        video_id = path.stem
        if video_id in video_paths:
            raise RuntimeError(f"Duplicate extracted video: {video_id}")
        video_paths[video_id] = path.relative_to(video_root)
    expected_ids = {item["video_id"] for item in videos}
    missing = sorted(expected_ids - set(video_paths))
    if missing or len(video_paths) != EXPECTED_VIDEOS:
        raise RuntimeError(
            f"Extracted video coverage mismatch: files={len(video_paths)}, "
            f"missing={missing[:10]}"
        )

    captions = defaultdict(list)
    for item in sentences:
        captions[item["video_id"]].append(item)
    if set(captions) != expected_ids:
        raise RuntimeError("Caption/video ID coverage mismatch")
    for values in captions.values():
        values.sort(key=lambda item: int(item["sen_id"]))
    caption_counts = {len(values) for values in captions.values()}
    if caption_counts != {20}:
        raise RuntimeError(f"Unexpected captions per video: {caption_counts}")

    ordered_ids = sorted(expected_ids, key=lambda value: int(value[5:]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for caption_round in range(20):
            for video_id in ordered_ids:
                item = captions[video_id][caption_round]
                record = {
                    "video_id": video_id,
                    "sentence_id": int(item["sen_id"]),
                    "videos": [str(video_paths[video_id])],
                    "text": f"<__dj__video> {item['caption']}",
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": "MSR-VTT",
        "source_repository": "nisav/MSR-VTT",
        "source_revision": DATASET_REVISION,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "annotation": str(annotation),
        "annotation_sha256": sha256(annotation),
        "video_count": len(video_paths),
        "video_caption_pair_count": EXPECTED_SENTENCES,
        "captions_per_video": 20,
        "ordering": "caption_round_then_numeric_video_id",
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("datasets/general_video_refine")
    parser.add_argument(
        "--archive", default=root / "source/MSRVTT_Videos.zip", type=Path
    )
    parser.add_argument(
        "--annotation",
        default=root / "source/raw_data/MSRVTT_data.json",
        type=Path,
    )
    parser.add_argument("--video-root", default=root / "videos", type=Path)
    parser.add_argument(
        "--output",
        default=root / "msrvtt-video-text-200000.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--metadata", default=root / "dataset_metadata.json", type=Path
    )
    parser.add_argument("--reextract", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
