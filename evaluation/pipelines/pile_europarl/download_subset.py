#!/usr/bin/env python3
"""Download a 2-GiB source-order The Pile EuroParl prefix to JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_ID = "timaeus/pile-europarl"
DEFAULT_OUTPUT = Path("datasets/pile_europarl/pile-europarl-raw.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")

    revision = HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    with partial.open("wb") as stream:
        for sample in dataset:
            text = sample.get("text")
            if not isinstance(text, str) or not text:
                continue
            payload = {"text": text, "meta": sample.get("meta", {})}
            encoded = (
                json.dumps(payload, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            if count and total_bytes + len(encoded) > args.max_bytes:
                break
            stream.write(encoded)
            digest.update(encoded)
            total_bytes += len(encoded)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())

    if count < 20_000:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Only {count} valid EuroParl records were found")
    partial.replace(args.output)
    metadata = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "split": "train",
        "selection": (
            f"largest source-order prefix not exceeding {args.max_bytes} bytes"
        ),
        "records": count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
