#!/usr/bin/env python3
"""Stream a bounded raw RedPajama GitHub Code subset to JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_ID = "togethercomputer/RedPajama-Data-1T"
CONFIG = "github"
DEFAULT_OUTPUT = Path(
    "datasets/redpajama_code/redpajama-github-raw-50000.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--max-bytes", type=int, default=3 * 1024**3)
    args = parser.parse_args()
    if args.max_samples <= 0 or args.max_bytes <= 0:
        parser.error("--max-samples and --max-bytes must be positive")

    revision = HfApi().dataset_info(DATASET_ID).sha
    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,
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
            payload = {
                "text": text,
                "meta": sample.get("meta", {}),
                "red_pajama_subset": sample.get(
                    "red_pajama_subset", CONFIG
                ),
            }
            encoded = (
                json.dumps(payload, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            if total_bytes + len(encoded) > args.max_bytes:
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    "50,000 records exceed the 3-GiB input policy"
                )
            stream.write(encoded)
            digest.update(encoded)
            total_bytes += len(encoded)
            count += 1
            if count >= args.max_samples:
                break
        stream.flush()
        os.fsync(stream.fileno())

    if count != args.max_samples:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Stream ended after {count}; expected {args.max_samples}"
        )
    partial.replace(args.output)
    metadata = {
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "revision": revision,
        "split": "train",
        "selection": f"first {count} valid records in streaming order",
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
