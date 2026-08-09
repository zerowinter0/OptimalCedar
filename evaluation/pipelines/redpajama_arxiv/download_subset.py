#!/usr/bin/env python3
"""Download a bounded source-order prefix of RedPajama's ArXiv shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


REVISION = "398f92572e94f4793e41c22ab7ea2a788d9e7de4"
SOURCE_LIST = "urls/arxiv.txt"
SOURCE_LIST_URL = (
    "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/"
    f"resolve/{REVISION}/{SOURCE_LIST}"
)
SOURCE_LIST_SHA256 = (
    "f6eaf3bce82f31a620a5679d6ed34552b3f5b0d2f12b1c4beeb21c767639b30d"
)
MAX_BYTES = 3 * 1024**3
MIN_RECORDS = 20_000
DEFAULT_OUTPUT = Path(
    "datasets/redpajama_arxiv/redpajama-arxiv-raw-3gib.jsonl"
)


def valid_existing(output: Path, metadata_path: Path) -> bool:
    if not output.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selection") != "largest source-order prefix at most 3 GiB":
        return False
    if int(metadata.get("records", 0)) < MIN_RECORDS:
        return False
    if metadata.get("bytes") != output.stat().st_size:
        return False
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == metadata.get("sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    metadata_path = output.with_suffix(".metadata.json")
    if valid_existing(output, metadata_path):
        print(metadata_path.read_text(encoding="utf-8"), end="")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    source_list_response = requests.get(SOURCE_LIST_URL, timeout=30)
    source_list_response.raise_for_status()
    source_list_bytes = source_list_response.content
    actual_list_sha256 = hashlib.sha256(source_list_bytes).hexdigest()
    if actual_list_sha256 != SOURCE_LIST_SHA256:
        raise RuntimeError(
            "Unexpected RedPajama ArXiv URL list digest: "
            f"{actual_list_sha256} != {SOURCE_LIST_SHA256}"
        )
    source_urls = [
        line.strip()
        for line in source_list_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    consumed_sources = []
    reached_limit = False
    try:
        with partial.open("wb") as stream:
            for source_url in source_urls:
                response = requests.get(
                    source_url, stream=True, timeout=(30, 120)
                )
                response.raise_for_status()
                source_record_count = 0
                source_size = int(response.headers.get("content-length", -1))
                try:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        sample = json.loads(line)
                        text = sample.get("text")
                        if not isinstance(text, str) or not text:
                            continue
                        encoded = json.dumps(
                            {"text": text, "meta": sample.get("meta", {})},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8") + b"\n"
                        if total_bytes + len(encoded) > MAX_BYTES:
                            reached_limit = True
                            break
                        stream.write(encoded)
                        digest.update(encoded)
                        total_bytes += len(encoded)
                        count += 1
                        source_record_count += 1
                finally:
                    response.close()
                consumed_sources.append(
                    {
                        "url": source_url,
                        "declared_bytes": source_size,
                        "records_consumed": source_record_count,
                        "complete": not reached_limit,
                    }
                )
                if reached_limit:
                    break
            stream.flush()
            os.fsync(stream.fileno())
        if count < MIN_RECORDS:
            raise RuntimeError(
                f"Expected at least {MIN_RECORDS} records, downloaded {count}"
            )
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    metadata = {
        "dataset_id": "togethercomputer/RedPajama-Data-1T",
        "revision": REVISION,
        "source_list": SOURCE_LIST,
        "source_list_url": SOURCE_LIST_URL,
        "source_list_sha256": SOURCE_LIST_SHA256,
        "selection": "largest source-order prefix at most 3 GiB",
        "max_bytes": MAX_BYTES,
        "sources": consumed_sources,
        "records": count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
