#!/usr/bin/env python3
"""Download and freeze the English Alpaca-CoT source used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REVISION = "18add89e3b884703ec869a5c6e2bcf1412ee7edc"
SOURCE_FILE = "Chain-of-Thought/CoT_data.json"
SOURCE_URL = (
    "https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/resolve/"
    f"{REVISION}/{SOURCE_FILE}"
)
DEFAULT_OUTPUT = Path(
    "datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output: Path = args.output
    metadata_path = output.with_suffix(".metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(SOURCE_URL, timeout=120) as response:
        records = json.load(response)
    if not isinstance(records, list) or len(records) < 20_000:
        raise RuntimeError(
            f"Unexpected Alpaca-CoT source size: {len(records)} records"
        )

    digest = hashlib.sha256()
    with output.open("wb") as stream:
        for record in records:
            row = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            encoded = (row + "\n").encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)

    metadata = {
        "dataset_id": "QingyiSi/Alpaca-CoT",
        "revision": REVISION,
        "source_file": SOURCE_FILE,
        "source_url": SOURCE_URL,
        "records": len(records),
        "jsonl_bytes": output.stat().st_size,
        "jsonl_sha256": digest.hexdigest(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
