#!/usr/bin/env python3
"""Download one frozen 100k-row Pile recipe source atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset

from evaluation.pipelines.pile_recipe_registry import RECIPES


EXPECTED_RECORDS = 100_000
MAX_BYTES = 3 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=sorted(RECIPES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    recipe = RECIPES[args.workload]
    output = args.output or recipe.dataset_path
    dataset = load_dataset(
        recipe.dataset_id,
        split="train",
        streaming=True,
        revision=recipe.revision,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    try:
        with partial.open("wb") as stream:
            for sample in dataset:
                text = sample.get("text")
                if not isinstance(text, str) or not text:
                    continue
                payload = {"text": text, "meta": sample.get("meta", {})}
                encoded = (
                    json.dumps(payload, ensure_ascii=False) + "\n"
                ).encode("utf-8")
                if total_bytes + len(encoded) > MAX_BYTES:
                    raise RuntimeError(
                        f"{args.workload} exceeds the fixed 3-GiB input cap"
                    )
                stream.write(encoded)
                digest.update(encoded)
                total_bytes += len(encoded)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())

        if count != EXPECTED_RECORDS:
            raise RuntimeError(
                f"Expected {EXPECTED_RECORDS} valid records, found {count}"
            )
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    metadata = {
        "workload": args.workload,
        "official_recipe": recipe.official_recipe,
        "data_juicer_hub_revision": "47fc345",
        "dataset_id": recipe.dataset_id,
        "revision": recipe.revision,
        "split": "train",
        "selection": "complete frozen 100000-row train split",
        "records": count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
