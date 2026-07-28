#!/usr/bin/env python3
"""Stream a bounded, unique RedPajama StackExchange subset to JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUTPUT = Path("datasets/stackexchange/redpajama-stackexchange-35000.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=35_000)
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".partial")
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        "stackexchange",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    with temp_path.open("w", encoding="utf-8") as stream:
        for sample in dataset:
            text = sample.get("text")
            if not isinstance(text, str) or not text:
                continue
            payload = {
                "text": text,
                "meta": sample.get("meta", {}),
                "red_pajama_subset": sample.get(
                    "red_pajama_subset", "stackexchange"
                ),
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.max_samples:
                break
        stream.flush()
        os.fsync(stream.fileno())

    if count != args.max_samples:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"stream ended after {count} samples; expected {args.max_samples}"
        )
    temp_path.replace(args.output)
    print(f"Wrote {count} unique source samples to {args.output}")


if __name__ == "__main__":
    main()
