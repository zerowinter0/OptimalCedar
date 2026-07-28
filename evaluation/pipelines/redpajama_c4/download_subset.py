#!/usr/bin/env python3
"""Stream a bounded raw RedPajama C4 subset to JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUTPUT = Path("datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=829_916)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=2 * 1024**3,
        help="Hard cap for serialized JSONL bytes (default: 2 GiB).",
    )
    args = parser.parse_args()
    if args.max_samples <= 0 or args.max_bytes <= 0:
        parser.error("--max-samples and --max-bytes must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        "c4",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

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
                "red_pajama_subset": sample.get("red_pajama_subset", "c4"),
            }
            encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
            if count and total_bytes + len(encoded) > args.max_bytes:
                break
            stream.write(encoded)
            total_bytes += len(encoded)
            count += 1
            if count >= args.max_samples:
                break
        stream.flush()
        os.fsync(stream.fileno())

    if count == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("stream did not yield any valid C4 samples")
    partial.replace(args.output)
    print(
        f"Wrote {count} raw C4 samples ({total_bytes} bytes) to {args.output}"
    )


if __name__ == "__main__":
    main()
