#!/usr/bin/env python3
"""Convert a bounded LLaVA source subset to Data-Juicer's caption-only format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_INPUT = Path(
    "evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl"
)
DEFAULT_OUTPUT = Path(
    "evaluation/datasets/llava_pretrain/"
    "blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl"
)


def convert_row(
    row: Dict[str, Any],
    image_root: Path | None = None,
) -> Dict[str, Any]:
    image = row.get("image")
    conversations = row.get("conversations")
    if not isinstance(image, str) or not image:
        raise ValueError("row has no image path")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("row has no assistant caption")
    caption = conversations[1].get("value")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("assistant caption is empty")
    image_path = Path(image)
    if image_root is not None and not image_path.is_absolute():
        image = str((image_root / image_path).resolve())
    return {
        "id": row.get("id"),
        "images": [image],
        "text": f"<image>\n{caption.strip()} <|__dj__eoc|>",
    }


def convert_subset(
    input_path: Path,
    output_path: Path,
    max_samples: int,
    image_root: Path | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    count = 0
    try:
        with input_path.open("r", encoding="utf-8") as source, partial.open(
            "w", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    converted = convert_row(
                        json.loads(line),
                        image_root=image_root,
                    )
                except (json.JSONDecodeError, ValueError, AttributeError) as exc:
                    raise ValueError(f"invalid source row {line_number}: {exc}") from exc
                target.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
                if count == max_samples:
                    break
            target.flush()
            os.fsync(target.fileno())
        if count != max_samples:
            raise RuntimeError(
                f"source ended after {count} rows; expected {max_samples}"
            )
        partial.replace(output_path)
        return count
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Resolve relative image paths against this directory. Use this "
            "for the shared Cedar/Data-Juicer/native-framework input file."
        ),
    )
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    count = convert_subset(
        args.input,
        args.output,
        args.max_samples,
        image_root=args.image_root,
    )
    print(f"Wrote {count} caption-only samples to {args.output}")


if __name__ == "__main__":
    main()
