"""Build the small JSONL manifest consumed by ``target_pipeline``.

The target workloads deliberately keep raw media on disk. A manifest contains
only an id, an image path, and an optional caption; this avoids copying a
multi-gigabyte image archive into an experiment output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _caption_from_conversations(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("LLaVA row must contain a user and assistant turn")
    assistant = value[1]
    if not isinstance(assistant, dict) or not isinstance(assistant.get("value"), str):
        raise ValueError("LLaVA assistant turn has no text value")
    return assistant["value"].strip()


def convert_llava(source: Path, destination: Path, image_root: Path | None = None) -> int:
    """Convert LLaVA conversations into the generic target manifest.

    Existing output is never replaced accidentally. Relative image paths are
    resolved against ``image_root`` when supplied; the image itself is not
    copied or decoded here.
    """

    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as stream, destination.open(
        "x", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                image = row["image"]
                if not isinstance(image, str) or not image:
                    raise ValueError("missing image")
                path = Path(image)
                if image_root is not None and not path.is_absolute():
                    path = (image_root / path).resolve()
                record = {
                    "id": row.get("id", str(path)),
                    "image": str(path),
                    "caption": _caption_from_conversations(row.get("conversations")),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid LLaVA row {line_number}: {exc}") from exc
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_records(records: Iterable[dict[str, Any]], destination: Path) -> int:
    """Write already-normalized records without silently overwriting output."""

    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("x", encoding="utf-8") as output:
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("image"), str):
                raise ValueError("each manifest record needs an image path")
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    args = parser.parse_args()
    print(convert_llava(args.source, args.destination, args.image_root))


if __name__ == "__main__":
    main()
