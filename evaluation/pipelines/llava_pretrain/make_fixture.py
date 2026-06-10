"""Create a small local JSONL/image fixture for the LLaVA Cedar workload."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil


DEFAULT_SOURCE_IMAGE_DIR = pathlib.Path("tests/data/images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_jsonl",
        default="/tmp/llava_pretrain_cedar_fixture.jsonl",
    )
    parser.add_argument("--image_root", default="/tmp/llava_pretrain_cedar_images")
    parser.add_argument("--source_image_dir", default=str(DEFAULT_SOURCE_IMAGE_DIR))
    args = parser.parse_args()

    output_jsonl = pathlib.Path(args.output_jsonl)
    image_root = pathlib.Path(args.image_root)
    source_image_dir = pathlib.Path(args.source_image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    caption = (
        "<image> Two people sitting at a train table using laptop computers "
        "beside water bottles and a window."
    )
    image_paths = sorted(source_image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(source_image_dir)
    src = image_paths[0]
    dst = image_root / src.name
    shutil.copyfile(src, dst)

    rows = []
    for idx in range(1, 11):
        rows.append(
            {
                "id": idx,
                "text": caption,
                "images": [src.name],
            }
        )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} samples to {output_jsonl}")
    print(f"image_root={image_root}")


if __name__ == "__main__":
    main()
