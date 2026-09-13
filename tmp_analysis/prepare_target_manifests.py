"""Build identical-size manifests for the five target_pipeline workloads.

simclr reads the Imagenette tree directly; the other four consume a JSONL
manifest. clip/blip need captions (COCO Karpathy annotations), dino/swav need
only an image path.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "evaluation/datasets/target_pipeline/coco"
ANNOTATION = COCO / "annotations/coco_karpathy_train.json"
IMAGENETTE_TRAIN = (
    ROOT / "evaluation/datasets/imagenette2/imagenette2/train"
)


def coco_pairs(limit):
    rows = json.loads(ANNOTATION.read_text())
    seen, selected = set(), []
    for row in rows:
        image = row["image"]
        if image in seen:
            continue
        seen.add(image)
        path = COCO / "images" / image
        if not path.is_file():
            continue
        caption = row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            continue
        selected.append(
            {"id": str(row.get("image_id", Path(image).stem)),
             "image": str(path), "caption": caption}
        )
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(f"only {len(selected)} usable COCO pairs for {limit}")
    return selected


def imagenette_images(limit):
    paths = sorted(IMAGENETTE_TRAIN.rglob("*.JPEG"))
    if len(paths) < limit:
        raise RuntimeError(f"only {len(paths)} Imagenette images for {limit}")
    return [
        {"id": path.stem, "image": str(path)}
        for path in paths[:limit]
    ]


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"{path}: {len(records)} records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    pairs = coco_pairs(args.samples)
    write(args.output / "clip.jsonl", pairs)
    write(args.output / "blip.jsonl", pairs)
    images = imagenette_images(args.samples)
    write(args.output / "dino.jsonl", images)
    write(args.output / "swav.jsonl", images)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "samples": args.samples,
                "clip_blip_source": str(ANNOTATION),
                "dino_swav_source": str(IMAGENETTE_TRAIN),
                "simclr_source": str(IMAGENETTE_TRAIN),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
