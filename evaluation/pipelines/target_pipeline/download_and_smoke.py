"""Download real paired samples and run five target entrypoints offline.

Functional smoke only: no optimizer comparison or throughput claims.
Run as a module from the repository root; see --help.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "evaluation/datasets/target_pipeline"
ANNOTATION_URL = (
    "https://storage.googleapis.com/sfr-vision-language-research/"
    "datasets/coco_karpathy_train.json"
)
NAMES = ("simclr", "dino", "swav", "clip", "blip")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def download(url, path):
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url} -> {path}", flush=True)
    subprocess.run([
        "curl", "--fail", "--location", "--retry", "8",
        "--retry-delay", "3", "--connect-timeout", "30",
        "--max-time", "86400", "--continue-at", "-", "--silent", "--show-error",
        "--output", str(part), url,
    ], check=True)
    part.replace(path)


def verify_image(path):
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.convert("RGB").load()


def manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare_images(out, count):
    train = ROOT / "evaluation/datasets/imagenette2/imagenette2/train"
    if not train.is_dir():
        download("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
                 DATA / "imagenette2.tgz")
        # GNU tar extracts archive members under the intended dataset directory.
        target = train.parent.parent
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xzf", str(DATA / "imagenette2.tgz"),
                        "-C", str(target)], check=True)
    images = sorted(train.glob("*/*.JPEG"))
    if len(images) < count:
        raise RuntimeError(f"Insufficient Imagenette images: {len(images)}")
    validation = sorted((train.parent / "val").glob("*/*.JPEG"))
    if len(images) != 9469 or len(validation) != 3925:
        raise RuntimeError(
            f"Imagenette incomplete: train={len(images)}, val={len(validation)}; "
            "expected 9469 and 3925")
    selected = images[:count]
    for path in selected:
        verify_image(path)
    records = [{"id": path.stem, "image": str(path)} for path in selected]
    for name in ("dino", "swav"):
        manifest(out / f"{name}.jsonl", records)
    return {"dataset": "Imagenette2 train", "root": str(train),
            "available_images": len(images), "validation_images": len(validation),
            "complete_dataset": True, "validated_sample_count": count,
            "source": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            "workloads": ["simclr", "dino", "swav"],
            "scope": "DINO/SwAV functional input; not full ImageNet training reproduction"}


def prepare_pairs(out, count, full):
    annotation = DATA / "coco/annotations/coco_karpathy_train.json"
    download(ANNOTATION_URL, annotation)
    rows = json.loads(annotation.read_text())
    selected, seen = [], set()
    for row in rows:
        if row["image"] in seen:
            continue
        seen.add(row["image"])
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("Not enough unique annotated images")
    archive_metadata = {}
    if full:
        expected = {"train2014": (82783, 13510573713),
                    "val2014": (40504, 6645013297)}

        def fetch_split(split):
            archive = DATA / "coco" / (split + ".zip")
            url = f"https://s3.amazonaws.com/images.cocodataset.org/zips/{split}.zip"
            download(url, archive)
            expected_count, expected_bytes = expected[split]
            if archive.stat().st_size != expected_bytes:
                raise RuntimeError(f"Unexpected archive size: {archive}")
            # unzip verifies CRC during extraction; overwrite to validate all
            # members, including any previously downloaded sample images.
            subprocess.run(["unzip", "-oq", str(archive),
                            "-d", str(DATA / "coco/images")], check=True)
            actual_count = len(list((DATA / "coco/images" / split).glob("*.jpg")))
            if actual_count != expected_count:
                raise RuntimeError(f"Incomplete {split}: {actual_count}")
            return split, {"url": url, "bytes": expected_bytes,
                           "images": actual_count, "crc_verified_on_extraction": True}

        with ThreadPoolExecutor(max_workers=2) as pool:
            archive_metadata.update(pool.map(fetch_split, expected))
        official_annotations = DATA / "coco/annotations_trainval2014.zip"
        download("https://s3.amazonaws.com/images.cocodataset.org/"
                 "annotations/annotations_trainval2014.zip", official_annotations)
        subprocess.run(["unzip", "-oq", str(official_annotations),
                        "-d", str(DATA / "coco")], check=True)
        for split in ("train2014", "val2014"):
            captions = DATA / "coco/annotations" / f"captions_{split}.json"
            content = json.loads(captions.read_text())
            if len(content["images"]) != expected[split][0]:
                raise RuntimeError(f"Incomplete captions: {captions}")
        # Verify all records of the BLIP training annotation can resolve images.
        missing = {r["image"] for r in rows
                   if not (DATA / "coco/images" / r["image"]).is_file()}
        if missing:
            raise RuntimeError(f"Missing {len(missing)} BLIP caption images")

    def fetch(row):
        relative = Path(row["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image path: {relative}")
        path = DATA / "coco/images" / relative
        download("https://s3.amazonaws.com/images.cocodataset.org/" + relative.as_posix(), path)
        verify_image(path)
        if not isinstance(row["caption"], str) or not row["caption"].strip():
            raise ValueError("Missing real caption")
        return {"id": row.get("image_id", relative.stem),
                "image": str(path), "caption": row["caption"]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fetch, selected))
    for name in ("clip", "blip"):
        manifest(out / f"{name}.jsonl", records)
    from transformers import AutoTokenizer
    tokenizer = DATA / "clip_tokenizer"
    AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(tokenizer)
    return {"dataset": "COCO 2014 with BLIP Karpathy training captions",
            "annotation_url": ANNOTATION_URL,
            "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
            "annotation_records": len(rows), "selected_unique_images": count,
            "complete_dataset": full, "archives": archive_metadata,
            "image_root": str(DATA / "coco/images"), "tokenizer_path": str(tokenizer),
            "workloads": ["clip", "blip"],
            "scope": "full COCO train/val images" if full else "real paired sample download"}


def test_one(args):
    import torch
    from evaluation.cedar_utils import CedarEvalSpec
    torch.set_num_threads(8)
    name = args.test_one
    module = importlib.import_module(
        f"evaluation.pipelines.target_pipeline.{name}.cedar_dataset")
    kwargs = {"dataset_path": str(args.output / f"{name}.jsonl"),
              "tokenizer_path": str(DATA / "clip_tokenizer")}
    spec = CedarEvalSpec(
        batch_size=4, num_total_samples=args.samples, num_epochs=1, kwargs=kwargs,
        disable_optimizer=True, disable_controller=True, disable_prefetch=True,
        disable_caching=True, disable_offload=True, disable_parallelism=True)
    start = time.time()
    dataset = module.get_dataset(spec)
    processed = 0

    def tensor(value, shape):
        assert isinstance(value, torch.Tensor)
        assert tuple(value.shape) == shape, (value.shape, shape)
        assert torch.isfinite(value).all()

    try:
        for batch in itertools.islice(dataset, args.samples // 4):
            for record in batch:
                if name == "simclr":
                    tensor(record, (1, 244, 244))
                elif name in ("dino", "swav"):
                    sides = [224, 224] + [96] * (8 if name == "dino" else 6)
                    assert len(record["views"]) == len(sides)
                    for view, side in zip(record["views"], sides):
                        tensor(view, (3, side, side))
                else:
                    tensor(record["pixel_values"], (3, 224, 224))
                    if name == "clip":
                        tensor(record["input_ids"], (77,))
                        tensor(record["attention_mask"], (77,))
                    else:
                        assert isinstance(record["caption"], str)
                        assert 0 < len(record["caption"].split(" ")) <= 30
                processed += 1
    finally:
        dataset._exit()
    assert processed == args.samples, processed
    save(args.output / "results" / f"{name}.json",
         {"status": "passed", "workload": name, "samples": processed,
          "batch_size": 4, "elapsed_seconds": time.time() - start,
          "mode": "real-data functional smoke, optimizer disabled"})
    print(f"PASS {name}: {processed} real samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--full-coco", action="store_true")
    parser.add_argument("--test-one", choices=NAMES)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.samples <= 0 or args.samples % 4:
        parser.error("--samples must be a positive multiple of 4")
    if args.test_one:
        test_one(args)
        return
    os.chdir(ROOT)
    save(args.output / "status.json", {"status": "running", "pid": os.getpid()})
    errors = {}
    metadata = {"samples_per_workload": args.samples, "CPU_BUDGET": 64,
                "W": 8, "purpose": "functional smoke; not a formal optimizer matrix"}
    for label, action in (
        ("imagenette", lambda: prepare_images(args.output, args.samples)),
        ("coco", lambda: prepare_pairs(args.output, args.samples, args.full_coco)),
    ):
        try:
            metadata[label] = action()
        except Exception:
            errors[label] = traceback.format_exc()
            print(errors[label], flush=True)
        save(args.output / "metadata.json", metadata)
    if errors and args.full_coco:
        save(args.output / "status.json", {
            "status": "failed", "phase": "full_dataset_preparation",
            "errors": errors, "tests_started": False})
        sys.exit(1)
    save(args.output / "status.json", {
        "status": "running", "phase": "smoke_tests", "pid": os.getpid()})
    for name in NAMES:
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run([
                sys.executable, "-m",
                "evaluation.pipelines.target_pipeline.download_and_smoke",
                "--output", str(args.output), "--samples", str(args.samples),
                "--test-one", name], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            errors[name] = f"Test exited {result.returncode}; see {name}.log"
            save(args.output / "results" / f"{name}.json",
                 {"status": "failed", "error": errors[name]})
    with (args.output / "regression.log").open("w") as log:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "-q",
            "evaluation/pipelines/target_pipeline/tests"],
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        errors["regression"] = "See regression.log"
    summary = {"status": "failed" if errors else "passed", "errors": errors,
               "results": {n: json.loads((args.output / "results" / f"{n}.json").read_text())
                           for n in NAMES}}
    save(args.output / "status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
