"""Measure input-size sensitivity for the six Figure 1 operators."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    read_jsonl,
    sha256_file,
)
from pico_multimodal.operators import ClipPredicate, TextNormalizer


OPERATOR_NAMES = (
    "normalize",
    "clip",
    "random_crop",
    "random_flip",
    "color_jitter",
    "gaussian_blur",
)
WORK_SCALES = (0.25, 1.0, 4.0, 16.0)
DEFAULT_TEXT_WORDS = (16, 64, 256, 1024)
DEFAULT_IMAGE_SIDES = (112, 224, 448, 896)


@dataclass(frozen=True)
class ScalingPoint:
    operator: str
    work_scale: float
    text_words: int | None
    image_side: int | None


def build_scaling_points(
    text_words: tuple[int, ...] = DEFAULT_TEXT_WORDS,
    image_sides: tuple[int, ...] = DEFAULT_IMAGE_SIDES,
) -> list[ScalingPoint]:
    if len(text_words) != len(WORK_SCALES):
        raise ValueError("The text grid must match WORK_SCALES")
    if len(image_sides) != len(WORK_SCALES):
        raise ValueError("The image grid must match WORK_SCALES")
    points: list[ScalingPoint] = []
    for operator in OPERATOR_NAMES:
        for scale, words, side in zip(WORK_SCALES, text_words, image_sides):
            if operator == "normalize":
                points.append(ScalingPoint(operator, scale, words, None))
            elif operator == "clip":
                points.append(ScalingPoint(operator, scale, words, side))
            else:
                points.append(ScalingPoint(operator, scale, None, side))
    return points


def summarize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    dimensions: dict[tuple[Any, ...], tuple[int | None, int | None]] = {}
    for row in rows:
        key = (row["operator"], float(row["work_scale"]))
        grouped[key].append(float(row["duration_ns_per_record"]))
        dimensions[key] = (row["text_words"], row["image_side"])
    summaries = []
    for key in sorted(grouped, key=lambda item: (OPERATOR_NAMES.index(item[0]), item[1])):
        durations = np.asarray(grouped[key], dtype=float)
        rates = 1e9 / durations
        text_words, image_side = dimensions[key]
        summaries.append(
            {
                "operator": key[0],
                "work_scale": key[1],
                "text_words": text_words,
                "image_side": image_side,
                "trials": len(durations),
                "median_ns_per_record": float(np.median(durations)),
                "median_records_per_sec": float(np.median(rates)),
                "q1_records_per_sec": float(
                    np.quantile(rates, 0.25, method="linear")
                ),
                "q3_records_per_sec": float(
                    np.quantile(rates, 0.75, method="linear")
                ),
            }
        )
    return summaries


def _caption_with_words(caption: str, target: int) -> str:
    words = caption.strip().split() or ["image"]
    repeated = (words * ((target + len(words) - 1) // len(words)))[:target]
    return " ".join(repeated)


def _materialize_images(
    records: list[dict[str, Any]],
    image_root: Path,
    output_root: Path,
    sides: tuple[int, ...],
) -> dict[tuple[int, int], Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[int, int], Path] = {}
    for record_idx, record in enumerate(records):
        source = image_root / str(record["image_path"])
        for side in sides:
            destination = output_root / f"record_{record_idx:04d}_{side}.jpg"
            if not destination.exists():
                with Image.open(source) as image:
                    rgb = image.convert("RGB")
                    contained = ImageOps.fit(
                        rgb,
                        (side, side),
                        method=Image.Resampling.LANCZOS,
                    )
                    contained.save(destination, format="JPEG", quality=92)
            paths[(record_idx, side)] = destination
    return paths


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _time_calls(function: Callable[[Any], Any], inputs: list[Any], cuda: bool) -> float:
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    for item in inputs:
        function(item)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / len(inputs)


def run_benchmark(
    fixture: Path,
    image_root: Path,
    output_root: Path,
    records: int = 128,
    repetitions: int = 7,
    text_words: tuple[int, ...] = DEFAULT_TEXT_WORDS,
    image_sides: tuple[int, ...] = DEFAULT_IMAGE_SIDES,
) -> dict[str, Any]:
    all_records = read_jsonl(fixture)
    if records < 1 or records > len(all_records):
        raise ValueError("records is outside the fixture size")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    selected = all_records[:records]
    points = build_scaling_points(text_words, image_sides)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _materialize_images(
        selected,
        image_root,
        output_root / "generated_inputs",
        image_sides,
    )

    # Inputs are materialized before timing so CPU image operators measure
    # transformation service rather than storage or decoding. CLIP receives a
    # path because image loading is part of the actual predicate implementation.
    tensors = {
        side: [
            read_image(str(paths[(record_idx, side)]), mode=ImageReadMode.RGB)
            for record_idx in range(records)
        ]
        for side in image_sides
    }
    normalize = TextNormalizer()
    clip = ClipPredicate(float("-inf"), image_root=None)
    functions: dict[str, Callable[[Any], Any]] = {
        "normalize": normalize,
        "clip": clip.score,
        "random_crop": transforms.RandomResizedCrop(224),
        "random_flip": transforms.RandomHorizontalFlip(),
        "color_jitter": transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        "gaussian_blur": transforms.GaussianBlur(kernel_size=23),
    }
    prepared: dict[ScalingPoint, list[Any]] = {}
    for point in points:
        if point.operator in {"normalize", "clip"}:
            assert point.text_words is not None
            point_records = []
            for record_idx, record in enumerate(selected):
                item = dict(record)
                item["caption"] = _caption_with_words(
                    str(record.get("caption", "")), point.text_words
                )
                if point.image_side is not None:
                    item["image_path"] = str(paths[(record_idx, point.image_side)])
                point_records.append(item)
            prepared[point] = point_records
        else:
            assert point.image_side is not None
            prepared[point] = tensors[point.image_side]

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    for operator in OPERATOR_NAMES:
        base = next(
            point
            for point in points
            if point.operator == operator and point.work_scale == 1.0
        )
        functions[operator](prepared[base][0])
        if operator == "clip":
            torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for trial in range(repetitions):
        torch.manual_seed(20260909 + trial)
        trial_points = list(points)
        random.Random(20260909 + trial).shuffle(trial_points)
        for point_idx, point in enumerate(trial_points, start=1):
            duration = _time_calls(
                functions[point.operator],
                prepared[point],
                cuda=point.operator == "clip",
            )
            row = asdict(point)
            row.update(
                {
                    "trial": trial,
                    "duration_ns_per_record": duration,
                    "records_per_sec": 1e9 / duration,
                }
            )
            rows.append(row)
            print(
                f"trial={trial + 1}/{repetitions} "
                f"point={point_idx}/{len(points)} operator={point.operator} "
                f"scale={point.work_scale:g} ns_per_record={duration:.1f}",
                flush=True,
            )

    raw_path = output_root / "raw.json"
    atomic_write_json(
        raw_path,
        {
            "schema_version": 1,
            "status": "success",
            "formal_protocol": (
                records == 128
                and repetitions == 7
                and text_words == DEFAULT_TEXT_WORDS
                and image_sides == DEFAULT_IMAGE_SIDES
            ),
            "records": records,
            "repetitions": repetitions,
            "text_words": text_words,
            "image_sides": image_sides,
            "metadata": command_metadata(),
            "rows": rows,
        },
    )
    summaries = summarize_rows(rows)
    summary_path = output_root / "summary.csv"
    _write_summary_csv(summary_path, summaries)
    result = {
        "status": "success",
        "formal_protocol": records == 128 and repetitions == 7,
        "operators": list(OPERATOR_NAMES),
        "points": len(points),
        "raw_rows": len(rows),
        "raw_sha256": sha256_file(raw_path),
        "summary_sha256": sha256_file(summary_path),
    }
    atomic_write_json(output_root / "result.json", result)
    return result


def _int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--text-words", type=_int_tuple, default=DEFAULT_TEXT_WORDS)
    parser.add_argument("--image-sides", type=_int_tuple, default=DEFAULT_IMAGE_SIDES)
    args = parser.parse_args()
    result = run_benchmark(
        fixture=args.fixture,
        image_root=args.image_root,
        output_root=args.output_root,
        records=args.records,
        repetitions=args.repetitions,
        text_words=args.text_words,
        image_sides=args.image_sides,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
