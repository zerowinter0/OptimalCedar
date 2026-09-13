"""Controlled text-size and image-size sweeps for Figure 1."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    read_jsonl,
    sha256_file,
)
from evaluation.pipelines.simclrv2_multimodal.operator_scaling import (
    OPERATOR_NAMES,
    _caption_with_words,
    _materialize_images,
    _time_calls,
    _write_summary_csv,
)
from pico_multimodal.operators import ClipPredicate, TextNormalizer


TEXT_SWEEP_WORDS = (128, 256, 384, 512, 640, 768, 896, 1024)
IMAGE_SWEEP_SIDES = (128, 256, 384, 512, 640, 768, 896, 1024)
FIXED_IMAGE_SIDE = 224
FIXED_TEXT_WORDS = 128


@dataclass(frozen=True)
class ModalityScalingPoint:
    operator: str
    varied_dimension: str
    input_value: int
    text_words: int
    image_side: int


def build_modality_scaling_points(
    text_words: tuple[int, ...] = TEXT_SWEEP_WORDS,
    image_sides: tuple[int, ...] = IMAGE_SWEEP_SIDES,
) -> list[ModalityScalingPoint]:
    if not text_words or not image_sides:
        raise ValueError("Both modality grids must be non-empty")
    points: list[ModalityScalingPoint] = []
    for operator in OPERATOR_NAMES:
        points.extend(
            ModalityScalingPoint(
                operator=operator,
                varied_dimension="text",
                input_value=words,
                text_words=words,
                image_side=FIXED_IMAGE_SIDE,
            )
            for words in text_words
        )
        points.extend(
            ModalityScalingPoint(
                operator=operator,
                varied_dimension="image",
                input_value=side,
                text_words=FIXED_TEXT_WORDS,
                image_side=side,
            )
            for side in image_sides
        )
    return points


def summarize_modality_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    dimensions: dict[tuple[str, str, int], tuple[int, int]] = {}
    for row in rows:
        key = (
            str(row["operator"]),
            str(row["varied_dimension"]),
            int(row["input_value"]),
        )
        grouped[key].append(float(row["duration_ns_per_record"]))
        dimensions[key] = (int(row["text_words"]), int(row["image_side"]))

    dimension_order = {"text": 0, "image": 1}
    summaries: list[dict[str, Any]] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            dimension_order[item[1]],
            OPERATOR_NAMES.index(item[0]),
            item[2],
        ),
    ):
        durations_ms = np.asarray(grouped[key], dtype=float) / 1e6
        text_words, image_side = dimensions[key]
        summaries.append(
            {
                "operator": key[0],
                "varied_dimension": key[1],
                "input_value": key[2],
                "text_words": text_words,
                "image_side": image_side,
                "trials": len(durations_ms),
                "median_ms_per_record": float(np.median(durations_ms)),
                "q1_ms_per_record": float(
                    np.quantile(durations_ms, 0.25, method="linear")
                ),
                "q3_ms_per_record": float(
                    np.quantile(durations_ms, 0.75, method="linear")
                ),
            }
        )
    return summaries


def _prepare_inputs(
    point: ModalityScalingPoint,
    selected: list[dict[str, Any]],
    paths: dict[tuple[int, int], Path],
) -> list[Any]:
    if point.operator in {"normalize", "clip"}:
        prepared = []
        for record_idx, record in enumerate(selected):
            item = dict(record)
            item["caption"] = _caption_with_words(
                str(record.get("caption", "")), point.text_words
            )
            item["image_path"] = str(paths[(record_idx, point.image_side)])
            prepared.append(item)
        return prepared
    return [
        read_image(
            str(paths[(record_idx, point.image_side)]),
            mode=ImageReadMode.RGB,
        )
        for record_idx in range(len(selected))
    ]


def run_modality_benchmark(
    fixture: Path,
    image_root: Path,
    output_root: Path,
    records: int = 128,
    repetitions: int = 7,
    text_words: tuple[int, ...] = TEXT_SWEEP_WORDS,
    image_sides: tuple[int, ...] = IMAGE_SWEEP_SIDES,
) -> dict[str, Any]:
    all_records = read_jsonl(fixture)
    if records < 1 or records > len(all_records):
        raise ValueError("records is outside the fixture size")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    selected = all_records[:records]
    points = build_modality_scaling_points(text_words, image_sides)
    output_root.mkdir(parents=True, exist_ok=True)
    all_sides = tuple(sorted(set(image_sides) | {FIXED_IMAGE_SIDE}))
    paths = _materialize_images(
        selected,
        image_root,
        output_root / "generated_inputs",
        all_sides,
    )

    normalize = TextNormalizer()
    clip = ClipPredicate(float("-inf"), image_root=None)
    functions = {
        "normalize": normalize,
        "clip": clip.score,
        "random_crop": transforms.RandomResizedCrop(224),
        "random_flip": transforms.RandomHorizontalFlip(),
        "color_jitter": transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        "gaussian_blur": transforms.GaussianBlur(kernel_size=23),
    }
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    warmup_by_operator = {
        operator: next(point for point in points if point.operator == operator)
        for operator in OPERATOR_NAMES
    }
    for operator, point in warmup_by_operator.items():
        functions[operator](_prepare_inputs(point, selected[:1], paths)[0])
        if operator == "clip":
            torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for trial in range(repetitions):
        torch.manual_seed(20260909 + trial)
        trial_points = list(points)
        random.Random(20260909 + trial).shuffle(trial_points)
        for point_idx, point in enumerate(trial_points, start=1):
            inputs = _prepare_inputs(point, selected, paths)
            duration = _time_calls(
                functions[point.operator],
                inputs,
                cuda=point.operator == "clip",
            )
            row = asdict(point)
            row.update(
                {
                    "trial": trial,
                    "duration_ns_per_record": duration,
                }
            )
            rows.append(row)
            print(
                f"trial={trial + 1}/{repetitions} "
                f"point={point_idx}/{len(points)} operator={point.operator} "
                f"dimension={point.varied_dimension} value={point.input_value} "
                f"ns_per_record={duration:.1f}",
                flush=True,
            )
            del inputs

    raw_path = output_root / "raw.json"
    formal_protocol = (
        records == 128
        and repetitions == 7
        and text_words == TEXT_SWEEP_WORDS
        and image_sides == IMAGE_SWEEP_SIDES
    )
    atomic_write_json(
        raw_path,
        {
            "schema_version": 1,
            "status": "success",
            "formal_protocol": formal_protocol,
            "records": records,
            "repetitions": repetitions,
            "text_words": text_words,
            "image_sides": image_sides,
            "fixed_text_words": FIXED_TEXT_WORDS,
            "fixed_image_side": FIXED_IMAGE_SIDE,
            "metadata": command_metadata(),
            "rows": rows,
        },
    )
    summaries = summarize_modality_rows(rows)
    summary_path = output_root / "summary.csv"
    _write_summary_csv(summary_path, summaries)
    result = {
        "status": "success",
        "formal_protocol": formal_protocol,
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
    parser.add_argument("--text-words", type=_int_tuple, default=TEXT_SWEEP_WORDS)
    parser.add_argument("--image-sides", type=_int_tuple, default=IMAGE_SWEEP_SIDES)
    args = parser.parse_args()
    result = run_modality_benchmark(
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
