"""Controlled operator input-sensitivity benchmark for Figure 5."""

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
from transformers import BlipProcessor, CLIPProcessor

from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    read_jsonl,
    sha256_file,
)
from evaluation.pipelines.multimodal_running_example import operators as ops
from evaluation.pipelines.redpajama_c4 import dj_operators as text_ops


DEFAULT_TEXT_GRID = (16, 32, 64, 128, 256, 512)
DEFAULT_IMAGE_GRID = (128, 256, 512, 1024)


def text_grid() -> tuple[int, ...]:
    return DEFAULT_TEXT_GRID


def image_side_grid() -> tuple[int, ...]:
    return DEFAULT_IMAGE_GRID


def multimodal_grid() -> tuple[tuple[int, int], ...]:
    return tuple(
        (tokens, side)
        for tokens in (16, 32, 64, 128)
        for side in DEFAULT_IMAGE_GRID
    )


@dataclass(frozen=True)
class ScalingRow:
    operator: str
    text_tokens: int | None
    image_side: int | None
    trial: int
    duration_ns_per_record: float


@dataclass(frozen=True)
class ScalingSummary:
    operator: str
    text_tokens: int | None
    image_side: int | None
    trials: int
    median_ns: float
    q1_ns: float
    q3_ns: float


def summarize_trials(rows: Iterable[ScalingRow]) -> list[ScalingSummary]:
    grouped: dict[tuple[str, int | None, int | None], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.operator, row.text_tokens, row.image_side)].append(
            float(row.duration_ns_per_record)
        )
    summaries = []
    for (operator, tokens, side), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1] if item[0][1] is not None else -1,
            item[0][2] if item[0][2] is not None else -1,
        ),
    ):
        summaries.append(
            ScalingSummary(
                operator=operator,
                text_tokens=tokens,
                image_side=side,
                trials=len(values),
                median_ns=float(np.median(values)),
                q1_ns=float(np.quantile(values, 0.25, method="linear")),
                q3_ns=float(np.quantile(values, 0.75, method="linear")),
            )
        )
    return summaries


def _repeat_and_trim_sentencepiece(caption: str, target: int) -> str:
    tokenizer = text_ops._get_sentencepiece_model("en")
    clause = caption.strip() or "A photograph with visible objects."
    repeated = clause
    while len(tokenizer.encode_as_pieces(repeated)) < target:
        repeated = f"{repeated} {clause}"
    return tokenizer.decode_pieces(tokenizer.encode_as_pieces(repeated)[:target])


def _repeat_and_trim_hf(caption: str, target: int, tokenizer: Any) -> str:
    clause = caption.strip() or "A photograph with visible objects."
    repeated = clause
    while len(tokenizer.encode(repeated, add_special_tokens=False)) < target:
        repeated = f"{repeated} {clause}"
    ids = tokenizer.encode(repeated, add_special_tokens=False)[:target]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _materialize_images(
    records: list[dict[str, Any]],
    image_root: Path,
    output_root: Path,
    sides: tuple[int, ...],
) -> dict[tuple[str, int], Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    materialized: dict[tuple[str, int], Path] = {}
    for record in records:
        image_id = str(record["image_id"])
        source = image_root / str(record["image_path"])
        for side in sides:
            key = (image_id, side)
            if key in materialized:
                continue
            destination = output_root / f"{image_id}_{side}.png"
            if not destination.exists():
                with Image.open(source) as image:
                    rgb = image.convert("RGB")
                    contained = ImageOps.contain(
                        rgb,
                        (side, side),
                        method=Image.Resampling.LANCZOS,
                    )
                    square = Image.new("RGB", (side, side), (0, 0, 0))
                    offset = (
                        (side - contained.width) // 2,
                        (side - contained.height) // 2,
                    )
                    square.paste(contained, offset)
                    square.save(destination, format="PNG", optimize=False)
            materialized[key] = destination
    return materialized


def _records_for_point(
    records: list[dict[str, Any]],
    operator: str,
    tokens: int | None,
    side: int | None,
    images: dict[tuple[str, int], Path],
    clip_processor: CLIPProcessor,
    blip_processor: BlipProcessor,
) -> list[dict[str, Any]]:
    output = []
    for original in records:
        record = dict(original)
        if side is not None:
            record["image_path"] = str(images[(str(record["image_id"]), side)])
        if tokens is not None:
            if operator in {"normalize", "perplexity"}:
                caption = _repeat_and_trim_sentencepiece(record["caption"], tokens)
            elif operator == "clip":
                caption = _repeat_and_trim_hf(
                    record["caption"], tokens, clip_processor.tokenizer
                )
            elif operator == "blip":
                caption = _repeat_and_trim_hf(
                    record["caption"], tokens, blip_processor.tokenizer
                )
            else:
                raise ValueError(f"Unexpected text-sensitive operator {operator}")
            record["caption"] = caption
        if operator in {"perplexity", "clip", "blip"}:
            record = ops.TextNormalizer()(record)
        output.append(record)
    return output


def _points(
    text_values: tuple[int, ...],
    image_values: tuple[int, ...],
) -> list[tuple[str, int | None, int | None]]:
    points = []
    for operator in ("normalize", "perplexity"):
        points.extend((operator, tokens, None) for tokens in text_values)
    for operator in ("safety", "aesthetic"):
        points.extend((operator, None, side) for side in image_values)
    mm_tokens = tuple(value for value in text_values if value <= 128)
    for operator in ("clip", "blip"):
        points.extend(
            (operator, tokens, side)
            for tokens in mm_tokens
            for side in image_values
        )
    return points


def _operator_functions() -> tuple[
    dict[str, Callable[[dict[str, Any]], Any]], set[str]
]:
    functions = {
        "normalize": ops.TextNormalizer(),
        "perplexity": ops.PerplexityPredicate(float("inf")).score,
        "safety": ops.SafetyPredicate(float("inf")).score,
        "aesthetic": ops.AestheticPredicate(float("-inf")).score,
        "clip": ops.ClipPredicate(float("-inf")).score,
        "blip": ops.BlipPredicate(float("-inf")).score,
    }
    return functions, {"aesthetic", "clip", "blip"}


def _time_point(
    function: Callable[[dict[str, Any]], Any],
    records: list[dict[str, Any]],
    cuda: bool,
) -> float:
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    for record in records:
        function(record)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter_ns() - started
    return elapsed / len(records)


def _write_summary_csv(path: Path, summaries: list[ScalingSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def run_scaling(
    fixture_root: Path,
    image_root: Path,
    output_root: Path,
    record_count: int = 128,
    repetitions: int = 7,
    text_values: tuple[int, ...] = DEFAULT_TEXT_GRID,
    image_values: tuple[int, ...] = DEFAULT_IMAGE_GRID,
) -> dict[str, Any]:
    all_records = read_jsonl(fixture_root / "scaling.jsonl")
    if record_count < 1 or record_count > len(all_records):
        raise ValueError("record_count is outside the scaling split")
    records = all_records[:record_count]
    output_root.mkdir(parents=True, exist_ok=True)
    images = _materialize_images(
        records,
        image_root,
        output_root / "generated_inputs" / "images",
        image_values,
    )
    clip_processor = CLIPProcessor.from_pretrained(
        ops._snapshot_path("clip"), local_files_only=True
    )
    blip_processor = BlipProcessor.from_pretrained(
        ops._snapshot_path("blip"), local_files_only=True
    )
    points = _points(text_values, image_values)
    prepared = {
        point: _records_for_point(
            records,
            *point,
            images,
            clip_processor,
            blip_processor,
        )
        for point in points
    }
    functions, cuda_operators = _operator_functions()

    # Model initialization and one complete operator call are excluded from
    # every reported interval.
    warmed = set()
    for operator, tokens, side in points:
        if operator in warmed:
            continue
        functions[operator](prepared[(operator, tokens, side)][0])
        if operator in cuda_operators:
            torch.cuda.synchronize()
        warmed.add(operator)

    rows: list[ScalingRow] = []
    for trial in range(repetitions):
        trial_points = list(points)
        random.Random(20260907 + trial).shuffle(trial_points)
        for index, (operator, tokens, side) in enumerate(trial_points, start=1):
            duration = _time_point(
                functions[operator],
                prepared[(operator, tokens, side)],
                operator in cuda_operators,
            )
            rows.append(ScalingRow(operator, tokens, side, trial, duration))
            print(
                f"scaling trial={trial + 1}/{repetitions} "
                f"point={index}/{len(points)} operator={operator} "
                f"tokens={tokens} side={side} ns_per_record={duration:.1f}",
                flush=True,
            )

    raw_payload = {
        "schema_version": 1,
        "status": "success",
        "formal_protocol": (
            record_count == 128
            and repetitions == 7
            and text_values == DEFAULT_TEXT_GRID
            and image_values == DEFAULT_IMAGE_GRID
        ),
        "records": record_count,
        "repetitions": repetitions,
        "text_grid": text_values,
        "image_grid": image_values,
        "metadata": command_metadata(),
        "rows": [asdict(row) for row in rows],
    }
    raw_path = output_root / "raw.json"
    atomic_write_json(raw_path, raw_payload)
    summaries = summarize_trials(rows)
    summary_path = output_root / "summary.csv"
    _write_summary_csv(summary_path, summaries)
    result = {
        "status": "success",
        "formal_protocol": raw_payload["formal_protocol"],
        "points": len(points),
        "raw_rows": len(rows),
        "raw_sha256": sha256_file(raw_path),
        "summary_sha256": sha256_file(summary_path),
    }
    atomic_write_json(output_root / "result.json", result)
    return result


def _parse_grid(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("grid values must be positive integers")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--text-grid", type=_parse_grid, default=DEFAULT_TEXT_GRID)
    parser.add_argument("--image-grid", type=_parse_grid, default=DEFAULT_IMAGE_GRID)
    args = parser.parse_args()
    result = run_scaling(
        args.fixture_root,
        args.image_root,
        args.output_root,
        args.records,
        args.repetitions,
        args.text_grid,
        args.image_grid,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
