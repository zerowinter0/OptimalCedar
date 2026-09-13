"""Score the held-out calibration split and materialize the declared grid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    read_jsonl,
    sha256_file,
)
from evaluation.motivation_multimodal.pilot import iter_threshold_configs
from evaluation.pipelines.multimodal_running_example.operators import (
    MODEL_NAMES,
    MODEL_REVISIONS,
    AestheticPredicate,
    BlipPredicate,
    ClipPredicate,
    PerplexityPredicate,
    SafetyPredicate,
    TextNormalizer,
)


def _quantile(rows: Iterable[Mapping[str, Any]], key: str, q: float) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Calibration scores for {key} are empty or non-finite")
    return float(np.quantile(values, q, method="linear"))


def derive_threshold_payloads(
    score_rows: list[Mapping[str, Any]],
    configs: Iterable[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert retained fractions into thresholds with explicit directions."""

    payloads: dict[str, dict[str, Any]] = {}
    if configs is None:
        configs = iter_threshold_configs()
    for config in configs:
        payloads[config.key] = {
            "key": config.key,
            "retention_percent": {
                "perplexity": config.p_retention,
                "safety": config.q_retention,
                "aesthetic": config.a_retention,
                "clip": config.c_retention,
                "blip": config.b_retention,
            },
            "thresholds": {
                "perplexity_max": _quantile(
                    score_rows, "perplexity", config.p_retention / 100.0
                ),
                "safety_max": _quantile(
                    score_rows, "safety", config.q_retention / 100.0
                ),
                "aesthetic_min": _quantile(
                    score_rows, "aesthetic", 1.0 - config.a_retention / 100.0
                ),
                "clip_min": _quantile(
                    score_rows, "clip", 1.0 - config.c_retention / 100.0
                ),
                "blip_min": _quantile(
                    score_rows, "blip", 1.0 - config.b_retention / 100.0
                ),
            },
            "quantile_method": "numpy.linear",
        }
    return payloads


def score_calibration_records(
    records: list[Mapping[str, Any]],
    image_root: str | Path,
) -> list[dict[str, Any]]:
    normalizer = TextNormalizer()
    perplexity = PerplexityPredicate(float("inf"))
    safety = SafetyPredicate(float("inf"), image_root)
    aesthetic = AestheticPredicate(float("-inf"), image_root)
    clip = ClipPredicate(float("-inf"), image_root)
    blip = BlipPredicate(float("-inf"), image_root)
    rows = []
    started = time.monotonic()
    for index, record in enumerate(records, start=1):
        normalized = normalizer(record)
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "perplexity": perplexity.score(normalized),
                "safety": safety.score(record),
                "aesthetic": aesthetic.score(record),
                "clip": clip.score(normalized),
                "blip": blip.score(normalized),
            }
        )
        if index % 10 == 0 or index == len(records):
            elapsed = time.monotonic() - started
            print(
                f"calibration progress={index}/{len(records)} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )
    return rows


def run_calibration(
    fixture_root: Path,
    image_root: Path,
    output_root: Path,
    configs: Iterable[Any] | None = None,
) -> dict[str, Any]:
    calibration_path = fixture_root / "calibration.jsonl"
    manifest_path = fixture_root / "manifest.json"
    records = read_jsonl(calibration_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(records) != 500 or manifest.get("splits", {}).get("calibration") != 500:
        raise ValueError("Formal calibration requires exactly 500 fixture records")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = score_calibration_records(records, image_root)
    scores_payload = {
        "schema_version": 1,
        "split": "calibration",
        "records": len(rows),
        "fixture_output_sha256": manifest["output_sha256"],
        "calibration_jsonl_sha256": sha256_file(calibration_path),
        "model_names": MODEL_NAMES,
        "model_revisions": MODEL_REVISIONS,
        "metadata": command_metadata(),
        "scores": rows,
    }
    scores_path = output_root / "calibration_scores.json"
    atomic_write_json(scores_path, scores_payload)

    thresholds = derive_threshold_payloads(rows, configs=configs)
    threshold_root = output_root / "thresholds"
    for key, payload in thresholds.items():
        artifact = {
            **payload,
            "schema_version": 1,
            "calibration_scores_sha256": sha256_file(scores_path),
            "fixture_output_sha256": manifest["output_sha256"],
            "model_revisions": MODEL_REVISIONS,
        }
        atomic_write_json(threshold_root / f"{key}.json", artifact)
    summary = {
        "status": "success",
        "records": len(rows),
        "configurations": len(thresholds),
        "scores_path": str(scores_path),
        "scores_sha256": sha256_file(scores_path),
        "threshold_root": str(threshold_root),
    }
    atomic_write_json(output_root / "calibration_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_calibration(args.fixture_root, args.image_root, args.output_root),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
