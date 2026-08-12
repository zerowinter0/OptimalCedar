#!/usr/bin/env python3
"""Inventory and classify recipes in a pinned Data-Juicer Hub checkout."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


MODEL_MARKERS = (
    "language_id",
    "perplexity",
    "image_text",
    "video_nsfw",
    "video_frames_text_similarity",
    "video_aesthetics",
    "video_watermark",
    "sentence_augmentation",
    "sdxl_",
    "mllm_",
)


def recipe_family(relative_path: Path) -> str:
    parts = relative_path.parts
    if "alpaca_cot" in parts:
        return "instruction_tuning_text"
    if "github_code" in parts:
        return "source_code"
    if "image" in parts:
        return "image_text"
    if "video" in parts:
        return "video_text"
    if "pretrain" in parts:
        return "pretraining_text"
    if "reproduced_bloom" in parts:
        return "multilingual_reproduction"
    if "reproduced_redpajama" in parts:
        return "corpus_reproduction"
    return "other"


def operator_names(document: Dict[str, Any]) -> List[str]:
    process = document.get("process", [])
    if not isinstance(process, list):
        raise ValueError("recipe process must be a list")
    names: List[str] = []
    for entry in process:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"invalid process entry: {entry!r}")
        names.extend(str(name) for name in entry)
    return names


def recipe_paths(hub_root: Path) -> Iterable[Path]:
    for relative_root in (
        Path("refined_recipes"),
        Path("reproduced_bloom"),
        Path("reproduced_redpajama"),
    ):
        yield from sorted((hub_root / relative_root).rglob("*.yaml"))


def inventory(hub_root: Path) -> Dict[str, Any]:
    rows = []
    for path in recipe_paths(hub_root):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            raise ValueError(f"recipe is not a mapping: {path}")
        operators = operator_names(document)
        relative = path.relative_to(hub_root)
        rows.append(
            {
                "recipe": relative.as_posix(),
                "family": recipe_family(relative),
                "operator_count": len(operators),
                "reorderable_per_record_count": sum(
                    "deduplicator" not in name for name in operators
                ),
                "global_deduplicator_count": sum(
                    "deduplicator" in name for name in operators
                ),
                "model_operator_count": sum(
                    any(marker in name for marker in MODEL_MARKERS)
                    for name in operators
                ),
                "operators": operators,
            }
        )

    family_counts = Counter(row["family"] for row in rows)
    commit = subprocess.run(
        ["git", "-C", str(hub_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "hub_root": str(hub_root),
        "hub_commit": commit,
        "recipe_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "recipes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, default=Path("data-juicer-hub"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()

    report = inventory(args.hub_root.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with args.tsv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "recipe",
                "family",
                "operator_count",
                "reorderable_per_record_count",
                "global_deduplicator_count",
                "model_operator_count",
                "operators",
            )
        )
        for row in report["recipes"]:
            writer.writerow(
                (
                    row["recipe"],
                    row["family"],
                    row["operator_count"],
                    row["reorderable_per_record_count"],
                    row["global_deduplicator_count"],
                    row["model_operator_count"],
                    ",".join(row["operators"]),
                )
            )

    print(
        f"classified {report['recipe_count']} recipes at "
        f"{report['hub_commit']}"
    )


if __name__ == "__main__":
    main()
