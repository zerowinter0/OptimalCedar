#!/usr/bin/env python3
"""Audit substantive input scale for the fresh Data-Juicer candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CANDIDATES = {
    "redpajama_code": {
        "source": "datasets/redpajama_code/redpajama-github-raw-50000.jsonl",
        "metadata": "datasets/redpajama_code/redpajama-github-raw-50000.metadata.json",
        "metadata_bytes_key": "bytes",
        "target_outputs": 20_000,
    },
    "redpajama_arxiv": {
        "source": "datasets/redpajama_arxiv/redpajama-arxiv-raw-3gib.jsonl",
        "metadata": "datasets/redpajama_arxiv/redpajama-arxiv-raw-3gib.metadata.json",
        "metadata_bytes_key": "bytes",
        "target_outputs": 2_500,
    },
    "alpaca_cot": {
        "source": "datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl",
        "metadata": "datasets/alpaca_cot/alpaca-cot-en-cot-data.metadata.json",
        "metadata_bytes_key": "jsonl_bytes",
        "target_outputs": 65_000,
    },
    "video_self_evolution": {
        "source": "datasets/general_video_refine/msrvtt-video-text-200000.jsonl",
        "metadata": "datasets/general_video_refine/dataset_metadata.json",
        "metadata_bytes_key": "output_bytes",
        "metadata_records_key": "video_caption_pair_count",
        "target_outputs": 5_000,
        "video_root": "datasets/general_video_refine/videos",
    },
}


def _prefix_stats(path: Path, count: int) -> tuple[int, int]:
    observed = 0
    byte_count = 0
    with path.open("rb") as stream:
        for line in stream:
            if observed == count:
                break
            observed += 1
            byte_count += len(line)
    if observed != count:
        raise RuntimeError(
            f"source exhausted at {observed}/{count} records: {path}"
        )
    return observed, byte_count


def _source_record_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _video_prefix_files(
    source: Path, count: int, video_root: Path
) -> tuple[int, int]:
    relative_paths = set()
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == count:
                break
            payload = json.loads(line)
            videos = payload.get("videos")
            if not isinstance(videos, list) or len(videos) != 1:
                raise RuntimeError(f"invalid video record {index}: {source}")
            relative_paths.add(str(videos[0]))
    if len(relative_paths) != count:
        raise RuntimeError(
            f"expected {count} distinct prefix videos, found "
            f"{len(relative_paths)}"
        )
    paths = [video_root / relative for relative in sorted(relative_paths)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return len(paths), sum(path.stat().st_size for path in paths)


def audit(repo: Path) -> dict[str, Any]:
    workloads = {}
    for workload, spec in CANDIDATES.items():
        source = repo / str(spec["source"])
        metadata_path = repo / str(spec["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_bytes = source.stat().st_size
        source_records = _source_record_count(source)
        metadata_bytes = int(metadata[str(spec["metadata_bytes_key"])])
        records_key = str(spec.get("metadata_records_key", "records"))
        metadata_records = int(metadata[records_key])
        if source_bytes != metadata_bytes or source_records != metadata_records:
            raise RuntimeError(
                f"source metadata mismatch for {workload}: "
                f"bytes={source_bytes}/{metadata_bytes}, "
                f"records={source_records}/{metadata_records}"
            )
        target = int(spec["target_outputs"])
        _, prefix_bytes = _prefix_stats(source, target)
        item = {
            "source": str(spec["source"]),
            "metadata": str(spec["metadata"]),
            "source_records": source_records,
            "source_bytes": source_bytes,
            "formal_output_target": target,
            "minimum_prefix_jsonl_bytes": prefix_bytes,
            "target_to_source_record_fraction": target / source_records,
        }
        if "video_root" in spec:
            video_count, video_bytes = _video_prefix_files(
                source, target, repo / str(spec["video_root"])
            )
            item["minimum_distinct_video_files"] = video_count
            item["minimum_distinct_video_bytes"] = video_bytes
        workloads[workload] = item
    return {
        "definition": (
            "Exact first-N source-order input prefix for N formal retained "
            "outputs; because no pipeline expands records, actual scanning "
            "is at least this large."
        ),
        "workloads": workloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
