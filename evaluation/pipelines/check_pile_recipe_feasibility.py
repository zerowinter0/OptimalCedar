#!/usr/bin/env python3
"""Source-only retained-record check for a frozen Pile recipe.

This is intentionally not an optimizer benchmark. It evaluates the official
recipe order only until 20,000 records have survived, or until the source is
exhausted, and records the rejection count at each filter.
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections import OrderedDict
from pathlib import Path

from evaluation.pipelines.pile_recipe_registry import (
    RECIPES,
    make_filter,
)
from evaluation.pipelines.stackexchange import dj_operators as ops


TARGET_RETAINED = 20_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=sorted(RECIPES))
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record-timeout",
        action="store_true",
        help=(
            "record a bounded serial-precheck timeout without claiming that "
            "the formal parallel workload is source-infeasible"
        ),
    )
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--source-bytes-read", type=int)
    args = parser.parse_args()

    recipe = RECIPES[args.workload]
    dataset_path = args.dataset_path or recipe.dataset_path
    output = args.output or dataset_path.with_suffix(".feasibility.json")
    if args.record_timeout:
        if args.elapsed_seconds is None or args.elapsed_seconds <= 0:
            parser.error("--record-timeout requires positive --elapsed-seconds")
        result = {
            "schema_version": 1,
            "status": "benchmarkable_timeout",
            "workload": args.workload,
            "dataset_path": str(dataset_path),
            "target_retained": TARGET_RETAINED,
            "source_records_scanned": None,
            "retained_records": None,
            "parse_failures": None,
            "official_order_filter_counts": None,
            "serial_precheck_elapsed_seconds": args.elapsed_seconds,
            "source_bytes_read_at_stop": args.source_bytes_read,
            "recorded_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "reason": (
                "The source-only official-order serial precheck reached its "
                "operational time bound. This does not establish source "
                "infeasibility; retain the workload for the formal W=8, "
                "CPU=64 benchmark, whose execution detects source exhaustion "
                "and has its own frozen 3600-second limit."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return
    filters = [
        (filter_spec.tag, make_filter(filter_spec))
        for filter_spec in recipe.filters
    ]
    filter_counts = OrderedDict(
        (tag, {"seen": 0, "kept": 0, "rejected": 0})
        for tag, _ in filters
    )

    fixed_mappers = [ops.CleanEmailMapper()]
    if recipe.clean_links:
        fixed_mappers.append(ops.CleanLinksMapper())
    fixed_mappers.extend(
        [
            ops.FixUnicodeMapper(),
            ops.PunctuationNormalizationMapper(),
            ops.WhitespaceNormalizationMapper(),
        ]
    )

    scanned = 0
    retained = 0
    parse_failures = 0
    with dataset_path.open(encoding="utf-8") as stream:
        for line in stream:
            scanned += 1
            try:
                sample = ops.parse_json_line(line)
                for mapper in fixed_mappers:
                    sample = mapper(sample)
            except Exception:
                parse_failures += 1
                continue

            accepted = True
            for tag, operator in filters:
                counts = filter_counts[tag]
                counts["seen"] += 1
                if operator(sample):
                    counts["kept"] += 1
                else:
                    counts["rejected"] += 1
                    accepted = False
                    break
            if accepted:
                retained += 1
                if retained >= TARGET_RETAINED:
                    break

    status = "feasible" if retained >= TARGET_RETAINED else "infeasible"
    result = {
        "schema_version": 1,
        "status": status,
        "workload": args.workload,
        "dataset_path": str(dataset_path),
        "target_retained": TARGET_RETAINED,
        "source_records_scanned": scanned,
        "retained_records": retained,
        "parse_failures": parse_failures,
        "official_order_filter_counts": filter_counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if status != "feasible":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
