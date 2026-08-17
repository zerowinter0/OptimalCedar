#!/usr/bin/env python3
"""Materialize workload-code operator scaling annotations into a profile."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.eval_cedar import import_module_from_path


def parse_kwargs(value: str) -> dict[str, str]:
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        key, separator, item_value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid dataset kwarg: {item!r}")
        result[key] = item_value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--dataset-kwargs", default="")
    parser.add_argument(
        "--require-all-operators-annotated", action="store_true"
    )
    args = parser.parse_args()

    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise RuntimeError(f"Profile is not a mapping: {args.profile}")
    old_metadata = profile.get("operator_compute_scaling", {})

    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=1,
        num_epochs=1,
        kwargs=parse_kwargs(args.dataset_kwargs),
        use_ray=False,
        profiled_stats=str(args.profile),
        disable_optimizer=True,
        disable_controller=True,
    )
    dataset = import_module_from_path(args.dataset_file).get_dataset(spec)
    try:
        feature = next(iter(dataset.features.values()))
        metadata = {}
        missing = []
        for p_id, pipe in feature.logical_pipes.items():
            explicit = bool(
                getattr(pipe, "compute_scaling_explicit", False)
            )
            if not pipe.is_source() and not explicit:
                missing.append((p_id, pipe.get_logical_name(), pipe.tag))
            entry = {
                "scaling": pipe.compute_scaling.value,
                "mode": "explicit" if explicit else "default",
                "annotation_source": (
                    "workload_code" if explicit else "documented_default"
                ),
            }
            old_entry = old_metadata.get(
                p_id, old_metadata.get(str(p_id), {})
            )
            if isinstance(old_entry, dict) and isinstance(
                old_entry.get("inference"), dict
            ):
                entry["inference"] = old_entry["inference"]
            metadata[p_id] = entry
        if args.require_all_operators_annotated and missing:
            raise RuntimeError(
                "Unannotated non-source operators: " + repr(missing)
            )
        profile["operator_compute_scaling"] = metadata
        profile["operator_compute_scaling_schema"] = {
            "schema_version": 2,
            "granularity": "operator",
            "default": "per_data",
            "manual_annotations_complete": not missing,
            "inference_retained_for_audit": True,
        }
    finally:
        dataset.close()

    args.profile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.profile.parent,
        prefix=f".{args.profile.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        yaml.safe_dump(profile, stream, sort_keys=False)
        temp_path = Path(stream.name)
    os.replace(temp_path, args.profile)


if __name__ == "__main__":
    main()
