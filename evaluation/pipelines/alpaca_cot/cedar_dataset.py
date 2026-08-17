"""Cedar implementation of Data-Juicer's Alpaca-CoT English recipe.

The two document-level deduplicators in the official recipe are deliberately
outside this feature: Cedar optimizes per-record operators.  The five
per-record filters and their arguments are otherwise kept exactly in recipe
order before optimization.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List

from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import (
    FilterPipe,
    MapperPipe,
    Pipe,
    PipeComputeScaling,
)
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.stackexchange import dj_operators as ops
from evaluation.pipelines.redpajama_c4.dj_operators import (
    FIELDS_CONTEXT,
    FIELDS_META,
    FIELDS_STATS,
    TEXT_KEY,
)


DEFAULT_DATASET_PATH = Path(
    "datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl"
)


def parse_and_format(line: Any) -> Dict[str, Any]:
    """Apply Data-Juicer's Alpaca formatter while parsing one JSONL row."""
    if isinstance(line, dict):
        line = line.get("text", "")
    sample = json.loads(line)
    instruction = str(sample.get("instruction", "")).strip()
    context = str(sample.get("input", "")).strip()
    response = str(sample.get("output", "")).strip()
    parts = [instruction]
    if context:
        parts.append(context)
    if response:
        parts.append(response)
    sample[TEXT_KEY] = "\n\n".join(part for part in parts if part)
    sample.setdefault(FIELDS_STATS, {})
    sample.setdefault(FIELDS_CONTEXT, {})
    sample.setdefault(FIELDS_META, {})
    return sample


class AlpacaCotFeature(Feature):
    """Eight-pipe Cedar feature for the five per-sample recipe filters."""

    def _compose(self, source_pipes: List[Pipe]):
        fp = MapperPipe(
            source_pipes[0], parse_and_format, tag="parse_and_format"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            ops.AlphanumericFilter(
                tokenization=False, min_ratio=0.1, max_ratio=sys.maxsize
            ),
            tag="alphanumeric",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.6),
            tag="char_repeat",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            ops.FlaggedWordsFilter(
                lang="en", tokenization=True, max_ratio=0.017
            ),
            tag="flagged_words",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            ops.MaximumLineLengthFilter(min_len=20),
            tag="max_line_len",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp, ops.TextLengthFilter(min_len=30), tag="text_length"
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        fp = MapperPipe(fp, ops.sync_text_key, tag="sync_text").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        fp = MapperPipe(
            fp, ops.extract_output_text, tag="extract_text"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = DEFAULT_DATASET_PATH
    if spec.kwargs and spec.kwargs.get("dataset_path"):
        dataset_path = Path(spec.kwargs["dataset_path"])

    feature = AlpacaCotFeature()
    feature.apply(LocalLineSource(str(dataset_path)))
    ctx = CedarContext(ray_config=spec.to_ray_config())
    if spec.config:
        return DataSet(
            ctx,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    return DataSet(
        ctx,
        {"feature": feature},
        enable_controller=not spec.disable_controller,
        enable_optimizer=not spec.disable_optimizer,
        profiled_data=spec.profiled_stats,
        run_profiling=spec.run_profiling,
        optimizer_options=OptimizerOptions(
            enable_prefetch=not spec.disable_prefetch,
            est_throughput=None,
            available_local_cpus=mp.cpu_count(),
            enable_offload=not spec.disable_offload,
            enable_reorder=not spec.disable_reorder,
            enable_caching=not spec.disable_caching,
            num_samples=getattr(spec, "num_total_samples", None),
            enable_local_parallelism=not spec.disable_parallelism,
            enable_fusion=not spec.disable_fusion,
            use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
            reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
        ),
        generate_plan=spec.generate_plan,
    )


__all__ = ["AlpacaCotFeature", "DEFAULT_DATASET_PATH", "get_dataset"]
