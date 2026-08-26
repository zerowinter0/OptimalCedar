"""Cedar migration of Data-Juicer's StackExchange refinement recipe.

Source recipe:
``refined_recipes/pretrain/redpajama-pile-stackexchange-refine.yaml`` in
``datajuicer/data-juicer-hub``.  The final global SimHash deduplicator is
omitted because it is not a per-sample operator.
"""

from __future__ import annotations

import multiprocessing as mp
import pathlib
from typing import List

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
from evaluation.pipelines.stackexchange import dj_operators as dj_ops


DEFAULT_DATASET_PATH = pathlib.Path(
    "datasets/stackexchange/redpajama-stackexchange-400000.jsonl"
)


class StackExchangeFeature(Feature):
    """Five normalization mappers followed by ten reorderable filters."""

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, dj_ops.parse_json_line, tag="parse").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)

        fp = MapperPipe(fp, dj_ops.CleanEmailMapper(), tag="clean_email").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(fp, dj_ops.CleanLinksMapper(), tag="clean_links").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(fp, dj_ops.FixUnicodeMapper(), tag="fix_unicode").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(
            fp, dj_ops.PunctuationNormalizationMapper(), tag="normalize_punct"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(
            fp, dj_ops.WhitespaceNormalizationMapper(), tag="normalize_space"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)

        fp = FilterPipe(
            fp,
            dj_ops.AlphanumericFilter(min_ratio=0.35, max_ratio=0.943),
            tag="alphanumeric",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.AverageLineLengthFilter(min_len=20, max_len=400),
            tag="avg_line_len",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.4),
            tag="char_repeat",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.FlaggedWordsFilter(
                lang="en", tokenization=True, max_ratio=0.01
            ),
            tag="flagged_words",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.LanguageIDScoreFilter(min_score=0.1),
            tag="language_id",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.MaximumLineLengthFilter(min_len=80),
            tag="max_line_len",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.PerplexityFilter(lang="en", max_ppl=10_000),
            tag="perplexity",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.SpecialCharactersFilter(min_ratio=0.232, max_ratio=0.7),
            tag="special_chars",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp, dj_ops.TextLengthFilter(min_len=200), tag="text_length"
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        fp = FilterPipe(
            fp,
            dj_ops.WordsNumFilter(
                lang="en",
                tokenization=True,
                min_num=100,
                max_num=2**63 - 1,
            ),
            tag="words_num",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = FilterPipe(
            fp,
            dj_ops.WordRepetitionFilter(
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.8,
            ),
            tag="word_repeat",
        )
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)

        fp = MapperPipe(fp, dj_ops.sync_text_key, tag="sync_text").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        fp = MapperPipe(
            fp, dj_ops.extract_output_text, tag="extract_text"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = DEFAULT_DATASET_PATH
    if spec.kwargs and spec.kwargs.get("dataset_path"):
        dataset_path = pathlib.Path(spec.kwargs["dataset_path"])

    feature = StackExchangeFeature()
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
