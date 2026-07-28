"""Cedar migration of Data-Juicer's RedPajama GitHub Code recipe.

The final global SimHash deduplicator is omitted because it is not a
per-sample operator and therefore is outside Cedar's optimizer search space.
"""

from __future__ import annotations

import multiprocessing as mp
import pathlib
from typing import List

from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import FilterPipe, MapperPipe, Pipe
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.redpajama_c4 import dj_operators as ops


DEFAULT_DATASET_PATH = pathlib.Path(
    "datasets/redpajama_code/redpajama-github-raw-50000.jsonl"
)


class RedPajamaCodeFeature(Feature):
    """Six fixed mappers followed by eight reorderable filters."""

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, ops.parse_json_line, tag="parse").fix()
        fp = MapperPipe(fp, ops.CleanEmailMapper(), tag="clean_email").fix()
        fp = MapperPipe(fp, ops.CleanLinksMapper(), tag="clean_links").fix()
        fp = MapperPipe(fp, ops.FixUnicodeMapper(), tag="fix_unicode").fix()
        fp = MapperPipe(
            fp, ops.PunctuationNormalizationMapper(), tag="normalize_punct"
        ).fix()
        fp = MapperPipe(
            fp, ops.WhitespaceNormalizationMapper(), tag="normalize_space"
        ).fix()
        fp = MapperPipe(
            fp, ops.CleanCopyrightMapper(), tag="clean_copyright"
        ).fix()

        fp = FilterPipe(
            fp,
            ops.AlphanumericFilter(
                min_ratio=0.4, max_ratio=0.8, tokenization=False
            ),
            tag="alphanumeric_chars",
        )
        fp = FilterPipe(
            fp,
            ops.AlphanumericFilter(
                min_ratio=1.5, max_ratio=3.0, tokenization=True
            ),
            tag="alphanumeric_tokens",
        )
        fp = FilterPipe(
            fp,
            ops.AverageLineLengthFilter(min_len=15, max_len=100),
            tag="avg_line_len",
        )
        fp = FilterPipe(
            fp,
            ops.CharacterRepetitionFilter(
                rep_len=10, min_ratio=0.05, max_ratio=0.3
            ),
            tag="char_repeat",
        )
        fp = FilterPipe(
            fp,
            ops.MaximumLineLengthFilter(min_len=50, max_len=500),
            tag="max_line_len",
        )
        fp = FilterPipe(
            fp,
            ops.TextLengthFilter(min_len=300),
            tag="text_length",
        )
        fp = FilterPipe(
            fp,
            ops.WordsNumFilter(
                lang="en",
                tokenization=False,
                min_num=30,
                max_num=5000,
            ),
            tag="words_num",
        )
        fp = FilterPipe(
            fp,
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=False,
                rep_len=10,
                max_ratio=0.1,
            ),
            tag="word_repeat",
        )

        fp = MapperPipe(fp, ops.sync_text_key, tag="sync_text").fix()
        return MapperPipe(
            fp, ops.extract_output_text, tag="extract_text"
        ).fix()


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = DEFAULT_DATASET_PATH
    if spec.kwargs and spec.kwargs.get("dataset_path"):
        dataset_path = pathlib.Path(spec.kwargs["dataset_path"])

    feature = RedPajamaCodeFeature()
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
