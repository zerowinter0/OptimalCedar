"""Cedar migration of Data-Juicer's The Pile EuroParl refinement recipe.

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
from evaluation.pipelines.stackexchange import dj_operators as ops


DEFAULT_DATASET_PATH = pathlib.Path(
    "datasets/pile_europarl/pile-europarl-raw.jsonl"
)


class PileEuroparlFeature(Feature):
    """Five fixed mappers followed by eleven reorderable filters."""

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

        fp = FilterPipe(
            fp,
            ops.AlphanumericFilter(
                min_ratio=0.75, max_ratio=0.90, tokenization=False
            ),
            tag="alphanumeric",
        )
        fp = FilterPipe(
            fp,
            ops.AverageLineLengthFilter(max_len=588),
            tag="avg_line_len",
        )
        fp = FilterPipe(
            fp,
            ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.16),
            tag="char_repeat",
        )
        fp = FilterPipe(
            fp,
            ops.FlaggedWordsFilter(
                lang="en", tokenization=True, max_ratio=0.0007
            ),
            tag="flagged_words",
        )
        fp = FilterPipe(
            fp,
            ops.LanguageIDScoreFilter(min_score=0.7),
            tag="language_id",
        )
        fp = FilterPipe(
            fp,
            ops.MaximumLineLengthFilter(max_len=4000),
            tag="max_line_len",
        )
        fp = FilterPipe(
            fp,
            ops.PerplexityFilter(lang="en", max_ppl=7596),
            tag="perplexity",
        )
        fp = FilterPipe(
            fp,
            ops.SpecialCharactersFilter(max_ratio=0.3),
            tag="special_chars",
        )
        fp = FilterPipe(
            fp,
            ops.TextLengthFilter(max_len=200_000),
            tag="text_length",
        )
        fp = FilterPipe(
            fp,
            ops.WordsNumFilter(
                lang="en",
                tokenization=True,
                min_num=20,
                max_num=100_000,
            ),
            tag="words_num",
        )
        fp = FilterPipe(
            fp,
            ops.WordRepetitionFilter(
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.2,
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

    feature = PileEuroparlFeature()
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
