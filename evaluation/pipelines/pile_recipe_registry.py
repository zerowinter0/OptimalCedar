"""Frozen Data-Juicer Pile recipes used by the Chapter 6 extension.

The recipes are transcribed from ``datajuicer/data-juicer-hub@47fc345``.
Only the final cross-record SimHash deduplicator is omitted: Cedar's optimizer
operates on per-record pipes and cannot reorder a global dataset operation.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

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


@dataclass(frozen=True)
class FilterSpec:
    tag: str
    operator: str
    kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class PileRecipe:
    workload: str
    official_recipe: str
    dataset_id: str
    revision: str
    dataset_path: Path
    logical_bytes: int
    clean_links: bool
    filters: Tuple[FilterSpec, ...]


def _filter(tag: str, operator: str, **kwargs: Any) -> FilterSpec:
    return FilterSpec(tag=tag, operator=operator, kwargs=kwargs)


RECIPES: Dict[str, PileRecipe] = {
    "pile_hackernews": PileRecipe(
        workload="pile_hackernews",
        official_recipe=(
            "refined_recipes/pretrain/pile-hackernews-refine.yaml"
        ),
        dataset_id="timaeus/pile-hackernews",
        revision="a5d6a32ee1039015b8037da6aa776af4cfb89df1",
        dataset_path=Path(
            "datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
        ),
        logical_bytes=500_518_158,
        clean_links=False,
        filters=(
            _filter(
                "alphanumeric",
                "AlphanumericFilter",
                tokenization=False,
                min_ratio=0.2,
                max_ratio=sys.maxsize,
            ),
            _filter(
                "avg_line_len",
                "AverageLineLengthFilter",
                min_len=15,
                max_len=sys.maxsize,
            ),
            _filter(
                "char_repeat",
                "CharacterRepetitionFilter",
                rep_len=10,
                max_ratio=0.3,
            ),
            _filter(
                "flagged_words",
                "FlaggedWordsFilter",
                lang="en",
                tokenization=True,
                max_ratio=0.05,
            ),
            _filter(
                "language_id", "LanguageIDScoreFilter", min_score=0.2
            ),
            _filter(
                "max_line_len",
                "MaximumLineLengthFilter",
                min_len=20,
                max_len=sys.maxsize,
            ),
            _filter(
                "perplexity",
                "PerplexityFilter",
                lang="en",
                max_ppl=10_000,
            ),
            _filter(
                "special_chars", "SpecialCharactersFilter", max_ratio=0.7
            ),
            _filter("text_length", "TextLengthFilter", min_len=100),
            _filter(
                "words_num",
                "WordsNumFilter",
                lang="en",
                tokenization=True,
                min_num=30,
                max_num=sys.maxsize,
            ),
            _filter(
                "word_repeat",
                "WordRepetitionFilter",
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.8,
            ),
        ),
    ),
    "pile_pubmed_abstracts": PileRecipe(
        workload="pile_pubmed_abstracts",
        official_recipe=(
            "refined_recipes/pretrain/pile-pubmed-abstract-refine.yaml"
        ),
        dataset_id="timaeus/pile-pubmed_abstracts",
        revision="2d733e1624c384e4a97acfd4b93c8e739420b32e",
        dataset_path=Path(
            "datasets/pile_pubmed_abstracts/"
            "pile-pubmed-abstracts-raw-100000.jsonl"
        ),
        logical_bytes=135_484_704,
        clean_links=True,
        filters=(
            _filter(
                "alphanumeric",
                "AlphanumericFilter",
                tokenization=False,
                min_ratio=0.7,
                max_ratio=0.881,
            ),
            _filter(
                "avg_line_len", "AverageLineLengthFilter", max_len=2100
            ),
            _filter(
                "char_repeat",
                "CharacterRepetitionFilter",
                rep_len=10,
                max_ratio=0.2,
            ),
            _filter(
                "flagged_words",
                "FlaggedWordsFilter",
                lang="en",
                tokenization=True,
                max_ratio=0.00232,
            ),
            _filter(
                "language_id", "LanguageIDScoreFilter", min_score=0.5
            ),
            _filter(
                "max_line_len", "MaximumLineLengthFilter", max_len=4000
            ),
            _filter(
                "perplexity",
                "PerplexityFilter",
                lang="en",
                max_ppl=4000,
            ),
            _filter(
                "special_chars", "SpecialCharactersFilter", max_ratio=0.38
            ),
            _filter("text_length", "TextLengthFilter", max_len=4000),
            _filter(
                "words_num",
                "WordsNumFilter",
                lang="en",
                tokenization=True,
                min_num=20,
                max_num=700,
            ),
            _filter(
                "word_repeat",
                "WordRepetitionFilter",
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.0887,
            ),
        ),
    ),
    "pile_freelaw": PileRecipe(
        workload="pile_freelaw",
        official_recipe="refined_recipes/pretrain/pile-freelaw-refine.yaml",
        dataset_id="timaeus/pile-freelaw",
        revision="e5cf633ac70c4659cb4761718bdd93d029df5150",
        dataset_path=Path(
            "datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl"
        ),
        logical_bytes=1_588_135_150,
        clean_links=True,
        filters=(
            _filter(
                "alphanumeric",
                "AlphanumericFilter",
                tokenization=False,
                min_ratio=0.3,
                max_ratio=sys.maxsize,
            ),
            _filter(
                "avg_line_len", "AverageLineLengthFilter", max_len=697
            ),
            _filter(
                "char_repeat",
                "CharacterRepetitionFilter",
                rep_len=10,
                max_ratio=0.4,
            ),
            _filter(
                "flagged_words",
                "FlaggedWordsFilter",
                lang="en",
                tokenization=True,
                max_ratio=0.0053,
            ),
            _filter(
                "language_id", "LanguageIDScoreFilter", min_score=0.5
            ),
            _filter(
                "max_line_len", "MaximumLineLengthFilter", max_len=4229
            ),
            _filter(
                "perplexity",
                "PerplexityFilter",
                lang="en",
                max_ppl=5322,
            ),
            _filter(
                "special_chars", "SpecialCharactersFilter", max_ratio=0.7
            ),
            _filter(
                "stopwords",
                "StopwordsFilter",
                lang="en",
                tokenization=True,
                min_ratio=0.1,
            ),
            _filter("text_length", "TextLengthFilter", max_len=84_026),
            _filter(
                "words_num",
                "WordsNumFilter",
                lang="en",
                tokenization=True,
                min_num=100,
                max_num=15_208,
            ),
            _filter(
                "word_repeat",
                "WordRepetitionFilter",
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.155,
            ),
        ),
    ),
    "pile_uspto_backgrounds": PileRecipe(
        workload="pile_uspto_backgrounds",
        official_recipe="refined_recipes/pretrain/pile-uspto-refine.yaml",
        dataset_id="timaeus/pile-uspto_backgrounds",
        revision="cb4f574c22debca066312ddccd0d048cbd7e148b",
        dataset_path=Path(
            "datasets/pile_uspto_backgrounds/"
            "pile-uspto-backgrounds-raw-100000.jsonl"
        ),
        logical_bytes=428_010_825,
        clean_links=True,
        filters=(
            _filter(
                "alphanumeric",
                "AlphanumericFilter",
                tokenization=False,
                min_ratio=0.7,
                max_ratio=sys.maxsize,
            ),
            _filter(
                "avg_line_len", "AverageLineLengthFilter", max_len=2000
            ),
            _filter(
                "char_repeat",
                "CharacterRepetitionFilter",
                rep_len=10,
                max_ratio=0.2,
            ),
            _filter(
                "flagged_words",
                "FlaggedWordsFilter",
                lang="en",
                tokenization=True,
                max_ratio=0.0016,
            ),
            _filter(
                "language_id", "LanguageIDScoreFilter", min_score=0.6
            ),
            _filter(
                "max_line_len", "MaximumLineLengthFilter", max_len=3061
            ),
            _filter(
                "perplexity",
                "PerplexityFilter",
                lang="en",
                max_ppl=4000,
            ),
            _filter(
                "special_chars", "SpecialCharactersFilter", max_ratio=0.3
            ),
            _filter("text_length", "TextLengthFilter", max_len=21_556),
            _filter(
                "words_num",
                "WordsNumFilter",
                lang="en",
                tokenization=True,
                min_num=100,
                max_num=6000,
            ),
            _filter(
                "word_repeat",
                "WordRepetitionFilter",
                lang="en",
                tokenization=True,
                rep_len=10,
                max_ratio=0.169,
            ),
        ),
    ),
}


def make_filter(spec: FilterSpec):
    constructor = getattr(ops, spec.operator)
    return constructor(**dict(spec.kwargs))


class PileRecipeFeature(Feature):
    """Cedar feature generated from one frozen recipe registry entry."""

    def __init__(self, workload: str):
        super().__init__()
        if workload not in RECIPES:
            raise ValueError(f"Unknown frozen Pile recipe: {workload}")
        self.workload = workload

    def _compose(self, source_pipes: List[Pipe]):
        recipe = RECIPES[self.workload]
        fp = source_pipes[0]
        fp = MapperPipe(fp, ops.parse_json_line, tag="parse").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(fp, ops.CleanEmailMapper(), tag="clean_email").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        if recipe.clean_links:
            fp = MapperPipe(
                fp, ops.CleanLinksMapper(), tag="clean_links"
            ).fix()
            fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(fp, ops.FixUnicodeMapper(), tag="fix_unicode").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(
            fp, ops.PunctuationNormalizationMapper(), tag="normalize_punct"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)
        fp = MapperPipe(
            fp, ops.WhitespaceNormalizationMapper(), tag="normalize_space"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_DATA)

        for filter_spec in recipe.filters:
            fp = FilterPipe(
                fp,
                make_filter(filter_spec),
                tag=filter_spec.tag,
            )
            fp.set_compute_scaling(
                PipeComputeScaling.PER_RECORD
                if filter_spec.operator == "TextLengthFilter"
                else PipeComputeScaling.PER_DATA
            )

        fp = MapperPipe(fp, ops.sync_text_key, tag="sync_text").fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        fp = MapperPipe(
            fp, ops.extract_output_text, tag="extract_text"
        ).fix()
        fp.set_compute_scaling(PipeComputeScaling.PER_RECORD)
        return fp


def get_pile_recipe_dataset(
    spec: CedarEvalSpec, workload: str
) -> DataSet:
    recipe = RECIPES[workload]
    dataset_path = recipe.dataset_path
    if spec.kwargs and spec.kwargs.get("dataset_path"):
        dataset_path = Path(spec.kwargs["dataset_path"])

    feature = PileRecipeFeature(workload)
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


__all__ = [
    "FilterSpec",
    "PileRecipe",
    "PileRecipeFeature",
    "RECIPES",
    "get_pile_recipe_dataset",
    "make_filter",
]
