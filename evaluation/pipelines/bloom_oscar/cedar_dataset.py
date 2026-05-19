"""
Cedar 版 BLOOM Oscar 清洗流水线，对应 DataJuicer
``configs/reproduced_bloom/bloom-oscar.yaml``（不含 SimHash 文档去重，见
``dj_operators`` 模块说明）。
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pathlib
from typing import List

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import FilterPipe, MapperPipe, Pipe
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.bloom_oscar import dj_operators as dj_ops

DATA_JUICER_ROOT = pathlib.Path("/data-juicer")
DEFAULT_DATASET_PATH = (
    DATA_JUICER_ROOT / "demos" / "data_mixture" / "data" / "redpajama-c4-refined.jsonl"
)
DEFAULT_PROFILE_PATH = "/tmp/bloom_oscar_feature_profile.yml"


class BloomOscarFeature(Feature):
    """与 ``bloom-oscar.yaml`` 中 process 顺序一致的 Feature（无 simhash 去重）。"""

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, dj_ops.parse_json_line, tag="parse").fix()

        fp = FilterPipe(
            fp,
            dj_ops.LanguageIDScoreFilter(min_score=0.8, lang="en"),
            tag="language_id",
        )

        fp = MapperPipe(fp, dj_ops.WhitespaceNormalizationMapper(), tag="whitespace_norm")
        fp = MapperPipe(
            fp,
            dj_ops.PunctuationNormalizationMapper(),
            tag="punctuation_norm",
        )
        fp = MapperPipe(fp, dj_ops.FixUnicodeMapper(), tag="fix_unicode")
        fp = MapperPipe(
            fp,
            dj_ops.RemoveWordsWithIncorrectSubstringsMapper(),
            tag="remove_bad_substrings",
        )
        fp = MapperPipe(
            fp,
            dj_ops.RemoveLongWordsMapper(max_len=25),
            tag="remove_long_words",
        ).fix()

        fp = FilterPipe(
            fp,
            dj_ops.WordsNumFilter(
                lang="en",
                tokenization=False,
                min_num=20,
                max_num=100_000,
            ),
            tag="words_num",
        )
        fp = FilterPipe(
            fp,
            dj_ops.CharacterRepetitionFilter(
                rep_len=10,
                min_ratio=0.0,
                max_ratio=0.106,
            ),
            tag="char_repeat",
        )
        fp = FilterPipe(
            fp,
            dj_ops.WordRepetitionFilter(
                lang="en",
                tokenization=False,
                rep_len=5,
                min_ratio=0.0,
                max_ratio=0.19,
            ),
            tag="word_repeat",
        )
        fp = FilterPipe(
            fp,
            dj_ops.SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.4),
            tag="special_chars",
        )
        fp = FilterPipe(
            fp,
            dj_ops.StopwordsFilter(lang="en", tokenization=False, min_ratio=0.3),
            tag="stopwords",
        )
        fp = FilterPipe(
            fp,
            dj_ops.FlaggedWordsFilter(lang="en", tokenization=False, max_ratio=0.01),
            tag="flagged_words",
        )
        fp = FilterPipe(
            fp,
            dj_ops.PerplexityFilter(lang="en", max_ppl=1500),
            tag="perplexity",
        )

        fp = MapperPipe(fp, dj_ops.sync_text_key, tag="sync_text")
        fp = MapperPipe(fp, dj_ops.extract_output_text, tag="extract_text").fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = DEFAULT_DATASET_PATH
    if spec.kwargs and spec.kwargs.get("dataset_path"):
        dataset_path = pathlib.Path(spec.kwargs["dataset_path"])

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = LocalLineSource(str(dataset_path))
    feature = BloomOscarFeature()
    feature.apply(source)

    if spec.config:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            spec.config,
            enable_controller=False,
            enable_optimizer=False,
        )
    else:
        dataset = DataSet(
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
                enable_local_parallelism=not spec.disable_parallelism,
                enable_fusion=not spec.disable_fusion,
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
                reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
            ),
            generate_plan=spec.generate_plan,
        )

    return dataset


def _build_default_spec(args: argparse.Namespace) -> CedarEvalSpec:
    use_ray = args.use_ray
    if not args.run_profiling and not args.disable_offload:
        use_ray = True

    return CedarEvalSpec(
        batch_size=1,
        num_total_samples=args.num_total_samples,
        num_epochs=1,
        config=args.config,
        kwargs={"dataset_path": args.dataset_path},
        use_ray=use_ray,
        profiled_stats=args.profiled_stats,
        run_profiling=args.run_profiling,
        disable_prefetch=args.disable_prefetch,
        disable_offload=args.disable_offload,
        disable_parallelism=args.disable_parallelism,
        disable_reorder=args.disable_reorder,
        disable_fusion=args.disable_fusion,
        disable_caching=args.disable_caching,
        use_my_optimizer=args.use_my_optimizer,
        generate_plan=args.generate_plan,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--profiled_stats", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_profiling", action="store_true")
    parser.add_argument("--generate_plan", action="store_true")
    parser.add_argument("--num_total_samples", type=int, default=20)
    parser.add_argument("--num_preview_samples", type=int, default=5)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument("--use_my_optimizer", type=int, choices=[0, 1, 2, 3, 4, 5], default=0)
    parser.add_argument("--disable_prefetch", action="store_true")
    parser.add_argument("--disable_offload", action="store_true")
    parser.add_argument("--disable_parallelism", action="store_true")
    parser.add_argument("--disable_reorder", action="store_true")
    parser.add_argument("--disable_fusion", action="store_true")
    parser.add_argument("--disable_caching", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    spec = _build_default_spec(args)
    ds = get_dataset(spec)

    if args.run_profiling:
        return

    for idx, sample in enumerate(ds):
        print(sample)
        if idx + 1 >= args.num_preview_samples:
            break


if __name__ == "__main__":
    main()
