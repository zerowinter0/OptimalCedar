"""
Cedar implementation of Data-Juicer Hub's LLaVA pretrain image-text refinement
recipe:

``/tmp/data-juicer-hub-latest/refined_recipes/image/llava-pretrain-refine.yaml``.

The Feature preserves the recipe's operator order, including the Hugging Face
CLIP/BLIP image-text filters.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import pathlib
from typing import List

from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.config import CedarContext
from cedar.pipes import (
    FilterPipe,
    MapperPipe,
    Pipe,
    PipeExecutionResource,
)
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.llava_pretrain import dj_operators as dj_ops

DEFAULT_DATASET_PATH = "/tmp/llava_pretrain_cedar_fixture.jsonl"
DEFAULT_IMAGE_ROOT = ""
DEFAULT_PROFILE_PATH = "/tmp/llava_pretrain_feature_profile.yml"

# LLaVA uses CUDA-backed CLIP/BLIP filters. Cedar local multiprocessing actors
# must not fork after CUDA initialization; spawn gives each actor a fresh CUDA
# context instead of inheriting a poisoned forked state.
mp.set_start_method(os.environ.get("LLAVA_MP_START_METHOD", "spawn"), force=True)


class LlavaPretrainFeature(Feature):
    """Cedar Feature matching the Data-Juicer LLaVA pretrain process list."""

    def __init__(self, image_root: str | pathlib.Path | None = None):
        super().__init__()
        self.image_root = str(image_root) if image_root else ""

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, dj_ops.parse_json_line, tag="parse").fix()
        fp = MapperPipe(
            fp,
            dj_ops.SetImageRootMapper(self.image_root),
            tag="image_root",
        ).fix()

        fp = MapperPipe(fp, dj_ops.FixUnicodeMapper(), tag="fix_unicode").fix()
        fp = MapperPipe(
            fp,
            dj_ops.PunctuationNormalizationMapper(),
            tag="punctuation_norm",
        ).fix()

        fp = FilterPipe(
            fp,
            dj_ops.AlphanumericFilter(min_ratio=0.60, max_ratio=1.0),
            tag="alphanumeric",
        )
        fp = FilterPipe(
            fp,
            dj_ops.CharacterRepetitionFilter(rep_len=10, max_ratio=0.09373663),
            tag="char_repeat",
        )
        fp = FilterPipe(
            fp,
            dj_ops.FlaggedWordsFilter(lang="en", tokenization=False, max_ratio=0.0),
            tag="flagged_words",
        )
        fp = FilterPipe(
            fp,
            dj_ops.PerplexityFilter(lang="en", max_ppl=14435.5806),
            tag="perplexity",
        )
        fp = FilterPipe(
            fp,
            dj_ops.SpecialCharactersFilter(
                min_ratio=0.16534802,
                max_ratio=0.42023757,
            ),
            tag="special_chars",
        )
        fp = FilterPipe(
            fp,
            dj_ops.WordRepetitionFilter(
                lang="en",
                tokenization=False,
                rep_len=10,
                max_ratio=0.03085751,
            ),
            tag="word_repeat",
        )

        fp = FilterPipe(
            fp,
            dj_ops.ImageAspectRatioFilter(
                min_ratio=0.333,
                max_ratio=3.0,
                any_or_all="any",
            ),
            tag="image_aspect_ratio",
        )
        fp = FilterPipe(
            fp,
            dj_ops.ImageShapeFilter(
                max_width=727,
                max_height=606,
                any_or_all="any",
            ),
            tag="image_shape",
        )
        fp = FilterPipe(
            fp,
            dj_ops.ImageSizeFilter(max_size="124KB", any_or_all="any"),
            tag="image_size",
        )
        fp = FilterPipe(
            fp,
            dj_ops.ImageTextSimilarityFilter(
                hf_clip="openai/clip-vit-base-patch32",
                min_score=0.20315419,
            ),
            tag="image_text_similarity",
        )
        fp.set_execution_resource(PipeExecutionResource.CUDA)
        fp = FilterPipe(
            fp,
            dj_ops.ImageTextMatchingFilter(
                hf_blip="Salesforce/blip-itm-base-coco",
                min_score=0.44930778,
            ),
            tag="image_text_matching",
        )
        fp.set_execution_resource(PipeExecutionResource.CUDA)

        fp = MapperPipe(fp, dj_ops.sync_text_key, tag="sync_text").fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    dataset_path = pathlib.Path(DEFAULT_DATASET_PATH)
    image_root = DEFAULT_IMAGE_ROOT
    if spec.kwargs:
        if spec.kwargs.get("dataset_path"):
            dataset_path = pathlib.Path(spec.kwargs["dataset_path"])
        if spec.kwargs.get("image_root"):
            image_root = spec.kwargs["image_root"]

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = LocalLineSource(str(dataset_path))
    feature = LlavaPretrainFeature(image_root=image_root)
    feature.apply(source)

    # Cedar implements local parallelism by cloning and sharding the complete
    # feature. Each clone would therefore materialize its own CUDA-backed CLIP
    # and BLIP models. Bound complete feature replicas to the GPU-operator
    # concurrency used by the experiment so every optimizer is evaluated under
    # the same GPU resource budget. CPU-only workloads remain unaffected.
    gpu_operator_parallelism = int(
        os.environ.get("LLAVA_GPU_OPERATOR_PARALLELISM", "1")
    )
    if gpu_operator_parallelism < 1:
        raise ValueError("LLAVA_GPU_OPERATOR_PARALLELISM must be at least 1")
    if (
        gpu_operator_parallelism > 1
        and os.environ.get("LLAVA_ALLOW_GPU_FEATURE_REPLICATION") != "1"
    ):
        raise RuntimeError(
            "LLaVA contains CUDA-backed operators inside INPROCESS/fused "
            "blocks. Replicating the complete Feature would duplicate models "
            "and invalidate the global GPU budget. Keep "
            "LLAVA_GPU_OPERATOR_PARALLELISM=1 for paper runs; set "
            "LLAVA_ALLOW_GPU_FEATURE_REPLICATION=1 only for explicit unsafe "
            "diagnostics."
        )
    local_worker_budget = min(mp.cpu_count(), gpu_operator_parallelism)
    logging.getLogger(__name__).info(
        "Using a uniform LLaVA GPU operator parallelism of %d for all "
        "optimizers (local worker budget=%d).",
        gpu_operator_parallelism,
        local_worker_budget,
    )

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
                available_local_cpus=local_worker_budget,
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
        kwargs={"dataset_path": args.dataset_path, "image_root": args.image_root},
        use_ray=use_ray,
        profiled_stats=args.profiled_stats,
        run_profiling=args.run_profiling,
        disable_prefetch=args.disable_prefetch,
        disable_offload=args.disable_offload,
        disable_parallelism=args.disable_parallelism,
        disable_reorder=args.disable_reorder,
        disable_fusion=args.disable_fusion,
        disable_caching=args.disable_caching,
        disable_optimizer=args.disable_optimizer,
        disable_controller=args.disable_controller,
        use_my_optimizer=args.use_my_optimizer,
        generate_plan=args.generate_plan,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--profiled_stats", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_profiling", action="store_true")
    parser.add_argument("--generate_plan", action="store_true")
    parser.add_argument("--num_total_samples", type=int, default=20)
    parser.add_argument("--num_preview_samples", type=int, default=5)
    parser.add_argument("--use_ray", action="store_true")
    parser.add_argument(
        "--use_my_optimizer",
        type=int,
        choices=list(range(9)),
        default=0,
    )
    parser.add_argument("--disable_optimizer", action="store_true")
    parser.add_argument("--disable_controller", action="store_true")
    parser.add_argument("--disable_prefetch", action="store_true")
    parser.add_argument("--disable_offload", action="store_true")
    parser.add_argument("--disable_parallelism", action="store_true")
    parser.add_argument("--disable_reorder", action="store_true")
    parser.add_argument("--disable_fusion", action="store_true")
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument(
        "--disable_smp_profile",
        action="store_true",
        help="Skip Cedar's SMP profiling pass; useful for HF model smoke tests.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    original_profile_smp = DataSet._profile_smp
    if args.disable_smp_profile:
        def skip_smp_profile(self, d, feature_to_profile, f_name, n_samples):
            d.setdefault("offloads", {})["SMP"] = {}
            return None

        DataSet._profile_smp = skip_smp_profile

    spec = _build_default_spec(args)
    try:
        ds = get_dataset(spec)
    finally:
        DataSet._profile_smp = original_profile_smp

    if args.run_profiling:
        return

    for idx, sample in enumerate(ds):
        print(sample)
        if idx + 1 >= args.num_preview_samples:
            break


if __name__ == "__main__":
    main()
