import pathlib
import logging

import torch.nn as nn
import multiprocessing as mp

from typing import List

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
    BatcherPipe,
)
from cedar.pipes.common import CedarPipeSpec
from cedar.pipes.context import PipeVariantType
from cedar.sources import LocalLineSource

from evaluation.cedar_utils import CedarEvalSpec


DATASET_LOC = "datasets/wikitext103"
DATASET_NAME = "wikitext103"
DATASET_FILE = "wikitext-103-v1.zip"
DATASET_SOURCE = "https://s3.amazonaws.com/research.metamind.io/wikitext/\
wikitext-103-v1.zip"


def _get_text(data):
    return data


class Wikitext1033Feature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        self.add_bos = T.AddToken(token=0, begin=True)
        self.add_bos_1 = T.AddToken(token=1, begin=True)
        self.add_bos_2 = T.AddToken(token=2, begin=True)
        self.add_bos_3 = T.AddToken(token=3, begin=True)
        self.add_bos_4 = T.AddToken(token=4, begin=True)
        self.add_bos_5 = T.AddToken(token=5, begin=True)
        self.add_bos_6 = T.AddToken(token=6, begin=True)
        self.add_bos_7 = T.AddToken(token=7, begin=True)
        self.add_bos_8 = T.AddToken(token=8, begin=True)
        self.add_bos_9 = T.AddToken(token=9, begin=True)
        self.add_bos_10 = T.AddToken(token=10, begin=True)
        self.batch_size = batch_size

    @staticmethod
    def _disable_ray_smp(pipe: Pipe) -> Pipe:
        """
        为单个 pipe 实例禁止 Ray/SMP 相关变体，避免影响同类其它实例。
        """
        spec = pipe.get_spec()
        blocked_variants = {
            PipeVariantType.RAY,
            PipeVariantType.SMP,
            PipeVariantType.TF_RAY,
            PipeVariantType.RAY_DS,
        }
        mutable_variants = [
            v for v in spec.mutable_variants if v not in blocked_variants
        ]
        fusable_source_variants = (
            [
                v
                for v in (spec.fusable_source_variants or [])
                if v != PipeVariantType.RAY_DS
            ]
            or None
        )
        pipe.pipe_spec = CedarPipeSpec(
            is_mutable=spec.mutable,
            mutable_variants=mutable_variants,
            is_fusable=spec.is_fusable,
            is_shardable=spec.is_shardable,
            is_fusable_source=spec.is_fusable_source,
            fusable_source_variants=fusable_source_variants,
        )
        return pipe

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, self.add_bos)
        fp = MapperPipe(fp, self.add_bos_1)
        fp = MapperPipe(fp, self.add_bos_2)
        fp = MapperPipe(fp, self.add_bos_3)
        # fp = MapperPipe(fp, self.add_bos_4)
        # fp = MapperPipe(fp, self.add_bos_5)
        # fp = MapperPipe(fp, self.add_bos_6)
        # fp = MapperPipe(fp, self.add_bos_7)
        # fp = MapperPipe(fp, self.add_bos_8)
        # fp = MapperPipe(fp, self.add_bos_9)
        # fp = MapperPipe(fp, self.add_bos_10)
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path(
        "wikitext-103/wiki.train.tokens"
    )

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = LocalLineSource(str(train_filepath))
    feature = Wikitext1033Feature(batch_size=spec.batch_size)
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
                enable_local_parallelism=not spec.disable_parallelism,
                enable_fusion=not spec.disable_fusion,
                enable_caching=not spec.disable_caching,
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
                reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
            ),
            generate_plan=spec.generate_plan,
        )

    return dataset


def main():
    logging.basicConfig(level=logging.INFO)
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    profiled_stats_path = (
        #repo_root / "cedar" / "compose" / "simple_five_ops_profile.yml"
        "/cedar/evaluation/failed_profiles/profile_case_0001.yml"
    )
    ds = get_dataset(CedarEvalSpec(1, None, 1, 
    run_profiling=False,
    use_ray=True,
    profiled_stats=str(profiled_stats_path),
    disable_offload=False,
    use_my_optimizer=1,
    disable_prefetch=True,
    disable_fusion=False,
    disable_caching=False,
    ))

    # i = 0
    # for f in ds:
    #     # print(f)
    #     print(f)
    #     print(f.size())
    #     if i == 10:
    #         break
    #     i += 1


if __name__ == "__main__":
    main()
