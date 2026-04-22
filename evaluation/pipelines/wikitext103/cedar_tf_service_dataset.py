import pathlib
import torch  # noqa: F401

import tensorflow as tf
import multiprocessing as mp

from typing import List

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
    TFOutputHint,
    TFTensorDontCare,
)
from cedar.sources import TFLocalLineSource

from evaluation.cedar_utils import CedarEvalSpec
from cedar.pipes.custom.wikitext103_tf_service import (
    _tokenize,
    _truncate,
    _embedding,
)


DATASET_LOC = "datasets/wikitext103"


class Wikitext103Feature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        source_pipes[0].set_output_tf_spec(
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
        fp = MapperPipe(
            fp,
            _tokenize,
            output_tf_hint=TFOutputHint([TFTensorDontCare()], dtype=tf.int32),
        ).fix()
        fp = MapperPipe(
            fp,
            _truncate,
            output_tf_hint=TFOutputHint(
                TFTensorDontCare(), TFTensorDontCare()
            ),
        ).fix()
        fp = MapperPipe(
            fp,
            _embedding,
            output_tf_hint=TFOutputHint(
                [TFTensorDontCare(), 764], TFTensorDontCare()
            ),
        ).fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path(
        "wikitext-103/wiki.train.tokens"
    )
    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = TFLocalLineSource(str(train_filepath))
    feature = Wikitext103Feature(batch_size=spec.batch_size)
    feature.apply(source)

    if spec.config:
        dataset = DataSet(
            ctx,
            {"feature": feature},
            feature_config=spec.config,
            enable_controller=False,
            enable_optimizer=False,
            prefetch=False,
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
                use_my_optimizer=getattr(spec, "use_my_optimizer", False),
            ),
            generate_plan=spec.generate_plan,
        )
    return dataset


if __name__ == "__main__":
    ds = get_dataset(CedarEvalSpec(1, None, 1))

    for i, x in enumerate(ds):
        print(x)
        print(tf.TensorSpec.from_tensor(x))
        if i == 10:
            break
