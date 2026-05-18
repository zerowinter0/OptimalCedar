import pathlib
import torch  # noqa: F401
import multiprocessing as mp

import tensorflow as tf
from tensorflow.python.ops import math_ops

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
from cedar.sources import COCOFileSource

from evaluation.cedar_utils import CedarEvalSpec

from cedar.pipes.custom.coco_tf import (
    read_image,
    distorted_bounding_box_crop,
    resize_image,
    random_flip,
    distort,
    normalize,
)


DATASET_LOC = "datasets/coco"


class COCOFeature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        source_pipes[0].set_output_tf_spec(
            (
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            )
        )
        fp = MapperPipe(
            fp,
            read_image,
            output_tf_hint=(
                TFOutputHint([None, None, 3], tf.float32),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            ),
        ).fix()
        fp = MapperPipe(
            fp,
            distorted_bounding_box_crop,
            output_tf_hint=(
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            ),
            tag="crop",
        )
        fp = MapperPipe(
            fp,
            resize_image,
            output_tf_hint=(
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            ),
            tag="resize",
        ).depends_on(["crop"])
        fp = MapperPipe(
            fp,
            random_flip,
            output_tf_hint=(
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            ),
            tag="flip",
        ).depends_on(["resize"])
        fp = MapperPipe(
            fp,
            distort,
            output_tf_hint=(
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            ),
        )
        fp = MapperPipe(
            fp,
            normalize,
            output_tf_hint=(
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
                TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            ),
            tag="normalize",
        ).fix()

        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = COCOFileSource(str(data_dir), tf_source=True)
    feature = COCOFeature(batch_size=spec.batch_size)
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
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
                reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
            ),
            generate_plan=spec.generate_plan,
        )
    return dataset


if __name__ == "__main__":
    ds = get_dataset(
        CedarEvalSpec(
            1, None, 1, disable_optimizer=True, disable_controller=True
        )
    )
    ds.save_config("/tmp/")

    for x in ds:
        print(x)
        break
