import pathlib
import torch  # noqa: F401
import multiprocessing as mp

import tensorflow as tf

from typing import List

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
    BatcherPipe,
    TFOutputHint,
    TFTensorDontCare,
)
from cedar.sources import LocalFSSource
from cedar.pipes.custom.simclrv2 import (
    decode_jpeg,
    crop_and_resize,
    convert_to_float,
    random_flip,
    color_jitter,
    gaussian_blur,
)

from evaluation.cedar_utils import CedarEvalSpec


DATASET_LOC = "datasets/imagenette2"


class SimCLRV2Feature(Feature):
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
            tf.io.read_file,
            output_tf_hint=TFOutputHint(
                TFTensorDontCare(), TFTensorDontCare()
            ),
        ).fix()
        fp = MapperPipe(
            fp,
            decode_jpeg,
            output_tf_hint=TFOutputHint([1, None, None, 3], tf.int8),
        ).fix()
        fp = MapperPipe(
            fp,
            convert_to_float,
            output_tf_hint=TFOutputHint(TFTensorDontCare(), tf.float32),
            tag="float",
        )
        fp = MapperPipe(
            fp,
            crop_and_resize,
            output_tf_hint=TFOutputHint(
                shape=[TFTensorDontCare(), 244, 244, TFTensorDontCare()],
                dtype=TFTensorDontCare(),
            ),
            tag="crop",
        )
        fp = MapperPipe(
            fp,
            random_flip,
            output_tf_hint=TFOutputHint(
                TFTensorDontCare(), TFTensorDontCare()
            ),
        ).depends_on(["crop"])
        fp = MapperPipe(
            fp,
            color_jitter,
            output_tf_hint=TFOutputHint(
                TFTensorDontCare(), TFTensorDontCare()
            ),
        )
        fp = MapperPipe(
            fp,
            tf.image.rgb_to_grayscale,
            output_tf_hint=TFOutputHint(
                shape=[
                    TFTensorDontCare(),
                    TFTensorDontCare(),
                    TFTensorDontCare(),
                    1,
                ],
                dtype=TFTensorDontCare(),
            ),
        )
        fp = MapperPipe(
            fp,
            gaussian_blur,
            output_tf_hint=TFOutputHint(
                TFTensorDontCare(), TFTensorDontCare()
            ),
        )
        fp = MapperPipe(
            fp,
            tf.image.per_image_standardization,
            output_tf_hint=TFOutputHint(
                TFTensorDontCare(), TFTensorDontCare()
            ),
        ).depends_on(["float"])
        fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )

    train_filepath = pathlib.Path(data_dir) / pathlib.Path("imagenette2/train")

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = LocalFSSource(str(train_filepath), recursive=True)
    feature = SimCLRV2Feature(batch_size=spec.batch_size)
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
    ds = get_dataset(
        CedarEvalSpec(
            1, None, 1, disable_optimizer=True, disable_controller=True
        )
    )

    for x in ds:
        print(tf.TensorSpec.from_tensor(x))
        print(tf.size(x))
        byte_tensor = tf.io.decode_raw(x, tf.uint8)
        size = tf.size(byte_tensor, out_type=tf.int32)
        print(size)
        break
