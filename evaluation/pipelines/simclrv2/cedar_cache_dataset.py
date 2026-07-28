import pathlib
import torch
import multiprocessing as mp
import logging

from typing import List

from torchvision import transforms
from torchvision.io import ImageReadMode

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
    BatcherPipe,
    ImageReaderPipe,
)
from cedar.sources import LocalFSSource

from evaluation.cedar_utils import CedarEvalSpec


DATASET_LOC = "datasets/imagenette2"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11


def to_float(x):
    return x.to(torch.float32)


class SimCLRV2Feature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = ImageReaderPipe(fp, mode=ImageReadMode.RGB, is_random=False).fix()
        fp = MapperPipe(fp, to_float, tag="float", is_random=False)
        fp = MapperPipe(
            fp,
            transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH)),
            tag="crop",
            is_random=True,
        )
        fp = MapperPipe(
            fp, transforms.RandomHorizontalFlip(), is_random=True
        ).depends_on(["crop"])
        fp = MapperPipe(
            fp,
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            tag="jitter",
            is_random=True,
        )
        fp = MapperPipe(
            fp, transforms.Grayscale(num_output_channels=1), is_random=False
        )
        fp = MapperPipe(
            fp,
            transforms.GaussianBlur(GAUSSIAN_BLUR_KERNEL_SIZE),
            is_random=False,
        )
        fp = MapperPipe(
            fp, transforms.Normalize((0.1307,), (0.3081,)), is_random=False
        ).depends_on(["float"])
        fp = BatcherPipe(fp, batch_size=self.batch_size, is_random=False).fix()
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
                num_samples=getattr(spec, "num_total_samples", None),
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
                reorder_timeout_sec=getattr(spec, "reorder_timeout_sec", None),
            ),
            generate_plan=spec.generate_plan,
        )
    return dataset

def main():
    logging.basicConfig(level=logging.INFO)
    ds = get_dataset(CedarEvalSpec(1, None, 1, 
    run_profiling=True,
    use_ray=True,
    profiled_stats="/tmp/sim_hash_profile.yml",
    disable_offload=False,
    use_my_optimizer=0,
    disable_prefetch=True,
    disable_fusion=False,
    disable_caching=False,
    ))

    i = 0
    for f in ds:
        # print(f)
        print(f)
        print(f.size())
        if i == 10:
            break
        i += 1


if __name__ == "__main__":
    main()
