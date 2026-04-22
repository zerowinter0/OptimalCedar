import pathlib
import torch
import multiprocessing as mp

from typing import List

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
)
from cedar.sources import COCOSource
from torchvision.transforms import v2

from evaluation.cedar_utils import CedarEvalSpec

DATASET_LOC = "datasets/coco"


def to_float(x):
    return x.to(torch.float32)


def to_tensor(x):
    x["image"] = v2.ToTensor()(x["image"])
    return x


def distort(x):
    x["image"] = v2.RandomPhotometricDistort(p=1)(x["image"])
    return x


def zoom_out(x):
    x["image"], x["boxes"] = v2.RandomZoomOut(fill=[123.0, 117.0, 104.0], p=1)(
        x["image"], x["boxes"]
    )
    return x


def crop(x):
    x["image"], x["boxes"] = v2.RandomIoUCrop()(x["image"], x["boxes"])
    return x


class COCOFeature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, zoom_out, tag="zoom")
        fp = MapperPipe(fp, crop, tag="crop").depends_on(["zoom"])
        fp = MapperPipe(
            fp, v2.SanitizeBoundingBox(labels_getter="boxes"), tag="sanitize"
        ).depends_on(["crop"])
        fp = MapperPipe(fp, v2.RandomHorizontalFlip(p=1)).depends_on(
            ["sanitize"]
        )
        fp = MapperPipe(fp, distort)
        fp = MapperPipe(fp, to_tensor)
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = COCOSource(str(data_dir))
    feature = COCOFeature(batch_size=spec.batch_size)
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
                use_my_optimizer=getattr(spec, "use_my_optimizer", False),
            ),
            generate_plan=spec.generate_plan,
        )
    return dataset


def main():
    dataset = get_dataset(CedarEvalSpec(1, None, 1))
    dataset.save_config("/tmp/")
    for x in dataset:
        print(x)
        break


if __name__ == "__main__":
    main()
