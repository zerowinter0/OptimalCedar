import pathlib
import matplotlib.pyplot as plt
import torch
import librosa
import numpy as np
import multiprocessing as mp

from typing import List

from cedar.client import DataSet
from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions
from cedar.pipes import (
    Pipe,
    MapperPipe,
)
from cedar.sources import LocalFSSource
from cedar.pipes.custom.commonvoice import (
    time_mask,
    _read,
    _resample,
    _spec,
    _stretch,
    frequency_mask,
    mel,
    SAMPLE_FREQ,
)

from evaluation.cedar_utils import CedarEvalSpec


DATASET_LOC = "datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips/"


class CommonvoiceFeature(Feature):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        fp = MapperPipe(fp, _read).fix()
        fp = MapperPipe(fp, _resample).fix()
        fp = MapperPipe(fp, _spec).fix()
        fp = MapperPipe(fp, _stretch)
        fp = MapperPipe(fp, time_mask)
        fp = MapperPipe(fp, frequency_mask)
        fp = MapperPipe(fp, mel).fix()
        return fp


def get_dataset(spec: CedarEvalSpec) -> DataSet:
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )

    ctx = CedarContext(ray_config=spec.to_ray_config())
    source = LocalFSSource(str(data_dir), recursive=True)
    feature = CommonvoiceFeature(batch_size=spec.batch_size)
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
                use_my_optimizer=getattr(spec, "use_my_optimizer", 0),
            ),
            generate_plan=spec.generate_plan,
        )
    return dataset


if __name__ == "__main__":
    dataset = get_dataset(CedarEvalSpec(1, None, 1))
    for x in dataset:
        print(x)
        print(torch.Tensor(x).size())

        fig, ax = plt.subplots()
        D = librosa.power_to_db(x, ref=np.max)
        img = librosa.display.specshow(
            D, y_axis="mel", x_axis="time", sr=SAMPLE_FREQ, ax=ax
        )
        fig.savefig("tmp2.png")
        break
