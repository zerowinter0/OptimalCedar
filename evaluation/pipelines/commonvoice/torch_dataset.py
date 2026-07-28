import pathlib
import torch
import torchdata.datapipes as dp
import librosa
import numpy as np
import random

from evaluation.torch_utils import TorchEvalSpec
from torch.utils.data import DataLoader

DATASET_LOC = "datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips/"
SAMPLE_FREQ = 8000
N_FFT = 400
FREQ_MASK_PARAM = 80
TIME_MASK_PARAM = 80
N_MELS = 256


def to_float(x):
    return x.to(torch.float32)


def time_mask(x):
    t = np.random.uniform(low=0.0, high=TIME_MASK_PARAM)
    t = int(t)
    tau = x.shape[1]
    t0 = random.randint(0, tau - t)
    x[:, t0 : t0 + t] = 0
    return x


def frequency_mask(x):
    f = np.random.uniform(low=0.0, high=FREQ_MASK_PARAM)
    f = int(f)
    v = x.shape[0]
    f0 = random.randint(0, v - f)
    x[f0 : f0 + f, :] = 0
    return x


def mel(x):
    return librosa.feature.melspectrogram(
        S=x, sr=SAMPLE_FREQ, n_mels=N_MELS, n_fft=N_FFT
    )


def build_datapipe(root, spec: TorchEvalSpec):
    datapipe = dp.iter.FileLister(root=root, recursive=True)
    # TODO: Evaluate where is a fair place to put this...
    datapipe = datapipe.sharding_filter()
    datapipe = dp.iter.Mapper(datapipe, lambda x: librosa.load(x))
    datapipe = dp.iter.Mapper(
        datapipe,
        lambda x: librosa.resample(
            y=x[0], orig_sr=x[1], target_sr=SAMPLE_FREQ
        ),
    )
    datapipe = dp.iter.Mapper(
        datapipe,
        lambda x: np.abs(librosa.stft(x, n_fft=N_FFT)) ** 2,
    )
    datapipe = dp.iter.Mapper(
        datapipe,
        lambda x: librosa.effects.time_stretch(x, rate=0.8, n_fft=N_FFT),
    )
    datapipe = dp.iter.Mapper(datapipe, time_mask)
    datapipe = dp.iter.Mapper(datapipe, frequency_mask)
    datapipe = dp.iter.Mapper(datapipe, mel)
    return datapipe


def get_dataset(spec: TorchEvalSpec):
    data_dir = spec.kwargs.get("dataset_path")
    if not data_dir:
        data_dir = (
            pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
        )

    datapipe = build_datapipe(str(data_dir), spec)

    dataloader = DataLoader(datapipe, num_workers=spec.num_workers)

    return dataloader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = get_dataset(TorchEvalSpec(8, 1))
    for x in dataset:
        print(x)
        print(x.size())

        fig, ax = plt.subplots()
        D = librosa.power_to_db(x.squeeze(0).numpy(), ref=np.max)
        img = librosa.display.specshow(
            D, y_axis="mel", x_axis="time", sr=SAMPLE_FREQ, ax=ax
        )
        fig.savefig("tmp.png")
        break
