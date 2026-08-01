import pathlib
import librosa
import numpy as np
import glob
import math
import random

import ray
import time

DATASET_LOC = "datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips/"
SAMPLE_FREQ = 8000
N_FFT = 400
FREQ_MASK_PARAM = 80
TIME_MASK_PARAM = 80
N_MELS = 256

WARMUP_SAMPLES = 1


def _read(x):
    return {"item": librosa.load(x["item"])}


def _resample(x):
    data = x["item"]
    return {
        "item": librosa.resample(
            y=data[0], orig_sr=data[1], target_sr=SAMPLE_FREQ
        )
    }


def _spec(x):
    return {"item": np.abs(librosa.stft(x["item"], n_fft=N_FFT)) ** 2}


def _stretch(x):
    return {
        "item": librosa.effects.time_stretch(x["item"], rate=0.8, n_fft=N_FFT)
    }


def mel(x):
    return {
        "item": librosa.feature.melspectrogram(
            S=x["item"], sr=SAMPLE_FREQ, n_mels=N_MELS, n_fft=N_FFT
        )
    }


def time_mask(x):
    x = x["item"].copy()
    t = np.random.uniform(low=0.0, high=TIME_MASK_PARAM)
    t = int(t)
    tau = x.shape[1]
    rand_int = max(0, tau - t)
    t0 = random.randint(0, rand_int)
    x[:, t0 : t0 + t] = 0  # noqa: E203
    return {"item": x}


def frequency_mask(x):
    x = x["item"].copy()
    f = np.random.uniform(low=0.0, high=FREQ_MASK_PARAM)
    f = int(f)
    v = x.shape[0]
    f0 = random.randint(0, v - f)
    x[f0 : f0 + f, :] = 0  # noqa: E203
    return {"item": x}


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def delta(self):
        if self._start is None or self._end is None:
            raise RuntimeError()
        return self._end - self._start


def build_ds(root, num_total_samples=None):
    files = sorted(glob.glob(f"{str(root)}/**/*.mp3", recursive=True))
    if not files:
        raise FileNotFoundError(f"No MP3 files found under {root}")
    if num_total_samples is not None:
        repeats = max(1, math.ceil(num_total_samples / len(files)))
        files = (files * repeats)[:num_total_samples]
    # Get list of dirs
    ds = ray.data.from_items(files)
    ds = ds.map(_read)
    ds = ds.map(_resample)
    ds = ds.map(_spec)
    ds = ds.map(_stretch)
    ds = ds.map(time_mask)
    ds = ds.map(frequency_mask)
    ds = ds.map(mel)

    return ds


def get_dataset(spec=None):
    data_dir = None if spec is None else spec.kwargs.get("dataset_path")
    if not data_dir:
        data_dir = (
            pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
        )
    num_total_samples = (
        None if spec is None else spec.num_total_samples
    )
    ds = build_ds(str(data_dir), num_total_samples=num_total_samples)

    return ds


if __name__ == "__main__":
    ds = get_dataset()

    # for row in ds.iter_rows():
    #     print(row)
    #     break
    timer = Timer()
    with timer:
        for idx, row in enumerate(ds.iter_rows()):
            if idx == WARMUP_SAMPLES:
                timer.reset()
            if idx == WARMUP_SAMPLES + 10000:
                break

            # if idx % 1000 == 0:
            #     print(idx)

    print(timer.delta())
