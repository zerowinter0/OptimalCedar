import numpy as np
import random
import librosa

SAMPLE_FREQ = 8000
N_FFT = 400
FREQ_MASK_PARAM = 80
TIME_MASK_PARAM = 80
N_MELS = 256


def time_mask(x):
    if isinstance(x, dict):
        x = x["item"].copy()
        ray_ds = True
    else:
        x = x.copy()
        ray_ds = False
    t = np.random.uniform(low=0.0, high=TIME_MASK_PARAM)
    t = int(t)
    tau = x.shape[1]
    rand_int = max(0, tau - t)
    t0 = random.randint(0, rand_int)
    x[:, t0 : t0 + t] = 0  # noqa: E203
    if ray_ds:
        return {"item": x}
    else:
        return x


def frequency_mask(x):
    if isinstance(x, dict):
        x = x["item"].copy()
        ray_ds = True
    else:
        x = x.copy()
        ray_ds = False
    f = np.random.uniform(low=0.0, high=FREQ_MASK_PARAM)
    f = int(f)
    v = x.shape[0]
    f0 = random.randint(0, v - f)
    x[f0 : f0 + f, :] = 0  # noqa: E203
    if ray_ds:
        return {"item": x}
    else:
        return x


def mel(x):
    if isinstance(x, dict):
        return {
            "item": librosa.feature.melspectrogram(
                S=x["item"], sr=SAMPLE_FREQ, n_mels=N_MELS, n_fft=N_FFT
            )
        }
    else:
        return librosa.feature.melspectrogram(
            S=x, sr=SAMPLE_FREQ, n_mels=N_MELS, n_fft=N_FFT
        )


def _read(x):
    if isinstance(x, dict):
        return {"item": librosa.load(x["item"])}
    else:
        return librosa.load(x)


def _resample(audio, orig_sr=None):
    """Resample Cedar tuple data and Ray dictionary data consistently."""
    if isinstance(audio, dict):
        data = audio["item"]
        return {
            "item": librosa.resample(
                y=data[0], orig_sr=data[1], target_sr=SAMPLE_FREQ
            )
        }

    # MapperPipe expands tuple-valued DataSample.data; direct callers may
    # still provide the original (audio, sample_rate) tuple.
    if orig_sr is None:
        audio, orig_sr = audio
    return librosa.resample(
        y=audio, orig_sr=orig_sr, target_sr=SAMPLE_FREQ
    )


def _spec(x):
    if isinstance(x, dict):
        return {"item": np.abs(librosa.stft(x["item"], n_fft=N_FFT)) ** 2}
    else:
        return np.abs(librosa.stft(x, n_fft=N_FFT)) ** 2


def _stretch(x):
    if isinstance(x, dict):
        return {
            "item": librosa.effects.time_stretch(
                x["item"], rate=0.8, n_fft=N_FFT
            )
        }
    else:
        return librosa.effects.time_stretch(x, rate=0.8, n_fft=N_FFT)
