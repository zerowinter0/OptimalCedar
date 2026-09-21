"""Is the ImageReader's traced output size a sampling artifact?

The reader reports its output size as the decoded pixel bytes (H x W x 3 for
uint8 RGB), so the number a run reports is the mean over the records the
trace sampler happened to pick -- roughly every ``rate * TRACE_FREQUENCY_SEC``
-th record of each worker's shard.  This script measures the population of
imagenette2/train and replays that stride sampler for the rates of the
order-transfer cells.

Usage (inside the container):
  python -u tmp_analysis/probe_reader_size_sampling.py
"""

import pathlib
import statistics
import sys

import PIL.Image

ROOT = pathlib.Path("/workspace/OptimalCedar")
TRAIN = ROOT / "evaluation/datasets/imagenette2/imagenette2/train"
WORKERS = 4


def decoded_bytes(path: pathlib.Path) -> int:
    with PIL.Image.open(path) as image:
        width, height = image.size
        bands = len(image.getbands())
    return width * height * bands


def stride_sample(sizes, stride: int, worker: int, workers: int = WORKERS):
    return sizes[worker::workers][::stride]


def main() -> int:
    files = sorted(
        p
        for p in TRAIN.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )
    sizes = [decoded_bytes(p) for p in files]
    mean = statistics.mean(sizes)
    stdev = statistics.stdev(sizes)
    print(f"files              : {len(files)}")
    print(f"population mean    : {mean / 1000:.1f} kB   stdev {stdev / 1000:.1f} kB "
          f"(CV {stdev / mean:.2f})")
    print(f"min / max          : {min(sizes) / 1000:.1f} / {max(sizes) / 1000:.1f} kB")
    print()
    # Per-worker shard, then the time-based sampler inside each worker.
    for stride in (17, 28):
        sample = []
        for worker in range(WORKERS):
            sample.extend(stride_sample(sizes, stride, worker))
        sample_mean = statistics.mean(sample)
        se = stdev / len(sample) ** 0.5
        print(
            f"stride {stride:>2} (≈{stride / 0.1:>6.0f} rec/s): n={len(sample):4d}  "
            f"mean {sample_mean / 1000:.1f} kB  ({sample_mean / mean - 1:+.1%} vs population, "
            f"SE {se / 1000:.1f} kB)"
        )
    print()
    print("reference: the order-transfer cells reported declared 646.6 kB, pico 678.8 kB,")
    print("           cedar 629.7 kB, old-dp 564.3 kB (means of 3 repeats).")
    print(f"one-sample SE at n=440: {stdev / 440 ** 0.5 / 1000:.1f} kB "
          f"({stdev / 440 ** 0.5 / mean:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
