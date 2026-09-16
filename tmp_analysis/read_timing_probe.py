"""Time the audio decode locally and on the remote node."""

import statistics
import sys
import time

import librosa

path = (
    "/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08"
    "/en/clips/common_voice_en_38024625.mp3"
)
samples = [path]
times = []
for _ in range(5):
    start = time.perf_counter()
    librosa.load(path)
    times.append((time.perf_counter() - start) * 1000.0)
print(f"local decode median={statistics.median(times):.3f} ms  all={[round(t,2) for t in times]}")
