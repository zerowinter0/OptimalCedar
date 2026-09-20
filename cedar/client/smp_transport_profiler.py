"""Aggregate IPC calibration with real Cedar objects and independent queue pairs."""
import hashlib
import logging
import math
import multiprocessing as mp
import os
import pickle
import statistics
import time

from cedar.pipes.common import DataSample
from cedar.service.actor import SMPActor

logger = logging.getLogger(__name__)


class _ObjectOutputActor(SMPActor):
    def __init__(self, values):
        super().__init__("smp_transport_profile", disable_torch_parallelism=False)
        self.values = values

    def process(self, index):
        return self.values[index]


def _pair_driver(index, snapshots, window, duration, repeats, barrier, ready, results):
    requests = mp.Queue(maxsize=window)
    responses = mp.Queue(maxsize=window)
    actor = None
    try:
        values = [pickle.loads(value) for value in snapshots]
        actor = _ObjectOutputActor(values)
        actor.daemon = True
        actor.register(requests, responses)
        actor.start()
        samples = [DataSample(i) for i in range(len(values))]
        sizes = [len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                 + len(pickle.dumps(DataSample(value), protocol=pickle.HIGHEST_PROTOCOL))
                 for sample, value in zip(samples, values)]
        cursor = 0

        def exchange():
            nonlocal cursor
            size = 0
            for _ in range(window):
                pos = cursor % len(samples)
                requests.put(samples[pos], timeout=60)
                size += sizes[pos]
                cursor += 1
            for _ in range(window):
                value = responses.get(timeout=60)
                if not isinstance(value, DataSample):
                    raise RuntimeError("SMP actor returned an invalid calibration sample")
            return size

        for _ in range(3):
            exchange()
        ready.put(index)
        for repeat in range(repeats):
            barrier.wait(timeout=180)
            started = time.perf_counter()
            count = wire_bytes = 0
            while time.perf_counter() - started < duration:
                wire_bytes += exchange()
                count += window
            finished = time.perf_counter()
            results.put({"index": index, "repeat": repeat, "start": started,
                         "end": finished, "samples": count,
                         "serialized_bytes": wire_bytes})
            barrier.wait(timeout=180)
    except BaseException as exc:
        ready.put({"error": repr(exc)})
        results.put({"error": repr(exc)})
        barrier.abort()
        raise
    finally:
        if actor is not None:
            actor.stop()
            actor.join(timeout=3)
            if actor.is_alive():
                actor.terminate()
                actor.join(timeout=2)
        for queue in (requests, responses):
            queue.close()
            queue.cancel_join_thread()


def _measure(snapshots, pairs, window, duration, repeats):
    ready, results = mp.Queue(), mp.Queue()
    barrier = mp.Barrier(pairs + 1)
    drivers = [mp.Process(target=_pair_driver, args=(
        i, snapshots, window, duration, repeats, barrier, ready, results))
        for i in range(pairs)]
    try:
        for driver in drivers:
            driver.start()
        for _ in drivers:
            message = ready.get(timeout=180)
            if isinstance(message, dict):
                raise RuntimeError(message)
        runs = []
        for repeat in range(repeats):
            barrier.wait(timeout=180)
            rows = [results.get(timeout=180) for _ in drivers]
            if any("error" in row for row in rows):
                raise RuntimeError(rows)
            if any(row["repeat"] != repeat for row in rows):
                raise RuntimeError("Unaligned SMP calibration repeat")
            elapsed = max(row["end"] for row in rows) - min(row["start"] for row in rows)
            count = sum(row["samples"] for row in rows)
            byte_count = sum(row["serialized_bytes"] for row in rows)
            runs.append({"samples": count, "elapsed_sec": elapsed,
                         "serialized_bytes": byte_count,
                         "throughput_bytes_per_sec": byte_count / elapsed})
            barrier.wait(timeout=180)
        return {"workers": pairs, "throughput_bytes_per_sec": statistics.median(
            run["throughput_bytes_per_sec"] for run in runs), "runs": runs}
    finally:
        barrier.abort()
        for driver in drivers:
            driver.join(timeout=5)
            if driver.is_alive():
                driver.terminate()
                driver.join(timeout=2)
        for queue in (ready, results):
            queue.close()
            queue.cancel_join_thread()


def profile_smp_aggregate_transport(snapshots, workers=(1, 2, 4, 8, 16, 32),
                                    max_inflight=10, duration_sec=1.0, repeats=3):
    """Measure aggregate output transport; includes legal object reducers.

    An independent driver/process queue pair represents each Feature replica.
    Input requests are small indices; outputs replay a balanced mixture of legal
    boundary objects. This is effective IPC throughput, not raw memory bandwidth.
    All objects retain their types and sizes; no payload truncation is performed.
    """
    snapshots = list(snapshots)
    workers = tuple(sorted(set(workers)))
    cpu_budget = len(os.sched_getaffinity(0))
    if (not snapshots or not workers or any(w < 1 or 2*w > cpu_budget for w in workers)
            or max_inflight < 1 or repeats < 1
            or not math.isfinite(duration_sec) or duration_sec <= 0):
        raise ValueError("Invalid SMP aggregate calibration configuration")
    points = []
    for pairs in workers:
        point = _measure(snapshots, pairs, max_inflight, duration_sec, repeats)
        points.append(point)
        logger.info("SMP aggregate transport W=%s throughput=%s bytes/sec",
                    pairs, point["throughput_bytes_per_sec"])
    return {"schema_version": 1, "max_inflight": max_inflight,
            "method": "independent_Cedar_SMPActor_real_legal_boundary_objects",
            "timing_excludes_startup": True, "cpu_budget": cpu_budget,
            "duration_per_repeat_sec": duration_sec, "measured_repeats": repeats,
            "object_samples": len(snapshots),
            "snapshot_bytes": [len(value) for value in snapshots],
            "snapshot_sha256": [hashlib.sha256(value).hexdigest() for value in snapshots],
            "points": points}
