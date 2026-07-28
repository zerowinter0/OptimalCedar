import math
import json
import threading
import time

from evaluation.compare_optimizer_perf import (
    _average_repeat_results,
    _wait_for_object_disk_cache_complete,
)


def test_average_repeat_results_preserves_repeated_timeout():
    repeats = [
        {
            "optimizer": "optimizer",
            "setup_time_sec": 300.0,
            "total_time_sec": float("inf"),
            "perf_time_sec": float("inf"),
            "num_samples": 0,
            "throughput_samples_per_sec": 0.0,
            "time_per_sample_us": float("inf"),
            "timed_out": True,
        }
        for _ in range(3)
    ]

    result = _average_repeat_results("optimizer", repeats)

    assert math.isinf(result["total_time_sec"])
    timeout_stats = result["statistics"]["total_time_sec"]
    assert timeout_stats["n"] == 3
    assert timeout_stats["finite_n"] == 0
    assert timeout_stats["nonfinite_n"] == 3
    assert math.isinf(timeout_stats["mean"])
    assert timeout_stats["stddev"] is None
    assert timeout_stats["ci95_low"] is None
    assert timeout_stats["ci95_high"] is None


def test_wait_for_object_disk_cache_complete_handles_async_worker_commit(
    tmp_path, monkeypatch
):
    namespace = "cache-race-regression"
    monkeypatch.setenv("CEDAR_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("CEDAR_CACHE_NAMESPACE", namespace)

    class FakeMpDataset:
        _iter_mode = "mp"
        features = {"feature_r0": object(), "feature_r1": object()}

    def commit_manifests():
        time.sleep(0.05)
        for shard in ("feature_r0", "feature_r1"):
            shard_dir = tmp_path / namespace / shard
            shard_dir.mkdir(parents=True)
            (shard_dir / ".manifest.json").write_text(
                json.dumps({"complete": True})
            )

    committer = threading.Thread(target=commit_manifests)
    committer.start()
    try:
        assert _wait_for_object_disk_cache_complete(
            FakeMpDataset(), timeout_sec=1.0, poll_interval_sec=0.01
        )
    finally:
        committer.join()
