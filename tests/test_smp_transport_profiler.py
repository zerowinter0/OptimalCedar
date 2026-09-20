import pickle
import numpy as np
from cedar.client.smp_transport_profiler import profile_smp_aggregate_transport

def test_real_legal_objects_use_actual_smp_actor_queues():
    snapshots = [pickle.dumps(np.zeros(4096, dtype=np.float32)),
                 pickle.dumps({"text": "sample text", "tokens": [1, 2, 3]})]
    curve = profile_smp_aggregate_transport(
        snapshots, workers=(1, 2), max_inflight=3,
        duration_sec=0.1, repeats=2)
    assert curve["max_inflight"] == 3
    assert curve["timing_excludes_startup"]
    assert [p["workers"] for p in curve["points"]] == [1, 2]
    for point in curve["points"]:
        assert point["throughput_bytes_per_sec"] > 0
        assert len(point["runs"]) == 2
        assert all(r["samples"] > 0 and r["serialized_bytes"] > 0
                   for r in point["runs"])
