"""Unit-level check of the representation-aware profiler on captured payloads.

Feeds the profiler's reservoir with the real payloads of a captured declared
run and runs only the new fitting block, so the class construction, the
transition table and the per-class curves can be inspected without a full
profile pass.

Usage (inside the container):
  python -u tmp_analysis/test_compute_model_profiler.py
"""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.client.dataset import DataSet  # noqa: E402
from cedar.pipes.common import ProfileInputReservoir  # noqa: E402
from block_mechanism_common import build_feature  # noqa: E402

# The profiling driver and every local worker run single-threaded; without this
# the fit measures a multithreaded kernel and comes out several times too cheap.
from cedar.utils.threading import limit_native_threadpools  # noqa: E402

_THREAD_LIMITER = limit_native_threadpools(1)  # noqa: F841
import torch  # noqa: E402

torch.set_num_threads(1)

CAPTURE = ROOT / "tmp_analysis/capture_declared/capture"
# Pipe p's input in the declared order is pipe p-1's output for this chain.
PRODUCER_OF = {7: 8, 6: 7, 5: 6, 4: 5, 3: 4, 2: 3, 1: 2, 0: 1}


def main() -> int:
    feature = build_feature(batch_size=4)
    reservoir = ProfileInputReservoir()
    for consumer, producer in PRODUCER_OF.items():
        for path in sorted(CAPTURE.glob(f"*_pipe_{consumer}.pkl"))[:1]:
            try:
                blob = pickle.loads(path.read_bytes())
            except Exception:  # noqa: BLE001
                continue
            for snapshot in blob["snapshots"][:2]:
                reservoir.capture(producer, pickle.loads(snapshot))
    print("reservoir pipe payload counts:", reservoir.metadata()["samples_per_pipe"])

    profile = {}
    dataset = object.__new__(DataSet)
    DataSet._profile_operator_compute_model(dataset, profile, feature, reservoir)
    model = profile.get("physical_model", {}).get("compute_model")
    if not model:
        print("FAIL: no compute_model section")
        return 1
    print("statistic:", model["statistic"])
    for p_id, entry in sorted(model["operators"].items(), key=lambda kv: int(kv[0])):
        classes = list(entry["by_class"])
        detail = {
            klass: (
                round(fit["k_ms_per_element"], 3e-9 and 9),
                round(fit["b_ms"], 4),
            )
            for klass, fit in entry["by_class"].items()
        }
        print(
            f"  pipe {p_id}: own={entry.get('own_class')} "
            f"elements={entry.get('own_elements')} classes={classes}"
        )
        for klass, fit in sorted(detail.items()):
            print(
                f"      {klass:14s} k={fit[0]:.3e} b={fit[1]:.4f} "
                f"points={[(round(x), round(y, 3)) for x, y in entry['by_class'][klass]['points_ms_per_element']]}"
            )
    print("transitions:", json.dumps(model["class_transition"], sort_keys=True))
    print("element ratios:", json.dumps(model["element_ratio"], sort_keys=True))
    print("measured classes:", json.dumps(model["measured_classes"], sort_keys=True))
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/test_compute_model_profile.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=1, default=float))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
