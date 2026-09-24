"""Materialise two reordered plans that the diagnosis never used.

The declared SimCLRv2 feature constrains only these edges: the reader precedes
everything, ``RandomHorizontalFlip`` follows ``crop``, and ``Normalize``
follows ``to_float``; the batcher is the sink.  Every linear extension of that
DAG is a legal plan for the system.  The diagnosis used declared / PICO /
cedar / old-dp; the two orders below are different linear extensions, chosen so
that they separate the dtype question from the channel question:

  v1  to_float stays where the image ops cannot see it: the same relative
      order as the declared plan, but every image transform runs on the
      reader's uint8 payload (dtype change only).
  v2  grayscale runs on the full-resolution image before crop, and to_float
      stays early, so blur and jitter see single-channel float32 input
      (channel change with the declared dtype).

Usage (inside the container):
  python -u tmp_analysis/build_validation_orders.py
"""

import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
PLANS = ROOT / "outputs/unopt_order_transfer_repeats_traceall_20260921/plans"
TARGET = ROOT / "tmp_analysis/validation_orders"

ORDERS = {
    # pipe ids: 9 lister, 8 reader, 7 to_float, 6 crop, 5 flip, 4 jitter,
    # 3 grayscale, 2 blur, 1 normalize, 0 batcher
    "v1": [9, 8, 6, 5, 4, 3, 2, 7, 1, 0],
    "v2": [9, 8, 7, 3, 6, 5, 4, 2, 1, 0],
}


def main() -> int:
    template = yaml.safe_load((PLANS / "declared.yaml").read_text())
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, order in ORDERS.items():
        plan = yaml.safe_load(yaml.safe_dump(template))
        payload = plan["physical_plan"]
        graph = {}
        for index, pid in enumerate(order):
            successor = order[index + 1] if index + 1 < len(order) else ""
            graph[str(pid)] = str(successor) if successor != "" else ""
        payload["graph"] = graph
        target = TARGET / f"{name}.yaml"
        target.write_text(yaml.safe_dump({"physical_plan": payload}))
        print(f"{name}: {' -> '.join(str(pid) for pid in order)}  -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
