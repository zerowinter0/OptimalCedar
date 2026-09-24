"""Build the semantics-preserving plan set: only ``to_float`` moves.

The SimCLRv2 augmentation applies crop -> flip -> jitter -> grayscale -> blur
to every image.  ``to_float`` is a cast; moving it does not change which
augmentation runs in which order, only the element type the transforms compute
in (float32 when the cast runs first, uint8 when it runs last).  Every such
plan is a legal plan for the system (the only edges are reader-first,
flip-after-crop, normalize-after-float, batcher-last) and it is the cleanest
possible controlled pair for the cost model: identical work, identical byte
*content*, different byte *volume*.

Usage (inside the container):
  python -u tmp_analysis/build_float_position_orders.py
"""

import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
PLANS = ROOT / "outputs/unopt_order_transfer_repeats_traceall_20260921/plans"
TARGET = ROOT / "tmp_analysis/semantic_orders"

# pipe ids: 9 lister, 8 reader, 7 to_float, 6 crop, 5 flip, 4 jitter,
# 3 grayscale, 2 blur, 1 normalize, 0 batcher
IMAGE_CHAIN = [6, 5, 4, 3, 2]


def orders():
    for position in range(len(IMAGE_CHAIN) + 1):
        chain = (
            IMAGE_CHAIN[:position] + [7] + IMAGE_CHAIN[position:]
        )
        name = f"a_f{position}"
        yield name, [9, 8] + chain + [1, 0]


def main() -> int:
    template = yaml.safe_load((PLANS / "declared.yaml").read_text())
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, order in orders():
        plan = yaml.safe_load(yaml.safe_dump(template))
        payload = plan["physical_plan"]
        graph = {}
        for index, p_id in enumerate(order):
            successor = order[index + 1] if index + 1 < len(order) else ""
            graph[str(p_id)] = str(successor) if successor != "" else ""
        payload["graph"] = graph
        target = TARGET / f"{name}.yaml"
        target.write_text(yaml.safe_dump({"physical_plan": payload}))
        print(f"{name}: {' '.join(str(p) for p in order)} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
