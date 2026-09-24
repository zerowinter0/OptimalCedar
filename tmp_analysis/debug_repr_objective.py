"""Why does the DP's chosen order differ from the brute-force argmin?

Prints, for two orders, the per-operator price computed by the DP's own
``_repr_compute_cost`` and by the independent reference used in the acceptance
test, so a mismatch can be attributed to one of the two.

Usage (inside the container):
  python -u tmp_analysis/debug_repr_objective.py <profile.yaml>
"""

import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import build_feature  # noqa: E402
from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpAffineReprOptimizer,
)


def main() -> int:
    profile_path = Path(sys.argv[1])
    profile = yaml.safe_load(profile_path.read_text())
    model = profile["physical_model"]["compute_model"]
    feature = build_feature(batch_size=4)
    optimizer = SimpleDpAffineReprOptimizer()
    feature.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=1,
        enable_offload=False,
        enable_reorder=True,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=False,
        num_samples=0,
        use_my_optimizer=0,
        reorder_timeout_sec=3600.0,
    )
    optimizer.run(str(profile_path), options)
    inner = list(optimizer._dp_inner_ops)
    print("inner ops:", inner)
    element_prod, class_state = optimizer._repr_tables()
    index_of = {int(p_id): index for index, p_id in enumerate(inner)}

    def dp_total(order):
        mask = 0
        total = 0.0
        rows = []
        for p_id in order:
            index = index_of[int(p_id)]
            value = optimizer._repr_compute_cost(
                int(p_id), element_prod[mask], class_state[mask]
            )
            rows.append(
                (int(p_id), class_state[mask], round(element_prod[mask]), round(value, 4))
            )
            total += value
            mask |= 1 << index
        return total, rows

    ratios = model["element_ratio"]
    transitions = model["class_transition"]
    elements = float(model["source_elements"])
    klass = str(model["source_class"])
    reader = int(model.get("reader_pipe", 8))
    elements *= float(ratios.get(str(reader), 1.0))
    klass = str(transitions.get(str(reader), {}).get(klass, klass))

    def ref_total(order):
        nonlocal elements, klass
        elements = float(model["source_elements"]) * float(
            ratios.get(str(reader), 1.0)
        )
        klass = str(
            transitions.get(str(reader), {}).get(str(model["source_class"]), str(model["source_class"]))
        )
        total = 0.0
        rows = []
        for p_id in order:
            curve = model["operators"][str(p_id)]["by_class"][klass]
            value = curve["k_ms_per_element"] * elements + curve["b_ms"]
            rows.append((p_id, klass, round(elements), round(value, 4)))
            total += value
            elements *= float(ratios.get(str(p_id), 1.0))
            klass = str(transitions.get(str(p_id), {}).get(klass, klass))
        return total, rows

    for order in ([6, 3, 7, 5, 1, 4, 2], [3, 6, 2, 5, 7, 1, 4]):
        ref, rows = ref_total(order)
        dp_value, dp_rows = dp_total([reader] + list(order))
        print(f"\norder {order}  (reader {reader} first)")
        print(f"  DP function total = {dp_value:.4f}")
        print(f"  reference total   = {ref:.4f}")
        for row in dp_rows:
            print(f"    DP   op {row[0]}: class={row[1]} el={row[2]} price={row[3]}")
        for row in rows:
            print(f"    REF  op {row[0]}: class={row[1]} el={row[2]} price={row[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
