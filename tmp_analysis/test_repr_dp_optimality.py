"""Acceptance test: the representation-aware DP matches brute force.

With fusion, offload and parallelism disabled, the only decision left is the
operator order, so every legal linear extension of the feature DAG can be
enumerated and scored with an independent reference implementation of the new
objective (elements inside a representation class).  The DP must return one of
the argmins.

Usage (inside the container):
  python -u tmp_analysis/test_repr_dp_optimality.py <profile.yaml>
"""

import itertools
import json
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

# Reorderable operators of the SimCLRv2 feature (pipe ids) and their dependency
# edges, both taken from the feature definition itself.
OPS = (1, 2, 3, 4, 5, 6, 7)
EDGES = ((6, 5), (7, 1))  # flip after crop, normalize after to_float


def _children(value):
    if not value:
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split(",")]
    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value]
    return []


def legal_orders():
    for permutation in itertools.permutations(OPS):
        position = {p_id: index for index, p_id in enumerate(permutation)}
        if all(position[a] < position[b] for a, b in EDGES):
            yield permutation


def reference_objective(profile, order, model_variant="affine"):
    model = profile["physical_model"]["compute_model"]
    ratios = model["element_ratio"]
    transitions = model["class_transition"]
    operators = model["operators"]
    elements = float(model["source_elements"])
    klass = str(model["source_class"])
    total = 0.0
    # The reader is fixed first: it turns one path into one image record.
    reader = int(model.get("reader_pipe", 8))
    elements *= float(ratios.get(str(reader), 1.0))
    klass = str(transitions.get(str(reader), {}).get(klass, klass))
    for p_id in order:
        entry = operators[str(p_id)]
        curve = entry["by_class"][klass]
        b = 0.0 if model_variant == "proportional" else curve["b_ms"]
        total += curve["k_ms_per_element"] * elements + b
        elements *= float(ratios.get(str(p_id), 1.0))
        klass = str(transitions.get(str(p_id), {}).get(klass, klass))
    return total


def main() -> int:
    profile_path = Path(sys.argv[1])
    profile = yaml.safe_load(profile_path.read_text())
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
        use_my_optimizer=36,
        reorder_timeout_sec=3600.0,
    )
    plan = optimizer.run(str(profile_path), options)
    chosen = []
    current = None
    graph = {int(k): v for k, v in plan.graph.items()}
    for p_id in sorted(graph):
        if p_id in OPS and not graph.get(p_id):
            current = p_id
    # Rebuild the executed chain order from the plan graph.
    children = {int(child) for value in graph.values() for child in _children(value)}
    node = next(p for p in graph if p not in children)
    chain = []
    while True:
        chain.append(node)
        value = graph.get(node)
        nxt = _children(value)
        if not nxt:
            break
        node = nxt[0]
    chosen = [p_id for p_id in chain if p_id in OPS]
    if sorted(chosen) != sorted(OPS):
        print(f"FAIL: DP plan does not contain every operator: {chosen}")
        return 1

    scores = {
        order: reference_objective(profile, order) for order in legal_orders()
    }
    best = min(scores.values())
    argmins = [order for order, value in scores.items() if abs(value - best) < 1e-9]
    chosen_score = reference_objective(profile, tuple(chosen))
    print(f"legal orders: {len(scores)}")
    print(f"DP order:     {chosen}  objective={chosen_score:.6f}")
    print(f"brute force:  {argmins[0]} objective={best:.6f} ({len(argmins)} argmins)")
    worst = max(scores.items(), key=lambda kv: kv[1])
    print(f"worst legal:  {list(worst[0])} objective={worst[1]:.6f}")
    result = {
        "chosen_order": list(chosen),
        "chosen_objective": chosen_score,
        "best_objective": best,
        "argmins": [list(order) for order in argmins],
        "worst_order": list(worst[0]),
        "worst_objective": worst[1],
        "legal_orders": len(scores),
        "passed": abs(chosen_score - best) < 1e-9,
    }
    target = ROOT / "outputs/affine_reorder_diagnosis_20260924/dp_optimality_test.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1))
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
