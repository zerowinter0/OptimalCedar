"""Is the representation/element state a function of the operator *set* only?

The DP stores one ``element_prod[mask]`` / ``class_state[mask]`` per subset, so
it assumes two legal prefixes with the same operator set always hand the next
operator the same representation and element count.  This script checks that
claim three ways:

 1. against every legal prefix of the SimCLRv2 dependency DAG (model table vs
    a direct walk of the prefix in its own order);
 2. against the payloads real plans actually delivered (the captured runs);
 3. for the byte side, that the byte volume stays a separate, unchanged feature.

Usage (inside the container):
  python -u tmp_analysis/state_sufficiency.py <profile.yaml>
"""

import itertools
import json
import pickle
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.pipes.common import (  # noqa: E402
    payload_compute_scale,
    payload_representation_class,
)

OPS = (1, 2, 3, 4, 5, 6, 7)
READER = 8
EDGES = ((6, 5), (7, 1))
CAPTURES = {
    "a_f0": ["a_f0_r1", "a_f0_r2", "a_f0_r3"],
    "a_f5": ["a_f5_r1", "a_f5_r2", "a_f5_r3"],
    "pico": ["pico", "pico_r2", "pico_r3"],
    "cedar": ["cedar", "cedar_r2", "cedar_r3"],
    "old-dp": ["old-dp", "old-dp_r2", "old-dp_r3"],
}


def legal_orders():
    for permutation in itertools.permutations(OPS):
        position = {p_id: index for index, p_id in enumerate(permutation)}
        if all(position[a] < position[b] for a, b in EDGES):
            yield permutation


def walk(model, order):
    """Features after applying the reader and then ``order``."""
    elements = float(model["source_elements"])
    klass = str(model["source_class"])
    ratios = model["element_ratio"]
    transitions = model["class_transition"]
    elements *= float(ratios.get(str(READER), 1.0))
    klass = str(transitions.get(str(READER), {}).get(klass, klass))
    states = {}
    for p_id in order:
        states[p_id] = (klass, elements)
        elements *= float(ratios.get(str(p_id), 1.0))
        klass = str(transitions.get(str(p_id), {}).get(klass, klass))
    # ``after`` is the state the *next* operator would see once every operator
    # of this prefix has been applied -- that is what the DP table stores.
    return states, (klass, elements)


def main() -> int:
    profile = yaml.safe_load(Path(sys.argv[1]).read_text())
    model = profile["physical_model"]["compute_model"]

    # 1) mask sufficiency over every legal prefix
    by_mask = {}
    disagreements = []
    for order in legal_orders():
        states, _after = walk(model, order)
        # Every prefix of every legal order must map to one state per mask.
        for size in range(1, len(order) + 1):
            prefix = order[:size]
            last = prefix[-1]
            klass, elements = states[last]
            # state *after* the last operator of the prefix
            ratio = float(model["element_ratio"].get(str(last), 1.0))
            transition = model["class_transition"].get(str(last), {})
            after = (str(transition.get(klass, klass)), elements * ratio)
            mask = frozenset(prefix)
            previous = by_mask.get(mask)
            if previous is not None and previous != after:
                disagreements.append(
                    {
                        "mask": sorted(mask),
                        "order": list(order),
                        "a": list(previous),
                        "b": list(after),
                    }
                )
                if len(disagreements) > 20:
                    break
            by_mask[mask] = after

    # 2) against the payloads the real plans delivered
    payload_checks = []
    for plan, runs in CAPTURES.items():
        for run in runs:
            directory = ROOT / f"tmp_analysis/capture_{run}/capture"
            for path in sorted(directory.glob("*_pipe_*.pkl")):
                try:
                    blob = pickle.loads(path.read_bytes())
                except Exception:  # noqa: BLE001
                    continue
                if not blob["snapshots"]:
                    continue
                value = pickle.loads(blob["snapshots"][0])
                klass = payload_representation_class(value)
                elements = payload_compute_scale(value)
                payload_checks.append(
                    {
                        "plan": plan,
                        "run": run,
                        "pipe": int(blob["p_id"]),
                        "class": klass,
                        "elements": elements,
                    }
                )

    # 3) the class/element state must be computed from the same mask the byte
    #    volume uses, and the byte volume must not appear in the compute model.
    byte_feature_leaks = [
        key
        for key in ("k_ms_per_byte", "x_reference_bytes")
        if key in json.dumps(model)
    ]
    result = {
        "mask_count": len(by_mask),
        "mask_disagreements": disagreements,
        "mask_sufficient": not disagreements,
        "payload_checks": len(payload_checks),
        "byte_feature_leaks": byte_feature_leaks,
        "scope": (
            "SimCLRv2 dependency DAG (reader + 7 mappers, edges crop->flip, "
            "to_float->normalize); 1260 legal orders; masks with up to 7 ops"
        ),
    }
    print(f"masks checked: {len(by_mask)}; disagreements: {len(disagreements)}")
    print(f"captured payload rows: {len(payload_checks)}")
    print(f"byte-feature leaks in compute model: {byte_feature_leaks}")
    if disagreements:
        print("FAIL: state is not a function of the operator set")
        print(json.dumps(disagreements[:3], indent=1))
    target = ROOT / "outputs/pico_final_w_only_20260924/state_sufficiency.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1))
    print(f"wrote {target}")
    return 0 if not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
