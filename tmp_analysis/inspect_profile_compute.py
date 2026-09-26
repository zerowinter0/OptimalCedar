"""Print the representation-aware compute model of a frozen profile."""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
PROFILE = ROOT / "outputs/affine_repr_profile_20260924"


def main(workloads):
    for workload in workloads:
        path = PROFILE / workload / "shared.yaml"
        if not path.exists():
            print(f"{workload}: MISSING {path}")
            continue
        data = yaml.safe_load(path.read_text())
        cm = (data.get("physical_model") or {}).get("compute_model") or {}
        ops = cm.get("operators") or {}
        print(f"== {workload}: method={cm.get('method')} statistic={cm.get('statistic')}")
        print(f"   operators={len(ops)} classes={json.dumps(cm.get('measured_classes'))[:200]}")
        for name, entry in ops.items():
            pieces = []
            for cls, curve in (entry.get("by_class") or {}).items():
                k = curve.get("k_ms_per_element")
                b = curve.get("b_ms")
                n = len(curve.get("points_ms_per_element") or [])
                pieces.append(f"{cls}: k={k:.3g} b={b:.4g} pts={n}")
            print(f"   op[{name}] own={entry.get('own_class')} :: " + " | ".join(pieces))
        print(f"   element_ratio={json.dumps(cm.get('element_ratio'))[:200]}")
        print(f"   class_transition={json.dumps(cm.get('class_transition'))[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
