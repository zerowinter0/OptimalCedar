"""Why can PICO not price the coco Ray plans?

Reproduces the per-operator cost-vector loop of ``_BlockCandidateProvider``
for every candidate backend and prints, per operator, which branch kept the
cost at ``inf``: the spec forbids the variant (strict), the profile has no
backend entry, or the cost hook raised.
"""

import logging
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from cedar.compose.optimizer import PipeDesc  # noqa: E402
from cedar.pipes.context import PipeVariantType  # noqa: E402
from score_plan_cost_models import build_scorers  # noqa: E402

CAMPAIGN = ROOT / "outputs/ultimate_eight_optimizers_fix_20260921"


def main() -> int:
    logging.disable(logging.INFO)
    workload = sys.argv[1] if len(sys.argv) > 1 else "coco"
    profile = yaml.safe_load((CAMPAIGN / workload / "profiles/shared.yaml").read_text())
    _cedar, pico, inner_ops = build_scorers(workload, profile)
    names = {
        p_id: pico.logical_pipes[p_id].get_logical_name() for p_id in inner_ops
    }
    print(f"inner ops: {[names[p] for p in inner_ops]}")
    for variant, backend_stats in pico._iter_candidate_backend_stats():
        print(f"\n=== {variant.name}: profile entries "
              f"{sorted((int(k) for k in backend_stats), key=int)}")
        for p_id in inner_ops:
            pipe = pico.logical_pipes[p_id]
            spec = pipe.pipe_spec if pipe.pipe_spec is not None else pipe.get_spec()
            strict = pico._dp_pipe_allows_variant(p_id, variant, strict=True)
            loose = pico._dp_pipe_allows_variant(p_id, variant, strict=False)
            cost = None
            error = None
            if strict:
                try:
                    cost = pico._calculate_pipe_cost(
                        p_id,
                        pico.profiled_stats["baseline"]["input_sizes"][p_id],
                        PipeDesc(name=None, variant_type=variant, variant_ctx=None),
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
            print(
                f"  pipe {p_id:>2} {names[p_id]:<34} "
                f"spec_mutable={getattr(spec, 'mutable', None)!s:<5} "
                f"spec_variants={sorted(v.name for v in spec.mutable_variants)} "
                f"can_mutate={pipe.can_mutate_to(variant)!s:<5} "
                f"strict={strict!s:<5} loose={loose!s:<5} "
                f"in_profile={p_id in backend_stats} "
                f"cost={None if cost is None else round(cost, 4)} {error or ''}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
