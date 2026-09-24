"""Compare how the models price a Ray-fused stage against a local W-way stage.

For each workload it prints the profile's boundary bandwidth and per-record
sizes, then replays two materialized plans through PICO's objective and shows
the local / ray / smp lane split plus Cedar's estimate.

Usage (inside the container):
  python -u tmp_analysis/probe_ray_local_pricing.py \
      outputs/ultimate_new_workloads_20260922/wikitext103/profiles/shared.yaml \
          outputs/ultimate_new_workloads_20260922/wikitext103/plans/round1__optimizer.yaml \
          outputs/ultimate_new_workloads_20260922/wikitext103/plans/round1__simple_dp_workers_width_boundary.yaml
"""

import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_plan_cost_models import build_scorers, load_plan, plan_chain  # noqa: E402


def main() -> int:
    workload = sys.argv[1]
    profile_path = Path(sys.argv[2])
    plans = sys.argv[3:]
    profile = yaml.safe_load(profile_path.read_text())

    boundary = profile.get("physical_model", {}).get("boundary", {})
    print(f"workload={workload}")
    for variant, entry in boundary.items():
        if isinstance(entry, dict):
            print(f"  boundary[{variant}] throughput_bytes_per_sec="
                  f"{entry.get('throughput_bytes_per_sec')}")
    base = profile.get("baseline", {})
    sizes = base.get("input_sizes", {})
    print("  baseline throughput=", base.get("throughput"))
    print("  per-record input sizes (bytes):",
          {k: round(v) for k, v in sorted(sizes.items(), key=lambda kv: int(kv[0]))})
    affine = profile.get("physical_model", {}).get("operator_affine", {}).get("operators", {})
    for pid, entry in sorted(affine.items(), key=lambda kv: int(kv[0])):
        print(f"  affine[{pid}] k={entry.get('k_ms_per_byte'):.3e} "
              f"b={entry.get('b_ms'):.5f} fixed={entry.get('fixed_fraction')}")

    cedar, pico, inner_ops = build_scorers(workload, profile)
    for raw in plans:
        path = Path(raw)
        plan = load_plan(path)
        ops = list(pico._dp_inner_ops)
        specs = pico._dp_blocks_from_physical_plan(plan, ops)
        previous = getattr(pico, "_dp_scoring_required_widths", None)
        pico._dp_scoring_required_widths = pico._dp_plan_stage_widths(plan)
        try:
            cost = pico._replay_dp_objective(specs, ops)
        finally:
            pico._dp_scoring_required_widths = previous
        cedar_cost = cedar.calculate_cost(plan.graph, plan=plan)
        print(f"\n{path.name}  W={plan.n_local_workers}")
        print(f"  chain: {plan_chain(plan)}")
        print(f"  cedar cost = {cedar_cost:.4f} ms/source-record")
        print(f"  pico lanes: local={cost.local_serial:.4f} ray={cost.ray_serial:.4f} "
              f"smp={cost.smp_serial:.4f} gpu={cost.gpu_serial:.4f} "
              f"score={cost.score:.4f} (S/W={cost.score / max(1, plan.n_local_workers):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
