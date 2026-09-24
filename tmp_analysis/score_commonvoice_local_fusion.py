"""Score two all-local CommonVoice fusion plans with Cedar's cost model.

Plans (pipe ids follow the materialized commonvoice plan):
  0 mel, 1 frequency_mask, 2 time_mask, 3 _stretch, 4 _spec, 5 _resample,
  6 _read, 7 LocalFSListerPipe (source), then FusedPipe(s) + PrefetcherPipe.

  A: one fused block [6,5,4,3,2,1,0], every stage INPROCESS (local).
  B: three fused blocks [6,5], [4,3], [2,1,0], every stage INPROCESS.

Cedar's model is Optimizer.calculate_cost: each pipe's local cost is its share
of the profiled whole-pipeline latency, and a materialized fused block is
charged sum(member costs) * (fused IO / baseline IO). W is not a model input.

Usage (inside the container):
  python -u tmp_analysis/score_commonvoice_local_fusion.py [profile.yaml]
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from cedar.compose.optimizer import Optimizer, PhysicalPlan  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.commonvoice.cedar_dataset import (  # noqa: E402
    CommonvoiceFeature,
)

PROFILE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else ROOT / "outputs/ultimate_eight_optimizers_20260920/commonvoice/profiles/shared.yaml"
)
PLAN_TEMPLATE = ROOT / (
    "outputs/ultimate_eight_optimizers_20260920/commonvoice/plans/"
    "round1__simple_dp_workers_width_boundary.yaml"
)
DATA_DIR = ROOT / "datasets/commonvoice/cv15_en_train_300000"
PIPE_NAMES = {
    0: "mel",
    1: "frequency_mask",
    2: "time_mask",
    3: "_stretch",
    4: "_spec",
    5: "_resample",
    6: "_read",
    7: "LocalFSListerPipe",
}
LOCAL = {"variant": "INPROCESS", "variant_ctx": {"variant_type": "INPROCESS"}}


def load_template() -> dict:
    data = yaml.safe_load(PLAN_TEMPLATE.read_text())
    payload = data["feature_r0"]
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    return payload


def build_plan(groups) -> dict:
    """Materialize one fused-block grouping with everything else INPROCESS."""
    template = load_template()
    pipes = {
        p_id: dict(rec)
        for p_id, rec in template["pipes"].items()
        if p_id < 8  # logical pipes plus the source; drop the template fusion
    }
    graph = {}
    previous = 7
    node = 8
    for group in groups:
        pipes[node] = {"name": "FusedPipe", "fused_pipes": list(group), **LOCAL}
        graph[previous] = str(node)
        previous = node
        node += 1
    pipes[node] = {"name": "PrefetcherPipe", **LOCAL}
    graph[previous] = str(node)
    graph[node] = ""
    return {
        "graph": graph,
        "pipes": pipes,
        "n_local_workers": template.get("n_local_workers", 64),
    }


def build_feature():
    feature = CommonvoiceFeature(batch_size=1)
    feature.apply(LocalFSSource(str(DATA_DIR), recursive=True, max_samples=64))
    return feature


def main() -> int:
    logging.disable(logging.INFO)
    profile = yaml.safe_load(PROFILE.read_text())
    feature = build_feature()
    optimizer = Optimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = profile
    optimizer._init_stats()

    total_cost = 1000.0 / profile["baseline"]["throughput"]
    print(f"profile            : {PROFILE}")
    print(
        "baseline throughput: "
        f"{profile['baseline']['throughput']:.2f} rec/s -> "
        f"whole-pipeline {total_cost:.4f} ms/source-record"
    )
    print("per-pipe local (INPROCESS) base cost, ms/source-record:")
    for p_id in sorted(PIPE_NAMES):
        print(
            f"  pipe {p_id:>2} {PIPE_NAMES[p_id]:<18} "
            f"{optimizer._base_cost_map[p_id]:.4f}"
        )
    print()

    for label, groups in (
        ("A  one block [6,5,4,3,2,1,0] local", [[6, 5, 4, 3, 2, 1, 0]]),
        ("B  [6,5] [4,3] [2,1,0] local", [[6, 5], [4, 3], [2, 1, 0]]),
    ):
        blocks = []
        original = optimizer._calculate_cost_fused

        def traced(specs, fused_pipes, input_size_map, output_size_map,
                   pipe_cost_map, _original=original):
            baseline, fused = _original(
                specs,
                fused_pipes,
                input_size_map,
                output_size_map,
                pipe_cost_map,
            )
            blocks.append((list(fused_pipes), baseline, fused))
            return baseline, fused

        optimizer._calculate_cost_fused = traced
        try:
            plan = PhysicalPlan.from_dict(build_plan(groups))
            cost = optimizer.calculate_cost(plan.graph, plan=plan)
        finally:
            optimizer._calculate_cost_fused = original

        print(f"plan {label}")
        print(
            f"  source (pipe 7)                  "
            f"{optimizer._base_cost_map[7]:.4f} ms"
        )
        for fused_pipes, baseline, fused in blocks:
            members = "{" + ",".join(str(p) for p in fused_pipes) + "}"
            print(
                f"  FusedPipe {members:<18} local  "
                f"sum(members)={baseline:.4f} -> charged {fused:.4f} ms"
            )
        print("  prefetcher (optimizer pipe)      0.0000 ms")
        print(
            f"  TOTAL                             {cost:.4f} ms/source-record "
            f"(=> {1000.0 / cost:.1f} rec/s single worker)"
        )
        print(
            "  model detail json                 "
            + json.dumps(
                [
                    {
                        "fused": b[0],
                        "sum_members": round(b[1], 4),
                        "charged": round(b[2], 4),
                    }
                    for b in blocks
                ]
            )
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
