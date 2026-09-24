"""Cedar cost for two requested CommonVoice plans.

  plan 1: [6,5,4,3] local + [2,1] local + [0] local
  plan 2: [6,5,4,3,2,1,0] SMP (one stage)

Pipe ids: 0 mel, 1 frequency_mask, 2 time_mask, 3 _stretch, 4 _spec,
5 _resample, 6 _read, 7 LocalFSListerPipe (source).

Cost unit is ms/source-record of Cedar's own model; width is not a model
input, so every stage is materialised the way the campaign plans do it
(RAY/SMP with the profile's stage width one).

Usage (inside the container):
  python -u tmp_analysis/score_commonvoice_two_plans.py [profile.yaml]
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from cedar.compose.optimizer import Optimizer, PhysicalPlan, PipeDesc  # noqa: E402
from cedar.pipes import PipeVariantType  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.commonvoice.cedar_dataset import (  # noqa: E402
    CommonvoiceFeature,
)

RUN = ROOT / "outputs/ultimate_eight_optimizers_20260920/commonvoice"
PROFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / "profiles/shared.yaml"
LOCAL_TEMPLATE = RUN / "plans/round1__simple_dp_workers_width_boundary.yaml"
SMP_TEMPLATE = RUN / "plans/round1__old_dp_boundary.yaml"
DATA_DIR = ROOT / "datasets/commonvoice/cv15_en_train_300000"
NAMES = {
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


def load_template(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())["feature_r0"]
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    return payload


def variant_spec(template: dict, variant: str) -> dict:
    if variant == "INPROCESS":
        return dict(LOCAL)
    for rec in template["pipes"].values():
        if rec.get("variant") == variant:
            return {
                "variant": variant,
                "variant_ctx": dict(rec["variant_ctx"]),
            }
    raise SystemExit(f"template has no {variant} stage")


def build_plan(template: dict, groups) -> dict:
    pipes = {p_id: dict(rec) for p_id, rec in template["pipes"].items() if p_id < 8}
    graph = {}
    previous, node = 7, 8
    for members, variant in groups:
        if len(members) == 1:
            # A one-operator stage is the original pipe carrying the variant,
            # not a FusedPipe: Cedar charges optimizer-inserted pipes that are
            # not multi-member fused nodes zero cost.
            p_id = members[0]
            pipes[p_id] = dict(pipes[p_id])
            pipes[p_id].update(variant_spec(template, variant))
            graph[previous] = str(p_id)
            previous = p_id
            continue
        pipes[node] = {
            "name": "FusedPipe",
            "fused_pipes": list(members),
            **variant_spec(template, variant),
        }
        graph[previous] = str(node)
        previous, node = node, node + 1
    pipes[node] = {"name": "PrefetcherPipe", **LOCAL}
    graph[previous] = str(node)
    graph[node] = ""
    return {
        "graph": graph,
        "pipes": pipes,
        "n_local_workers": template.get("n_local_workers", 64),
    }


def main() -> int:
    logging.disable(logging.INFO)
    profile = yaml.safe_load(PROFILE.read_text())
    feature = CommonvoiceFeature(batch_size=1)
    feature.apply(LocalFSSource(str(DATA_DIR), recursive=True, max_samples=64))
    optimizer = Optimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = profile
    optimizer._init_stats()
    baseline = profile["baseline"]

    print(f"profile : {PROFILE}")
    print(
        f"baseline: {baseline['throughput']:.2f} rec/s -> "
        f"{1000.0 / baseline['throughput']:.4f} ms/source-record unfused local"
    )
    print("\nper-operator cost at profiled chain sizes (ms/source-record)")
    print(f"{'pipe':<18}{'local':>10}{'SMP':>10}")
    smp_ctx = variant_spec(load_template(SMP_TEMPLATE), "SMP")["variant_ctx"]
    for p_id in sorted(p for p in NAMES if p < 7):
        size = float(baseline["input_sizes"][p_id])
        local = optimizer._calculate_pipe_cost(p_id, size, None)
        smp = optimizer._calculate_pipe_cost(
            p_id,
            size,
            PipeDesc(
                name=None,
                variant_type=PipeVariantType.SMP,
                variant_ctx=None,
            ),
        )
        print(f"{NAMES[p_id]:<18}{local:>10.4f}{smp:>10.4f}")

    plans = {
        "plan 1  [6,5,4,3] L + [2,1] L + [0] L": (
            LOCAL_TEMPLATE,
            [
                ([6, 5, 4, 3], "INPROCESS"),
                ([2, 1], "INPROCESS"),
                ([0], "INPROCESS"),
            ],
        ),
        "plan 2  [6,5,4,3,2,1,0] SMP": (
            SMP_TEMPLATE,
            [([6, 5, 4, 3, 2, 1, 0], "SMP")],
        ),
    }
    for label, (template_path, groups) in plans.items():
        template = load_template(template_path)
        blocks = []
        original = optimizer._calculate_cost_fused

        def traced(specs, fused_pipes, input_size_map, output_size_map,
                   pipe_cost_map, _original=original):
            result = _original(
                specs,
                fused_pipes,
                input_size_map,
                output_size_map,
                pipe_cost_map,
            )
            blocks.append(
                (list(fused_pipes), result[0], result[1], dict(pipe_cost_map))
            )
            return result

        optimizer._calculate_cost_fused = traced
        try:
            plan = PhysicalPlan.from_dict(build_plan(template, groups))
            cost = optimizer.calculate_cost(plan.graph, plan=plan)
        finally:
            optimizer._calculate_cost_fused = original

        print(f"\n{label}")
        print(f"  source (pipe 7, local)          "
              f"{optimizer._base_cost_map[7]:.4f} ms")
        for fused_pipes, members_sum, charged, cost_map in blocks:
            detail = ", ".join(
                f"{p}:{cost_map[p]:.4f}" for p in fused_pipes
            )
            print(
                f"  FusedPipe {str(fused_pipes):<22} sum={members_sum:.4f} "
                f"held={charged:.4f} ms  [{detail}]"
            )
        print(f"  prefetcher                     0.0000 ms")
        print(f"  TOTAL                          {cost:.4f} ms/source-record")
        print(f"  detail json                    "
              + json.dumps([
                  {
                      "fused": b[0],
                      "sum_members": round(b[1], 4),
                      "charged": round(b[2], 4),
                  }
                  for b in blocks
              ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
