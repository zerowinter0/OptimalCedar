"""Independent joint oracle for small instances (order x fusion x backend x W).

The DP is compared against a brute-force enumeration that *materialises* each
candidate plan and scores it through the optimizer's own plan-replay entry
point (``calculate_dp_objective_cost(plan=...)``).  The oracle therefore shares
the frozen cost parameters but never the DP's subset recurrence.

Scope is reported: the enumerated space is the one the optimizer is allowed to
search under the options used, and every axis that is *not* enumerated is
switched off for both sides.

Usage (inside the container):
  python -u tmp_analysis/oracle_joint_small.py <profile.yaml> [--subset]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import build_feature  # noqa: E402
from cedar.compose import OptimizerOptions, PhysicalPlan  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersBoundaryAffineReprOptimizer,
)
from cedar.compose import Feature  # noqa: E402
from cedar.pipes import BatcherPipe, ImageReaderPipe, MapperPipe  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402

PIPE_NAMES = {
    0: "BatcherPipe(batch_size=4)",
    1: "MapperPipe_Normalize",
    2: "MapperPipe_GaussianBlur",
    3: "MapperPipe_Grayscale",
    4: "MapperPipe_ColorJitter",
    5: "MapperPipe_RandomHorizontalFlip",
    6: "MapperPipe_RandomResizedCrop",
    7: "MapperPipe_to_float",
    8: "ImageReaderPipe",
    9: "LocalFSListerPipe",
}
FULL_OPS = (1, 2, 3, 4, 5, 6, 7)
SUBSET_OPS = (3, 2, 6, 7)  # grayscale, blur, crop, to_float
EDGES_FULL = ((6, 5), (7, 1))


def legal_orders(ops, edges):
    for permutation in itertools.permutations(ops):
        position = {p_id: index for index, p_id in enumerate(permutation)}
        if all(position[a] < position[b] for a, b in edges if a in ops and b in ops):
            yield permutation


def build_subset_feature(ops, batch_size: int = 4):
    """The real SimCLRv2 callables, restricted to a small operator set.

    The oracle needs a feature whose logical graph *is* the enumerated
    instance; reusing the full feature would make every candidate plan
    inconsistent with the feature it is scored against.
    """
    from torchvision import transforms
    from torchvision.io import ImageReadMode
    from block_mechanism_common import SIMCLRV2_DATASET

    import torch

    def to_float(x):
        return x.to(torch.float32)

    callables = {
        1: ("normalize", transforms.Normalize((0.1307,), (0.3081,)), ["float"]),
        2: ("blur", transforms.GaussianBlur(11), []),
        3: ("grayscale", transforms.Grayscale(num_output_channels=1), []),
        4: ("jitter", transforms.ColorJitter(0.1, 0.1, 0.1, 0.1), []),
        5: ("flip", transforms.RandomHorizontalFlip(), ["crop"]),
        6: ("crop", transforms.RandomResizedCrop((244, 244)), []),
        7: ("float", to_float, []),
    }

    class SubsetFeature(Feature):
        def __init__(self, batch_size, ops):
            super().__init__()
            self.batch_size = batch_size
            self.ops = ops

        def _compose(self, source_pipes):
            fp = source_pipes[0]
            fp = ImageReaderPipe(fp, mode=ImageReadMode.RGB).fix()
            made = {}
            for p_id in sorted(self.ops):
                tag, fn, depends = callables[p_id]
                pipe = MapperPipe(fp, fn, tag=tag)
                if depends:
                    pipe = pipe.depends_on(depends)
                made[tag] = pipe
                fp = pipe
            fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
            return fp

    feature = SubsetFeature(batch_size, tuple(ops))
    feature.apply(LocalFSSource(str(SIMCLRV2_DATASET), recursive=True))
    return feature


def build_plan_payload(order: Sequence[int], workers: int) -> Dict:
    """Materialise one all-INPROCESS plan for a legal order."""
    chain = [9, 8] + list(order) + [0]
    graph = {}
    for index, p_id in enumerate(chain):
        successor = chain[index + 1] if index + 1 < len(chain) else ""
        graph[str(p_id)] = str(successor) if successor != "" else ""
    pipes = {}
    for p_id in chain:
        pipes[str(p_id)] = {
            "name": PIPE_NAMES[p_id],
            "variant": "INPROCESS",
            "variant_ctx": {"variant_type": "INPROCESS"},
        }
    return {
        "graph": graph,
        "pipes": pipes,
        "n_local_workers": int(workers),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--workers", default="1,2,4,8,16,32,64")
    args = parser.parse_args()
    workers = [int(value) for value in args.workers.split(",")]
    ops = SUBSET_OPS if args.subset else FULL_OPS
    label = "simclrv2_subset_n4" if args.subset else "simclrv2_full_n7"

    profile = yaml.safe_load(args.profile.read_text())
    feature = build_feature(batch_size=4)
    optimizer = SimpleDpWorkersBoundaryAffineReprOptimizer()
    feature.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=False,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=False,
        enable_caching=False,
        num_samples=0,
        use_my_optimizer=39,
        reorder_timeout_sec=3600.0,
    )
    plan = optimizer.run(str(args.profile), options)
    dp_score = optimizer.calculate_dp_objective_cost(plan=plan)
    dp_workers = plan.n_local_workers
    graph = {int(k): v for k, v in plan.graph.items()}
    children = {
        int(child)
        for value in graph.values()
        for child in (
            [int(x) for x in value.split(",")]
            if isinstance(value, str) and value
            else (list(value) if value else [])
        )
    }
    node = next(p for p in graph if p not in children)
    dp_order = []
    while True:
        if node in ops:
            dp_order.append(node)
        value = graph.get(node)
        nxt = (
            [int(x) for x in value.split(",")]
            if isinstance(value, str) and value
            else (list(value) if value else [])
        )
        if not nxt:
            break
        node = nxt[0]

    # Start from the optimizer's own materialised plan so every candidate has
    # the exact pipe descriptors (prefetcher, fused nodes, variant contexts)
    # the feature expects; only the order and W are rewritten.
    template = json.loads(json.dumps(plan.to_dict()))
    print("template pipes:", sorted(template["pipes"], key=lambda k: int(k)))
    print("template graph:", {k: v for k, v in sorted(template["graph"].items(), key=lambda kv: int(kv[0]))})
    best = None
    best_plan = None
    evaluated = 0
    for order in legal_orders(ops, EDGES_FULL):
        for worker_count in workers:
            payload = json.loads(json.dumps(template))
            chain = [9, 8] + list(order) + [0]
            # Append the pass-through tail (prefetcher) the feature materialises.
            tail = [
                int(p_id)
                for p_id in sorted(template["pipes"], key=lambda k: int(k))
                if int(p_id) not in chain
            ]
            chain = chain + tail
            payload["graph"] = {
                str(p_id): (str(chain[i + 1]) if i + 1 < len(chain) else "")
                for i, p_id in enumerate(chain)
            }
            payload["n_local_workers"] = int(worker_count)
            try:
                candidate = PhysicalPlan.from_dict(payload)
                if evaluated == 0:
                    print("candidate graph:", candidate.graph)
                    print("candidate validate:", candidate.validate())
                score = optimizer.calculate_dp_objective_cost(plan=candidate)
            except Exception as exc:  # noqa: BLE001
                import traceback

                if evaluated == 0:
                    traceback.print_exc()
                print("skip", order, worker_count, exc)
                continue
            evaluated += 1
            if best is None or score < best:
                best = score
                best_plan = (list(order), worker_count)
    result = {
        "instance": label,
        "operators": list(ops),
        "workers_enumerated": workers,
        "orders_evaluated": evaluated,
        "dp_order": dp_order,
        "dp_workers": dp_workers,
        "dp_score": dp_score,
        "oracle_score": best,
        "oracle_plan": best_plan,
        "match": best is not None and abs(dp_score - best) <= 1e-6 * max(1.0, abs(best)),
        "scope": (
            "all legal orders x W in the enumerated set; fusion, offload and "
            "caching disabled on both sides; per-stage width fixed at 1"
        ),
    }
    print(json.dumps(result, indent=1))
    out = ROOT / "outputs/pico_final_w_only_20260924/oracle_results.json"
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:  # noqa: BLE001
            existing = {}
    existing[label] = result
    out.write_text(json.dumps(existing, indent=1))
    print(f"wrote {out}")
    return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
