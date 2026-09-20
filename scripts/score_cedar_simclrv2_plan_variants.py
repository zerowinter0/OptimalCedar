"""Cedar cost for SimCLRv2 plans given as operator-order strings.

Letters follow the paper figure legend (build_pipeline_cooptimization_simclr.py):
  R reader(ImageReader) | F to_float | C Crop(RandomResizedCrop) | H Flip |
  J Jitter(ColorJitter) | G Grayscale | B Blur(GaussianBlur) | N Normalize |
  T Batcher(4)

Every plan starts with the source and R, then the declared operator order, and
ends with the prefetcher.  ``--fuse-all`` materialises the declared operators as
one FusedPipe; otherwise each operator is its own INPROCESS stage.

Usage (inside the container):
  python -u scripts/score_cedar_simclrv2_plan_variants.py \
      --profile outputs/.../simclrv2/profiles/shared.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from cedar.compose.optimizer import Optimizer, PhysicalPlan, PipeDesc  # noqa: E402
from cedar.pipes import PipeVariantType  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2.cedar_dataset import (  # noqa: E402
    SimCLRV2Feature,
)

PIPE_OF_LETTER = {
    "F": 7,  # to_float
    "N": 1,  # Normalize
    "B": 2,  # GaussianBlur
    "G": 3,  # Grayscale
    "J": 4,  # ColorJitter
    "H": 5,  # RandomHorizontalFlip
    "C": 6,  # RandomResizedCrop
    "T": 0,  # BatcherPipe(4)
}
READER = 8
SOURCE = 9
DATA_DIR = ROOT / "evaluation/datasets/imagenette2/imagenette2/train"
LOCAL = {"variant": "INPROCESS", "variant_ctx": {"variant_type": "INPROCESS"}}
SMP_CTX = {
    "disable_torch_parallelism": True,
    "max_inflight": 10,
    "max_prefetch": 10,
    "n_procs": 1,
    "use_threads": True,
    "variant_type": "SMP",
}


def base_pipes(template: dict) -> dict:
    return {p_id: dict(rec) for p_id, rec in template.items() if p_id < 10}


def build_plan(
    template: dict,
    order: str,
    mode: str = "plain",
    fused_variant: str = "INPROCESS",
    workers: int = 64,
) -> dict:
    """mode: plain (one stage per letter), fuse_all (one fused block with every
    letter), fuse_mappers (fuse the mappable operators, keep the batcher T as
    its own stage at the position it appears)."""
    pipes = base_pipes(template)
    for p_id in range(10):
        pipes[p_id].update(LOCAL)
    graph = {SOURCE: str(READER)}
    previous = READER
    next_node = 10

    def fused(members):
        nonlocal previous, next_node
        pipes[next_node] = {
            "name": "FusedPipe",
            "fused_pipes": [PIPE_OF_LETTER[m] if isinstance(m, str) else m for m in members],
            "variant": fused_variant,
            "variant_ctx": (
                dict(SMP_CTX) if fused_variant == "SMP" else {"variant_type": "INPROCESS"}
            ),
        }
        graph[previous] = str(next_node)
        previous = next_node
        next_node += 1

    if mode == "fuse_all" and order:
        fused(list(order))
    elif mode == "fuse_mappers":
        groups, current = [], []
        for letter in order:
            if letter == "T":
                if current:
                    groups.append(current)
                    current = []
                groups.append(["T"])
            else:
                current.append(letter)
        if current:
            groups.append(current)
        for group in groups:
            if group == ["T"]:
                stage = PIPE_OF_LETTER["T"]
                graph[previous] = str(stage)
                previous = stage
            elif len(group) == 1:
                # a single operator is its own stage, not a one-member merge
                stage = PIPE_OF_LETTER[group[0]]
                graph[previous] = str(stage)
                previous = stage
            else:
                fused(group)
    else:
        for letter in order:
            stage = PIPE_OF_LETTER[letter]
            graph[previous] = str(stage)
            previous = stage
    graph[previous] = str(next_node)
    pipes[next_node] = {"name": "PrefetcherPipe", **LOCAL}
    graph[next_node] = ""
    return {"graph": graph, "pipes": pipes, "n_local_workers": workers}


def build_cedar_plan_with_smp(template: dict, workers: int = 64) -> dict:
    """cedar plan: R G C Fused{B,H,J} F N T, fused block changed to SMP."""
    pipes = base_pipes(template)
    for p_id in range(10):
        pipes[p_id].update(LOCAL)
    # source -> R(8) -> G(3) -> C(6) -> Fused(10) -> F(7) -> N(1) -> T(0) -> sink
    graph = {
        SOURCE: "8",
        8: "3",
        3: "6",
        6: str(10),
        10: "7",
        7: "1",
        1: "0",
        0: "11",
        11: "",
    }
    pipes[10] = {
        "name": "FusedPipe",
        "fused_pipes": [2, 5, 4],
        "variant": "SMP",
        "variant_ctx": dict(SMP_CTX),
    }
    pipes[11] = {"name": "PrefetcherPipe", **LOCAL}
    return {"graph": graph, "pipes": pipes, "n_local_workers": workers}


def plan_chain(plan: PhysicalPlan) -> str:
    starts = [p for p in plan.graph if not any(p in s for s in plan.graph.values())]
    node, out = starts[0], []
    while True:
        desc = plan.pipe_descs[node]
        name = (desc.name or f"pipe{node}").replace("MapperPipe_", "")
        if desc.fused_pipes:
            name = "Fused{" + ",".join(str(m) for m in desc.fused_pipes) + "}"
        variant = desc.variant_type
        if variant is not None and variant.name != "INPROCESS":
            ctx = desc.variant_ctx
            width = getattr(ctx, "n_actors", None) or getattr(ctx, "n_procs", None)
            name = f"{name}[{variant.name} w={width}]"
        out.append(name)
        if not plan.graph[node]:
            break
        node = next(iter(plan.graph[node]))
    return " -> ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--debug-plan",
        help="print the per-pipe walk for one plan label (prefix match)",
    )
    args = parser.parse_args()
    logging.disable(logging.INFO)

    profile = yaml.safe_load(args.profile.read_text())
    feature = SimCLRV2Feature(batch_size=4)
    feature.apply(LocalFSSource(str(DATA_DIR), recursive=True, max_samples=64))
    optimizer = Optimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = profile
    optimizer._init_stats()

    template = plan_payloads()
    plans = [
        ("1  G C B H J F T N (no fuse, local)", build_plan(template, "GCBHJFTN")),
        ("1b G C B H J F N T (no fuse, local)", build_plan(template, "GCBHJFNT")),
        ("2  F N B G J C H T (no fuse, local)", build_plan(template, "FNBGJCHT")),
        ("3  plan1, all fused, local", build_plan(template, "GCBHJFTN", "fuse_all")),
        ("3b plan1, mappers fused, local", build_plan(template, "GCBHJFTN", "fuse_mappers")),
        ("4  cedar plan, fuse block SMP", build_cedar_plan_with_smp(template)),
        ("5  plan2, mappers fused, local", build_plan(template, "FNBGJCHT", "fuse_mappers")),
        ("5b plan2, all fused, local", build_plan(template, "FNBGJCHT", "fuse_all")),
    ]
    print(f"profile: {args.profile}")
    print(f"{'plan':<36}{'W':>4}{'cedar cost':>12}")
    for label, payload in plans:
        plan = PhysicalPlan.from_dict(payload)
        if args.debug_plan and label.startswith(args.debug_plan):
            original = optimizer._calculate_pipe_cost

            def traced(p_id, size, desc, _original=original):
                cost = _original(p_id, size, desc)
                variant = (
                    desc.variant_type.name
                    if desc is not None and desc.variant_type is not None
                    else "INPROCESS"
                )
                print(
                    f"      pipe {p_id:>2} size={size:>12.1f} "
                    f"variant={variant:<9} cost={cost:.4f}"
                )
                return cost

            optimizer._calculate_pipe_cost = traced
            try:
                plan_cost = optimizer.calculate_cost(plan.graph, plan=plan)
            finally:
                optimizer._calculate_pipe_cost = original
            print(f"{label:<36}{plan.n_local_workers:>4}{plan_cost:>12.4f}")
            print(f"      {plan_chain(plan)}")
            continue
        cost = optimizer.calculate_cost(plan.graph, plan=plan)
        print(f"{label:<36}{plan.n_local_workers:>4}{cost:>12.4f}")
        print(f"      {plan_chain(plan)}")
    print(
        "\n(单位 ms/source-record，单 worker；unfused local 基线 = "
        f"{1000.0 / profile['baseline']['throughput']:.4f})"
    )
    print("\nper-operator costs in this profile (ms/source-record):")
    print(f"{'pipe':<22}{'local':>10}{'SMP':>10}{'RAY':>10}")
    names = {v: k for k, v in PIPE_OF_LETTER.items()}
    for p_id in sorted(names):
        size = float(profile["baseline"]["input_sizes"][p_id])
        row = []
        for variant in (None, PipeVariantType.SMP, PipeVariantType.RAY):
            desc = (
                None
                if variant is None
                else PipeDesc(name=None, variant_type=variant, variant_ctx=None)
            )
            try:
                row.append(optimizer._calculate_pipe_cost(p_id, size, desc))
            except Exception as exc:  # noqa: BLE001
                row.append(float("nan"))
        print(
            f"{names[p_id] + ' (pipe ' + str(p_id) + ')':<22}"
            f"{row[0]:>10.4f}{row[1]:>10.4f}{row[2]:>10.4f}"
        )
    return 0


def plan_payloads() -> dict:
    """Pipe descriptors for the SimCLRv2 operators, taken from a campaign plan."""
    path = (
        ROOT
        / "outputs/ultimate_eight_optimizers_20260920/simclrv2/plans/round1__unopti.yaml"
    )
    payload = yaml.safe_load(path.read_text())["feature"]
    return {int(k): v for k, v in payload["pipes"].items()}


if __name__ == "__main__":
    raise SystemExit(main())
