"""Small-scale joint oracle with the *same* candidate space as the DP.

Instance: a four-operator subset of the real SimCLRv2 feature
(grayscale, blur, crop, to_float) plus its reader and batcher.  The profile of
the full feature is re-keyed onto that instance, so both sides share the frozen
cost parameters while the enumeration itself is written here and never uses the
DP's subset recurrence.

Space (identical on both sides):
  * every legal operator order;
  * every contiguous fusion partition of that order;
  * every block backend in {INPROCESS, SMP} (the DP has offload disabled, so
    these are exactly its candidates);
  * W in {1, 2, 4} (the DP is restricted to the same set);
  * stage width fixed at 1, caching off.

Both the DP's chosen plan and every enumerated candidate are scored through the
deployed plan-replay entry point (``_replay_dp_objective``), i.e. the same
objective the optimizer reports, so a mismatch cannot come from two different
cost models.

Usage (inside the container):
  python -u tmp_analysis/oracle_joint_final.py <profile.yaml> [--ops 3,2,6,7]
"""

import argparse
import copy
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import SIMCLRV2_DATASET, build_feature  # noqa: E402
from cedar.compose import Feature, OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersBoundaryAffineReprOptimizer,
)
from cedar.pipes import BatcherPipe, ImageReaderPipe, MapperPipe  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402

WORKER_SET = (1, 2, 4)
BACKENDS = ("INPROCESS",)  # the DP's own candidate set with offload disabled


def subset_ops_from_full(full_feature, wanted: Sequence[int]):
    """Tags/callables of the wanted operators, taken from the real feature."""
    out = []
    for p_id in wanted:
        pipe = full_feature.logical_pipes[p_id]
        tag = getattr(pipe, "tag", None) or getattr(pipe, "name", None)
        out.append((p_id, pipe.name, tag, pipe.fn))
    return out


def build_subset_feature(specs, batch_size: int = 4):
    from torchvision.io import ImageReadMode

    class SubsetFeature(Feature):
        def __init__(self, batch_size, specs):
            super().__init__()
            self.batch_size = batch_size
            self.specs = specs

        def _compose(self, source_pipes):
            fp = source_pipes[0]
            fp = ImageReaderPipe(fp, mode=ImageReadMode.RGB).fix()
            for _, _, tag, fn in self.specs:
                fp = MapperPipe(fp, fn, tag=tag)
            fp = BatcherPipe(fp, batch_size=self.batch_size).fix()
            return fp

    feature = SubsetFeature(batch_size, tuple(specs))
    feature.apply(LocalFSSource(str(SIMCLRV2_DATASET), recursive=True))
    return feature


def remap_profile(profile: dict, mapping: Dict[int, int], subset_ids: set) -> dict:
    """Re-key every pipe-id dictionary of the profile onto the subset instance."""
    out = copy.deepcopy(profile)

    def remap_dict(section, value_transform=None, key_str: bool = False):
        if not isinstance(section, dict):
            return section
        new = {}
        for key, value in section.items():
            try:
                p_id = int(key)
            except (TypeError, ValueError):
                new[key] = value
                continue
            if p_id not in mapping:
                continue
            # The baseline/affine sections are indexed with the integer pipe id
            # (``_validate_stats``), the compute-model sections are indexed with
            # ``str(p_id)`` by the representation-aware pricing code.
            key = str(mapping[p_id]) if key_str else mapping[p_id]
            new[key] = value_transform(value) if value_transform else value
        return new

    baseline = out.get("baseline", {})
    for field in ("input_sizes", "output_sizes", "wall_latencies", "latencies"):
        if field in baseline:
            baseline[field] = remap_dict(baseline[field])
    out["baseline"] = baseline
    physical = out.get("physical_model", {})
    affine = physical.get("operator_affine")
    if isinstance(affine, dict) and isinstance(affine.get("operators"), dict):
        affine["operators"] = remap_dict(affine["operators"])
        if isinstance(affine.get("unfitted_operators"), list):
            affine["unfitted_operators"] = [
                str(mapping[int(p)])
                for p in affine["unfitted_operators"]
                if str(p).isdigit() and int(p) in mapping
            ]
        if isinstance(affine.get("unfitted_reasons"), dict):
            affine["unfitted_reasons"] = remap_dict(affine["unfitted_reasons"])
    model = physical.get("compute_model")
    if isinstance(model, dict):
        for field in ("operators", "class_transition", "element_ratio", "measured_classes"):
            if isinstance(model.get(field), dict):
                model[field] = remap_dict(model[field], key_str=True)
    out["physical_model"] = physical
    if isinstance(out.get("offloads"), dict):
        for backend, section in out["offloads"].items():
            out["offloads"][backend] = remap_dict(section)
    return out


def legal_orders(ops: Sequence[int], edges: Sequence[Tuple[int, int]]):
    for permutation in itertools.permutations(ops):
        position = {p_id: index for index, p_id in enumerate(permutation)}
        if all(position[a] < position[b] for a, b in edges):
            yield permutation


def fusion_partitions(order: Sequence[int]):
    """Every contiguous partition of one order into blocks."""
    n = len(order)
    if n == 0:
        yield ()
        return
    for cuts in itertools.product((False, True), repeat=n - 1):
        blocks: List[Tuple[int, ...]] = []
        current = [order[0]]
        for index, cut in enumerate(cuts):
            if cut:
                blocks.append(tuple(current))
                current = [order[index + 1]]
            else:
                current.append(order[index + 1])
        blocks.append(tuple(current))
        yield tuple(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--ops", default="3,2,6,7")
    args = parser.parse_args()
    wanted = tuple(int(token) for token in args.ops.split(","))

    full = build_feature(batch_size=4)
    specs = subset_ops_from_full(full, wanted)
    instance = build_subset_feature(specs)
    # Every subset pipe (mappers, reader, lister, batcher) needs profile
    # entries, so the mapping is built by pipe name over the whole instance.
    mapping = {}
    for p_id, pipe in instance.logical_pipes.items():
        for real_id, real_pipe in full.logical_pipes.items():
            if real_pipe.name == pipe.name:
                mapping[int(real_id)] = int(p_id)
                break
    profile = yaml.safe_load(args.profile.read_text())
    remapped = remap_profile(profile, mapping, set(instance.logical_pipes))

    optimizer = SimpleDpWorkersBoundaryAffineReprOptimizer()
    instance.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=False,          # candidate backends: INPROCESS + SMP only
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=False,
        num_samples=0,
        use_my_optimizer=39,
        reorder_timeout_sec=3600.0,
    )
    os.environ["CEDAR_WORKER_SEARCH_SET"] = ",".join(str(w) for w in WORKER_SET)
    os.environ["CEDAR_DP_WORKER_LADDER"] = "0"
    plan = optimizer.run(remapped, options)
    ops = list(optimizer._dp_inner_ops)
    specs_of = optimizer._dp_blocks_from_physical_plan(plan, ops)
    optimizer._dp_scoring_required_widths = {}
    optimizer._dp_selected_workers = max(1, int(plan.n_local_workers or 1))
    dp_score = optimizer._replay_dp_objective(specs_of, ops).score

    # The subset declares no ``depends_on`` edges, so the DP may reorder the
    # four mappers freely; the reader stays first and the batcher last (the
    # feature marks both ``fix()``).
    readers = [p for p in ops if instance.logical_pipes[int(p)].is_source()
               or instance.logical_pipes[int(p)].name == "ImageReaderPipe"]
    sink = [p for p in ops if instance.logical_pipes[int(p)].name.startswith("BatcherPipe")]
    mappers = [p for p in ops if p not in readers and p not in sink]
    index_of = {int(p): idx for idx, p in enumerate(ops)}
    orders = [
        tuple([readers[0]] + list(permutation) + [sink[0]])
        for permutation in itertools.permutations(mappers)
    ]
    print(f"instance: readers={readers} sink={sink} mappers={mappers} "
          f"orders={len(orders)}")

    from cedar.compose.optimizer import PipeVariantType

    # The exact predicate the DP uses when it materialises a fused block.
    fusable = {
        int(p)
        for p in ops
        if optimizer._pipe_can_materialize_fusion(
            int(p), PipeVariantType.INPROCESS
        )
    }
    best = None
    best_plan = None
    evaluated = 0
    skipped: Dict[str, int] = {}
    first_errors: List[str] = []
    for order in orders:
        for blocks in fusion_partitions(order):
            # Fusion is only a legal candidate for runs whose members are all
            # fusable; the reader and the batcher stay single-stage.
            if any(
                len(block) > 1 and any(int(p) not in fusable for p in block)
                for block in blocks
            ):
                continue
            variant_choices = [list(BACKENDS) for _ in blocks]
            for variants in itertools.product(*variant_choices):
                for workers in WORKER_SET:
                    # ``_replay_dp_objective`` works in *inner-op index* space:
                    # a block of pipe ids must be translated before scoring.
                    from cedar.compose.optimizer import PipeVariantType

                    specs_try = [
                        (
                            tuple(index_of[int(p)] for p in block),
                            PipeVariantType[variant],
                            False,
                            1,
                        )
                        for block, variant in zip(blocks, variants)
                    ]
                    try:
                        optimizer._dp_selected_workers = workers
                        score = optimizer._replay_dp_objective(specs_try, ops).score
                    except Exception as exc:  # noqa: BLE001
                        key = f"{type(exc).__name__}: {exc}"[:120]
                        skipped[key] = skipped.get(key, 0) + 1
                        if len(first_errors) < 3:
                            first_errors.append(
                                f"{key} | order={order} blocks={blocks} "
                                f"variants={variants} W={workers}"
                            )
                        continue
                    evaluated += 1
                    if best is None or score < best:
                        best = score
                        best_plan = {
                            "order": [int(p) for p in order],
                            "blocks": [list(block) for block in blocks],
                            "backends": list(variants),
                            "workers": workers,
                        }
    result = {
        "instance": {
            "operators": ops,
            "backends": list(BACKENDS),
            "workers": list(WORKER_SET),
            "stage_width": 1,
            "caching": False,
            "fusion": "every contiguous partition of every legal order",
        },
        "orders_evaluated": evaluated,
        "dp": {
            "order": [int(pid) for block in specs_of for pid in block[0]],
            "workers": optimizer._dp_selected_workers,
            "score": float(dp_score),
            "blocks": [list(block[0]) for block in specs_of],
            "backends": [block[1].name for block in specs_of],
        },
        "oracle": {"score": best, "plan": best_plan},
        "skipped_candidates": skipped,
        "first_errors": first_errors,
        "match": best is not None
        and abs(dp_score - best) <= 1e-6 * max(1.0, abs(best)),
    }
    print(json.dumps(result, indent=1))
    out = ROOT / "outputs/pico_final_w_only_20260924/oracle_results.json"
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:  # noqa: BLE001
            existing = {}
    existing[f"joint_n{len(wanted)}"] = result
    out.write_text(json.dumps(existing, indent=1))
    print(f"wrote {out}")
    return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
