"""Shared pieces of the SimCLRv2 B/H/J fusion-offload mechanism experiment.

The experiment fixes the pipeline order to the Cedar plan's order

    LocalFSLister(9) -> ImageReader(8) -> Grayscale(3) -> RandomResizedCrop(6)
    -> [ B(2) H(5) J(4) ] -> to_float(7) -> Normalize(1) -> Batcher(0)

and only changes how the three augmentations B=GaussianBlur, H=RandomHorizontalFlip
and J=ColorJitter are organised:

    L-U  local,  three native stages
    L-F  local,  one native fused stage
    R-U  remote Ray, three actor stages (driver in the middle, as the runtime does)
    R-F  remote Ray, one actor running the fused stage
"""
from __future__ import annotations

import copy
import json
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]

SIMCLRV2_DATASET = REPO / "evaluation/datasets/imagenette2/imagenette2/train"
SIMCLRV2_MODULE = REPO / "evaluation/pipelines/simclrv2/cedar_dataset.py"

# Pipe ids of the SimCLRv2 feature (see SimCLRV2Feature._compose).
PIPE_LOCAL_FS = 9
PIPE_IMAGE_READER = 8
PIPE_TO_FLOAT = 7
PIPE_CROP = 6
PIPE_FLIP = 5
PIPE_JITTER = 4
PIPE_GRAYSCALE = 3
PIPE_BLUR = 2
PIPE_NORMALIZE = 1
PIPE_BATCHER = 0
PIPE_PREFETCH = 10

# The fused block of the Cedar plan is FusedPipe{2,5,4} = B -> H -> J.
BLOCK_ORDER: Tuple[int, ...] = (PIPE_BLUR, PIPE_FLIP, PIPE_JITTER)
BLOCK_NAMES = {
    PIPE_BLUR: "B_blur",
    PIPE_FLIP: "H_flip",
    PIPE_JITTER: "J_jitter",
}

# Per-operator seed offsets: the random sequence of one operator on one record
# must not depend on the configuration, the actor count or the fusion.
OP_SEED_OFFSET = {
    "B_blur": 11,
    "H_flip": 23,
    "J_jitter": 37,
    "crop": 53,
    "grayscale": 61,
}
SEED_MODULUS = 2**31 - 1
SEED_BASE = 20260923

# Fixed surrounding order of every configuration (the Cedar plan's order).
FIXED_ORDER_PREFIX: Tuple[int, ...] = (PIPE_IMAGE_READER, PIPE_GRAYSCALE, PIPE_CROP)
FIXED_ORDER_SUFFIX: Tuple[int, ...] = (PIPE_TO_FLOAT, PIPE_NORMALIZE, PIPE_BATCHER)

CONFIGS = ("L-U", "L-F", "R-U", "R-F")

_thread_state = threading.local()


def record_seed(record_id: int, op_name: str) -> int:
    return (SEED_BASE + record_id * 1_000_003 + OP_SEED_OFFSET[op_name]) % SEED_MODULUS


def set_current_record(record_id: int, offset: int = 0) -> int:
    """Bind the deterministic seed of the next operator call (thread local)."""
    _thread_state.record_id = record_id
    _thread_state.offset = offset
    return record_id


def current_record() -> Tuple[int, int]:
    return (
        getattr(_thread_state, "record_id", 0),
        getattr(_thread_state, "offset", 0),
    )


def seed_for_current(op_name: str) -> int:
    record_id, offset = current_record()
    return record_seed(record_id + offset, op_name)


def build_feature(dataset_path: Path = SIMCLRV2_DATASET, batch_size: int = 4):
    """The real SimCLRv2 feature with its sources applied."""
    import sys

    sys.path.insert(0, str(REPO))
    from cedar.sources import LocalFSSource
    from evaluation.eval_cedar import import_module_from_path

    module = import_module_from_path(str(SIMCLRV2_MODULE))
    feature = module.SimCLRV2Feature(batch_size=batch_size)
    source = LocalFSSource(str(dataset_path), recursive=True)
    feature.apply(source)
    return feature


def payload_bytes(value: Any) -> Optional[int]:
    """Serialized size of one payload, using the Ray serializer when present."""
    try:
        import ray

        return len(ray.cloudpickle.dumps(value))
    except Exception:  # noqa: BLE001 - fall back to the stdlib serializer
        import pickle

        try:
            return len(pickle.dumps(value))
        except Exception:  # noqa: BLE001
            return None


def tensor_nbytes(value: Any) -> Optional[int]:
    import torch

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        total = 0
        for item in value.values():
            size = tensor_nbytes(item)
            if size is None:
                return None
            total += size
        return total
    return None


def load_base_plan(plan_path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(plan_path).read_text())
    payload = payload.get("physical_plan", payload)
    if "feature" in payload:
        payload = payload["feature"]
    if "feature_r0" in payload:
        payload = payload["feature_r0"]
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    return payload


def _inprocess_desc(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "variant": "INPROCESS",
        "variant_ctx": {"variant_type": "INPROCESS"},
        "execution_resource": "cpu",
    }


def _ray_desc(name: str, submit_batch_size: int = 16) -> Dict[str, Any]:
    return {
        "name": name,
        "variant": "RAY",
        "variant_ctx": {
            "variant_type": "RAY",
            "n_actors": 1,
            "max_inflight": 100,
            "max_prefetch": 100,
            "use_threads": True,
            "submit_batch_size": submit_batch_size,
            "num_gpus": 0.0,
        },
        "execution_resource": "cpu",
    }


def block_plan(
    base_plan: Dict[str, Any], config: str, workers: int
) -> Dict[str, Any]:
    """Rewrite the Cedar plan so only the B/H/J block changes.

    ``base_plan`` must be the Cedar plan (``FusedPipe{2,5,4}`` after
    ``Grayscale -> RandomResizedCrop``); the surrounding order is kept.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown configuration {config}")
    plan = copy.deepcopy(base_plan)
    plan["n_local_workers"] = int(workers)
    pipes = plan["pipes"]
    graph = plan["graph"]

    fused_id = next(
        p_id
        for p_id, desc in pipes.items()
        if desc.get("fused_pipes") and list(desc["fused_pipes"]) == list(BLOCK_ORDER)
    )
    head_id = BLOCK_ORDER[0]
    tail_id = BLOCK_ORDER[-1]

    # Remove the fused block, keep the bookkeeping consistent.
    del pipes[fused_id]
    predecessor = next(
        p_id for p_id, succ in graph.items() if fused_id in _succ(succ)
    )
    successor = _succ(graph.get(fused_id, ""))

    if config in ("L-F", "R-F"):
        desc = (
            _inprocess_desc("FusedPipe")
            if config == "L-F"
            else _ray_desc("FusedPipe")
        )
        # The optimizer-pipe registry keys fused stages by this exact name.
        desc["fused_pipes"] = list(BLOCK_ORDER)
        pipes[fused_id] = desc
        block_nodes = [fused_id]
    else:
        for p_id in BLOCK_ORDER:
            prev_name = pipes[p_id].get("name") or f"pipe{p_id}"
            desc = (
                _inprocess_desc(prev_name)
                if config == "L-U"
                else _ray_desc(prev_name)
            )
            pipes[p_id] = desc
        block_nodes = list(BLOCK_ORDER)
        # The fused node disappears when the block runs as single stages.
        graph.pop(fused_id, None)

    # Rebuild the linear graph: predecessor -> block nodes -> successor.
    graph[predecessor] = [str(block_nodes[0])]
    for left, right in zip(block_nodes, block_nodes[1:]):
        graph[left] = [str(right)]
    graph[block_nodes[-1]] = [str(item) for item in successor]

    # Pipes that are no longer on the graph are dead weight; the runtime keys
    # variants off the graph, but keep the file readable.
    for p_id in list(graph):
        if p_id not in pipes:
            raise RuntimeError(f"plan references unknown pipe {p_id}")
    return plan


def _succ(value: Any) -> List[int]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).split(",") if item.strip()]


def write_plan(plan: Dict[str, Any], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    # DataSet._load_config expects the plan under a top-level key.
    path.write_text(
        yaml.safe_dump({"physical_plan": plan_payload(plan)}, sort_keys=False)
    )


def plan_payload(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Serializable ``PhysicalPlan.from_dict`` payload for a plan dict."""
    serializable = {
        # Keep integer keys: DataSet._load_config reads the plan straight into
        # PhysicalPlan.from_dict, which matches pipe ids against the feature's
        # logical pipes by integer id.
        "graph": {
            int(k): ",".join(str(x) for x in _succ(v))
            for k, v in plan["graph"].items()
        },
        "pipes": {int(k): v for k, v in plan["pipes"].items()},
        "n_local_workers": plan["n_local_workers"],
    }
    # Pipes that were fused away stay in the file as dead entries; the loader
    # still wants a variant for every desc.
    for desc in serializable["pipes"].values():
        desc.setdefault("variant", "INPROCESS")
        desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})
        desc.setdefault("execution_resource", "cpu")
    return serializable


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
