#!/usr/bin/env python3
"""Cedar-backed reorder/fusion/offload co-optimization example.

The script builds a small image-like pipeline with real Cedar pipes, profiles
it with Cedar's profiler, and compares modeled costs from Cedar optimizers.
It intentionally avoids a separate hand-written execution or cost model.

The joint strategy directly uses ``cedar.compose.dp_optimizer.DpOptimizer`` with
cache disabled.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cedar.client import DataSet
from cedar.compose import Feature, OptimizerOptions
from cedar.compose.dp_optimizer import DpOptimizer
from cedar.compose.dp_seperate_optimizer import DpSeperateOptimizer
from cedar.compose.optimizer import Optimizer, PhysicalPlan
from cedar.config import CedarContext, RayConfig
from cedar.pipes import MapperPipe, Pipe
from cedar.sources import IterSource


OUT_DIR = REPO_ROOT / "evaluation" / "plots"
DEFAULT_PROFILE = OUT_DIR / "reorder_fusion_offload_cedar_profile.yml"
DEFAULT_CSV = OUT_DIR / "reorder_fusion_offload_cedar_costs.csv"


Record = Dict[str, Any]


def make_record(idx: int, image_size: int, roi_side_fraction: float) -> Record:
    rng = np.random.default_rng(idx)
    compressed_size = max(8, int(round(image_size * 0.7)))
    compressed = rng.normal(
        loc=0.5,
        scale=0.15,
        size=(compressed_size, compressed_size, 3),
    )
    compressed = np.clip(compressed, 0.0, 1.0).astype(np.float32)
    return {"idx": idx, "compressed": compressed, "roi_side_fraction": roi_side_fraction}


def make_records(num_samples: int, image_size: int, roi_side_fraction: float) -> List[Record]:
    return [make_record(idx, image_size, roi_side_fraction) for idx in range(num_samples)]


def get_image_array(record: Record) -> np.ndarray:
    if "image" in record:
        return record["image"]
    tensor = record["tensor"]
    return np.ascontiguousarray(np.transpose(tensor.astype(np.float32), (1, 2, 0)))


def set_image_array(record: Record, image: np.ndarray, prefer_tensor: bool) -> None:
    if prefer_tensor:
        record.pop("image", None)
        record["tensor"] = np.ascontiguousarray(np.transpose(image.astype(np.float16), (2, 0, 1)))
    else:
        record.pop("tensor", None)
        record["image"] = np.ascontiguousarray(image.astype(np.float32, copy=False))


def decode_image(x: Record) -> Record:
    out = dict(x)
    compressed = out.pop("compressed")
    target = int(out["target_size"])
    scale = int(np.ceil(target / compressed.shape[0]))
    image = np.repeat(np.repeat(compressed, scale, axis=0), scale, axis=1)
    out["image"] = np.ascontiguousarray(image[:target, :target])
    return out


def edge_enhance(x: Record) -> Record:
    out = dict(x)
    img = get_image_array(out)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    score = 0.0
    work = gray
    for _ in range(2):
        gx = work[:, 2:] - work[:, :-2]
        gy = work[2:, :] - work[:-2, :]
        score += float(np.abs(gx[:, 1:-1]).mean() + np.abs(gy[1:-1, :]).mean())
        work = (work + np.roll(work, 1, axis=0) + np.roll(work, -1, axis=1)) / 3.0
    out["edge_score"] = score
    out["edge_features"] = np.ascontiguousarray(
        np.stack(
            [
                work,
                np.roll(work, 1, axis=0),
                np.roll(work, -1, axis=0),
                np.roll(work, 1, axis=1),
                np.roll(work, -1, axis=1),
                np.abs(np.roll(work, 1, axis=0) - work),
                np.abs(np.roll(work, -1, axis=1) - work),
                work * work,
            ],
            axis=-1,
        ).astype(np.float32)
    )
    return out


def color_convert(x: Record) -> Record:
    out = dict(x)
    prefer_tensor = "tensor" in out and "image" not in out
    img = get_image_array(out)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    converted = np.empty_like(img)
    converted[..., 0] = 0.299 * r + 0.587 * g + 0.114 * b
    converted[..., 1] = -0.168736 * r - 0.331264 * g + 0.5 * b
    converted[..., 2] = 0.5 * r - 0.418688 * g - 0.081312 * b
    for _ in range(80):
        converted = (
            3.0 * converted
            + np.roll(converted, 1, axis=0)
            + np.roll(converted, -1, axis=0)
            + np.roll(converted, 1, axis=1)
            + np.roll(converted, -1, axis=1)
        ) / 7.0
    set_image_array(out, converted, prefer_tensor)
    edge_features = out.pop("edge_features", None)
    if edge_features is not None:
        out["edge_summary"] = np.ascontiguousarray(edge_features[..., :3])
    return out


def normalize_image(x: Record) -> Record:
    out = dict(x)
    prefer_tensor = "tensor" in out and "image" not in out
    image = get_image_array(out)
    side = max(8, int(round(image.shape[0] * 0.7)))
    rows = np.linspace(0, image.shape[0] - 1, side).astype(np.int64)
    cols = np.linspace(0, image.shape[1] - 1, side).astype(np.int64)
    resized = image[np.ix_(rows, cols)]
    normalized = np.ascontiguousarray(
        (resized - resized.mean(axis=(0, 1))) / (resized.std(axis=(0, 1)) + 1e-4)
    )
    set_image_array(out, normalized, prefer_tensor)
    return out


class ToTensorPrune:
    def __call__(self, x: Record) -> Record:
        out = dict(x)
        side_fraction = float(out.get("roi_side_fraction", 0.32))
        image = get_image_array(out)
        probe = image
        for _ in range(4):
            probe = (
                probe
                + np.roll(probe, 1, axis=0)
                + np.roll(probe, -1, axis=1)
            ) / 3.0
        h, w = image.shape[:2]
        side = max(4, min(h, w, int(round(min(h, w) * side_fraction))))
        score = int(abs(out.get("edge_score", float(probe.mean()))) * 1000)
        top = 0 if h == side else score % (h - side + 1)
        left = 0 if w == side else (score // 7) % (w - side + 1)
        cropped = np.ascontiguousarray(image[top : top + side, left : left + side])
        out.pop("image", None)
        out.pop("edge_features", None)
        out.pop("edge_summary", None)
        out["tensor"] = np.ascontiguousarray(np.transpose(cropped.astype(np.float16), (2, 0, 1)))
        out["roi_pruned"] = True
        return out


def detail_smooth(x: Record) -> Record:
    out = dict(x)
    arr = get_image_array(out)
    for _ in range(3):
        arr = (
            4.0 * arr
            + np.roll(arr, 1, axis=0)
            + np.roll(arr, -1, axis=0)
            + np.roll(arr, 1, axis=1)
            + np.roll(arr, -1, axis=1)
        ) / 8.0
    detail = np.concatenate(
        [
            arr,
            np.roll(arr, 1, axis=0),
            np.roll(arr, -1, axis=0),
            np.roll(arr, 1, axis=1),
            np.roll(arr, -1, axis=1),
            arr * arr,
        ],
        axis=-1,
    )
    out.pop("image", None)
    out.pop("tensor", None)
    out["detail"] = np.ascontiguousarray(detail)
    return out


class ToTensor:
    def __call__(self, x: Record) -> Record:
        out = dict(x)
        detail = out.pop("detail")
        out["tensor"] = np.ascontiguousarray(np.transpose(detail, (2, 0, 1)))
        return out


class CedarImageCooptFeature(Feature):
    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        fp = source_pipes[0]
        fp = MapperPipe(fp, decode_image, tag="decode").fix()
        fp = MapperPipe(fp, edge_enhance, tag="edge")
        fp = MapperPipe(fp, color_convert, tag="color").depends_on(["decode"])
        fp = MapperPipe(fp, normalize_image, tag="normalize").depends_on(["color"])
        fp = MapperPipe(fp, ToTensorPrune(), tag="prune").depends_on(["edge"])
        fp = MapperPipe(fp, detail_smooth, tag="detail").depends_on(["normalize", "prune"])
        fp = MapperPipe(fp, ToTensor(), tag="tensor").depends_on(["detail"])
        return fp


class FixedOrderPhysicalOptimizer(DpSeperateOptimizer):
    """Run the fixed-order physical DP on the original order."""

    def _dp_reorder_offload_cache_fusion(self, inner_ops: List[int]) -> Tuple[List[int], Optional[int]]:
        cache_p_id = self._dp_fixed_order_offload_cache_fusion(inner_ops)
        return inner_ops, cache_p_id


@dataclass(frozen=True)
class Result:
    strategy: str
    cost_ms: float
    order: str
    fused_blocks: str


def build_feature(num_samples: int, image_size: int, keep_fraction: float) -> CedarImageCooptFeature:
    feature = CedarImageCooptFeature()
    records = make_records(num_samples, image_size, keep_fraction)
    for record in records:
        record["target_size"] = image_size
    feature.apply(IterSource(records))
    return feature


def optimizer_options(
    enable_reorder: bool = True,
    enable_fusion: bool = True,
    enable_offload: bool = False,
) -> OptimizerOptions:
    return OptimizerOptions(
        enable_prefetch=False,
        enable_offload=enable_offload,
        enable_reorder=enable_reorder,
        enable_local_parallelism=False,
        enable_fusion=enable_fusion,
        enable_caching=False,
        disable_physical_opt=False,
    )


def profile_with_cedar(
    profile_path: Path,
    num_samples: int,
    image_size: int,
    keep_fraction: float,
    use_ray: bool,
    force: bool,
) -> Dict[str, Any]:
    if profile_path.exists() and not force:
        with profile_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    feature = build_feature(num_samples, image_size, keep_fraction)
    ctx = CedarContext(RayConfig(n_cpus=4) if use_ray else None)
    try:
        DataSet(
            ctx,
            {"feature": feature},
            prefetch=False,
            enable_controller=False,
            enable_optimizer=False,
            profiled_data=str(profile_path),
            run_profiling=True,
            optimizer_options=OptimizerOptions(num_samples=num_samples),
        )
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise

    with profile_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_optimizer(
    optimizer_cls: Type[Optimizer],
    stats: Dict[str, Any],
    num_samples: int,
    image_size: int,
    keep_fraction: float,
) -> Optimizer:
    feature = build_feature(num_samples, image_size, keep_fraction)
    opt = optimizer_cls()
    opt.init(feature.logical_pipes, feature.logical_adj_list)
    opt.profiled_stats = stats
    return opt


def plan_cost(opt: Optimizer, plan: PhysicalPlan) -> float:
    caching_on = opt._get_cache_pid(plan) is not None
    fused_blocks = [
        list(desc.fused_pipes)
        for desc in plan.pipe_descs.values()
        if getattr(desc, "fused_pipes", None) and len(getattr(desc, "fused_pipes", [])) > 1
    ]
    return opt.calculate_cost(
        plan.graph,
        physical_specs=plan.pipe_descs,
        fused_pipes=fused_blocks if fused_blocks else None,
        caching_on=caching_on,
        plan=plan,
    )


def format_plan(opt: Optimizer, plan: PhysicalPlan) -> Tuple[str, str]:
    path = opt._get_critical_path(
        plan.graph,
        opt._get_source_p_id(),
        opt._get_output_p_id(plan.graph),
        plan,
    )[0]
    names = [plan.pipe_descs[p_id].name for p_id in path]
    fused = []
    for p_id in path:
        desc = plan.pipe_descs[p_id]
        if getattr(desc, "fused_pipes", None) and len(desc.fused_pipes) > 1:
            fused.append("[" + ",".join(str(x) for x in desc.fused_pipes) + "]")
    return " -> ".join(names), ";".join(fused) if fused else "-"


def run_base_cost(
    stats: Dict[str, Any],
    num_samples: int,
    image_size: int,
    keep_fraction: float,
    enable_offload: bool,
) -> Result:
    opt = init_optimizer(Optimizer, stats, num_samples, image_size, keep_fraction)
    opt.options = optimizer_options(
        enable_reorder=False,
        enable_fusion=False,
        enable_offload=enable_offload,
    )
    opt._validate_stats()
    opt._init_stats()
    cost = opt.calculate_cost(opt.physical_plan.graph, physical_specs=opt.physical_plan.pipe_descs)
    order, fused = format_plan(opt, opt.physical_plan)
    return Result("unoptimized", cost, order, fused)


def run_optimizer(
    strategy: str,
    optimizer_cls: Type[Optimizer],
    stats: Dict[str, Any],
    num_samples: int,
    image_size: int,
    keep_fraction: float,
    options: OptimizerOptions,
) -> Result:
    opt = init_optimizer(optimizer_cls, stats, num_samples, image_size, keep_fraction)

    compose_logger = logging.getLogger("cedar.compose")
    old_level = compose_logger.level
    old_handlers = list(compose_logger.handlers)
    old_propagate = compose_logger.propagate
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    compose_logger.handlers = [handler]
    compose_logger.setLevel(logging.INFO)
    compose_logger.propagate = False
    try:
        plan = opt.run(stats, options)
    finally:
        compose_logger.handlers = old_handlers
        compose_logger.setLevel(old_level)
        compose_logger.propagate = old_propagate

    cost = plan_cost(opt, plan)
    order, fused = format_plan(opt, plan)
    return Result(strategy, cost, order, fused)


def write_results(results: Sequence[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "cost_ms", "order", "fused_blocks"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "strategy": result.strategy,
                    "cost_ms": f"{result.cost_ms:.6f}",
                    "order": result.order,
                    "fused_blocks": result.fused_blocks,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--keep-fraction", type=float, default=0.3)
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    stats = profile_with_cedar(
        args.profile_path,
        args.num_samples,
        args.image_size,
        args.keep_fraction,
        args.use_ray,
        args.force_profile,
    )

    common_options = optimizer_options(
        enable_reorder=True,
        enable_fusion=True,
        enable_offload=args.use_ray,
    )
    results = [
        run_base_cost(
            stats,
            args.num_samples,
            args.image_size,
            args.keep_fraction,
            enable_offload=args.use_ray,
        ),
        run_optimizer(
            "reorder -> fusion/offload",
            DpSeperateOptimizer,
            stats,
            args.num_samples,
            args.image_size,
            args.keep_fraction,
            common_options,
        ),
        run_optimizer(
            "fusion/offload -> reorder-units",
            FixedOrderPhysicalOptimizer,
            stats,
            args.num_samples,
            args.image_size,
            args.keep_fraction,
            common_options,
        ),
        run_optimizer(
            "joint reorder + fusion/offload",
            DpOptimizer,
            stats,
            args.num_samples,
            args.image_size,
            args.keep_fraction,
            common_options,
        ),
    ]
    write_results(results, args.csv)

    for result in results:
        print(f"{result.strategy:34s} {result.cost_ms:10.6f}  fused={result.fused_blocks}")
        print(f"  {result.order}")
    print(f"Wrote {args.profile_path}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
