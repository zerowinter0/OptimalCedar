from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
import itertools
from typing import Dict, List, Optional, Set, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cedar.client.dataset import _DataSetIter
from cedar.client.profiler import FeatureProfiler
from cedar.compose import Feature, OptimizerOptions, PhysicalPlan
from cedar.compose.optimizer import Optimizer, PipeDesc
from cedar.config import CedarContext
from cedar.pipes import (
    MapperPipe,
    Pipe,
    PipeVariantContextFactory,
    PipeVariantType,
    TFOutputHint,
    TFTensorDontCare,
)
from cedar.sources import LocalFSSource

import tensorflow as tf


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
DEFAULT_DATASET_DIR = REPO_ROOT / "evaluation" / "datasets" / "imagenette2-160"
DEFAULT_OUT_DIR = Path("/tmp/pico_motivation_reorder_cache")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
GAUSSIAN_BLUR_RADIUS = 32
UNSHARP_BLUR_RADIUS = 24
RANDOM_CROP_SIZE = 96


def decode_jpeg(x):
    img = tf.image.decode_jpeg(x, channels=3)
    return tf.expand_dims(img, axis=0)


def resize_image(x):
    x = tf.image.convert_image_dtype(x, tf.float32)
    return tf.image.resize(x, [64, 64])


def random_crop_image(x):
    return tf.image.random_crop(x, [1, RANDOM_CROP_SIZE, RANDOM_CROP_SIZE, 3])


def gaussian_blur_tensor(x, radius):
    x = tf.image.convert_image_dtype(x, tf.float32)
    sigma = tf.cast(radius, x.dtype) / 3.0
    coords = tf.cast(tf.range(-radius, radius + 1), x.dtype)
    gaussian_1d = tf.exp(-(coords**2) / (2.0 * sigma**2))
    gaussian_1d = gaussian_1d / tf.reduce_sum(gaussian_1d)
    kernel = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = tf.reshape(kernel, [2 * radius + 1, 2 * radius + 1, 1, 1])
    channels = tf.shape(x)[-1]
    kernel = tf.tile(kernel, [1, 1, channels, 1])
    return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")


def gaussian_filter2d(x):
    return gaussian_blur_tensor(x, GAUSSIAN_BLUR_RADIUS)


def unsharp_mask_image(x):
    x = tf.image.convert_image_dtype(x, tf.float32)
    blurred = gaussian_blur_tensor(x, UNSHARP_BLUR_RADIUS)
    sharpened = x + 1.5 * (x - blurred)
    return tf.clip_by_value(sharpened, 0.0, 1.0)


class FiveOpImageFeature(Feature):
    """Fixed source/read step plus five image operators used in the motivation figure."""

    def _compose(self, source_pipes: List[Pipe]):
        fp = source_pipes[0]
        source_pipes[0].set_output_tf_spec(tf.TensorSpec(shape=(), dtype=tf.string))

        fp = MapperPipe(
            fp,
            tf.io.read_file,
            output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            tag="read_file",
            is_random=False,
        ).fix()
        fp = MapperPipe(
            fp,
            decode_jpeg,
            output_tf_hint=TFOutputHint([1, None, None, 3], tf.int8),
            tag="decode_jpeg",
            is_random=False,
        )
        fp = MapperPipe(
            fp,
            random_crop_image,
            output_tf_hint=TFOutputHint([1, RANDOM_CROP_SIZE, RANDOM_CROP_SIZE, 3], TFTensorDontCare()),
            tag="random_crop_image",
            is_random=True,
        )
        fp = MapperPipe(
            fp,
            resize_image,
            output_tf_hint=TFOutputHint([1, 64, 64, 3], tf.float32),
            tag="resize_image",
            is_random=False,
        )
        fp = MapperPipe(
            fp,
            unsharp_mask_image,
            output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            tag="unsharp_mask_image",
            is_random=False,
        )
        fp = MapperPipe(
            fp,
            gaussian_filter2d,
            output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
            tag="gaussian_filter2d",
            is_random=False,
        )
        return fp


def ensure_imagenette(dataset_dir: Path) -> Path:
    train_dir = dataset_dir / "train"
    if train_dir.exists() and any(train_dir.rglob("*.JPEG")):
        return train_dir
    if train_dir.exists() and any(train_dir.rglob("*.jpg")):
        return train_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir.parent / "imagenette2-160.tgz"
    if not archive.exists():
        logging.info("Downloading %s to %s", IMAGENETTE_URL, archive)
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)

    logging.info("Extracting %s to %s", archive, dataset_dir.parent)
    with tarfile.open(archive, "r:gz") as tfp:
        tfp.extractall(dataset_dir.parent)

    if not train_dir.exists():
        raise FileNotFoundError(train_dir)
    return train_dir


def prepare_image_subset(train_dir: Path, out_dir: Path, num_samples: int) -> Path:
    image_paths = sorted(
        p
        for p in train_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) < num_samples:
        raise RuntimeError(
            f"Need {num_samples} images under {train_dir}, found {len(image_paths)}"
        )
    subset_dir = out_dir / f"profile_images_{num_samples}"
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True)
    for idx, src in enumerate(image_paths[:num_samples]):
        dst = subset_dir / f"{idx:05d}_{src.name}"
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    return subset_dir


def build_feature(train_dir: Path) -> FiveOpImageFeature:
    feature = FiveOpImageFeature()
    source = LocalFSSource(str(train_dir), recursive=True)
    feature.apply(source)
    return feature


def profile_feature(train_dir: Path, profile_path: Path, num_samples: int) -> None:
    if profile_path.exists():
        logging.info("Using existing profile %s", profile_path)
        return
    logging.info("Building feature for baseline profile")
    feature = build_feature(train_dir)
    logging.info("Profiling baseline over %s samples", num_samples)
    ctx = CedarContext()
    loaded_feature = feature.profile(ctx, None)
    source_pipe = feature.get_source_pipes()
    profiler = FeatureProfiler(feature, profile_mode=True)
    batch_size = profiler.get_batch_size()
    data_iter = _DataSetIter(
        loaded_features={"feature": loaded_feature},
        profilers={"feature": profiler},
        return_datasample=False,
        source_pipes={"feature": source_pipe},
    )
    n_batches = 0
    start_time = time.time()
    for _ in data_iter:
        if n_batches == 0:
            start_time = time.time()
        n_batches += 1
        if n_batches * batch_size >= num_samples:
            break
    end_time = time.time()
    throughput = (n_batches * batch_size) / max(end_time - start_time, 1e-12)
    baseline = {
        "latencies": profiler.calculate_avg_latency_per_sample(),
        "input_sizes": profiler.calculate_avg_data_size()[0],
        "output_sizes": profiler.calculate_avg_data_size()[1],
        "throughput": throughput,
    }
    feature.reset()
    logging.info("Profiling local disk IO")
    write_latency, read_latency = profile_io()
    profile = {
        "baseline": baseline,
        "offloads": {},
        "disk_info": {
            "read_latency": read_latency,
            "write_latency": write_latency,
        },
    }
    with profile_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False)


def profile_io(character: str = "a", file_size_mb: int = 10):
    temp_path = Path("/tmp") / "pico_motivation_profile_io.tmp"
    file_size_bytes = file_size_mb * 1024 * 1024
    start = time.time()
    with temp_path.open("w", encoding="utf-8") as f:
        f.write(character * file_size_bytes)
    write_latency = (time.time() - start) / file_size_bytes
    start = time.time()
    with temp_path.open("r", encoding="utf-8") as f:
        _ = f.read()
    read_latency = (time.time() - start) / file_size_bytes
    temp_path.unlink(missing_ok=True)
    return write_latency, read_latency


def init_optimizer(feature: FiveOpImageFeature, profile_path: Path) -> Optimizer:
    optimizer = Optimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    with profile_path.open("r", encoding="utf-8") as f:
        optimizer.profiled_stats = yaml.safe_load(f)
    optimizer.options = OptimizerOptions(
        enable_prefetch=False,
        enable_offload=False,
        enable_reorder=False,
        enable_local_parallelism=False,
        enable_fusion=False,
        enable_caching=False,
        num_samples=None,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    return optimizer


def pipe_id_by_tag(feature: FiveOpImageFeature, tag: str) -> int:
    matches = [p_id for p_id, p in feature.logical_pipes.items() if getattr(p, "tag", None) == tag]
    if len(matches) != 1:
        names = {p_id: (p.get_logical_name(), getattr(p, "tag", None)) for p_id, p in feature.logical_pipes.items()}
        raise RuntimeError(f"Expected one pipe with tag {tag!r}, got {matches}; pipes={names}")
    return matches[0]


def linear_graph(order: List[int]) -> Dict[int, Set[int]]:
    return {p_id: ({order[idx + 1]} if idx + 1 < len(order) else set()) for idx, p_id in enumerate(order)}


def cache_desc() -> PipeDesc:
    return PipeDesc(
        name="ObjectDiskCachePipe",
        variant_type=PipeVariantType.INPROCESS,
        variant_ctx=PipeVariantContextFactory.create_context(PipeVariantType.INPROCESS),
    )


def make_plan(
    optimizer: Optimizer,
    order: List[int],
    cache_after: Optional[int],
) -> PhysicalPlan:
    graph = linear_graph(order)
    pipe_descs = {
        p_id: PipeDesc(
            name=optimizer.physical_plan.pipe_descs[p_id].name,
            variant_type=PipeVariantType.INPROCESS,
            variant_ctx=PipeVariantContextFactory.create_context(PipeVariantType.INPROCESS),
        )
        for p_id in order
    }
    if cache_after is not None:
        cache_p_id = max(pipe_descs) + 1
        next_nodes = graph[cache_after]
        graph[cache_after] = {cache_p_id}
        graph[cache_p_id] = next_nodes
        pipe_descs[cache_p_id] = cache_desc()
    return PhysicalPlan(graph=graph, pipe_descs=pipe_descs, n_local_workers=1)


def save_plan(plan: PhysicalPlan, path: Path) -> None:
    data = plan.to_dict()
    for _, p_dict in data["pipes"].items():
        if "variant" not in p_dict:
            p_dict["variant"] = "INPROCESS"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"physical_plan": data}, f, sort_keys=False)


def ordered_names(feature: FiveOpImageFeature, order: List[int]) -> List[str]:
    return [feature.logical_pipes[p_id].get_logical_name() for p_id in order]


def operator_profile(
    feature: FiveOpImageFeature,
    optimizer: Optimizer,
    pipe_ids: List[int],
) -> Dict[str, Dict[str, float]]:
    profile = {}
    for p_id in pipe_ids:
        name = feature.logical_pipes[p_id].get_logical_name()
        profile[name] = {
            "pipe_id": p_id,
            "cost": float(optimizer._base_cost_map[p_id]),
            "selectivity": (
                float(optimizer._data_size_ratio_map[p_id])
                if optimizer._data_size_ratio_map[p_id] is not None
                else None
            ),
        }
    return profile


def plan_costs(
    optimizer: Optimizer,
    order: List[int],
    cache_after: Optional[int],
    num_epochs: int,
) -> Tuple[PhysicalPlan, float, Optional[float], float]:
    cold_plan = make_plan(optimizer, order, cache_after=None)
    cold_cost = optimizer.calculate_cost(
        cold_plan.graph,
        physical_specs=cold_plan.pipe_descs,
        caching_on=False,
        plan=cold_plan,
    )
    plan = make_plan(optimizer, order, cache_after)
    if cache_after is None:
        return plan, cold_cost, None, cold_cost * num_epochs
    cached_cost = optimizer.calculate_cost(
        plan.graph,
        physical_specs=plan.pipe_descs,
        caching_on=True,
        plan=plan,
    )
    total_cost = cold_cost + (num_epochs - 1) * cached_cost
    return plan, cold_cost, cached_cost, total_cost


def legal_orders(
    fixed_prefix: List[int],
    decode_p_id: int,
    mutable_ops: List[int],
) -> List[List[int]]:
    remaining = [p_id for p_id in mutable_ops if p_id != decode_p_id]
    return [fixed_prefix + [decode_p_id] + list(perm) for perm in itertools.permutations(remaining)]


def legal_cache_points(order: List[int], random_p_id: int) -> List[Optional[int]]:
    random_pos = order.index(random_p_id)
    return [None] + order[2:random_pos]


def choose_best_total(
    optimizer: Optimizer,
    candidates: List[Tuple[List[int], Optional[int]]],
    num_epochs: int,
) -> Tuple[List[int], Optional[int], PhysicalPlan, float, Optional[float], float]:
    best = None
    for order, cache_after in candidates:
        plan, cold_cost, cached_cost, total_cost = plan_costs(
            optimizer, order, cache_after, num_epochs
        )
        cached_tie_breaker = cached_cost if cached_cost is not None else float("inf")
        row_key = (total_cost, cold_cost, cached_tie_breaker)
        row = (row_key, total_cost, cold_cost, cached_cost, order, cache_after, plan)
        if best is None or row_key < best[0]:
            best = row
    if best is None:
        raise RuntimeError("No candidate plans were generated.")
    _, total_cost, cold_cost, cached_cost, order, cache_after, plan = best
    return order, cache_after, plan, cold_cost, cached_cost, total_cost


def choose_best_cold(
    optimizer: Optimizer,
    orders: List[List[int]],
    num_epochs: int,
) -> List[int]:
    best = None
    for order in orders:
        _, cold_cost, _, _ = plan_costs(optimizer, order, None, num_epochs)
        row = (cold_cost, order)
        if best is None or row < best:
            best = row
    if best is None:
        raise RuntimeError("No reorder candidates were generated.")
    return best[1]


def strategy_unoptimized(
    optimizer: Optimizer,
    original_order: List[int],
    random_p_id: int,
    num_epochs: int,
) -> Tuple[List[int], Optional[int], PhysicalPlan, float, Optional[float], float]:
    order = original_order
    plan, cold_cost, cached_cost, total_cost = plan_costs(
        optimizer, order, None, num_epochs
    )
    return order, None, plan, cold_cost, cached_cost, total_cost


def strategy_reorder_then_cache(
    optimizer: Optimizer,
    fixed_prefix: List[int],
    decode_p_id: int,
    random_p_id: int,
    mutable_ops: List[int],
    num_epochs: int,
) -> Tuple[List[int], Optional[int], PhysicalPlan, float, Optional[float], float]:
    orders = legal_orders(fixed_prefix, decode_p_id, mutable_ops)
    reordered = choose_best_cold(optimizer, orders, num_epochs)
    candidates = [
        (reordered, cache_after)
        for cache_after in legal_cache_points(reordered, random_p_id)
    ]
    return choose_best_total(optimizer, candidates, num_epochs)


def strategy_cache_then_reorder(
    optimizer: Optimizer,
    fixed_prefix: List[int],
    original_order: List[int],
    random_p_id: int,
    num_epochs: int,
) -> Tuple[List[int], Optional[int], PhysicalPlan, float, Optional[float], float]:
    original = original_order
    cache_candidates = [
        (original, cache_after)
        for cache_after in legal_cache_points(original, random_p_id)
    ]
    _, fixed_cache, _, _, _, _ = choose_best_total(optimizer, cache_candidates, num_epochs)
    if fixed_cache is None:
        return choose_best_total(optimizer, [(original, None)], num_epochs)

    cache_idx = original.index(fixed_cache)
    random_idx = original.index(random_p_id)
    prefix_ops = original[2 : cache_idx + 1]
    middle_ops = original[cache_idx + 1 : random_idx]
    suffix_ops = original[random_idx + 1 :]
    candidates = []
    for prefix_perm in itertools.permutations(prefix_ops):
        for suffix_perm in itertools.permutations(suffix_ops):
            order = (
                fixed_prefix
                + list(prefix_perm)
                + middle_ops
                + [random_p_id]
                + list(suffix_perm)
            )
            candidates.append((order, fixed_cache))
    return choose_best_total(optimizer, candidates, num_epochs)


def strategy_joint_reorder_cache(
    optimizer: Optimizer,
    fixed_prefix: List[int],
    decode_p_id: int,
    random_p_id: int,
    mutable_ops: List[int],
    num_epochs: int,
) -> Tuple[List[int], Optional[int], PhysicalPlan, float, Optional[float], float]:
    candidates = []
    for order in legal_orders(fixed_prefix, decode_p_id, mutable_ops):
        for cache_after in legal_cache_points(order, random_p_id):
            candidates.append((order, cache_after))
    return choose_best_total(optimizer, candidates, num_epochs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num_profile_samples", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--force_profile", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Preparing dataset at %s", args.dataset_dir)
    raw_train_dir = ensure_imagenette(args.dataset_dir)
    train_dir = prepare_image_subset(raw_train_dir, args.out_dir, args.num_profile_samples)
    profile_path = args.out_dir / f"profile_{args.num_profile_samples}.yml"
    if args.force_profile and profile_path.exists():
        profile_path.unlink()
    profile_feature(train_dir, profile_path, args.num_profile_samples)

    feature = build_feature(train_dir)
    optimizer = init_optimizer(feature, profile_path)

    source = optimizer._get_source_p_id()
    read = pipe_id_by_tag(feature, "read_file")
    d = pipe_id_by_tag(feature, "decode_jpeg")
    c = pipe_id_by_tag(feature, "random_crop_image")
    r = pipe_id_by_tag(feature, "resize_image")
    u = pipe_id_by_tag(feature, "unsharp_mask_image")
    b = pipe_id_by_tag(feature, "gaussian_filter2d")

    fixed_prefix = [source, read]
    mutable_ops = [d, c, r, u, b]
    original_order = fixed_prefix + mutable_ops
    strategy_fns = {
        "unoptimized": lambda: strategy_unoptimized(
            optimizer, original_order, c, args.num_epochs
        ),
        "reorder_then_cache": lambda: strategy_reorder_then_cache(
            optimizer, fixed_prefix, d, c, mutable_ops, args.num_epochs
        ),
        "cache_then_reorder": lambda: strategy_cache_then_reorder(
            optimizer, fixed_prefix, original_order, c, args.num_epochs
        ),
        "joint_reorder_cache": lambda: strategy_joint_reorder_cache(
            optimizer, fixed_prefix, d, c, mutable_ops, args.num_epochs
        ),
    }

    results = {
        "dataset_train_dir": str(train_dir),
        "profile_path": str(profile_path),
        "num_profile_samples": args.num_profile_samples,
        "num_epochs_for_total_cost": args.num_epochs,
        "fixed_prefix": ["LocalFSSource", "tf.io.read_file"],
        "optimized_ops": [
            "decode_jpeg",
            "random_crop_image",
            "resize_image",
            "unsharp_mask_image",
            "gaussian_filter2d",
        ],
        "operator_profile": operator_profile(feature, optimizer, [d, c, r, u, b]),
        "search_constraints": [
            "LocalFSSource and tf.io.read_file are fixed prefix operators.",
            "decode_jpeg is kept before image tensor operators.",
            "cache candidates are limited to deterministic prefixes before random_crop_image.",
            "reorder candidates enumerate legal permutations of the five image operators under these constraints.",
        ],
        "strategies": {},
    }

    for name, strategy_fn in strategy_fns.items():
        order, cache_after, plan, cold_cost, cached_cost, total_cost = strategy_fn()
        plan_path = args.out_dir / f"{name}_plan.yml"
        save_plan(plan, plan_path)
        results["strategies"][name] = {
            "plan_path": str(plan_path),
            "order_pipe_ids": order,
            "order_names": ordered_names(feature, order),
            "cache_after_pipe_id": cache_after,
            "cache_after_name": (
                feature.logical_pipes[cache_after].get_logical_name()
                if cache_after is not None
                else None
            ),
            "cold_epoch_cost": float(cold_cost),
            "cached_epoch_cost": float(cached_cost) if cached_cost is not None else None,
            "repeat_epoch_cost": float(cached_cost) if cached_cost is not None else float(cold_cost),
            "total_cost": float(total_cost),
        }

    summary_path = args.out_dir / "summary.yml"
    with summary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "strategy,cold_epoch_cost,cached_epoch_cost,repeat_epoch_cost,total_cost,plan_path\n"
        )
        for name, row in results["strategies"].items():
            f.write(
                f"{name},{row['cold_epoch_cost']},{row['cached_epoch_cost']},"
                f"{row['repeat_epoch_cost']},{row['total_cost']},{row['plan_path']}\n"
            )

    print(f"profile_path: {profile_path}")
    print(f"summary_path: {summary_path}")
    for name, row in results["strategies"].items():
        print(
            f"{name}: cold={row['cold_epoch_cost']:.6f}, "
            f"cached={row['cached_epoch_cost']}, repeat={row['repeat_epoch_cost']:.6f}, "
            f"total={row['total_cost']:.6f}, "
            f"plan={row['plan_path']}"
        )


if __name__ == "__main__":
    main()
