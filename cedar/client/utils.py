import torch
import multiprocessing as mp
import logging
import os
from ray import cloudpickle
from typing import Optional, Union, Dict

from cedar.config import CedarContext
from cedar.compose import Feature, PhysicalPlan
from cedar.pipes import PipeVariantType, DataSample

from .profiler import FeatureProfiler
from .controller import FeatureController
from .logger import DataSetLogger

logger = logging.getLogger(__name__)


class Sentinel:
    def __init__(self, idx):
        self.idx = idx


def multiprocess_worker_loop_from_serialized_feature(
    idx: int,
    ctx: CedarContext,
    queue: mp.Queue,
    startup_queue: mp.Queue,
    serialized_feature: bytes,
    feature_name: str,
    feature_plan: Optional[PhysicalPlan],
    done: mp.Event,
    epoch_start: mp.Event,
    enable_controller: bool,
    available_scale: Dict[PipeVariantType, int],
    ray_init_lock=None,
):
    """Start a spawn-safe worker for feature graphs containing callables."""
    feature = cloudpickle.loads(serialized_feature)
    multiprocess_worker_loop(
        idx,
        ctx,
        queue,
        startup_queue,
        feature,
        feature_name,
        feature_plan,
        done,
        epoch_start,
        enable_controller,
        available_scale,
        ray_init_lock,
    )


def multiprocess_worker_loop(
    idx: int,
    ctx: CedarContext,
    queue: mp.Queue,
    startup_queue: mp.Queue,
    feature: Feature,
    feature_name: str,
    feature_plan: Optional[PhysicalPlan],
    done: mp.Event,
    epoch_start: mp.Event,
    enable_controller: bool,
    available_scale: Dict[PipeVariantType, int],
    ray_init_lock=None,
):
    logger.info(f"Starting multiprocess worker {idx}...")
    torch.set_num_threads(1)
    # Give every local worker a stable cache shard inside the shared
    # workload/optimizer cache namespace. This is independent of the process
    # PID, so the same materialized cache is reusable across repeats.
    os.environ["CEDAR_CACHE_SHARD"] = feature_name

    plan_uses_ray = feature_plan is None or any(
        feature_plan.pipe_descs[p_id].variant_type
        in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
        for p_id in feature_plan.graph
    )
    if ctx.ray_config is not None and plan_uses_ray:
        logger.info(f"Initializing Ray at worker {idx}")
        # Ray driver registration is not robust when dozens of forked local
        # workers connect to the same GCS concurrently. Serialize only the
        # initialization handshake; workers execute concurrently afterwards.
        if ray_init_lock is None:
            ctx.init_ray()
        else:
            with ray_init_lock:
                ctx.init_ray()
    elif ctx.ray_config is not None:
        logger.info(f"Skipping Ray initialization at worker {idx}; plan is local-only")

    if feature_plan is not None:
        logger.info(f"Loading feature {feature_name} from plan.")
        feat = feature.load_from_plan(ctx, feature_plan)
    else:
        feat = feature.load(ctx, False)

    if enable_controller:
        path = f"/tmp/cedar_{feature_name}_log.txt"
        with open(path, "w") as _:
            pass
        ds_logger = DataSetLogger(path)
        profiler = FeatureProfiler(feature, ds_logger)
        controller = FeatureController(  # noqa: F841
            profiler=profiler,
            feature=feature,
            logger=ds_logger,
            test_mode=False,
            available_scale=available_scale,
        )

    # Report only active physical stages. Fused-away logical pipe descriptors
    # intentionally remain in the plan so FusedPipe can recover their
    # callables, but they must not count as runtime Ray operators.
    actor_counts = {}
    if feature_plan is not None:
        for p_id in feature_plan.graph:
            desc = feature_plan.pipe_descs[p_id]
            if desc.variant_type not in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            ):
                continue
            actual = feature.physical_pipes[p_id].pipe_variant.get_scale()
            expected = desc.variant_ctx.n_actors
            if actual != expected:
                raise RuntimeError(
                    f"Worker {idx} created {actual} actors for active Ray "
                    f"pipe {p_id}; expected {expected}"
                )
            actor_counts[p_id] = actual
    startup_queue.put((idx, actor_counts))

    while True:
        # Wait for dataset to signal start
        epoch_start.wait()
        epoch_start.clear()
        logger.info(f"MP worker {idx} starting epoch.")

        # For torch tensors, background process needs to be alive while
        # main process reads the queue. Keep this process alive until
        # signaled by the main process.
        if done.is_set():
            break

        for x in feat:
            if isinstance(x, DataSample):
                if x.dummy:
                    continue
                if enable_controller:
                    profiler.update_ds(x)
                queue.put(x.data)
            else:
                queue.put(x)

        logger.info(f"MP worker {idx} finished epoch.")
        queue.put(Sentinel(idx))

    logger.info(f"Terminating worker {idx}")


def unpack_feature_map(
    feature_name: str,
    feature_map: Optional[
        Union[
            str,
            Dict[
                str,
                str,
            ],
        ]
    ],
) -> Optional[str]:
    if isinstance(feature_map, dict) and feature_name in feature_map:
        map = feature_map[feature_name]
    elif isinstance(feature_map, str):
        map = feature_map
    else:
        map = None
    return map
