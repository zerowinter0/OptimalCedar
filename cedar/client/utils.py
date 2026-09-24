import torch
import multiprocessing as mp
import logging
import os
import psutil
from ray import cloudpickle
from typing import Any, List, Optional, Union, Dict

from cedar.config import CedarContext
from cedar.compose import Feature, PhysicalPlan
from cedar.pipes import PipeVariantType, DataSample

from .profiler import FeatureProfiler
from .controller import FeatureController
from .logger import DataSetLogger
from cedar.utils.threading import limit_native_threadpools

logger = logging.getLogger(__name__)


def kill_process_tree(pid: int) -> List[int]:
    """SIGKILL ``pid`` together with every live descendant.

    Local dataset workers start SMP actor processes of their own. A worker
    that has to be killed cannot clean those actors up, and an orphaned actor
    keeps holding its core for the rest of the campaign, so the whole subtree
    is killed.
    """
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []

    # Enumerate before signalling: killing the parent re-parents its children
    # to init, which would hide them from a later tree walk.
    victims = root.children(recursive=True) + [root]
    killed = []
    for victim in victims:
        try:
            victim.kill()
        except psutil.NoSuchProcess:
            continue
        killed.append(victim.pid)
    return killed


class Sentinel:
    def __init__(self, idx):
        self.idx = idx


def _collect_service_stats(feature) -> Dict[str, Any]:
    """Return worker-side service statistics per physical pipe."""
    stats: Dict[str, Any] = {}
    for p_id, pipe in getattr(feature, "physical_pipes", {}).items():
        variant = getattr(pipe, "pipe_variant", None)
        service = getattr(variant, "service", None)
        if service is None:
            service = getattr(
                getattr(variant, "variant_ctx", None), "service", None
            )
        entry: Dict[str, Any] = {}
        getter = getattr(service, "get_backend_compute_stats", None)
        if callable(getter):
            try:
                backend = getter()
            except Exception:  # noqa: BLE001
                backend = None
            if backend:
                entry.update(backend)
        # Client-side Ray path: serialize+submit per batch, and ray.get per
        # batch (queue wait + actor compute + return transfer + deserialize).
        path_getter = getattr(service, "get_path_timing_stats", None)
        if callable(path_getter):
            try:
                path = path_getter()
            except Exception:  # noqa: BLE001
                path = None
            if path:
                entry["path_timing"] = path
                entry.setdefault("method", path["method"])
                entry.setdefault("observation_unit", "sample")
                entry.setdefault("count", path["samples"])
        if entry:
            stats[str(p_id)] = entry
    return stats


def _dump_reconcile_profile(profiler, idx: int, directory: str, feature=None) -> None:
    """Write one worker's per-stage trace so the model can be reconciled."""
    import json
    import pathlib

    try:
        input_sizes, output_sizes = profiler.calculate_avg_data_size()
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        input_sizes, output_sizes = {}, {}
    # Per-pipe execution counters: how many records a worker submitted to each
    # parallel stage and how many results it consumed.  Comparing them with
    # the harness's sample count and the backend's own service time is what
    # separates "the plan did less work" from "the plan did the work faster",
    # which the aggregate throughput alone cannot distinguish.
    counters = {}
    for p_id, pipe in getattr(feature, "physical_pipes", {}).items():
        variant = getattr(pipe, "pipe_variant", None)
        if variant is None:
            continue
        issued = getattr(variant, "issued_tasks", None)
        completed = getattr(variant, "completed_tasks", None)
        if issued is None and completed is None:
            continue
        counters[str(p_id)] = {
            "issued": issued,
            "completed": completed,
            "max_inflight": getattr(variant, "max_inflight", None),
            "max_prefetch": getattr(variant, "max_prefetch", None),
        }
    payload = {
        "worker": idx,
        "samples": profiler.get_sample_count(),
        "pipe_counters": counters,
        "batch_size": profiler.get_batch_size(),
        "wall_latency_ns_per_sample": profiler.calculate_avg_wall_latency_per_sample(),
        "process_latency_ns_per_sample": profiler.calculate_avg_latency_per_sample(),
        "buffer_sizes": profiler.calculate_avg_buffer_size(),
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        # Raw per-pipe samples let the reconciliation separate service from
        # queueing instead of hiding it inside a mean.
        "wall_latency_samples": {
            str(p_id): list(values)
            for p_id, values in profiler.wall_latencies.items()
            if values
        },
        "buffer_size_samples": {
            str(p_id): list(values)
            for p_id, values in profiler.buffer_sizes.items()
            if values
        },
        "observations": {
            str(p_id): len(values)
            for p_id, values in profiler.wall_latencies.items()
        },
    }
    if feature is not None:
        payload["service_stats"] = _collect_service_stats(feature)
    try:
        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"worker_{idx}.json").write_text(json.dumps(payload))
        logger.info("Wrote reconciliation trace for worker %s", idx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write reconciliation trace for %s: %s", idx, exc)


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
    # Keep this controller alive for the entire worker.  In fork mode NumPy
    # may already have initialized OpenBLAS, so environment variables alone
    # are insufficient.
    native_threadpool_limiter = limit_native_threadpools(1)  # noqa: F841
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
        # Reconciliation also wants worker-side service timings, which are only
        # collected when the parallel variants are created in profiling mode.
        if os.environ.get("CEDAR_RECONCILE_DIR"):
            for desc in feature_plan.pipe_descs.values():
                if desc.variant_type not in (
                    PipeVariantType.RAY,
                    PipeVariantType.TF_RAY,
                    PipeVariantType.SMP,
                ):
                    continue
                variant_ctx = getattr(desc, "variant_ctx", None)
                if variant_ctx is None:
                    continue
                try:
                    variant_ctx.profile_backend_compute = True
                except Exception:  # noqa: BLE001
                    pass
                # Ray contexts build their client-side service when the plan is
                # unpickled, which happens before this point; update it too.
                service = getattr(variant_ctx, "service", None)
                if service is not None and hasattr(
                    service, "profile_backend_compute"
                ):
                    service.profile_backend_compute = True
        feat = feature.load_from_plan(ctx, feature_plan)
    else:
        feat = feature.load(ctx, False)

    # Cost-model reconciliation (P5): when CEDAR_RECONCILE_DIR is set, trace the
    # executed plan in this worker and dump per-stage wall-clock service so that
    # every modeled term can be compared against the same run that produced it.
    reconcile_dir = os.environ.get("CEDAR_RECONCILE_DIR")
    reconcile_profiler = None
    if reconcile_dir:
        try:
            # Mirror Feature.profile(): tracing data sizes requires the source
            # variant to enable profiling, otherwise the DataSample carries no
            # size dictionary and the profiler refuses the update.
            for source_pipe in getattr(feature, "source_pipes", None) or []:
                variant = source_pipe.get_variant()
                enable = getattr(variant, "enable_profiling", None)
                if callable(enable):
                    enable()
            # The profiler reads the Feature's physical pipes, so it must be
            # constructed on the feature that load_from_plan() mutated rather
            # than on the returned iterator.
            reconcile_profiler = FeatureProfiler(feature, profile_mode=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reconciliation tracing disabled: %s", exc)
            reconcile_profiler = None

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

    # Diagnostic operator capture: record the representation and the direct
    # callable time of every operator of the executing plan.  Disabled unless
    # CEDAR_OP_CAPTURE_DIR is set.
    from .op_capture import maybe_attach_operator_capture

    operator_capture = maybe_attach_operator_capture(feature, idx)

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

        reconcile_updates = 0
        for x in feat:
            if isinstance(x, DataSample):
                if x.dummy:
                    continue
                if reconcile_profiler is not None:
                    try:
                        reconcile_profiler.update_ds(x)
                        reconcile_updates += 1
                        # The driver may stop the iterator early once the
                        # requested sample count is reached and then terminate
                        # this worker, so flush the trace periodically instead
                        # of only at the end of the epoch.
                        if reconcile_updates % 10 == 0:
                            _dump_reconcile_profile(
                                reconcile_profiler,
                                idx,
                                reconcile_dir,
                                feature,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Reconciliation update skipped: %s", exc)
                        reconcile_profiler = None
                if enable_controller:
                    profiler.update_ds(x)
                queue.put(x.data)
            else:
                queue.put(x)

        if reconcile_profiler is not None:
            _dump_reconcile_profile(
                reconcile_profiler, idx, reconcile_dir, feature
            )
        if operator_capture is not None:
            operator_capture.dump()
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
