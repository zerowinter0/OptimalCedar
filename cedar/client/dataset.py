import gc
import logging
import os
import pathlib
import statistics
import threading
import multiprocessing as mp
import math
import copy
import pickle
import tempfile
import time
import yaml
import ray
import sys
from typing import Dict, Optional, Iterable, Any, List, Union, Tuple
from queue import Queue, Empty

from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions, PhysicalPlan
from cedar.compose import constants as compose_constants
from cedar.pipes import (
    BatcherPipe,
    FilterPipe,
    ImageReaderPipe,
    MapperPipe,
    Pipe,
    PipeVariant,
    DataSample,
    PipeExecutionResource,
    PipeVariantType,
    PipeVariantContext,
    InProcessPipeVariantContext,
    RayPipeVariantContext,
    TFRayPipeVariantContext,
    SMPPipeVariantContext,
)
from cedar.pipes.common import (
    ProfileInputReservoir,
    payload_compute_scale,
    payload_representation_class,
    set_profile_input_reservoir,
)
from .profiler import FeatureProfiler
from .boundary_profiler import (
    profile_object_marshalling,
    profile_stage_boundary_cached,
)
from .controller import FeatureController
from .logger import DataSetLogger
from .utils import (
    kill_process_tree,
    multiprocess_worker_loop,
    multiprocess_worker_loop_from_serialized_feature,
    Sentinel,
    unpack_feature_map,
)
from .constants import (
    RAY_PROFILE_N_ACTORS,
    RAY_PROFILE_INFLIGHT,
    RAY_PROFILE_PREFETCH,
    RAY_PROFILE_SUBMIT_BATCH_SIZE,
    AVAILABLE_RAY_SCALE,
    SMP_PROFILE_N_PROCS,
    SMP_PROFILE_INFLIGHT,
    SMP_PROFILE_PREFETCH,
)

logger = logging.getLogger(__name__)


MP_QUEUE_MAX_SIZE = 100

# A local worker that is blocked in ``result_queue.put()`` (the consumer
# stopped at ``num_total_samples``) never observes the done event, and a
# worker blocked in a threading wait does not act on SIGTERM either. Shutdown
# therefore escalates on a deadline: the interpreter joins surviving children
# without a timeout at exit, so an unbounded shutdown hangs the cell.
MP_WORKER_EXIT_GRACE_SEC = 2.0
MP_WORKER_TERMINATE_GRACE_SEC = 5.0
MP_WORKER_KILL_GRACE_SEC = 5.0


def _profile_time_sec_from_env() -> float:
    """Return the independently configurable duration of each profile stage."""
    raw = os.environ.get("CEDAR_PROFILE_TIME_SEC", "10")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "CEDAR_PROFILE_TIME_SEC must be numeric"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(
            "CEDAR_PROFILE_TIME_SEC must be finite and positive"
        )
    return value


PROFILE_TIME_SEC = _profile_time_sec_from_env()


def _minimum_parallel_epoch_records(
    variant_type: PipeVariantType,
    width: int,
    ray_batch_size: int,
    minimum_records_per_worker: Optional[int] = None,
) -> int:
    """Return an epoch size that exercises every worker sufficiently.

    A global observation count is misleading at high width: with 48 Ray
    actors and batch size 10, the old rule supplied only two batches to each
    actor.  Require a per-worker record floor and preserve complete Ray
    batches so width curves represent sustained rather than startup behavior.
    """
    if width < 1 or ray_batch_size < 1:
        raise ValueError("width and ray_batch_size must be positive")
    if variant_type in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
        per_worker = ray_batch_size * 2
        if minimum_records_per_worker is not None:
            per_worker = max(per_worker, minimum_records_per_worker)
        per_worker = (
            (per_worker + ray_batch_size - 1) // ray_batch_size
        ) * ray_batch_size
    else:
        per_worker = max(4, minimum_records_per_worker or 0)
    return width * per_worker


def _ray_profile_gpu_fraction(pipe: Pipe, width: int) -> float:
    """Reserve one GPU in total when profiling a CUDA Ray stage."""

    if width < 1:
        raise ValueError("Ray profile width must be positive")
    if pipe.execution_resource == PipeExecutionResource.CUDA:
        return 1.0 / width
    return 0.0


def _accept_profile_value(value: Any) -> bool:
    """Pickle-safe pass-through predicate for boundary profiling.

    A ``MapperPipe`` cannot be a backend-neutral identity for tuple values:
    Cedar's SMP mapper expands tuple inputs as positional arguments, whereas
    its Ray mapper passes the tuple as one value.  A filter actor evaluates
    the predicate but returns the original object unchanged on both backends.
    """
    return True


class _ProfileReplayPipeVariant(PipeVariant):
    """Finite replay source backed by immutable serialized legal inputs."""

    def __init__(self, snapshots: List[bytes], record_count: int) -> None:
        super().__init__(None)
        self.source = True
        self.snapshots = snapshots
        self.record_count = record_count
        self.variant_ctx = InProcessPipeVariantContext()

    def _iter_impl(self):
        for idx in range(self.record_count):
            value = pickle.loads(self.snapshots[idx % len(self.snapshots)])
            yield DataSample(value)

    def get_scale(self) -> int:
        return 0

    def is_scalable(self) -> bool:
        return False

    def shutdown(self) -> None:
        return


class _ProfiledFilterCallable:
    """Count filter decisions without changing normal execution variants."""

    def __init__(self, fn):
        self.fn = fn
        self.input_count = 0
        self.output_count = 0

    def __call__(self, value):
        self.input_count += 1
        keep = bool(self.fn(value))
        if keep:
            self.output_count += 1
        return keep


def _consolidate_filter_selectivity(profile: Dict[str, Any]) -> None:
    """Keep the highest-coverage conditional-selectivity observation.

    Every baseline/Ray/SMP profiling pass evaluates the same logical
    pipeline over the same source order. Backend mutations can change how
    many records a ten-second pass reaches, so the observation with the
    largest input count provides the strongest evidence without adding any
    profiling time or combining duplicated source prefixes.
    """
    baseline = profile.get("baseline")
    if not isinstance(baseline, dict):
        return
    baseline_inputs = baseline.get("input_counts")
    baseline_outputs = baseline.get("output_counts")
    if not isinstance(baseline_inputs, dict) or not isinstance(
        baseline_outputs, dict
    ):
        return

    observations = [("baseline", baseline)]
    offloads = profile.get("offloads", {})
    if isinstance(offloads, dict):
        for variant, variant_profiles in offloads.items():
            if not isinstance(variant_profiles, dict):
                continue
            for profiled_pipe_id, observation in variant_profiles.items():
                if isinstance(observation, dict):
                    observations.append(
                        (f"{variant}:{profiled_pipe_id}", observation)
                    )

    selected_inputs = {}
    selected_outputs = {}
    selected_sources = {}
    for filter_id in baseline_inputs:
        best_inputs = -1
        best_outputs = 0
        best_source = "baseline"
        for source, observation in observations:
            inputs = observation.get("input_counts", {}).get(filter_id)
            outputs = observation.get("output_counts", {}).get(filter_id)
            if (
                isinstance(inputs, int)
                and isinstance(outputs, int)
                and 0 <= outputs <= inputs
                and inputs > best_inputs
            ):
                best_inputs = inputs
                best_outputs = outputs
                best_source = source
        if best_inputs < 0:
            best_inputs = 0
        selected_inputs[filter_id] = best_inputs
        selected_outputs[filter_id] = best_outputs
        selected_sources[filter_id] = best_source

    baseline["input_counts"] = selected_inputs
    baseline["output_counts"] = selected_outputs
    baseline["selectivities"] = {
        filter_id: (
            selected_outputs[filter_id] / selected_inputs[filter_id]
            if selected_inputs[filter_id]
            else 1.0
        )
        for filter_id in selected_inputs
    }
    baseline["selectivity_observation_sources"] = selected_sources


class _DataSetIter:
    """
    Abstraction for DataSet iteration in order to allow
    for the processing of multiple epochs.

    Args:
        loaded_features: Dict of feature names to
            loaded Features.
        return_datasample: Bool indicating whether
            DataSample objects or the data contained
            in it should be returned.
        source_pipes: Dict of feature names to a
            list of corresponding source pipes.

    Attributes:
        feature_iters: Dict from feature names to
            iterable feature pipe
        feature_names: List of all feature names
        return_datasample: Bool indicating whether
            DataSample objects or the data contained
            in it should be returned.
        source_pipes: Dict of feature names to a
            list of corresponding source pipes.
        expected_output_lengths: Dict from feature
            names to the amount of expected samples
            being produced by a given feature.
        outputs_left: Dict from feature names to
            a set containg the sample IDs left to
            process.
    """

    def __init__(
        self,
        loaded_features: Dict[str, Iterable],
        profilers: Optional[Dict[str, FeatureProfiler]] = None,
        return_datasample: bool = False,
        source_pipes: Dict[str, List] = {},
    ) -> None:
        self.feature_iters = {k: iter(f) for k, f in loaded_features.items()}
        self.feature_names = list(self.feature_iters.keys())
        self.feature_profilers = profilers
        self._return_datasample = return_datasample

        if len(loaded_features) != 1:
            raise NotImplementedError

        # NOTE: Multiple source pipes not supported yet
        self.source_pipes = source_pipes
        for feature_name, feature_sources in self.source_pipes.items():
            for source in feature_sources:
                source.pipe_variant.reset_for_new_epoch()

        # TODO: Think about how to handle last partition

        # map from partition id to set of received samples for given partition;
        # partition is marked as sealed when all samples have been received
        self.partitions_received = {}

    def _get_source_partition_size(self):
        # NOTE: Multiple sources per feature not supported yet
        logging.warning(
            "Using deprecated function _get_source_partition_size!"
        )
        partition_sizes = {}
        for feature_name, source_pipes in self.source_pipes.items():
            partition_sizes[feature_name] = source_pipes[
                0
            ].pipe_variant.get_num_samples_in_partition()
        return partition_sizes

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: This is getting pretty heavy... clean this up
        f_name = self.feature_names[0]  # support only one feature for now
        samples_per_partition = self.source_pipes[f_name][
            0
        ].pipe_variant.get_num_samples_in_partition()
        try:
            ds = next(self.feature_iters[f_name])
            try:
                while ds.dummy:
                    ds = next(self.feature_iters[f_name])
                self.feature_profilers[f_name].update_ds(ds)
                if ds.sample_id is not None:
                    # NOTE: Deactive tracking for cache reads
                    if not ds.read_from_cache:
                        self._mark_sample_id_as_received(
                            ds,
                            samples_per_partition,
                            self.source_pipes[f_name][
                                0
                            ],  # NOTE: only one source pipe
                        )
                return ds if self._return_datasample else ds.data
            except AttributeError:
                return ds
            except TypeError:
                return ds
        except StopIteration:
            source_pipe_variant = self._get_source_pipe_variant()
            if len(source_pipe_variant.get_in_flight_partitions()) > 0:
                source_pipe_variant.seal_last_partition()

            # Wait for any ongoing mutations to finish
            raise StopIteration

    def _mark_sample_id_as_received(
        self, sample: DataSample, samples_per_partition: int, source_pipe: Pipe
    ) -> None:
        sample_id = sample.sample_id
        ds_partition_id = sample_id // samples_per_partition
        if ds_partition_id not in self.partitions_received:
            self.partitions_received[ds_partition_id] = set()
        if sample_id in self.partitions_received[ds_partition_id]:
            raise RuntimeError(f"Sample with ID {sample_id} received twice.")

        self.partitions_received[ds_partition_id].add(sample_id)

        if (
            len(self.partitions_received[ds_partition_id])
            == samples_per_partition
        ):
            source_pipe.pipe_variant.seal_partition(ds_partition_id)
            # TODO: Maybe delete data about this source here?

    # NOTE: Currently only one source for one feature supported
    def _get_source_pipe_variant(self) -> PipeVariant:
        f_name = self.feature_names[0]  # support only one feature for now
        source_pipe = self.source_pipes[f_name][0].pipe_variant
        return source_pipe

    def checkpoint_partitions(
        self, checkpoint_only_sealed: bool = True
    ) -> None:
        """
        Checkpoints partitions by saving partition information to pkl file.
        If checkpoint_only_sealed is True, only sealed partition information
        is stored. Otherwise, information about in-flight and fully-sent
        partitions is also stored.
        """
        source_pipe_variant = self._get_source_pipe_variant()
        source_pipe_variant.checkpoint_partitions(checkpoint_only_sealed)

    def are_partitions_remaining(self) -> bool:
        """
        Checks whether there are any samples sent by the source,
        but not fully consumed by the iterator. True if there
        are such samples. False otherwise.
        """
        source_pipe_variant = self._get_source_pipe_variant()
        empty_in_flight = (
            len(source_pipe_variant.get_in_flight_partitions()) == 0
        )
        empty_fully_sent = (
            len(source_pipe_variant.get_fully_sent_partitions()) == 0
        )
        return empty_fully_sent and empty_in_flight

    def get_source_pipes(self) -> Dict[str, List[Pipe]]:
        """
        Returns a dictionary, mapping feature names to the
        list of source pipes. Should only be used for testing!
        """
        logging.warning(
            "Using function get_source_pipes: Should only be used\
                        for testing purposes."
        )

        return self.source_pipes

    def get_feature_names(self) -> List[str]:
        """
        Returns a list of strings, containing the feature names
        of this DataSetIter. Should only be used for testing!
        """
        logging.warning(
            "Using function get_feature_names: Should only be used\
                        for testing purposes."
        )

        return self.feature_names


class _ThreadedDataSetIter:
    """
    Runs the entire pipeline in a thread.
    """

    def __init__(self, features: List[Iterable]):
        self.threads = []
        self.queue = Queue()
        self.features = features

    def __iter__(self):
        # Start threads
        logger.info("Calling iter on dataset iter")
        for feature in self.features:
            t = threading.Thread(
                target=self._worker_fn, args=(feature, self.queue)
            )
            t.start()
            self.threads.append(t)
        return self

    def __next__(self):
        while (
            any(t.is_alive() for t in self.threads) or not self.queue.empty()
        ):
            try:
                return self.queue.get(timeout=1)
            except Empty:
                continue
        else:
            raise StopIteration

    def _worker_fn(self, feature: Iterable, queue: Queue):
        logger.info("Starting worker thread")
        for x in feature:
            if x.dummy:
                continue
            queue.put(x.data)


class _MultiprocessDataSetIter:
    """
    This Iterable manages a pool of processes, each of which executes an entire
    feature.
    """

    def __init__(
        self,
        ctx: CedarContext,
        features: Dict[str, Feature],
        plans: Optional[Dict[str, PhysicalPlan]],
        enable_controller: bool,
    ):
        self._ctx = ctx
        self._plans = plans
        # Ray and gRPC create background native threads. Forking after those
        # libraries have been imported can copy inconsistent synchronization
        # state into a child and crash in ray.init(). Use spawn whenever a
        # worker can initialize Ray. Preserve the default context for strictly
        # local plans, which may legitimately contain non-pickleable callables.
        worker_can_use_ray = ctx.ray_config is not None and (
            plans is None
            or any(
                any(
                    plan.pipe_descs[p_id].variant_type
                    in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
                    for p_id in plan.graph
                )
                for plan in plans.values()
            )
        )
        self._mp_ctx = (
            mp.get_context("spawn") if worker_can_use_ray else mp.get_context()
        )
        self._spawn_ray_workers = worker_can_use_ray
        self._result_queue = self._mp_ctx.Queue(maxsize=MP_QUEUE_MAX_SIZE)
        self._startup_queue = self._mp_ctx.Queue()

        self._workers = {}
        self._features = features
        self._done = self._mp_ctx.Event()
        self._num_done = 0
        self._epoch_active = False
        self._enable_controller = enable_controller

        self._worker_epoch_start = {}
        self._ray_init_lock = self._mp_ctx.Lock()

        try:
            self._init_workers()
        except BaseException:
            self._shutdown()
            raise

    def _init_workers(self):
        idx = 0

        ray_parallelism = math.ceil(AVAILABLE_RAY_SCALE / len(self._features))
        for f_name, feature in self._features.items():
            if self._plans is not None:
                plan = self._plans[f_name]
            else:
                plan = None
            epoch_start = self._mp_ctx.Event()
            self._worker_epoch_start[idx] = epoch_start
            if self._spawn_ray_workers:
                from ray import cloudpickle

                worker_target = multiprocess_worker_loop_from_serialized_feature
                worker_feature = cloudpickle.dumps(feature)
            else:
                worker_target = multiprocess_worker_loop
                worker_feature = feature
            worker = self._mp_ctx.Process(
                target=worker_target,
                args=(
                    idx,
                    self._ctx,
                    self._result_queue,
                    self._startup_queue,
                    worker_feature,
                    f_name,
                    plan,
                    self._done,
                    epoch_start,
                    self._enable_controller,
                    {PipeVariantType.RAY: ray_parallelism},
                    self._ray_init_lock,
                ),
            )
            # Workers may load plans containing SMP operators. SMP variants
            # start their own child processes, which Python forbids from a
            # daemon process. Shutdown is handled explicitly by _shutdown().
            worker.daemon = False
            worker.start()
            self._workers[idx] = worker
            idx += 1

        self._await_workers_ready()

    def _await_workers_ready(self, timeout_sec: Optional[float] = None) -> None:
        """Wait until every worker has constructed and verified its stages.

        A plan that gives every worker its own actor pool makes the remote node
        start hundreds of processes at once, which needs longer than the
        historical 180 s (COCO's 32-worker plan needs more, and the same limit
        makes the 10k-record text subsets fail).  ``CEDAR_WORKER_READY_TIMEOUT_SEC``
        raises the bound without changing what is measured.
        """
        if timeout_sec is None:
            timeout_sec = float(
                os.environ.get("CEDAR_WORKER_READY_TIMEOUT_SEC", "600")
            )
        reports = {}
        deadline = time.monotonic() + timeout_sec
        while len(reports) < len(self._workers):
            failed_workers = [
                (idx, worker.exitcode)
                for idx, worker in self._workers.items()
                if worker.exitcode is not None
            ]
            if failed_workers:
                raise RuntimeError(
                    "Multiprocess dataset worker exited during startup: "
                    f"{failed_workers}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Timed out waiting for multiprocess dataset workers to "
                    "initialize their execution stages"
                )
            try:
                worker_idx, actor_counts = self._startup_queue.get(
                    timeout=min(0.1, remaining)
                )
            except Empty:
                continue
            if worker_idx in reports:
                raise RuntimeError(
                    f"Worker {worker_idx} sent duplicate startup reports"
                )
            reports[worker_idx] = actor_counts

        expected = {}
        if self._plans is not None:
            for plan in self._plans.values():
                for p_id in plan.graph:
                    desc = plan.pipe_descs[p_id]
                    if desc.variant_type in (
                        PipeVariantType.RAY,
                        PipeVariantType.TF_RAY,
                    ):
                        expected[p_id] = (
                            expected.get(p_id, 0) + desc.variant_ctx.n_actors
                        )

        actual = {}
        for actor_counts in reports.values():
            for p_id, count in actor_counts.items():
                actual[p_id] = actual.get(p_id, 0) + count
        if actual != expected:
            raise RuntimeError(
                "Ray actor startup accounting mismatch: "
                f"expected={expected}, actual={actual}"
            )
        if actual:
            logger.info(
                "Verified global Ray actor counts for all active stages: %s",
                actual,
            )

    def __iter__(self):
        # A consumer is allowed to stop an epoch early (for example, the
        # cache-materialization pass stops after num_total_samples).  The
        # workers may already have queued the remaining data and their epoch
        # sentinels.  Reusing this iterator without draining those messages
        # makes stale sentinels count toward the next epoch and can terminate
        # it before all of its samples are delivered.
        if self._epoch_active:
            logger.info("Draining unfinished MP epoch before starting a new one...")
            self._drain_epoch()

        logger.info("New epoch for MP iter...")
        for _, event in self._worker_epoch_start.items():
            # Signal all workers to start next epoch
            event.set()
        self._num_done = 0
        self._epoch_active = True
        return self

    def _get_result(self):
        while self._num_done < len(self._workers):
            try:
                data = self._result_queue.get(timeout=0.1)
                if isinstance(data, Sentinel):
                    self._num_done += 1
                else:
                    return True, data
            except Empty:
                failed_workers = [
                    (idx, worker.exitcode)
                    for idx, worker in self._workers.items()
                    if worker.exitcode is not None
                ]
                if failed_workers:
                    raise RuntimeError(
                        "Multiprocess dataset worker exited before completing "
                        f"its epoch: {failed_workers}"
                    )
                continue

        # multiprocessing.Queue.empty() is intentionally not used here: its
        # result is not reliable across processes.  Every worker enqueues its
        # sentinel after its data, and multiprocessing.Queue preserves each
        # producer's order, so receiving all worker sentinels is the precise
        # epoch boundary.
        self._epoch_active = False
        return False, None

    def _drain_epoch(self):
        while self._epoch_active:
            has_data, _ = self._get_result()
            if not has_data:
                break

    def __next__(self):
        has_data, data = self._get_result()
        if not has_data:
            logger.info("Finished fetching from workers...")
            raise StopIteration
        return data

    def _shutdown(self):
        if not self._workers:
            return

        workers = dict(self._workers)
        self._done.set()
        for _, event in self._worker_epoch_start.items():
            # Need to signal start for workers to check for done signal
            event.set()

        # Workers can be blocked in queue.put() after the consumer stops at
        # num_total_samples. Every escalation step is bounded, and the process
        # tree of a worker that ignores SIGTERM is killed, so the driver can
        # never end up waiting for a child at interpreter exit.
        self._join_workers(workers, MP_WORKER_EXIT_GRACE_SEC)
        for idx in self._alive_workers(workers):
            logger.info(f"Terminating worker {idx}...")
            workers[idx].terminate()
        self._join_workers(workers, MP_WORKER_TERMINATE_GRACE_SEC)
        for idx in self._alive_workers(workers):
            logger.warning(
                f"Worker {idx} ignored SIGTERM; killing its process tree "
                f"(pids {kill_process_tree(workers[idx].pid)})"
            )
        self._join_workers(workers, MP_WORKER_KILL_GRACE_SEC)
        survivors = self._alive_workers(workers)
        if survivors:
            raise RuntimeError(
                "Dataset workers survived SIGKILL and would block interpreter "
                f"exit: {[(idx, workers[idx].pid) for idx in survivors]}"
            )

        self._workers.clear()
        self._worker_epoch_start.clear()
        self._result_queue.cancel_join_thread()
        self._result_queue.close()
        self._startup_queue.close()

    @staticmethod
    def _alive_workers(workers):
        return [idx for idx, w in workers.items() if w.is_alive()]

    @staticmethod
    def _join_workers(workers, timeout_sec):
        """Join every worker against one shared deadline."""
        deadline = time.monotonic() + timeout_sec
        for _, w in workers.items():
            w.join(max(0.0, deadline - time.monotonic()))

    def __del__(self):
        self._shutdown()


class DataSet:
    """
    A DataSet is the user (i.e., ML job) facing API of cedar.
    It represents a collection of Features, and coordinates
    with a set of executors to retrieve preprocessed data
    for each feature.

    It exposes an iterator interface to allow iteration
    of samples that have been transformed by
    the corresponding feature(s).

    Args:
        ctx: CedarContext containing runtime context.
        features: Dict of feature name to
            Features that compose the DataSet.
        feature_config: If provided, map of feature name
            to path of yaml file for that feature,
            containing a physical plan. The feature
            will be loaded according to the plan.
        prefetch: Insert a prefetch pipe at the end of each feature.
            Defaults to true. This option only applies if feature_config is not
            provided and optimizer is disabled.

    Attributes:
        ctx: CedarContext containing runtime context.
        features: Dict of feature name to
            Features that compose the DataSet.
        feature_names: List of all feature names.
        feature_config: Dict of feature name to
            file with saved plan config.
        loaded_features: Dict of feature names to
            loaded Features.
        source_pipes: Dict of feature names to a
            list of corresponding source pipes.
        curr_epoch: Int representing current training epoch.
        iter_mode: String specifying which iterator to use.
            "default": Default iterator, runs in main process
            "thread": Threaded iterator, runs feature in thread
            "mp": Multiprocess iterator, runs feature in process
        enable_optimizer: True if the dataset should enable the static
            optimizer
        profiled_data: Dict mapping feature name to path to YAML with
            profiled results.
        run_profiling: If true, only run profiler and exit immediately
        optimizer_options: Options for the optimizer
        generate_plan: If true, only run optimizer and exit immediately
    """

    def __init__(
        self,
        ctx: CedarContext,
        features: Dict[str, Feature],
        feature_config: Optional[Union[str, Dict[str, str]]] = None,
        prefetch: bool = True,
        enable_controller: bool = False,
        test_mode: bool = False,
        iter_mode: str = "default",
        enable_optimizer: bool = False,
        profiled_data: Optional[str] = None,
        run_profiling: bool = False,
        optimizer_options: Optional[OptimizerOptions] = None,
        generate_plan: bool = False,
    ):
        self._log_file = pathlib.Path("/tmp/cedar_log.txt")
        # Overwrite the file
        with open(self._log_file.as_posix(), "w"):
            pass
        self._logger = DataSetLogger(self._log_file.as_posix())

        self.ctx = ctx
        self.prefetch = prefetch
        self.features = features
        self.feature_names = list(self.features.keys())
        self.curr_epoch = -1
        self.dataset_iter = None
        self.test_mode = test_mode
        self.enable_optimizer = enable_optimizer
        self.optimizer_options = optimizer_options
        self._iter_mode = iter_mode
        self._test_iter = False
        self.feature_plans = None
        self.ctx_initialized = False

        self._mp_iter = None

        # Create feature plans
        self.use_config = feature_config is not None
        self._load_config(feature_config)

        # Optionally swap the optimizer implementation. The selector is:
        # 0/default Optimizer, 1/MyOptimizer, 2/DpOptimizer, 3/DjOptimizer,
        # 4/DpTwoStageOptimizer, 5/DpCedarOptimizer, 6/CedarJointOptimizer,
        # 7/ExpOptimizer, 8/PecanOptimizer, 9/PecanTwoStageOptimizer,
        # 10/DjTwoStageOptimizer, 11/SimpleDpOptimizer,
        # 12/SequentialExhaustiveOptimizer, 13/MinimalParallelDpOptimizer,
        # 14/SingleWorkerCudaDpOptimizer, 15/SimpleDpRayCandidateOptimizer, 16/CmOptimizer.
        optimizer_selector = 0
        if self.optimizer_options is not None:
            optimizer_selector = int(
                getattr(self.optimizer_options, "use_my_optimizer", 0)
            )
        self._legacy_cedar_profile = optimizer_selector == 11
        self._cm_profile = optimizer_selector in (2, 16)
        if optimizer_selector == 1:
            from cedar.compose.my_optimizer import MyOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(MyOptimizer())
        elif optimizer_selector == 2:
            from cedar.compose.dp_optimizer import DpOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(DpOptimizer())
        elif optimizer_selector == 3:
            from cedar.compose.dj_optimizer import DjOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(DjOptimizer())
        elif optimizer_selector == 4:
            from cedar.compose.dp_two_stage_optimizer import DpTwoStageOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(DpTwoStageOptimizer())
        elif optimizer_selector == 5:
            from cedar.compose.dp_cedar_optimizer import DpCedarOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(DpCedarOptimizer())
        elif optimizer_selector == 6:
            from cedar.compose.cedar_joint_optimizer import CedarJointOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(CedarJointOptimizer())
        elif optimizer_selector == 7:
            from cedar.compose.exp_optimizer import ExpOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(ExpOptimizer())
        elif optimizer_selector == 8:
            from cedar.compose.pecan_optimizer import PecanOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(PecanOptimizer())
        elif optimizer_selector == 9:
            from cedar.compose.policy_two_stage_optimizer import (
                PecanTwoStageOptimizer,
            )

            for _, feature in self.features.items():
                feature.set_optimizer(PecanTwoStageOptimizer())
        elif optimizer_selector == 10:
            from cedar.compose.policy_two_stage_optimizer import (
                DjTwoStageOptimizer,
            )

            for _, feature in self.features.items():
                feature.set_optimizer(DjTwoStageOptimizer())
        elif optimizer_selector == 11:
            from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(SimpleDpOptimizer())
        elif optimizer_selector == 12:
            from cedar.compose.sequential_exhaustive_optimizer import (
                SequentialExhaustiveOptimizer,
            )

            for _, feature in self.features.items():
                feature.set_optimizer(SequentialExhaustiveOptimizer())
        elif optimizer_selector == 13:
            from cedar.compose.sequential_exhaustive_optimizer import (
                MinimalParallelDpOptimizer,
            )

            for _, feature in self.features.items():
                feature.set_optimizer(MinimalParallelDpOptimizer())
        elif optimizer_selector == 14:
            from cedar.compose.sequential_exhaustive_optimizer import (
                SingleWorkerCudaDpOptimizer,
            )

            for _, feature in self.features.items():
                feature.set_optimizer(SingleWorkerCudaDpOptimizer())
        elif optimizer_selector == 15:
            from cedar.compose.simple_dp_ray_candidate_optimizer import (
                SimpleDpRayCandidateOptimizer,
            )

            for _, feature in self.features.items():
                feature.set_optimizer(SimpleDpRayCandidateOptimizer())
        elif optimizer_selector == 16:
            from cedar.compose.cm_optimizer import CmOptimizer
            for feature in self.features.values():
                feature.set_optimizer(CmOptimizer())
        elif optimizer_selector == 17:
            from cedar.compose.old_dp_optimizer import OldDpOptimizer
            for feature in self.features.values():
                feature.set_optimizer(OldDpOptimizer())
        elif optimizer_selector == 18:
            from cedar.compose.plumber_optimizer import PlumberOptimizer
            for feature in self.features.values():
                feature.set_optimizer(PlumberOptimizer())
        elif optimizer_selector == 19:
            from cedar.compose.raydata_optimizer import RayDataOptimizer
            for feature in self.features.values():
                feature.set_optimizer(RayDataOptimizer())
        elif optimizer_selector in (20, 21, 22, 23, 24, 25, 26, 27, 28, 29):
            from cedar.compose.simple_dp_ablation_optimizer import (
                SimpleDpWorkersOptimizer, SimpleDpBoundaryOptimizer,
                SimpleDpVariantOptimizer, SimpleDpWidthOptimizer,
                UnoptimizedOptimizer, SimpleDpWorkersBoundaryOptimizer,
                SimpleDpMaxWorkersBoundaryOptimizer,
                SimpleDpWorkersWidthBoundaryOptimizer,
                OldDpBoundaryOptimizer, LayeredSimpleDpOptimizer,
            )
            cls = (SimpleDpWorkersOptimizer, SimpleDpBoundaryOptimizer,
                   SimpleDpVariantOptimizer, SimpleDpWidthOptimizer,
                   UnoptimizedOptimizer,
                   SimpleDpWorkersBoundaryOptimizer,
                   SimpleDpMaxWorkersBoundaryOptimizer,
                   SimpleDpWorkersWidthBoundaryOptimizer,
                   OldDpBoundaryOptimizer, LayeredSimpleDpOptimizer,
                   )[optimizer_selector - 20]
            for feature in self.features.values():
                feature.set_optimizer(cls())
        elif optimizer_selector in (31, 32, 33):
            from cedar.compose.staged_ablation_optimizer import (
                StagedBoundaryOptimizer,
                StagedBoundaryAffineOptimizer,
                StagedWorkersBoundaryAffineOptimizer,
            )
            staged_cls = (
                StagedBoundaryOptimizer,
                StagedBoundaryAffineOptimizer,
                StagedWorkersBoundaryAffineOptimizer,
            )[optimizer_selector - 31]
            for feature in self.features.values():
                feature.set_optimizer(staged_cls())
        elif optimizer_selector in (34, 35, 36, 37):
            from cedar.compose.simple_dp_ablation_optimizer import (
                SimpleDpBoundaryAffineElementsOptimizer,
                SimpleDpBoundaryAffineReprOptimizer,
                SimpleDpBoundaryAffineReprProportionalOptimizer,
                SimpleDpWorkersWidthBoundaryAffineReprOptimizer,
            )
            repr_cls = (
                SimpleDpBoundaryAffineElementsOptimizer,
                SimpleDpBoundaryAffineReprProportionalOptimizer,
                SimpleDpBoundaryAffineReprOptimizer,
                SimpleDpWorkersWidthBoundaryAffineReprOptimizer,
            )[optimizer_selector - 34]
            for feature in self.features.values():
                feature.set_optimizer(repr_cls())
        elif optimizer_selector in (38, 39, 40, 41):
            from cedar.compose.simple_dp_ablation_optimizer import (
                SimpleDpAffineReprOptimizer,
                SimpleDpWorkersAffineReprOptimizer,
                SimpleDpWorkersBoundaryAffineReprOptimizer,
                SimpleDpWorkersByteProportionalOptimizer,
            )
            final_cls = (
                SimpleDpAffineReprOptimizer,
                SimpleDpWorkersBoundaryAffineReprOptimizer,
                SimpleDpWorkersAffineReprOptimizer,
                SimpleDpWorkersByteProportionalOptimizer,
            )[optimizer_selector - 38]
            for feature in self.features.values():
                feature.set_optimizer(final_cls())
        elif optimizer_selector == 42:
            from cedar.compose.staged_ablation_optimizer import (
                StagedWorkersBoundaryAffineReprOptimizer,
            )
            for feature in self.features.values():
                feature.set_optimizer(StagedWorkersBoundaryAffineReprOptimizer())
        elif optimizer_selector != 0:
            raise ValueError(
                "OptimizerOptions.use_my_optimizer must be 0-29, 31-42."
            )

        if len(self.features) == 0:
            raise ValueError("No features provided")
        if len(self.features) != 1 and self._iter_mode == "default":
            raise NotImplementedError(
                "Can only create a dataset with one feature."
            )  # noqa: E501

        if self.use_config and enable_optimizer:
            raise RuntimeError("Cannot load from config and use optimizer")

        if (
            profiled_data is None
            and not run_profiling
            and self.ctx.use_ray()
            and not self.use_config
        ):
            raise ValueError(
                "Cannot use ray without profiled data. "
                "First run profiling and provide the YAML file."
            )

        self.enable_controller = enable_controller

        if run_profiling:
            # Just run profiling and exit
            for f_name in self.feature_names:
                if profiled_data is None or profiled_data == "":
                    profiled_data = f"/tmp/{f_name}_profile.yml"
                self._profile(f_name, output_file=profiled_data)
                self.features[f_name].to_yaml(f"/tmp/cedar_{f_name}_plan.yml")
            exit(0)

        # If the optimizer is enabled, we need profiled data
        if self.enable_optimizer:
            self._run_optimizer(profiled_data)
            if generate_plan:
                exit(0)

        # Initialize context if necessary
        self._init_ctx()

        self._init_features()

    def _load_config(
        self, feature_config
    ) -> Optional[Dict[str, PhysicalPlan]]:
        if feature_config is None:
            return

        self.feature_plans = {}
        for f_name in self.feature_names:
            feature_config = unpack_feature_map(f_name, feature_config)
            with open(feature_config, "r") as f:
                d = yaml.safe_load(f)

            plan = PhysicalPlan.from_dict(d["physical_plan"])
            logger.info(
                f"Using feature config {feature_config} for feature {f_name}."
            )

            if plan.n_local_workers > 1:
                if len(self.feature_names) > 1:
                    raise RuntimeError(
                        "Cannot use multiple workers with multiple features"
                    )
                self._shard_feature(plan)
                break
            else:
                self.feature_plans[f_name] = plan

    def _run_optimizer(self, profiled_data: Optional[str]):
        if self._iter_mode == "mp" or self._iter_mode == "thread":
            raise RuntimeError("Cannot optimize non-default iter.")
        if len(self.features) != 1:
            raise RuntimeError("Cannot optimize more than 1 feature.")
        if self.feature_plans is not None:
            raise RuntimeError("Running optimizer and config provided.")

        f_name = self.feature_names[0]
        feature = self.features[f_name]
        self.feature_plans = {}

        # Don't automatically run profiler in test mode
        if not self.test_mode and (
            profiled_data is None or profiled_data == ""
        ):
            raise RuntimeError(
                "Profiled data not provided. "
                "Please run profile and provide YAML file."
            )

        # Run the optimizer for each feature
        # Always enable prefetching if using optimizer
        if self.optimizer_options is None:
            self.optimizer_options = OptimizerOptions(
                enable_prefetch=True,
                est_throughput=None,
                available_local_cpus=mp.cpu_count() - 1,
            )

        plan = feature.optimize(
            self.optimizer_options,
            profiled_data,
        )

        # If the plan calls for more than 1 local worker, duplicate and shard
        # the feature
        if plan.n_local_workers > 1:
            self._shard_feature(plan)
        else:
            self.feature_plans[f_name] = plan

        # Save the plan
        save_path = "/tmp/cedar_optimized_plan.yml"
        logger.info(f"Saving optimized plan to {save_path}")
        p = plan.to_dict()

        # For fused pipes, variant is not set, set to inprocess
        for _, p_dict in p["pipes"].items():
            if "variant" not in p_dict:
                p_dict["variant"] = "INPROCESS"
        with open(save_path, "w") as f:
            yaml.dump({"physical_plan": p}, f)

    def _shard_feature(self, plan: PhysicalPlan):
        if len(self.features) != 1:
            raise RuntimeError("Cannot shard more than 1 feature.")
        if plan.n_local_workers < 2:
            raise RuntimeError("Cannot shard with fewer than 2 workers")

        f_name = self.feature_names[0]
        feature = self.features[f_name]
        self.feature_plans[f_name] = plan

        self.feature_names = []
        self.features = {}
        self.feature_plans = {}

        for i in range(plan.n_local_workers):
            rank_spec = (plan.n_local_workers, i)
            logger.info(rank_spec)
            feature_copy = feature.create_copy()
            f_name_copy = f_name + f"_r{i}"
            feature_copy.shard_source(rank_spec)

            self.feature_names.append(f_name_copy)
            self.features[f_name_copy] = feature_copy
            self.feature_plans[f_name_copy] = plan

        self._iter_mode = "mp"

    def _init_features(self):
        if self._iter_mode != "mp":
            self.loaded_features = self._load_features()
            self.source_pipes = self._get_feature_source_pipes()
        else:
            self.loaded_features = None
            self.source_pipes = None

            self._logger.log(
                f"Using MP with {len(self.features)} local workers"
            )
            if self.feature_plans is not None:
                for f_name, plan in self.feature_plans.items():
                    self._logger.log(f"Feature {f_name}")
                    self._logger.log(f"Physical Plan: {plan.graph}")
                    self._logger.log(f"n_local_workers {plan.n_local_workers}")
                    for p_id, desc in plan.pipe_descs.items():
                        self._logger.log(f"Pipe {p_id} = {desc.serialize()}")

        # Feature profiling/controllers
        if self._iter_mode == "default":
            self.profilers = {
                f_name: FeatureProfiler(
                    self.features[f_name], logger=self._logger
                )
                for f_name in self.feature_names
            }
        if (
            self.enable_controller
            # and not self.use_config
            and self._iter_mode == "default"
        ):
            self.controllers = {
                f_name: FeatureController(
                    self.profilers[f_name],
                    self.features[f_name],
                    logger=self._logger,
                    test_mode=self.test_mode,
                    available_scale={PipeVariantType.RAY: AVAILABLE_RAY_SCALE},
                )
                for f_name in self.feature_names
            }

        # Return the raw datasample, for testing
        self._return_datasample = False

    def __iter__(self):
        self.curr_epoch += 1
        logger.info(
            f"Creating new iterator (epoch {self.curr_epoch}) for DataSet."
        )
        if self._iter_mode == "thread":
            logger.warning("Using thread iterable. Use with caution!")
            features = [v for k, v in self.loaded_features.items()]
            self.dataset_iter = iter(_ThreadedDataSetIter(features))
        elif self._iter_mode == "mp":
            # Don't create a new dataset_iter, keep proc alive
            if self._mp_iter is None:
                self._mp_iter = _MultiprocessDataSetIter(
                    self.ctx,
                    self.features,
                    self.feature_plans,
                    self.enable_controller,
                )
            self.dataset_iter = iter(self._mp_iter)
        elif self._iter_mode == "default":
            if not self._test_iter:
                self.dataset_iter = _DataSetIter(
                    loaded_features=self.loaded_features,
                    profilers=self.profilers,
                    return_datasample=self._return_datasample,
                    source_pipes=self.source_pipes,
                )
        else:
            raise ValueError(
                "Unsupported iter mode {}".format(self._iter_mode)
            )

        return self.dataset_iter

    def _load_features(self):
        loaded_features = {}
        for f_name in self.feature_names:
            if self.use_config or self.enable_optimizer:
                plan = self.feature_plans[f_name]
                feat = self.features[f_name].load_from_plan(self.ctx, plan)
            else:
                feat = self.features[f_name].load(
                    ctx=self.ctx,
                    prefetch=self.prefetch,
                )

            loaded_features[f_name] = feat

            # Log the loaded features
            self._logger.log("Feature {} Logical Plan...".format(f_name))
            logical_plan, physical_plan = self.features[
                f_name
            ].serialize_plan()
            self._logger.log(str(logical_plan))
            if physical_plan is not None:
                self._logger.log("Physical Plan...")
                self._logger.log(str(physical_plan))

        return loaded_features

    def _get_feature_source_pipes(self):
        """
        Gets source pipes for each feature.
        Should be called after load_features.
        NOTE: Support for multiple pipes not yet implemented.
        """
        source_pipes = {}
        for f_name in self.feature_names:
            source_pipes[f_name] = self.features[f_name].get_source_pipes()

        return source_pipes

    def viz_logical_plan(self, path: str):
        """
        Visualizes the logical plans for all
        features of the DataSet and saves them
        to the given path.
        """
        for f_name, f in self.features.items():
            log_path = pathlib.Path(path) / f"{f_name}_log_plan.png"
            f.viz_logical_plan(str(log_path))

    def viz_physical_plan(self, path: str):
        """
        Visualizes the physical plans for all
        features of the DataSet and saves thems
        to the given path.
        """
        for f_name, f in self.features.items():
            phys_path = pathlib.Path(path) / f"{f_name}_phys_plan.png"
            f.viz_physical_plan(str(phys_path))

    def save_config(self, path: str):
        """
        Saves the feature config for all
        features of the DataSet to a yaml
        file at the given path.
        """
        for f_name, f in self.features.items():
            config_path = pathlib.Path(path) / f"{f_name}_config.yaml"
            print("Saving config to {}".format(config_path))
            f.to_yaml(str(config_path))

    def get_plan(self):
        """
        Returns a dict mapping each feature name to its plan.
        """
        d = {}
        for f_name, f in self.features.items():
            plan = f.serialize_plan()
            d[f_name] = plan
        return d

    def load_feature_from_dict(
        self, name: str, plan: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Load a specific feature from a physical plan.
        Args:
            name: feature name
            plan: Dict representing physical plan of feature
        """
        self.loaded_features[name] = self.features[name].load_from_dict(
            self.ctx, plan
        )

    def reset_feature(self, name: str):
        """
        Resets the physical plan of a given feature.
        Args:
            name: feature name
        """
        self.features[name].reset()

    def save_plan(self):
        """
        Saves the physical plan to disk.
        """
        raise NotImplementedError

    def check_remaining_samples(self) -> bool:
        """
        Checks whether there are any samples sent by the source,
        but not fully consumed by the iterator. True if there
        are such samples. False otherwise.
        """
        if self.dataset_iter is None:
            raise RuntimeError("DataSetIter not yet created.")
        return self.dataset_iter.are_partitions_remaining()

    def checkpoint(self, checkpoint_only_sealed: bool) -> None:
        """
        Checkpoints partitions by saving partition information to pkl file.
        If checkpoint_only_sealed is True, only sealed partition information
        is stored. Otherwise, information about in-flight and fully-sent
        partitions is also stored.
        """
        if self.dataset_iter is None:
            raise RuntimeError("DataSetIter not yet created.")
        self.dataset_iter.checkpoint_partitions(checkpoint_only_sealed)

    def _del_iter(self):
        """
        Explicitly deeletes the _DataSetIter stored by this DataSet.
        Should only be used for testing purposes!
        """
        del self.dataset_iter

    def _get_source_pipes(self) -> Dict[str, List[Pipe]]:
        """
        Returns a dictionary, mapping feature names to the
        list of source pipes. Should only be used for testing!
        """
        return self.dataset_iter.get_source_pipes()

    def _get_feature_names(self) -> List[str]:
        """
        Returns a list of strings, containing the feature names
        of this DataSetIter. Should only be used for testing!
        """
        return self.dataset_iter.get_feature_names()

    def _create_dataset_iter(self) -> None:
        """
        Creates the _DataSetIter. Only used for testing!
        """
        logging.warning(
            "Creating _DataSetIter explicitly.\
                Should only be used for testing."
        )
        self.dataset_iter = _DataSetIter(
            loaded_features=self.loaded_features,
            profilers=self.profilers,
            return_datasample=self._return_datasample,
            source_pipes=self.source_pipes,
        )

        self._test_iter = True

    def _init_ctx(self) -> None:
        # If using MP, children will init ray
        if self.ctx_initialized:
            return
        if self._iter_mode != "mp" and self.ctx.use_ray():
            self.ctx.init_ray()
        self.ctx_initialized = True

    def _profile(self, f_name, n_samples=None, output_file=None):
        if not (getattr(self, "_cm_profile", False)
                or os.environ.get("CEDAR_PROFILE_CM") == "1"):
            return self._profile_standard(f_name, n_samples, output_file)
        from cedar.client.linear_cost_profile import profile_linear_feature
        # Reuse the enhanced DP profiling route (including backend worker timing).
        # The additional instrumented local pass is excluded from its baseline.
        from threadpoolctl import threadpool_limits
        import torch
        env = {"CEDAR_PROFILE_FILTER_SELECTIVITY": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1"}
        previous = {key: os.environ.get(key) for key in env}
        settings = {key: globals()[key] for key in
                    ("RAY_PROFILE_N_ACTORS", "SMP_PROFILE_N_PROCS")}
        old_threads = torch.get_num_threads()
        os.environ.update(env)
        globals().update(RAY_PROFILE_N_ACTORS=1, SMP_PROFILE_N_PROCS=1)
        try:
            torch.set_num_threads(1)
            with threadpool_limits(limits=1):
                profile = self._profile_standard(f_name, n_samples, output_file)
                profile["cm_model"] = profile_linear_feature(
                    self.features[f_name], self.ctx, PROFILE_TIME_SEC, n_samples)
        finally:
            globals().update(settings)
            torch.set_num_threads(old_threads)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        destination = output_file or f"/tmp/{f_name}_profile.yml"
        with open(destination, "w") as stream:
            yaml.safe_dump(profile, stream)
        return profile

    def _profile_standard(
        self,
        f_name: str,
        n_samples: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> Dict:
        """
        Runs a short profiling step on this dataset.

        Args:
            n_samples: Runs the profiler for n_samples if provided, otherwise
                will run for PROFILE_TIME_SEC
            output_file: If provided, output a YAML file with profiled results.
                Otherwise, will output to "/tmp/<feature_name>_profile.yml"
        """
        if len(os.sched_getaffinity(0)) != 1 and "pytest" not in sys.modules:
            # Ignore if in pytest
            # raise RuntimeError(
            #     "Please run profiling with proc taskset to 1 cpu"
            # )
            logger.warning("Running profiling without taskset to 1 cpu...")
            logger.warning("Not recommended if using non-Pythonops")

        # Need to initialize ctx before profiling
        self._init_ctx()

        if getattr(self, "_legacy_cedar_profile", False):
            return self._profile_legacy_cedar(
                f_name, n_samples=n_samples, output_file=output_file
            )

        incremental_from = os.environ.get(
            "CEDAR_INCREMENTAL_PROFILE_FROM"
        )
        if incremental_from:
            if os.environ.get("CEDAR_INCREMENTAL_WALL_BASELINE") == "1":
                return self._profile_wall_baseline_incremental(
                    f_name=f_name,
                    feature_to_profile=self.features[f_name],
                    n_samples=n_samples,
                    output_file=output_file,
                    existing_profile=incremental_from,
                )
            return self._profile_backend_compute_incremental(
                f_name=f_name,
                feature_to_profile=self.features[f_name],
                n_samples=n_samples,
                output_file=output_file,
                existing_profile=incremental_from,
            )

        # Enable profiling for the feature
        logger.info(
            "Profiling feature {}, output to {}...".format(f_name, output_file)
        )
        feature_to_profile = self.features[f_name]

        d = {}

        # A profile is only meaningful together with the resources used to
        # produce it. Runtime resource-matching mode consumes this signature
        # and refuses to execute a plan whose per-stage width differs.
        d["resource_config"] = {
            "schema_version": 1,
            "profile_scope": "single_local_worker",
            "profile_local_workers": 1,
            "actors_per_stage": (
                RAY_PROFILE_N_ACTORS
                if RAY_PROFILE_N_ACTORS == SMP_PROFILE_N_PROCS
                else None
            ),
            "ray_actors_per_stage": RAY_PROFILE_N_ACTORS,
            "smp_procs_per_stage": SMP_PROFILE_N_PROCS,
        }
        d["profile_metadata"] = {
            "stage_duration_sec": PROFILE_TIME_SEC,
        }
        logger.info("Profile resource signature: %s", d["resource_config"])

        layered_profile = (
            os.environ.get("CEDAR_LAYERED_ADAPTIVE_PROFILE", "1") == "1"
        )
        reservoir = None
        if layered_profile:
            reservoir = ProfileInputReservoir(
                max_samples_per_pipe=int(
                    os.environ.get("CEDAR_PROFILE_POOL_SAMPLES", "64")
                ),
                max_bytes_per_pipe=int(
                    os.environ.get(
                        "CEDAR_PROFILE_POOL_BYTES_PER_PIPE",
                        str(64 * 1024 * 1024),
                    )
                ),
                max_bytes_total=int(
                    os.environ.get(
                        "CEDAR_PROFILE_POOL_BYTES_TOTAL",
                        str(512 * 1024 * 1024),
                    )
                ),
            )
        baseline_profile = self._profile_feature(
            f_name, feature_to_profile, n_samples, None
        )
        d["baseline"] = baseline_profile
        # The timing passes are bounded by the adaptive profiler's confidence
        # rule, so on a slow recipe they see only a few dozen records and every
        # filter looks non-selective.  Selectivity is what lets the joint DP
        # know that running a selective filter early removes work from every
        # later operator, so pay for one extra baseline pass that is bounded by
        # wall time instead of by the timing rule.
        if os.environ.get("CEDAR_PROFILE_FILTER_SELECTIVITY") == "1":
            selectivity_seconds = float(
                os.environ.get("CEDAR_PROFILE_SELECTIVITY_SEC", "90")
            )
            previous_profile_time = PROFILE_TIME_SEC
            try:
                globals()["PROFILE_TIME_SEC"] = selectivity_seconds
                selectivity_counts = self._profile_feature(
                    f_name, feature_to_profile, None, None
                )
            finally:
                globals()["PROFILE_TIME_SEC"] = previous_profile_time
            for key in ("input_counts", "output_counts", "selectivities"):
                counts = selectivity_counts.get(key)
                if isinstance(counts, dict) and counts:
                    baseline_profile[key] = counts
            logger.info(
                "Selectivity pass for %s: %s records seen per filter",
                f_name,
                {
                    pipe_id: (selectivity_counts.get("input_counts") or {}).get(
                        pipe_id
                    )
                    for pipe_id in (
                        selectivity_counts.get("input_counts") or {}
                    )
                },
            )
        boundary_profile_setting = os.environ.get(
            "CEDAR_PROFILE_BOUNDARY_MODEL"
        )
        if boundary_profile_setting is None:
            profile_boundaries = "pytest" not in sys.modules
        else:
            profile_boundaries = boundary_profile_setting == "1"
        if profile_boundaries:
            d["physical_model"] = {
                "schema_version": 1,
                "boundary": {},
            }

        if layered_profile:
            # Cedar and old_dp_boundary consume measured whole-pipeline
            # throughput through Cedar's Amdahl inversion. The three current
            # Simple-DP variants consume the isolated layers attached below.
            # Keep both in one profile so every optimizer shares one run.
            if self.ctx.use_ray():
                self._profile_ray(d, feature_to_profile, f_name, n_samples,
                              profile_backend_compute=False)
            self._profile_smp(d, feature_to_profile, f_name, n_samples,
                              profile_backend_compute=False)
            # TF fusion is another legacy whole-pipeline candidate. Measure
            # it before isolated passes can warm its caches or models.
            self._profile_tf(d, feature_to_profile, f_name, n_samples)

        if layered_profile and profile_boundaries:
            if self.ctx.use_ray():
                self._profile_boundary_model(
                    d, PipeVariantType.RAY, RAY_PROFILE_N_ACTORS
                )
            self._profile_boundary_model(
                d, PipeVariantType.SMP, SMP_PROFILE_N_PROCS
            )

        if layered_profile:
            if reservoir is None:
                raise RuntimeError("Layered profile input reservoir is absent")
            # Preserve the exact legacy measurement order above. Snapshot
            # collection is a separate discarded pass after baseline,
            # whole-pipeline offloads and boundaries, so serialization and
            # any extra cache warming cannot affect those compatibility data.
            self._collect_profile_input_reservoir(
                f_name, feature_to_profile, n_samples, reservoir
            )
            d["profile_metadata"]["input_reservoir_capture"] = (
                "post_legacy_separate_discarded_pipeline_pass"
            )
            self._profile_layered_backends(
                d, feature_to_profile, reservoir
            )
        else:
            # If using ray, profile each op
            if self.ctx.use_ray():
                self._profile_ray(d, feature_to_profile, f_name, n_samples)

            self._profile_smp(d, feature_to_profile, f_name, n_samples)

        if (
            not layered_profile
            and self.ctx.use_ray()
            and profile_boundaries
        ):
            self._profile_boundary_model(
                d,
                PipeVariantType.RAY,
                RAY_PROFILE_N_ACTORS,
            )
        if not layered_profile and profile_boundaries:
            self._profile_boundary_model(
                d,
                PipeVariantType.SMP,
                SMP_PROFILE_N_PROCS,
            )

        if os.environ.get("CEDAR_PROFILE_FILTER_SELECTIVITY") == "1":
            _consolidate_filter_selectivity(d)

        if not layered_profile:
            self._profile_tf(d, feature_to_profile, f_name, n_samples)

        # TODO: ENote: Profile reading / writing disk
        write_time_per_byte, read_time_per_byte = self._profile_io()
        d["disk_info"] = {}
        d["disk_info"]["read_latency"] = read_time_per_byte
        d["disk_info"]["write_latency"] = write_time_per_byte

        if output_file is None:
            output_file = f"/tmp/{f_name}_profile.yml"

        with open(output_file, "w") as outfile:
            yaml.dump(d, outfile)
        return d

    def _collect_profile_input_reservoir(
        self,
        f_name: str,
        feature_to_profile: Feature,
        n_samples: Optional[int],
        reservoir: ProfileInputReservoir,
    ) -> None:
        """Collect layered inputs in a discarded, unmeasured profile pass.

        ``capture_profile_input`` executes before ``x.trace()``. Enabling it
        during a compatibility measurement would attribute pickle cost to
        operators. Running this after every legacy measurement also prevents
        its extra cache warming from changing Cedar's offload observations.
        """
        set_profile_input_reservoir(reservoir)
        try:
            self._profile_feature(
                f_name, feature_to_profile, n_samples, None
            )
        finally:
            set_profile_input_reservoir(None)

    def _profile_legacy_cedar(
        self,
        f_name: str,
        n_samples: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Cedar's original baseline/Ray/SMP/TF profiling protocol."""
        logger.info(
            "Profiling feature %s with original Cedar profile schema", f_name
        )
        feature = self.features[f_name]
        disabled_env = (
            "CEDAR_PROFILE_INFER_COMPUTE_SCALING",
            "CEDAR_PROFILE_FILTER_SELECTIVITY",
        )
        old_env = {key: os.environ.pop(key, None) for key in disabled_env}
        try:
            profile: Dict[str, Any] = {
                "baseline": self._profile_feature(
                    f_name, feature, n_samples, None
                )
            }
            if self.ctx.use_ray():
                self._profile_ray(
                    profile,
                    feature,
                    f_name,
                    n_samples,
                    profile_backend_compute=False,
                )
            self._profile_smp(
                profile,
                feature,
                f_name,
                n_samples,
                profile_backend_compute=False,
            )
            self._profile_tf(profile, feature, f_name, n_samples)
            write_latency, read_latency = self._profile_io()
            profile["disk_info"] = {
                "read_latency": read_latency,
                "write_latency": write_latency,
            }
        finally:
            for key, value in old_env.items():
                if value is not None:
                    os.environ[key] = value

        measurement_keys = {
            "latencies",
            "input_sizes",
            "output_sizes",
            "throughput",
        }
        profile["baseline"] = {
            key: value
            for key, value in profile["baseline"].items()
            if key in measurement_keys
        }
        for backend_entries in profile.get("offloads", {}).values():
            for p_id, measurement in list(backend_entries.items()):
                backend_entries[p_id] = {
                    key: value
                    for key, value in measurement.items()
                    if key in measurement_keys
                }

        if output_file is None:
            output_file = f"/tmp/{f_name}_profile.yml"
        with open(output_file, "w") as outfile:
            yaml.dump(profile, outfile)
        return profile

    def _profile_wall_baseline_incremental(
        self,
        f_name: str,
        feature_to_profile: Feature,
        n_samples: Optional[int],
        output_file: Optional[str],
        existing_profile: str,
    ) -> Dict[str, Any]:
        """Add baseline wall-clock operator timings to an existing profile.

        The isolated offload throughputs, backend worker measurements,
        selectivities, sizes, and boundary calibration remain unchanged. This
        makes the extension inexpensive and preserves the common profile used
        by every optimizer in the comparison.
        """
        with open(existing_profile, "r") as stream:
            profile = yaml.safe_load(stream)
        if not isinstance(profile, dict):
            raise RuntimeError(
                f"Incremental profile is not a mapping: {existing_profile}"
            )
        expected_resources = {
            "schema_version": 1,
            "profile_scope": "single_local_worker",
            "profile_local_workers": 1,
            "actors_per_stage": (
                RAY_PROFILE_N_ACTORS
                if RAY_PROFILE_N_ACTORS == SMP_PROFILE_N_PROCS
                else None
            ),
            "ray_actors_per_stage": RAY_PROFILE_N_ACTORS,
            "smp_procs_per_stage": SMP_PROFILE_N_PROCS,
        }
        if profile.get("resource_config") != expected_resources:
            raise RuntimeError(
                "Incremental wall profiling resource mismatch: "
                f"existing={profile.get('resource_config')}, "
                f"current={expected_resources}"
            )
        baseline = profile.get("baseline")
        if not isinstance(baseline, dict):
            raise RuntimeError("Existing profile has no baseline mapping.")

        fresh_baseline = self._profile_feature(
            f_name, feature_to_profile, n_samples, None
        )
        wall_latencies = fresh_baseline.get("wall_latencies")
        if not isinstance(wall_latencies, dict) or not wall_latencies:
            raise RuntimeError("No baseline wall-clock timings were collected.")
        if set(wall_latencies) != set(baseline.get("latencies", {})):
            raise RuntimeError(
                "Wall-clock baseline pipe set does not match the existing "
                "profile."
            )
        baseline["wall_latencies"] = wall_latencies
        profile["incremental_wall_baseline"] = {
            "schema_version": 1,
            "source_profile": os.path.abspath(existing_profile),
            "clock": "perf_counter_ns",
            "updated_operators": len(wall_latencies),
        }

        if output_file is None:
            output_file = f"/tmp/{f_name}_profile.yml"
        with open(output_file, "w") as outfile:
            yaml.dump(profile, outfile)
        return profile

    def _profile_backend_compute_incremental(
        self,
        f_name: str,
        feature_to_profile: Feature,
        n_samples: Optional[int],
        output_file: Optional[str],
        existing_profile: str,
    ) -> Dict[str, Any]:
        """Add worker-side backend timings without perturbing old profile data.

        This deliberately preserves baseline throughput, selectivities, data
        sizes, disk measurements, and isolated offload throughput.  As a
        result, optimizers that do not understand ``backend_compute`` see
        byte-for-byte equivalent numeric inputs, while DpOptimizer can consume
        the new direct measurement.
        """
        with open(existing_profile, "r") as stream:
            profile = yaml.safe_load(stream)
        if not isinstance(profile, dict):
            raise RuntimeError(
                f"Incremental profile is not a mapping: {existing_profile}"
            )
        expected_resources = {
            "schema_version": 1,
            "profile_scope": "single_local_worker",
            "profile_local_workers": 1,
            "actors_per_stage": (
                RAY_PROFILE_N_ACTORS
                if RAY_PROFILE_N_ACTORS == SMP_PROFILE_N_PROCS
                else None
            ),
            "ray_actors_per_stage": RAY_PROFILE_N_ACTORS,
            "smp_procs_per_stage": SMP_PROFILE_N_PROCS,
        }
        if profile.get("resource_config") != expected_resources:
            raise RuntimeError(
                "Incremental profiling resource mismatch: "
                f"existing={profile.get('resource_config')}, "
                f"current={expected_resources}"
            )

        fresh: Dict[str, Any] = {}
        if self.ctx.use_ray():
            self._profile_ray(
                fresh, feature_to_profile, f_name, n_samples
            )
        self._profile_smp(fresh, feature_to_profile, f_name, n_samples)

        existing_offloads = profile.get("offloads")
        if not isinstance(existing_offloads, dict):
            raise RuntimeError("Existing profile has no offload mapping.")
        updated = 0
        for variant_name, pipe_profiles in fresh.get(
            "offloads", {}
        ).items():
            existing_variant = existing_offloads.get(variant_name)
            if not isinstance(existing_variant, dict):
                if pipe_profiles:
                    raise RuntimeError(
                        f"Existing profile has no {variant_name} section."
                    )
                continue
            for p_id, new_pipe_profile in pipe_profiles.items():
                direct = new_pipe_profile.get("backend_compute")
                if direct is None:
                    raise RuntimeError(
                        "No worker-side backend timing was collected for "
                        f"{variant_name} pipe {p_id}."
                    )
                if p_id not in existing_variant:
                    raise RuntimeError(
                        f"Existing profile has no {variant_name} pipe {p_id}."
                    )
                existing_variant[p_id]["backend_compute"] = direct
                updated += 1
        if updated == 0:
            raise RuntimeError("Incremental profiling found no mutable backends.")

        physical_model = profile.setdefault(
            "physical_model", {"schema_version": 1, "boundary": {}}
        )
        physical_model["schema_version"] = 1
        physical_model.setdefault("boundary", {})
        if self.ctx.use_ray():
            self._profile_boundary_model(
                profile, PipeVariantType.RAY, RAY_PROFILE_N_ACTORS
            )
        self._profile_boundary_model(
            profile, PipeVariantType.SMP, SMP_PROFILE_N_PROCS
        )
        profile["incremental_backend_compute"] = {
            "schema_version": 1,
            "source_profile": os.path.abspath(existing_profile),
            "updated_operator_variants": updated,
            "confidence_bound": "one_sided_normal_95pct",
        }

        if output_file is None:
            output_file = f"/tmp/{f_name}_profile.yml"
        with open(output_file, "w") as outfile:
            yaml.dump(profile, outfile)
        return profile

    def _profile_boundary_model(
        self,
        profile: Dict[str, Any],
        variant: PipeVariantType,
        width: int,
    ) -> None:
        """Attach a measured stage-boundary model without failing profiling.

        Boundary calibration is platform-specific. A failed calibration leaves
        the corresponding entry absent so optimizers can use their
        backward-compatible constants for old or partially collected profiles.
        """

        physical_model = profile.setdefault(
            "physical_model",
            {"schema_version": 1, "boundary": {}},
        )
        boundaries = physical_model.setdefault("boundary", {})
        try:
            boundaries[variant.name] = profile_stage_boundary_cached(
                ctx=self.ctx,
                variant=variant,
                width=width,
            )
        except Exception as exc:
            physical_model.setdefault("calibration_errors", {})[
                variant.name
            ] = f"{type(exc).__name__}: {exc}"
            if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
                raise RuntimeError(
                    "Remote Ray boundary calibration failed; refusing unmeasured "
                    "bandwidth fallback"
                ) from exc
            logger.warning(
                "Failed to profile %s stage boundary; optimizer will use "
                "its compatibility fallback: %s",
                variant.name,
                exc,
            )

    def _profile_io(
        self, character: str = "a", file_size_mb: int = 10
    ) -> Tuple[int, int]:
        """
        Generates file of specified size filled with a predetermined
        character, measures the time taken to write and read the file,
        then deletes the file.

        Returns the time per byte for both writing and reading.

        Args:
            character: Character to fill the file with.
            file_size_mb: Size of the file in megabytes.
        """
        file_size_bytes = (
            file_size_mb * 1024 * 1024
        )  # Convert size from MB to bytes

        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "temp_file.txt")

        # Write to the file and time the operation
        start_write = time.time()
        with open(temp_file_path, "w") as file:
            file.write(character * file_size_bytes)
        end_write = time.time()

        # Calculate time taken to write
        write_time = end_write - start_write
        write_time_per_byte = write_time / file_size_bytes

        # Read the file and time the operation
        start_read = time.time()
        with open(temp_file_path, "r") as file:
            _ = file.read()
        end_read = time.time()

        # Calculate time taken to read
        read_time = end_read - start_read
        read_time_per_byte = read_time / file_size_bytes

        # Delete the file
        os.remove(temp_file_path)

        return write_time_per_byte, read_time_per_byte

    def _profile_tf(
        self,
        d: Dict[str, Any],
        feature_to_profile: Feature,
        f_name: str,
        n_samples: Optional[int],
    ):
        loaded_feature = feature_to_profile.profile_tf(self.ctx)
        if loaded_feature is None:
            return

        source_pipe = feature_to_profile.get_source_pipes()

        # Create an Iterable
        dataset_iter = _DataSetIter(
            loaded_features={f_name: loaded_feature},
            return_datasample=False,
            source_pipes={f_name: source_pipe},
        )
        b_sz = feature_to_profile.get_batch_size()

        n_batches = 0
        start_time = None
        for x in dataset_iter:
            # Warm up time
            if n_batches == 0:
                start_time = time.time()
            n_batches += 1
            curr_time = time.time()

            if n_samples is not None:
                if n_batches * b_sz >= n_samples:
                    break
            elif (curr_time - start_time) >= PROFILE_TIME_SEC:
                break
        end_time = time.time()
        if start_time is None:
            feature_to_profile.reset()
            raise RuntimeError(
                f"Profiling feature {f_name} produced no batches. "
                "Check that the input dataset exists and is not fully filtered out."
            )

        throughput_samples_per_sec = (n_batches * b_sz) / (
            end_time - start_time
        )

        # Reset the feature and init
        feature_to_profile.reset()
        time.sleep(5)  # Sleep in case we need some time to shutdown

        d["tf_fuse"] = {
            "throughput": throughput_samples_per_sec,
        }

    @staticmethod
    def _modeled_offload_profile(
        baseline: Dict[str, Any],
        p_id: int,
        backend_compute: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the legacy schema from independently measured components."""
        result = copy.deepcopy(baseline)
        baseline_throughput = float(baseline["throughput"])
        total_sec = 1.0 / max(baseline_throughput, 1e-12)
        wall_latencies = baseline.get("wall_latencies", {})
        local_ns = float(
            wall_latencies.get(p_id, wall_latencies.get(str(p_id), 0.0))
        )
        backend_sec = (
            float(backend_compute["mean_ms_per_sample"]) / 1000.0
        )
        modeled_sec = max(1e-12, total_sec - local_ns / 1e9 + backend_sec)
        result["throughput"] = 1.0 / modeled_sec
        result["backend_compute"] = backend_compute
        result["throughput_provenance"] = {
            "method": "component_substitution",
            "measured_baseline_throughput": baseline_throughput,
            "replaced_local_wall_ns_per_sample": local_ns,
            "measured_backend_ms_per_sample": backend_compute[
                "mean_ms_per_sample"
            ],
        }
        return result

    def _adaptive_operator_benchmark(
        self,
        pipe: Pipe,
        snapshots: List[bytes],
        variant_type: PipeVariantType,
        width: int,
        min_duration: float,
        max_duration: float,
        target_rse: float,
        min_observations: int,
        ray_submit_batch_size: Optional[int] = None,
        minimum_records_per_worker: Optional[int] = None,
        record_actor_locations: bool = False,
        snapshot_sequence: Optional[List[Tuple[str, List[bytes]]]] = None,
    ) -> Dict[str, Any]:
        """Measure one backend with fixed inputs until confidence converges."""
        replay = _ProfileReplayPipeVariant(snapshots, 1)
        predecessor = pipe.input_pipes[0]
        # Replaying a TF operator needs an explicit input signature: the
        # synthetic replay source is not a TF pipe, and this path builds the
        # variant directly instead of going through Pipe.mutate(), so derive
        # the signature from the logical predecessor once.
        if pipe.is_tf() and getattr(pipe, "_input_tf_spec", None) is None:
            try:
                derived_spec = predecessor.generate_output_tf_spec()
            except Exception:  # noqa: BLE001 - only a fallback safeguard
                derived_spec = None
            if derived_spec is not None:
                pipe._input_tf_spec = derived_spec
        replay.p_id = predecessor.id
        old_predecessor_variant = predecessor.pipe_variant
        predecessor.pipe_variant = replay
        if variant_type == PipeVariantType.RAY:
            variant_ctx = RayPipeVariantContext(
                n_actors=width,
                max_inflight=max(RAY_PROFILE_INFLIGHT, width * 5),
                max_prefetch=RAY_PROFILE_PREFETCH,
                use_threads=True,
                submit_batch_size=(
                    ray_submit_batch_size
                    if ray_submit_batch_size is not None
                    else RAY_PROFILE_SUBMIT_BATCH_SIZE
                ),
                profile_backend_compute=True,
                num_gpus=_ray_profile_gpu_fraction(pipe, width),
            )
        elif variant_type == PipeVariantType.TF_RAY:
            variant_ctx = TFRayPipeVariantContext(
                n_actors=width,
                max_inflight=max(RAY_PROFILE_INFLIGHT, width * 5),
                max_prefetch=RAY_PROFILE_PREFETCH,
                use_threads=True,
                submit_batch_size=RAY_PROFILE_SUBMIT_BATCH_SIZE,
                profile_backend_compute=True,
                num_gpus=_ray_profile_gpu_fraction(pipe, width),
            )
        elif variant_type == PipeVariantType.SMP:
            variant_ctx = SMPPipeVariantContext(
                n_procs=width,
                max_inflight=max(SMP_PROFILE_INFLIGHT, width * 3),
                max_prefetch=SMP_PROFILE_PREFETCH,
                use_threads=True,
                disable_torch_parallelism=True,
                profile_backend_compute=True,
            )
        else:
            raise ValueError(f"Unsupported adaptive backend {variant_type}")
        try:
            variant = pipe._create_pipe_variant(variant_type, variant_ctx)
        finally:
            predecessor.pipe_variant = old_predecessor_variant
        variant.p_id = pipe.id
        variant.pipe_spec = pipe.pipe_spec
        service = getattr(variant, "service", None)
        if service is None:
            service = getattr(variant_ctx, "service", None)
        if service is None:
            raise RuntimeError(
                f"{variant_type.name} variant exposes no profiling service"
            )
        try:
            actor_locations = None
            if record_actor_locations:
                if variant_type not in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
                    raise ValueError("Actor locations require a Ray backend")
                actor_locations = ray.get([
                    actor.get_runtime_location.remote()
                    for actor in service._actors
                ])
            if variant_type in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            ):
                effective_batch = (
                    ray_submit_batch_size
                    if ray_submit_batch_size is not None
                    else RAY_PROFILE_SUBMIT_BATCH_SIZE
                )
            else:
                effective_batch = 1
            minimum_parallel_epoch = _minimum_parallel_epoch_records(
                variant_type,
                width,
                effective_batch,
                minimum_records_per_worker,
            )

            trials = (
                snapshot_sequence
                if snapshot_sequence is not None
                else [("single", snapshots)]
            )
            results = []
            for trial_label, trial_snapshots in trials:
                replay.snapshots = trial_snapshots
                # Warm the same per-worker work quantum used by measurement. Ray
                # dispatches batches randomly, so the historical one-record warmup
                # reached only one actor; at width 48, half of a two-batch/actor
                # measurement could then be cold starts. A sustained warmup makes
                # the confidence test operate on a stationary population.
                warm_started = time.perf_counter()
                replay.record_count = minimum_parallel_epoch
                for _ in variant:
                    pass
                warm_elapsed = max(time.perf_counter() - warm_started, 1e-6)
                warm_sec_per_record = warm_elapsed / minimum_parallel_epoch
                reset_stats = getattr(
                    service, "reset_backend_compute_stats", None
                )
                if reset_stats is None:
                    raise RuntimeError(
                        f"{variant_type.name} service cannot reset timing stats"
                    )
                reset_stats()

                # Aim for roughly half-second epochs, but a parallel epoch must
                # actually exercise every worker. The old upper bound of 256
                # records could produce e.g. seven records for width=8 with a Ray
                # submit batch of ten: one tail task ran on one actor and the
                # resulting measurement was incorrectly labelled width=8.
                epoch_records = max(
                    minimum_parallel_epoch,
                    min(4096, int(0.5 / warm_sec_per_record)),
                )
                started = time.perf_counter()
                converged = False
                rse = math.inf
                stats = None
                measured_input_records = 0
                while True:
                    replay.record_count = epoch_records
                    for _ in variant:
                        pass
                    measured_input_records += epoch_records
                    elapsed = time.perf_counter() - started
                    stats = service.get_backend_compute_stats()
                    if stats is not None:
                        mean = float(stats["mean_ms_per_sample"])
                        stderr = float(stats["stderr_ms_per_sample"])
                        rse = stderr / mean if mean > 0 else math.inf
                        converged = (
                            elapsed >= min_duration
                            and int(stats["count"]) >= min_observations
                            and rse <= target_rse
                        )
                    if converged or elapsed >= max_duration:
                        break
                if stats is None:
                    raise RuntimeError(
                        f"No {variant_type.name} worker timing for pipe {pipe.id}"
                    )
                stats = dict(stats)
                stats["end_to_end_ms_per_input_sample"] = (
                    elapsed * 1000.0 / measured_input_records
                )
                if actor_locations is not None:
                    stats["actor_locations"] = actor_locations
                stats["measured_input_records"] = measured_input_records
                stats["adaptive_profile"] = {
                    "width": width,
                    "elapsed_sec": elapsed,
                    "warmup_sec": warm_elapsed,
                    "warmup_records": minimum_parallel_epoch,
                    "epoch_records": epoch_records,
                    "minimum_parallel_epoch_records": minimum_parallel_epoch,
                    "unique_input_records": len(trial_snapshots),
                    "target_rse": target_rse,
                    "observed_rse": rse,
                    "min_duration_sec": min_duration,
                    "max_duration_sec": max_duration,
                    "min_observations": min_observations,
                    "converged": converged,
                    "stop_reason": "confidence" if converged else "max_duration",
                }
                stats["trial_label"] = trial_label
                results.append(stats)
            return results if snapshot_sequence is not None else results[0]
        finally:
            variant.shutdown()

    def _time_operator_on_snapshots(
        self,
        fn,
        snapshots: List[bytes],
        min_calls: int,
        max_calls: int,
        repeats: int,
        target_sec: float,
        max_batch_bytes: int,
        records_per_call: int = 1,
    ) -> Optional[float]:
        """Median milliseconds per record for one operator callable.

        ``records_per_call`` normalizes callables that process several records
        per invocation (a batcher assembles a whole batch at once).
        """
        if not snapshots:
            return None
        records_per_call = max(1, int(records_per_call))
        record_bytes = max(
            1.0, statistics.median(len(snapshot) for snapshot in snapshots)
        )
        for snapshot in snapshots[:3]:
            self._time_operator_fresh_snapshot(fn, snapshot)
        calibration_calls = 3
        calibration_sec = 0.0
        for _ in range(calibration_calls):
            for snapshot in snapshots:
                calibration_sec += self._time_operator_fresh_snapshot(
                    fn, snapshot
                )
        per_call_sec = max(
            calibration_sec / (calibration_calls * len(snapshots)), 1e-9
        )
        calls = int(min(max_calls, max(min_calls, target_sec / per_call_sec)))
        bytes_per_pass = record_bytes * len(snapshots)
        calls = max(
            1,
            min(calls, int(max(1.0, max_batch_bytes / bytes_per_pass))),
        )
        rates: List[float] = []
        for _ in range(max(1, repeats)):
            was_enabled = gc.isenabled()
            gc.disable()
            duration = 0.0
            try:
                for _ in range(calls):
                    for snapshot in snapshots:
                        duration += self._time_operator_fresh_snapshot(
                            fn, snapshot
                        )
            finally:
                if was_enabled:
                    gc.enable()
                    gc.collect()
            rates.append(calls * len(snapshots) / max(duration, 1e-9))
        return 1000.0 / statistics.median(rates) / records_per_call

    @staticmethod
    def _time_operator_fresh_snapshot(fn, snapshot: bytes) -> float:
        """Time ``fn`` on a fresh value without charging deserialization.

        Profiling inputs are immutable serialized snapshots because mapper
        functions are allowed to mutate their argument. Reusing one decoded
        value makes transforms such as COCO RandomZoomOut repeatedly enlarge
        their own output and can exhaust host memory. Decode before starting
        the clock and release the mutated value immediately after the call.
        """
        value = pickle.loads(snapshot)
        start = time.perf_counter()
        result = fn(value)
        duration = time.perf_counter() - start
        if result is NotImplemented:
            raise RuntimeError("Unexpected operator result")
        del result
        del value
        return duration

    def _time_operator_value(
        self,
        fn,
        value,
        min_calls: int,
        max_calls: int,
        repeats: int,
        target_sec: float,
        max_batch_bytes: int,
        payload_bytes: int = 0,
    ) -> Optional[float]:
        """Median milliseconds per record for one operator on one payload."""
        snapshot = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._time_operator_fresh_snapshot(fn, snapshot)
        calibration_calls = 3
        calibration_sec = 0.0
        for _ in range(calibration_calls):
            calibration_sec += self._time_operator_fresh_snapshot(fn, snapshot)
        per_call_sec = max(calibration_sec / calibration_calls, 1e-9)
        calls = int(min(max_calls, max(min_calls, target_sec / per_call_sec)))
        if payload_bytes > 0:
            calls = max(
                1,
                min(calls, int(max(1.0, max_batch_bytes / payload_bytes))),
            )
        rates: List[float] = []
        for _ in range(max(1, repeats)):
            was_enabled = gc.isenabled()
            gc.disable()
            duration = 0.0
            try:
                for _ in range(calls):
                    duration += self._time_operator_fresh_snapshot(
                        fn, snapshot
                    )
            finally:
                if was_enabled:
                    gc.enable()
                    gc.collect()
            rates.append(calls / max(duration, 1e-9))
        return 1000.0 / statistics.median(rates)

    def _affine_rescale_payload(self, value, factor: float, depth: int = 0):
        """Return ``value`` with its data-carrying fields scaled by ``factor``.

        The affine calibration asks one question per operator: how much of this
        operator's cost is size-independent?  Image pipelines answer it only if
        the same operator is also fed a differently sized payload, because every
        record that reaches an operator after a resize has the same shape.  This
        helper builds such payloads from the operator's own legal input, keeping
        the container structure so the operator sees a real record: images are
        resampled, text is truncated or repeated, and anything unsupported makes
        the caller fall back to its own snapshots.
        """
        if depth > 3 or factor <= 0.0:
            return None
        if isinstance(value, dict):
            scaled = {}
            changed = False
            for key, item in value.items():
                replacement = self._affine_rescale_payload(
                    item, factor, depth + 1
                )
                if replacement is None:
                    scaled[key] = item
                else:
                    scaled[key] = replacement
                    changed = True
            return scaled if changed else None
        if isinstance(value, (list, tuple)):
            items = []
            changed = False
            for item in value:
                replacement = self._affine_rescale_payload(
                    item, factor, depth + 1
                )
                if replacement is None:
                    items.append(item)
                else:
                    items.append(replacement)
                    changed = True
            if not changed:
                return None
            return type(value)(items) if not isinstance(value, tuple) else tuple(items)
        if isinstance(value, str):
            if factor < 1.0:
                cut = max(1, int(len(value) * factor))
                return value[:cut] if cut < len(value) else None
            repeats = max(2, int(round(factor)))
            return value * repeats
        if isinstance(value, bytes):
            if factor < 1.0:
                cut = max(1, int(len(value) * factor))
                return value[:cut] if cut < len(value) else None
            return value * max(2, int(round(factor)))
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            Image = None
        if Image is not None and isinstance(value, Image.Image):
            width, height = value.size
            new_size = (
                max(1, int(round(width * factor))),
                max(1, int(round(height * factor))),
            )
            if new_size == value.size:
                return None
            return value.resize(new_size, Image.BILINEAR)
        try:
            import torch
            import torch.nn.functional as torch_functional
        except Exception:  # noqa: BLE001
            return None
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            channel_last = value.dim() == 3 and value.shape[-1] in (1, 3, 4)
            if channel_last:
                moved = value.permute(2, 0, 1)
            else:
                moved = value
            height, width = int(moved.shape[-2]), int(moved.shape[-1])
            new_size = (
                max(1, int(round(height * factor))),
                max(1, int(round(width * factor))),
            )
            if new_size == (height, width):
                return None
            floating = moved.dtype.is_floating_point
            try:
                resized = torch_functional.interpolate(
                    moved.unsqueeze(0).float(),
                    size=new_size,
                    mode="bilinear" if floating else "nearest",
                    align_corners=False if floating else None,
                ).squeeze(0)
            except Exception as exc:  # noqa: BLE001
                # Payloads that are tensors but not images (bounding boxes,
                # token ids, masks without spatial layout) have no spatial
                # counterfactual: the caller keeps the measured constant cost
                # instead of failing the whole profile.
                logger.info(
                    "Rescale counterfactual skipped for tensor shape %s: %s",
                    tuple(value.shape),
                    exc,
                )
                return None
            if not floating:
                resized = resized.round().to(moved.dtype)
            if channel_last:
                resized = resized.permute(1, 2, 0)
            return resized.contiguous()
        return None

    @staticmethod
    def _operator_affine_measurement(pipe):
        """Return ``(callable, tag, records_per_call)`` for one operator.

        A batcher or an image reader has no Python ``fn``, yet its per-record
        cost is real and measurable: the batcher assembles a batch and the
        reader decodes the file it is handed. Measuring those callables keeps
        every priced operator on the same fitted ``kx+b`` layer instead of
        exempting the operator from the model. Readers are timed on a warm
        page cache, so their fitted slope describes decode work rather than
        disk latency.
        """
        fn = getattr(pipe, "fn", None)
        if fn is not None:
            return fn, "callable", 1
        if isinstance(pipe, BatcherPipe):
            from .linear_cost_profile import NativeBatchCall

            # One call assembles a full batch, so its cost covers that many
            # records and is normalized to a per-record price below.
            return (
                NativeBatchCall(pipe.batch_size, pipe.drop_last),
                "batcher",
                int(pipe.batch_size),
            )
        if isinstance(pipe, ImageReaderPipe):
            from cedar.pipes.io import read_image

            mode = getattr(pipe, "mode", None)
            return (
                lambda value, mode=mode: read_image(value, mode=mode),
                "image_reader",
                1,
            )
        return None, f"unsupported_pipe_type:{type(pipe).__name__}", 1

    @staticmethod
    def _readable_affine_snapshots(snapshots: List[bytes]) -> List[bytes]:
        """Keep captured records that name an existing readable file.

        A file lister hands its reader both directory entries and files. Only
        the files describe decode work, and timing the reader on a directory
        raises instead of measuring anything.
        """
        kept: List[bytes] = []
        for raw in snapshots:
            try:
                value = pickle.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            path = value
            if isinstance(path, pathlib.Path):
                path = str(path)
            if isinstance(path, (str, bytes)) and os.path.isfile(path):
                kept.append(raw)
        return kept

    def _affine_usable_snapshots(
        self, fn, snapshots: List[bytes]
    ) -> List[bytes]:
        """Drop legal records this operator cannot process at all.

        A lister can hand its consumer records the consumer rejects (a
        directory entry, a zero-length file). One such record must not cost the
        operator its measured coefficients, so the pool is validated with one
        call per record and the rejects are counted instead of timed.
        """
        usable: List[bytes] = []
        for raw in snapshots:
            try:
                self._time_operator_fresh_snapshot(fn, raw)
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "Affine measurement ignores an unusable legal record: %s",
                    exc,
                )
                continue
            usable.append(raw)
        return usable

    def _profile_operator_input_size_affine(
        self,
        profile: Dict[str, Any],
        feature: Feature,
        reservoir: ProfileInputReservoir,
    ) -> None:
        """Fit ``cost(record) = k * input_bytes + b`` for every operator.

        The DP prices an operator's compute by the byte volume that reaches it,
        which is what lets it prefer orders that shrink the payload before the
        expensive operators.  A purely per-byte price assumes every operator's
        cost vanishes as its input shrinks; real operators keep a fixed part
        (call overhead, kernel launch, per-record bookkeeping).  Each operator
        is therefore measured on the smallest and the largest legal input it
        sees, and, when its own inputs carry no size contrast, on rescaled
        versions of its own legal input.  ``fixed_fraction`` is the share of the
        profiled cost that does not shrink with the payload.
        """
        enabled = os.environ.get("CEDAR_PROFILE_OPERATOR_AFFINE", "1")
        if enabled.strip() not in ("1", "true", "True", "yes"):
            return
        min_calls = int(os.environ.get("CEDAR_PROFILE_AFFINE_MIN_CALLS", "5"))
        max_calls = int(
            os.environ.get("CEDAR_PROFILE_AFFINE_MAX_CALLS", "400")
        )
        repeats = int(os.environ.get("CEDAR_PROFILE_AFFINE_REPEATS", "5"))
        target_sec = float(
            os.environ.get("CEDAR_PROFILE_AFFINE_TARGET_SEC", "0.05")
        )
        max_batch_bytes = int(
            os.environ.get(
                "CEDAR_PROFILE_AFFINE_MAX_BATCH_BYTES",
                str(64 * 1024 * 1024),
            )
        )
        max_operators = int(
            os.environ.get("CEDAR_PROFILE_AFFINE_MAX_OPERATORS", "80")
        )
        include_cuda = os.environ.get(
            "CEDAR_PROFILE_AFFINE_INCLUDE_CUDA", "0"
        ).strip() in ("1", "true", "True", "yes")
        min_contrast = float(
            os.environ.get("CEDAR_PROFILE_AFFINE_MIN_CONTRAST", "1.10")
        )
        baseline = profile.get("baseline", {})
        input_sizes = baseline.get("input_sizes", {}) or {}
        operators: Dict[str, Dict[str, Any]] = {}
        unfitted_reasons: Dict[str, str] = {}
        for p_id, pipe in feature.logical_pipes.items():
            if len(operators) >= max_operators:
                unfitted_reasons[str(p_id)] = "beyond_max_operators"
                break
            if pipe.pipe_spec is None or len(pipe.input_pipes) != 1:
                unfitted_reasons[str(p_id)] = "not_a_single_input_stage"
                continue
            if (
                not include_cuda
                and pipe.execution_resource
                == PipeExecutionResource.CUDA
            ):
                unfitted_reasons[str(p_id)] = "cuda_operator_excluded"
                continue
            fn = getattr(pipe, "fn", None)
            measurement = "callable"
            records_per_call = 1
            if fn is None:
                fn, measurement, records_per_call = (
                    self._operator_affine_measurement(pipe)
                )
                if fn is None:
                    # Record why this operator has no measured kx+b instead of
                    # silently leaving the DP to invent a cost regime.
                    logger.info(
                        "Input-size affine fit has no measurement for pipe "
                        "%s (%s): %s",
                        p_id,
                        type(pipe).__name__,
                        measurement,
                    )
                    unfitted_reasons[str(p_id)] = measurement
                    continue
            predecessor_id = pipe.input_pipes[0].id
            snapshots = reservoir.values_for(predecessor_id)
            if measurement == "image_reader":
                # Directories in a lister's output carry no decode work.
                snapshots = self._readable_affine_snapshots(snapshots)
            snapshots = self._affine_usable_snapshots(fn, snapshots)
            if not snapshots:
                unfitted_reasons[str(p_id)] = "no_usable_legal_input"
                continue
            ordered = sorted(snapshots, key=len)
            stratum = max(1, len(ordered) // 4)
            low = ordered[:stratum]
            high = ordered[-stratum:]
            x_low = float(statistics.median(len(s) for s in low))
            x_high = float(statistics.median(len(s) for s in high))
            points: List[Tuple[float, float]] = []
            try:
                if x_low > 0.0 and x_high >= x_low * min_contrast:
                    cost_low = self._time_operator_on_snapshots(
                        fn, low, min_calls, max_calls, repeats,
                        target_sec, max_batch_bytes, records_per_call,
                    )
                    cost_high = self._time_operator_on_snapshots(
                        fn, high, min_calls, max_calls, repeats,
                        target_sec, max_batch_bytes, records_per_call,
                    )
                    if cost_low is not None and cost_high is not None:
                        points = [(x_low, cost_low), (x_high, cost_high)]
                if not points and measurement == "callable":
                    # Rescaling only makes sense for data-carrying payloads: a
                    # truncated path string would name a different (or
                    # non-existent) file, so path-consuming operators keep the
                    # measured constant fallback below instead.
                    try:
                        reference = pickle.loads(ordered[len(ordered) // 2])
                        factor_low = float(
                            os.environ.get(
                                "CEDAR_PROFILE_AFFINE_DOWNSCALE", "0.5"
                            )
                        )
                        factor_high = float(
                            os.environ.get(
                                "CEDAR_PROFILE_AFFINE_UPSCALE", "2.0"
                            )
                        )
                        scaled_pairs = []
                        for factor in (factor_low, factor_high):
                            scaled = self._affine_rescale_payload(
                                reference, factor
                            )
                            if scaled is None:
                                continue
                            scaled_pairs.append(
                                (
                                    len(
                                        pickle.dumps(
                                            scaled,
                                            protocol=(
                                                pickle.HIGHEST_PROTOCOL
                                            ),
                                        )
                                    ),
                                    scaled,
                                )
                            )
                        if len(scaled_pairs) >= 2:
                            scaled_pairs.sort(key=lambda item: item[0])
                            for size_bytes, payload in scaled_pairs:
                                cost = self._time_operator_value(
                                    fn, payload, min_calls, max_calls,
                                    repeats, target_sec, max_batch_bytes,
                                    size_bytes,
                                )
                                if cost is not None:
                                    points.append((float(size_bytes), cost))
                    except Exception as exc:  # noqa: BLE001
                        # A rescaled payload the operator rejects (for example a
                        # truncated path) is not a legal input; keep the
                        # measured constant cost of its real inputs instead.
                        logger.info(
                            "Rescaled affine counterfactual unusable for pipe "
                            "%s: %s",
                            p_id,
                            exc,
                        )
                        points = []
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "Input-size affine fit skipped for pipe %s: %s", p_id, exc
                )
                unfitted_reasons[str(p_id)] = f"measurement_failed:{exc}"[:200]
                continue
            if len(points) < 2:
                # Legal inputs with no usable size contrast still carry a
                # measured per-record cost. Record it as the k = 0 member of
                # the same family instead of leaving the operator unpriced.
                constant = self._time_operator_on_snapshots(
                    fn, ordered, min_calls, max_calls, repeats,
                    target_sec, max_batch_bytes, records_per_call,
                )
                if constant is None:
                    unfitted_reasons[str(p_id)] = "constant_measurement_failed"
                    continue
                operators[str(p_id)] = {
                    "fixed_fraction": 1.0,
                    "k_ms_per_byte": 0.0,
                    "b_ms": constant,
                    "x_reference_bytes": float(
                        input_sizes.get(
                            p_id, input_sizes.get(str(p_id), 0.0)
                        )
                        or statistics.median(len(s) for s in ordered)
                    ),
                    "measurement": measurement,
                    "status": "constant_measured",
                    "reason": "no_legal_size_contrast",
                }
                continue
            points.sort()
            (x_low_pt, cost_low_pt), (x_high_pt, cost_high_pt) = (
                points[0],
                points[-1],
            )
            if x_high_pt <= x_low_pt:
                continue
            # Every operator gets kx+b, including operators whose cost is flat
            # in the payload size: a flat cost is the k = 0 member of the same
            # family, not a separate cost regime.
            slope = max(
                0.0,
                (cost_high_pt - cost_low_pt) / (x_high_pt - x_low_pt),
            )
            intercept = max(0.0, cost_low_pt - slope * x_low_pt)
            reference_bytes = float(
                input_sizes.get(p_id, input_sizes.get(str(p_id), 0.0)) or 0.0
            )
            if reference_bytes <= 0.0:
                reference_bytes = float(
                    statistics.median(len(s) for s in ordered)
                )
            reference_cost = slope * reference_bytes + intercept
            if not math.isfinite(reference_cost) or reference_cost <= 0.0:
                continue
            fixed_fraction = min(
                0.95, max(0.0, intercept / reference_cost)
            )
            operators[str(p_id)] = {
                "fixed_fraction": round(fixed_fraction, 6),
                "k_ms_per_byte": slope,
                "b_ms": intercept,
                "x_reference_bytes": reference_bytes,
                "measurement": measurement,
                "points_ms_per_byte": [
                    [x_low_pt, cost_low_pt], [x_high_pt, cost_high_pt]
                ],
                "source": (
                    "legal_inputs"
                    if len(points) == 2
                    and x_low == x_low_pt
                    and x_high == x_high_pt
                    else "rescaled_legal_input"
                ),
            }
        unfitted = sorted(
            p_id
            for p_id, pipe in feature.logical_pipes.items()
            if not pipe.is_source() and str(p_id) not in operators
        )
        profile.setdefault(
            "physical_model", {"schema_version": 1, "boundary": {}}
        )["operator_affine"] = {
            "schema_version": 1,
            "method": "two_stratum_operator_affine_fit",
            "note": (
                "cost(record) = k * input_bytes + b, fitted on the smallest and "
                "largest legal inputs that reached the operator, or on rescaled "
                "versions of its own legal input when those inputs have no size "
                "contrast. Operators whose cost is flat in the payload keep "
                "k = 0 and are handled by the same equation. "
                "fixed_fraction = b / (k * x_reference + b) is the share of the "
                "operator's reference cost that does not shrink with the payload"
            ),
            "min_contrast": min_contrast,
            # Operators with no measurable callable (for example pass-through
            # NoopPipe/BatcherPipe stages) cannot be fitted. They are listed
            # here so an optimizer that is asked to price them raises instead
            # of silently falling back to byte-proportional compute.
            "unfitted_operators": sorted(unfitted),
            "unfitted_reasons": {
                key: unfitted_reasons[key]
                for key in sorted(unfitted_reasons, key=int)
            },
            "operators": operators,
        }
        logger.info(
            "Input-size affine calibration: %s/%s operators fitted",
            len(operators),
            len(feature.logical_pipes),
        )

    @staticmethod
    def _fit_width_curve(measured):
        """Fit the measured actor-width curve of one stage."""
        from cedar.compose.my_optimizer import fit_width_curve

        return fit_width_curve(measured)

    def _time_operator_grid_mean_ms(
        self,
        fn,
        grid: List[bytes],
        budget_sec: float,
        max_calls_per_point: int,
    ) -> List[Optional[float]]:
        """Interleaved mean milliseconds for every point of one operator.

        The planning objective is the mean service time, and several kernels
        switch into a slower mode only over multi-second windows.  Measuring
        every point in a short burst therefore reports a biased (usually too
        cheap) mean -- this was the defect that made the first version of this
        model fit a slope three times too small.  All points of one operator
        are consequently measured round-robin inside one window, so every point
        sees the same slow-mode episodes, and each point keeps the raw mean of
        all its calls.
        """
        if not grid:
            return []
        for snapshot in grid[:3]:
            self._time_operator_fresh_snapshot(fn, snapshot)
        durations: List[List[float]] = [[] for _ in grid]
        invalid: List[bool] = [False] * len(grid)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            while True:
                for index, snapshot in enumerate(grid):
                    if invalid[index]:
                        continue
                    if len(durations[index]) >= max_calls_per_point:
                        continue
                    try:
                        durations[index].append(
                            self._time_operator_fresh_snapshot(fn, snapshot)
                        )
                    except Exception:  # noqa: BLE001
                        # A rescaled counterfactual the operator rejects (for
                        # example a list container an image transform cannot
                        # read) is not a legal input: drop that point only.
                        invalid[index] = True
                        logger.info(
                            "Representation compute point rejected; skipped"
                        )
                # The budget is *per point*: bursts shorter than a few seconds
                # miss the slow-mode episodes of kernels such as GaussianBlur
                # and report a mean that is several times too cheap.
                if all(
                    invalid[index]
                    or
                    len(values) >= max_calls_per_point
                    or sum(values) >= budget_sec
                    for index, values in enumerate(durations)
                ):
                    break
        finally:
            if gc_was_enabled:
                gc.enable()
                gc.collect()
        return [
            1000.0 * statistics.fmean(values)
            if values and not invalid[index]
            else None
            for index, values in enumerate(durations)
        ]

    @staticmethod
    def _fit_compute_coefficients(
        points: List[Tuple[float, float]]
    ) -> Optional[Dict[str, float]]:
        """Two-stratum least squares for ``k * elements + b``."""
        if len(points) < 2:
            return None
        points = sorted(points)
        (x_low, y_low), (x_high, y_high) = points[0], points[-1]
        if x_high <= x_low:
            return None
        slope = max(0.0, (y_high - y_low) / (x_high - x_low))
        intercept = max(0.0, y_low - slope * x_low)
        return {
            "k_ms_per_element": slope,
            "b_ms": intercept,
            "points_ms_per_element": [[x_low, y_low], [x_high, y_high]],
        }

    def _profile_operator_compute_model(
        self,
        profile: Dict[str, Any],
        feature: Feature,
        reservoir: ProfileInputReservoir,
    ) -> None:
        """Fit ``cost = k_(operator, representation) * elements + b``.

        Reordering a pipeline can hand an operator the same record in a
        different representation (moving ``to_float`` behind the image
        transforms turns a float32 payload into a uint8 one).  A byte-shaped
        curve cannot express that, so every operator is measured inside each
        representation class it can legally receive, on payloads the pipeline
        itself materialises, and only the spatial extent is rescaled to obtain
        the two strata.  Boundaries and transport keep using bytes.
        """
        enabled = os.environ.get("CEDAR_PROFILE_COMPUTE_MODEL", "1")
        if enabled.strip() not in ("1", "true", "True", "yes"):
            return
        budget_sec = float(
            os.environ.get("CEDAR_PROFILE_COMPUTE_TARGET_SEC", "5.0")
        )
        max_calls_per_point = int(
            os.environ.get("CEDAR_PROFILE_COMPUTE_MAX_CALLS", "400")
        )
        max_classes = int(
            os.environ.get("CEDAR_PROFILE_COMPUTE_MAX_CLASSES", "6")
        )
        pool_per_pipe = int(
            os.environ.get("CEDAR_PROFILE_COMPUTE_POOL_PER_PIPE", "2")
        )
        downscale = float(
            os.environ.get("CEDAR_PROFILE_AFFINE_DOWNSCALE", "0.5")
        )
        upscale = float(os.environ.get("CEDAR_PROFILE_AFFINE_UPSCALE", "2.0"))

        # 1) Candidate payloads: the pipeline's own intermediate values plus one
        # application of every callable, so a representation the declared order
        # never materialises (uint8 single channel, say) is still measured on a
        # real record rather than invented.
        callables: Dict[int, Any] = {}
        records_per_call: Dict[int, int] = {}
        for p_id, pipe in feature.logical_pipes.items():
            fn, _, per_call = self._operator_affine_measurement(pipe)
            if fn is not None:
                callables[int(p_id)] = fn
                records_per_call[int(p_id)] = max(1, int(per_call or 1))

        def _is_single_record(value) -> bool:
            """Batches are not per-record payloads; they have no fit curve."""
            dim = getattr(value, "dim", None)
            if callable(dim):
                return int(dim()) <= 3
            return True

        pool: List[Any] = []
        seen: set = set()
        for p_id in sorted(reservoir.samples):
            for raw in reservoir.values_for(p_id)[:pool_per_pipe]:
                try:
                    value = pickle.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                if not _is_single_record(value):
                    continue
                klass = payload_representation_class(value)
                scale = payload_compute_scale(value)
                if klass is None or scale is None:
                    continue
                key = (klass, round(scale, 3))
                if key in seen:
                    continue
                seen.add(key)
                pool.append(value)
        for p_id in sorted(callables):
            fn = callables[p_id]
            for value in list(pool):
                try:
                    produced = fn(value)
                except Exception:  # noqa: BLE001
                    continue
                if not _is_single_record(produced):
                    continue
                klass = payload_representation_class(produced)
                scale = payload_compute_scale(produced)
                if klass is None or scale is None:
                    continue
                key = (klass, round(scale, 3))
                if key in seen:
                    continue
                seen.add(key)
                pool.append(produced)
        if not pool:
            logger.info("Compute-model profiling found no usable payloads")
            return

        operators: Dict[str, Dict[str, Any]] = {}
        transitions: Dict[str, Dict[str, str]] = {}
        element_ratios: Dict[str, float] = {}
        coverage: Dict[str, Any] = {}
        source_class = None
        source_elements = None
        for package_id, pipe in sorted(feature.logical_pipes.items(), key=lambda kv: int(kv[0])):
            if pipe.input_pipes:
                continue
            for raw in reservoir.values_for(int(package_id)):
                try:
                    value = pickle.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                source_class = payload_representation_class(value)
                source_elements = payload_compute_scale(value)
                break
            if source_class is not None:
                break
        for p_id, fn in sorted(callables.items()):
            per_call_divisor = records_per_call.get(p_id, 1)
            pipe = feature.logical_pipes[p_id]
            if len(pipe.input_pipes) != 1:
                coverage[str(p_id)] = "not_a_single_input_stage"
                continue
            entry: Dict[str, Any] = {}
            transition: Dict[str, str] = {}
            own_input = None
            own_inputs = reservoir.values_for(pipe.input_pipes[0].id)
            for raw in own_inputs:
                try:
                    own_input = pickle.loads(raw)
                    break
                except Exception:  # noqa: BLE001
                    continue
            by_class: Dict[str, List[Any]] = {}
            rejections: Dict[str, str] = {}
            for value in pool:
                klass = payload_representation_class(value)
                if klass is None:
                    continue
                try:
                    produced = fn(value)
                except Exception as exc:  # noqa: BLE001
                    # Keep the first failure per class so a profile that cannot
                    # cover a representation says why instead of only "no
                    # measured representation".
                    rejections.setdefault(
                        klass, f"{type(exc).__name__}: {exc}"[:200]
                    )
                    continue
                out_class = payload_representation_class(produced)
                if out_class is not None and _is_single_record(produced):
                    transition.setdefault(klass, out_class)
                if len(by_class.get(klass, [])) < 4:
                    by_class.setdefault(klass, []).append(value)
            fits: Dict[str, Dict[str, float]] = {}
            grid_specs: List[Tuple[str, float, bytes]] = []
            for klass, values in sorted(by_class.items())[:max_classes]:
                values.sort(
                    key=lambda item: payload_compute_scale(item) or 0.0
                )
                base = values[len(values) // 2]
                for factor in (downscale, upscale):
                    scaled = self._affine_rescale_payload(base, factor)
                    candidate = scaled if scaled is not None else base
                    if scaled is not None:
                        try:
                            fn(candidate)
                        except Exception:  # noqa: BLE001
                            # The rescaled form is not a legal input for this
                            # operator (a container it cannot read); only the
                            # real payload is usable.
                            candidate = base
                    try:
                        snapshot = pickle.dumps(
                            candidate, protocol=pickle.HIGHEST_PROTOCOL
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    scale = payload_compute_scale(candidate)
                    if scale is None:
                        continue
                    if any(
                        spec[0] == klass and abs(spec[1] - float(scale)) < 1e-6
                        for spec in grid_specs
                    ):
                        # A payload that cannot be rescaled (a file path) has
                        # no contrast; one point is enough for the k = 0 fit.
                        continue
                    grid_specs.append((klass, float(scale), snapshot))
            if grid_specs:
                means = self._time_operator_grid_mean_ms(
                    fn,
                    [spec[2] for spec in grid_specs],
                    budget_sec,
                    max_calls_per_point,
                )
                per_class: Dict[str, List[Tuple[float, float]]] = {}
                for (klass, scale, _), cost in zip(grid_specs, means):
                    if cost is None:
                        continue
                    per_class.setdefault(klass, []).append(
                        (scale, cost / per_call_divisor)
                    )
                points_by_class = per_class
            else:
                points_by_class = {}
            for klass, points in points_by_class.items():
                fit = self._fit_compute_coefficients(points)
                if fit is None and len(points) == 1:
                    # A payload with no size contrast (a file path handed to a
                    # reader) still costs something: it is the k = 0 member of
                    # the same family, exactly like the byte model's constant
                    # fallback, and it must not leave the operator unpriced.
                    scale, cost = points[0]
                    fit = {
                        "k_ms_per_element": 0.0,
                        "b_ms": float(cost),
                        "points_ms_per_element": [[scale, cost], [scale, cost]],
                    }
                if fit is None:
                    continue
                fit["samples"] = len(by_class.get(klass, []))
                fits[klass] = fit
            if not fits:
                coverage[str(p_id)] = {
                    "status": "no_measured_representation",
                    "pool": len(pool),
                    "classes_seen": sorted(rejections)
                    or sorted(
                        {
                            str(payload_representation_class(value))
                            for value in pool
                        }
                    ),
                    "rejections": rejections,
                }
                logger.info(
                    "Compute model: operator %s has no measurable "
                    "representation (pool=%s, classes=%s, first rejections=%s)",
                    p_id,
                    len(pool),
                    sorted(rejections)
                    or sorted(
                        {
                            str(payload_representation_class(value))
                            for value in pool
                        }
                    ),
                    rejections,
                )
                continue
            if own_input is not None:
                own_class = payload_representation_class(own_input)
                own_scale = payload_compute_scale(own_input)
                entry["own_class"] = own_class
                entry["own_elements"] = own_scale
                try:
                    produced = fn(own_input)
                    out_scale = payload_compute_scale(produced)
                    if own_scale and out_scale is not None:
                        # ``records_per_call`` operators (a batcher) consume
                        # several records in one call: their per-record compute
                        # scale must not grow with the batch.
                        element_ratios[str(p_id)] = float(out_scale) / (
                            float(own_scale) * per_call_divisor
                        )
                except Exception:  # noqa: BLE001
                    pass
            entry["by_class"] = fits
            operators[str(p_id)] = entry
            transitions[str(p_id)] = transition
            coverage[str(p_id)] = sorted(fits)
        profile.setdefault("physical_model", {})["compute_model"] = {
            "schema_version": 1,
            "statistic": "mean_ms_per_callable_call",
            "scale": "payload elements (C*H*W for images, bytes for text)",
            "method": "per_representation_two_stratum_element_fit",
            "note": (
                "cost = k_(operator, representation_class) * elements + b; the "
                "class comes from payload dtype/channels/container, which a "
                "planner sees without running the plan. Byte volumes remain the "
                "unit for boundaries and transport."
            ),
            "operators": operators,
            "class_transition": transitions,
            "element_ratio": element_ratios,
            "measured_classes": coverage,
            "source_class": source_class,
            "source_elements": source_elements,
        }
        fitted = sum(
            len(entry.get("by_class", {})) for entry in operators.values()
        )
        logger.info(
            "Representation-aware compute model: %s operators, %s class curves",
            len(operators),
            fitted,
        )

    def _profile_layered_backends(
        self,
        profile: Dict[str, Any],
        feature: Feature,
        reservoir: ProfileInputReservoir,
    ) -> None:
        """Collect isolated adaptive costs and targeted width calibration."""
        self._profile_operator_input_size_affine(profile, feature, reservoir)
        self._profile_operator_compute_model(profile, feature, reservoir)
        affine_section = profile.get("physical_model", {}).get(
            "operator_affine"
        )
        if (
            not isinstance(affine_section, dict)
            or affine_section.get("schema_version") != 1
        ):
            raise RuntimeError(
                "Layered profiling must fit an operator kx+b layer; "
                "CEDAR_PROFILE_OPERATOR_AFFINE cannot be disabled for the "
                "shared profile contract"
            )
        min_duration = float(
            os.environ.get("CEDAR_ADAPTIVE_PROFILE_MIN_SEC", "3")
        )
        max_duration = float(
            os.environ.get("CEDAR_ADAPTIVE_PROFILE_MAX_SEC", "30")
        )
        target_rse = float(
            os.environ.get("CEDAR_ADAPTIVE_PROFILE_TARGET_RSE", "0.10")
        )
        min_observations = int(
            os.environ.get("CEDAR_ADAPTIVE_PROFILE_MIN_OBS", "30")
        )
        if not (0 < min_duration <= max_duration):
            raise RuntimeError("Invalid adaptive profile duration bounds")
        if not (0 < target_rse < 1) or min_observations < 2:
            raise RuntimeError("Invalid adaptive profile confidence settings")

        modeled_offloads = {
            PipeVariantType.RAY.name: {},
            PipeVariantType.TF_RAY.name: {},
            PipeVariantType.SMP.name: {},
        }
        isolated = {}
        candidates = {
            PipeVariantType.RAY: [],
            PipeVariantType.SMP: [],
        }
        for variant_type in (
            PipeVariantType.RAY,
            PipeVariantType.SMP,
        ):
            if variant_type == PipeVariantType.RAY and not self.ctx.use_ray():
                continue
            if variant_type == PipeVariantType.SMP:
                # Do not inherit Cedar's historical eight-CPU profile mask.
                # Formal W=8 execution uses the full container cpuset, and
                # restricting a width-48 curve to CPUs 0-7 measures a wholly
                # different resource configuration.
                pass
            for p_id, pipe in feature.logical_pipes.items():
                if pipe.pipe_spec is None or len(pipe.input_pipes) != 1:
                    continue
                if variant_type == PipeVariantType.SMP and pipe.is_tf():
                    # A TensorFlow operator keeps its tf.data iterator and
                    # symbolic tensors inside the pipeline process; the SMP
                    # process pool cannot host it, and replaying the operator
                    # there deadlocks instead of returning a measurement.
                    logger.info(
                        "Skipping SMP profile for TF pipe %s", p_id
                    )
                    continue
                effective_variant = variant_type
                if (
                    variant_type == PipeVariantType.RAY
                    and pipe.is_tf()
                    and PipeVariantType.TF_RAY
                    in pipe.pipe_spec.mutable_variants
                ):
                    effective_variant = PipeVariantType.TF_RAY
                elif variant_type not in pipe.pipe_spec.mutable_variants:
                    continue
                predecessor_id = pipe.input_pipes[0].id
                snapshots = reservoir.values_for(predecessor_id)
                if not snapshots:
                    raise RuntimeError(
                        "No legal replay inputs captured for pipe "
                        f"{p_id} from predecessor {predecessor_id}"
                    )
                logger.info(
                    "Adaptive isolated profile pipe=%s backend=%s inputs=%s",
                    p_id,
                    effective_variant.name,
                    len(snapshots),
                )
                timing = self._adaptive_operator_benchmark(
                    pipe,
                    snapshots,
                    effective_variant,
                    1,
                    min_duration,
                    max_duration,
                    target_rse,
                    min_observations,
                )
                legacy_entry = profile.get("offloads", {}).get(
                    effective_variant.name, {}
                ).get(p_id)
                if legacy_entry is None:
                    raise RuntimeError(
                        "Layered profile has no legacy whole-pipeline "
                        f"measurement for pipe {p_id} backend "
                        f"{effective_variant.name}"
                    )
                # DP/PICO and the three current Simple-DP variants read this
                # isolated worker timing. Cedar and old_dp_boundary read the
                # original measured ``throughput`` from the same entry.
                legacy_entry["backend_compute"] = timing
                modeled_offloads[effective_variant.name][p_id] = (
                    self._modeled_offload_profile(
                        profile["baseline"], p_id, timing
                    )
                )
                isolated[f"{effective_variant.name}:{p_id}"] = timing
                if variant_type in candidates:
                    candidates[variant_type].append(
                        (float(timing["mean_ms_per_sample"]), p_id, pipe,
                         snapshots, effective_variant)
                    )

        physical = profile.setdefault(
            "physical_model", {"schema_version": 1, "boundary": {}}
        )
        object_boundaries = physical.setdefault("object_boundary", {})
        identity_min_duration = float(
            os.environ.get("CEDAR_IDENTITY_BOUNDARY_MIN_SEC", "0.5")
        )
        identity_max_duration = float(
            os.environ.get("CEDAR_IDENTITY_BOUNDARY_MAX_SEC", "3")
        )
        if not (0 < identity_min_duration <= identity_max_duration):
            raise RuntimeError("Invalid identity boundary profile duration")
        for variant_type in (PipeVariantType.RAY, PipeVariantType.SMP):
            if variant_type == PipeVariantType.RAY and not self.ctx.use_ray():
                continue
            values_by_pipe = {}
            marshalling_cache = {}
            identity_cache = {}

            def identity_boundary_for(boundary_p_id: int) -> Dict[str, Any]:
                cached = identity_cache.get(boundary_p_id)
                if cached is not None:
                    return cached
                boundary_snapshots = reservoir.values_for(boundary_p_id)
                if not boundary_snapshots:
                    raise ValueError(
                        f"No legal boundary inputs for pipe {boundary_p_id}"
                    )
                boundary_pipe = feature.logical_pipes[boundary_p_id]
                identity_pipe = FilterPipe(
                    boundary_pipe, _accept_profile_value
                )
                submit_batch_size = None
                if variant_type == PipeVariantType.RAY:
                    serialized_size = statistics.median(
                        len(value) for value in boundary_snapshots
                    )
                    submit_batch_size = min(
                        max(
                            int(
                                compose_constants.RAY_SUBMIT_BATCH_SCALING_FACTOR
                                // max(2 * serialized_size, 1)
                            ),
                            1,
                        ),
                        500,
                    )
                timing = self._adaptive_operator_benchmark(
                    identity_pipe,
                    boundary_snapshots,
                    variant_type,
                    1,
                    identity_min_duration,
                    identity_max_duration,
                    min(0.10, target_rse),
                    min_observations,
                    ray_submit_batch_size=submit_batch_size,
                )
                result = {
                    "method": "cedar_identity_stage_real_objects",
                    "mean_ms_per_sample": float(
                        timing["end_to_end_ms_per_input_sample"]
                    ),
                    "backend_compute_ms_per_sample": float(
                        timing["mean_ms_per_sample"]
                    ),
                    "input_records": int(timing["measured_input_records"]),
                    "submit_batch_size": (
                        submit_batch_size
                        if submit_batch_size is not None
                        else 1
                    ),
                    "adaptive_profile": timing["adaptive_profile"],
                }
                identity_cache[boundary_p_id] = result
                return result

            for p_id, pipe in feature.logical_pipes.items():
                if pipe.pipe_spec is None or len(pipe.input_pipes) != 1:
                    continue
                predecessor_id = pipe.input_pipes[0].id
                try:
                    if predecessor_id not in marshalling_cache:
                        marshalling_cache[predecessor_id] = (
                            profile_object_marshalling(
                                reservoir.values_for(predecessor_id),
                                variant_type,
                            )
                        )
                    if p_id not in marshalling_cache:
                        marshalling_cache[p_id] = profile_object_marshalling(
                            reservoir.values_for(p_id), variant_type
                        )
                    input_identity = identity_boundary_for(predecessor_id)
                    output_identity = identity_boundary_for(p_id)
                except (ValueError, TypeError, pickle.PickleError) as exc:
                    logger.warning(
                        "Skipping real-object %s boundary for pipe %s: %s",
                        variant_type.name,
                        p_id,
                        exc,
                    )
                    continue
                values_by_pipe[p_id] = {
                    "input_pipe_id": predecessor_id,
                    "input_serialize_ms_per_sample": marshalling_cache[
                        predecessor_id
                    ]["serialize_ms_per_sample"],
                    "output_deserialize_ms_per_sample": marshalling_cache[
                        p_id
                    ]["deserialize_ms_per_sample"],
                    "input_serialized_bytes_per_sample": marshalling_cache[
                        predecessor_id
                    ]["serialized_bytes_per_sample"],
                    "output_serialized_bytes_per_sample": marshalling_cache[
                        p_id
                    ]["serialized_bytes_per_sample"],
                    "input_identity_stage": input_identity,
                    "output_identity_stage": output_identity,
                }
            object_boundaries[variant_type.name] = {
                "schema_version": 2,
                "method": "cedar_identity_stage_real_legal_objects",
                "operators": values_by_pipe,
            }
        scaling = physical.setdefault("scaling", {})
        top_k = int(os.environ.get("CEDAR_PROFILE_SCALING_TOP_K", "2"))
        raw_widths = os.environ.get("CEDAR_PROFILE_SCALING_WIDTHS")
        if raw_widths is None:
            raw_widths = os.environ.get("CEDAR_PROFILE_SCALING_WIDTH", "8")
        try:
            scaling_widths = sorted(
                {int(value.strip()) for value in raw_widths.split(",")}
            )
        except ValueError as exc:
            raise RuntimeError("Invalid scaling profile widths") from exc
        if not scaling_widths or scaling_widths[0] < 1:
            raise RuntimeError("Scaling profile widths must be positive")
        scaling_max_duration = float(
            os.environ.get("CEDAR_PROFILE_SCALING_MAX_SEC", "20")
        )
        if scaling_max_duration <= 0:
            raise RuntimeError("Invalid scaling profile max duration")
        scaling_ray_batch_size = int(
            os.environ.get("CEDAR_PROFILE_SCALING_RAY_BATCH_SIZE", "1")
        )
        scaling_min_records_per_worker = int(
            os.environ.get(
                "CEDAR_PROFILE_SCALING_MIN_RECORDS_PER_WORKER", "4"
            )
        )
        if scaling_ray_batch_size < 1 or scaling_min_records_per_worker < 1:
            raise RuntimeError("Invalid scaling profile work quantum")
        for family, entries in candidates.items():
            scaling[family.name] = {}
            ranked_entries = sorted(
                entries, reverse=True, key=lambda item: item[0]
            )
            if top_k > 0:
                ranked_entries = ranked_entries[:top_k]
            for _, p_id, pipe, snapshots, effective_variant in ranked_entries:
                width_timings = {}
                for scaling_width in scaling_widths:
                    if scaling_width == 1:
                        timing = isolated.get(
                            f"{effective_variant.name}:{p_id}"
                        )
                    else:
                        timing = self._adaptive_operator_benchmark(
                            pipe,
                            snapshots,
                            effective_variant,
                            scaling_width,
                            min(1.0, min_duration),
                            min(scaling_max_duration, max_duration),
                            min(0.15, max(target_rse, 0.01)),
                            min_observations,
                            ray_submit_batch_size=(
                                scaling_ray_batch_size
                                if effective_variant
                                in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
                                else None
                            ),
                            minimum_records_per_worker=(
                                scaling_min_records_per_worker
                            ),
                        )
                    if timing is not None:
                        width_timings[scaling_width] = timing
                scaling[family.name][p_id] = {
                    "schema_version": 2,
                    "method": "legal_input_multiwidth_actor_curve",
                    # Fitted cost(actors) = scale / actors ** exponent over the
                    # converged points, so a plan can price the widths its
                    # resource slice allows without enumerating unmeasured
                    # integers. Measured points always win over this fit.
                    "fit": self._fit_width_curve({
                        int(width): float(timing["mean_ms_per_sample"])
                        for width, timing in width_timings.items()
                        if timing.get("adaptive_profile", {}).get(
                            "converged"
                        ) is True
                    }),
                    "widths": width_timings,
                }

        if os.environ.get("CEDAR_PROFILE_CUDA_WORK", "1") == "1":
            from cedar.client.cuda_work_profiler import profile_cuda_metadata_invariance
            cuda_work = profile_cuda_metadata_invariance(self, feature, reservoir)
            physical["cuda_workload"] = cuda_work
            operators = physical["operator_affine"]["operators"]
            for raw_pid, result in cuda_work["operators"].items():
                coefficients = result.get("affine_coefficients")
                if not isinstance(coefficients, dict):
                    continue
                operators[str(int(raw_pid))] = {
                    "k_ms_per_byte": float(coefficients["k_ms_per_byte"]),
                    "b_ms": float(coefficients["b_ms"]),
                    "x_reference_bytes": float(
                        coefficients["x_reference_bytes"]
                    ),
                    "fixed_fraction": float(coefficients["fixed_fraction"]),
                    "source": "remote_actor_metadata_counterfactual",
                    "evidence": "physical_model.cuda_workload",
                    "metadata_invariance": result.get("reason"),
                }
            # The CUDA counterfactual can price operators the local sweep had
            # to skip, so keep the diagnostic list of unpriced operators in
            # step with the coefficients the profile actually carries.
            affine_section = physical["operator_affine"]
            affine_section["unfitted_operators"] = sorted(
                p_id
                for p_id in affine_section.get("unfitted_operators", [])
                if str(int(p_id)) not in operators
            )

        if os.environ.get("CEDAR_PROFILE_SMP_AGGREGATE_TRANSPORT", "1") == "1":
            from cedar.client.smp_transport_profiler import profile_smp_aggregate_transport
            # Balanced legal object classes from all boundaries that can touch
            # an SMP stage. Keep actual types/sizes; sample the median capture.
            boundary_ids = set()
            for p_id, pipe in feature.logical_pipes.items():
                if (pipe.pipe_spec is not None
                        and PipeVariantType.SMP in pipe.pipe_spec.mutable_variants):
                    boundary_ids.add(p_id)
                    boundary_ids.update(p.id for p in pipe.input_pipes)
            transport_snapshots = []
            for boundary_id in sorted(boundary_ids):
                captured = sorted(reservoir.values_for(boundary_id), key=len)
                if captured:
                    transport_snapshots.append(captured[len(captured) // 2])
            if boundary_ids and not transport_snapshots:
                raise RuntimeError("No legal objects captured for SMP aggregate profiling")
            if transport_snapshots:
                context = SMPPipeVariantContext()
                max_pairs = min(32, len(os.sched_getaffinity(0)) // 2)
                widths = [w for w in (1, 2, 4, 8, 16, 32) if w <= max_pairs]
                if max_pairs not in widths:
                    widths.append(max_pairs)
                curve = profile_smp_aggregate_transport(
                    transport_snapshots, workers=widths,
                    max_inflight=context.max_inflight)
                curve["boundary_pipe_ids"] = sorted(boundary_ids)
                physical.setdefault("boundary", {}).setdefault("SMP", {})[
                    "aggregate_transport"] = curve

        profile.setdefault("profile_metadata", {})["profile_protocol"] = (
            "dual_legacy_whole_pipeline_plus_adaptive_layered")
        profile["profile_metadata"]["optimizer_profile_consumers"] = {
            "legacy": "baseline.latencies/throughput, offloads.*.throughput, tf_fuse",
            "simple_dp_and_pico": (
                "baseline.wall_latencies, physical_model.operator_affine, "
                "offloads.*.backend_compute, physical_model.scaling/boundary"),
        }
        profile["layered_profile"] = {
            "schema_version": 1,
            "method": "fixed_legal_input_adaptive_microbenchmark",
            "input_pool": reservoir.metadata(),
            "isolated_operator_costs": isolated,
            "modeled_offloads": modeled_offloads,
            "component_layers": {
                "operator_compute": "isolated_replay",
                "stage_boundary": "physical_model.boundary",
                "object_marshalling": "physical_model.object_boundary",
                "parallel_scaling": "physical_model.scaling",
                "contention_and_fusion": "deferred_to_selected_plan_validation",
            },
            "compatibility_throughput": "legacy_whole_pipeline_measurement",
        }

    def _profile_smp(
        self,
        d: Dict[str, Any],
        feature_to_profile: Feature,
        f_name: str,
        n_samples: Optional[int],
        profile_backend_compute: bool = True,
    ) -> None:
        if "offloads" not in d:
            d["offloads"] = {}
        d["offloads"][PipeVariantType.SMP.name] = {}
        for p_id, pipe in feature_to_profile.logical_pipes.items():
            if pipe.is_tf():
                # TF operators keep their tf.data iterators in-process; the SMP
                # process pool cannot host them (the profile replay deadlocks),
                # so this workload is profiled without SMP entries.
                logger.info("Skipping SMP profile for TF pipe %s", p_id)
                continue
            if (
                pipe.pipe_spec is not None
                and PipeVariantType.SMP in pipe.pipe_spec.mutable_variants
            ):
                logger.info(f"Profiling feature {p_id} with SMP")
                mutation_dict = {}
                # TODO: Choose some reasonable values for these...
                mutation_dict[p_id] = SMPPipeVariantContext(
                    n_procs=SMP_PROFILE_N_PROCS,
                    max_inflight=SMP_PROFILE_INFLIGHT,
                    max_prefetch=SMP_PROFILE_PREFETCH,
                    use_threads=True,
                    disable_torch_parallelism=True,
                    profile_backend_compute=profile_backend_compute,
                )
                profile = self._profile_feature(
                    f_name,
                    feature_to_profile,
                    n_samples,
                    mutation_dict,
                )
                d["offloads"][PipeVariantType.SMP.name][p_id] = profile

    def _profile_ray(
        self,
        d: Dict[str, Any],
        feature_to_profile: Feature,
        f_name: str,
        n_samples: Optional[int],
        profile_backend_compute: bool = True,
    ) -> None:
        if "offloads" not in d:
            d["offloads"] = {}
        d["offloads"][PipeVariantType.RAY.name] = {}
        d["offloads"][PipeVariantType.TF_RAY.name] = {}
        for p_id, pipe in feature_to_profile.logical_pipes.items():
            if pipe.is_tf():
                if (
                    pipe.pipe_spec is not None
                    and PipeVariantType.TF_RAY
                    in pipe.pipe_spec.mutable_variants
                ):
                    logger.info(
                        f"Profiling feature {p_id} with ray TF offload"
                    )
                    mutation_dict = {}
                    # TODO: Choose some reasonable values for these...
                    mutation_dict[p_id] = TFRayPipeVariantContext(
                        n_actors=RAY_PROFILE_N_ACTORS,
                        max_inflight=RAY_PROFILE_INFLIGHT,
                        max_prefetch=RAY_PROFILE_PREFETCH,
                        use_threads=True,
                        submit_batch_size=RAY_PROFILE_SUBMIT_BATCH_SIZE,
                        profile_backend_compute=profile_backend_compute,
                    )

                    profile = self._profile_feature(
                        f_name,
                        feature_to_profile,
                        n_samples,
                        mutation_dict,
                    )
                    d["offloads"][PipeVariantType.TF_RAY.name][p_id] = profile
            else:
                if (
                    pipe.pipe_spec is not None
                    and PipeVariantType.RAY in pipe.pipe_spec.mutable_variants
                ):
                    logger.info(f"Profiling feature {p_id} with ray offload")
                    mutation_dict = {}
                    # TODO: Choose some reasonable values for these...
                    mutation_dict[p_id] = RayPipeVariantContext(
                        n_actors=RAY_PROFILE_N_ACTORS,
                        max_inflight=RAY_PROFILE_INFLIGHT,
                        max_prefetch=RAY_PROFILE_PREFETCH,
                        use_threads=True,
                        submit_batch_size=RAY_PROFILE_SUBMIT_BATCH_SIZE,
                        profile_backend_compute=profile_backend_compute,
                        num_gpus=(
                            1.0 / RAY_PROFILE_N_ACTORS
                            if pipe.execution_resource
                            == PipeExecutionResource.CUDA
                            else 0.0
                        ),
                    )

                    profile = self._profile_feature(
                        f_name,
                        feature_to_profile,
                        n_samples,
                        mutation_dict,
                    )
                    d["offloads"][PipeVariantType.RAY.name][p_id] = profile

    def _profile_feature(
        self,
        f_name: str,
        feature_to_profile: Feature,
        n_samples: Optional[int],
        mutation_dict: Optional[Dict[int, PipeVariantContext]],
    ):
        filter_counters = {}
        original_filter_fns = {}
        collect_filter_selectivity = (
            os.environ.get("CEDAR_PROFILE_FILTER_SELECTIVITY") == "1"
        )
        if collect_filter_selectivity:
            for p_id, pipe in feature_to_profile.logical_pipes.items():
                if isinstance(pipe, FilterPipe):
                    original_filter_fns[p_id] = pipe.fn
                    counter = _ProfiledFilterCallable(pipe.fn)
                    filter_counters[p_id] = counter
                    pipe.fn = counter
        try:
            loaded_feature = feature_to_profile.profile(
                self.ctx, mutation_dict
            )
        finally:
            # Materialized variants retain the wrapped callable. Restore the
            # logical graph immediately so later offload profiles and formal
            # executions use the original operator object.
            for p_id, fn in original_filter_fns.items():
                feature_to_profile.logical_pipes[p_id].fn = fn
        source_pipe = feature_to_profile.get_source_pipes()

        # Create a profiler
        profiler = FeatureProfiler(feature_to_profile, profile_mode=True)
        b_sz = profiler.get_batch_size()

        # Create an Iterable
        dataset_iter = _DataSetIter(
            loaded_features={f_name: loaded_feature},
            profilers={f_name: profiler},
            return_datasample=False,
            source_pipes={f_name: source_pipe},
        )

        try:
            n_batches = 0
            for x in dataset_iter:
                # Warm up time
                if n_batches == 0:
                    start_time = time.time()
                n_batches += 1
                curr_time = time.time()

                if n_samples is not None:
                    if n_batches * b_sz >= n_samples:
                        break
                elif (curr_time - start_time) >= PROFILE_TIME_SEC:
                    break
            end_time = time.time()

            throughput_samples_per_sec = (n_batches * b_sz) / (
                end_time - start_time
            )
            # Per-pipe latencies
            pipe_latencies = profiler.calculate_avg_latency_per_sample()
            wall_pipe_latencies = (
                profiler.calculate_avg_wall_latency_per_sample()
            )
            input_sizes, output_sizes = profiler.calculate_avg_data_size()

            # A backend profile mutates exactly one logical operator.  Read
            # worker-side timings before reset tears down its service.
            backend_compute = None
            if mutation_dict is not None and len(mutation_dict) == 1:
                profiled_p_id = next(iter(mutation_dict))
                physical_pipe = feature_to_profile.physical_pipes.get(
                    profiled_p_id
                )
                if physical_pipe is not None:
                    variant = physical_pipe.get_variant()
                    service = getattr(variant, "service", None)
                    if service is None:
                        service = getattr(
                            getattr(variant, "variant_ctx", None),
                            "service",
                            None,
                        )
                    stats_fn = getattr(
                        service, "get_backend_compute_stats", None
                    )
                    if stats_fn is not None:
                        backend_compute = stats_fn()
        finally:
            # Backend profiling repeatedly rebuilds the same Feature. Besides
            # stopping Cedar workers, give the workload a chance to clear
            # process-global accelerator caches before the next trial. This
            # must also run after an operator exception or CUDA OOM.
            if feature_to_profile.loaded:
                feature_to_profile.reset()
            feature_to_profile.release_profile_resources()
            time.sleep(5)  # Allow worker shutdown and CUDA frees to settle.

        result = {
            "latencies": pipe_latencies,
            "wall_latencies": wall_pipe_latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
            "throughput": throughput_samples_per_sec,
        }
        if backend_compute is not None:
            result["backend_compute"] = backend_compute
        if collect_filter_selectivity:
            input_counts = {
                p_id: counter.input_count
                for p_id, counter in filter_counters.items()
            }
            output_counts = {
                p_id: counter.output_count
                for p_id, counter in filter_counters.items()
            }
            result["input_counts"] = input_counts
            result["output_counts"] = output_counts
            result["selectivities"] = {
                p_id: (
                    output_counts[p_id] / input_counts[p_id]
                    if input_counts[p_id]
                    else 1.0
                )
                for p_id in filter_counters
            }
        return result

    def close(self):
        """Release worker and pipe resources owned by this dataset."""
        if self._mp_iter is not None:
            self._mp_iter._shutdown()
            self._mp_iter = None

        # Ray actor handles must be released while the current Ray driver is
        # still connected. Relying on Python destructors after ray.shutdown()
        # can reconnect a fresh driver that does not own the old handles.
        if self._iter_mode != "mp":
            for feature in self.features.values():
                if getattr(feature, "loaded", False):
                    feature.reset()
        self.dataset_iter = None

    def _exit(self):
        # Backwards-compatible test helper.
        self.close()
