import logging
import os
import pathlib
import threading
import multiprocessing as mp
import math
import copy
import pickle
import tempfile
import time
import yaml
import sys
from typing import Dict, Optional, Iterable, Any, List, Union, Tuple
from queue import Queue, Empty

from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions, PhysicalPlan
from cedar.pipes import (
    FilterPipe,
    Pipe,
    PipeVariant,
    DataSample,
    PipeVariantType,
    PipeVariantContext,
    InProcessPipeVariantContext,
    RayPipeVariantContext,
    TFRayPipeVariantContext,
    SMPPipeVariantContext,
)
from cedar.pipes.common import (
    ProfileInputReservoir,
    set_profile_input_reservoir,
)
from .profiler import FeatureProfiler
from .boundary_profiler import profile_stage_boundary_cached
from .controller import FeatureController
from .logger import DataSetLogger
from .utils import (
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
    SMP_TASKSET_MASK,
)

logger = logging.getLogger(__name__)


MP_QUEUE_MAX_SIZE = 100


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

    def _await_workers_ready(self, timeout_sec: float = 180.0) -> None:
        """Wait until every worker has constructed and verified its stages."""
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

        self._done.set()
        for _, event in self._worker_epoch_start.items():
            # Need to signal start for workers to check for done signal
            event.set()
        # Use one shared grace period. Workers can be blocked in queue.put()
        # after the consumer stops at num_total_samples.
        deadline = time.monotonic() + 1.0
        for _, w in self._workers.items():
            w.join(max(0.0, deadline - time.monotonic()))
        for idx, w in self._workers.items():
            if w.is_alive():
                logger.info(f"Terminating worker {idx}...")
                w.terminate()
        for _, w in self._workers.items():
            w.join(5)
        self._workers.clear()
        self._worker_epoch_start.clear()
        self._result_queue.cancel_join_thread()
        self._result_queue.close()
        self._startup_queue.close()

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
        # 10/DjTwoStageOptimizer, 11/SimpleDpOptimizer.
        optimizer_selector = 0
        if self.optimizer_options is not None:
            optimizer_selector = int(
                getattr(self.optimizer_options, "use_my_optimizer", 0)
            )
        self._legacy_cedar_profile = optimizer_selector == 11
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
        elif optimizer_selector != 0:
            raise ValueError(
                "OptimizerOptions.use_my_optimizer must be between 0 and 11."
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

    def _profile(
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
            if (
                os.environ.get(
                    "CEDAR_INCREMENTAL_COMPUTE_SCALING"
                )
                == "1"
            ):
                return self._profile_compute_scaling_incremental(
                    f_name=f_name,
                    feature_to_profile=self.features[f_name],
                    n_samples=n_samples,
                    output_file=output_file,
                    existing_profile=incremental_from,
                )
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
            os.environ.get("CEDAR_LAYERED_ADAPTIVE_PROFILE") == "1"
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
            set_profile_input_reservoir(reservoir)
        try:
            baseline_profile = self._profile_feature(
                f_name, feature_to_profile, n_samples, None
            )
        finally:
            if layered_profile:
                set_profile_input_reservoir(None)
        d["baseline"] = baseline_profile
        inferred_scalings = baseline_profile.get(
            "compute_scaling_inference", {}
        )
        d["operator_compute_scaling"] = {}
        for p_id, pipe in feature_to_profile.logical_pipes.items():
            explicit = bool(
                getattr(pipe, "compute_scaling_explicit", False)
            )
            inference = inferred_scalings.get(
                p_id, inferred_scalings.get(str(p_id))
            )
            if explicit:
                scaling = pipe.compute_scaling.value
                mode = "explicit"
            elif isinstance(inference, dict):
                scaling = inference.get("scaling", "per_data")
                mode = (
                    "inferred"
                    if inference.get("reason") == "classified"
                    else "default"
                )
            else:
                scaling = "per_data"
                mode = "default"
            entry = {"scaling": scaling, "mode": mode}
            if isinstance(inference, dict):
                entry["inference"] = inference
            d["operator_compute_scaling"][p_id] = entry

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
            if reservoir is None:
                raise RuntimeError("Layered profile input reservoir is absent")
            self._profile_layered_backends(
                d, feature_to_profile, reservoir
            )
        else:
            # If using ray, profile each op
            if self.ctx.use_ray():
                self._profile_ray(d, feature_to_profile, f_name, n_samples)

            # NOTE: Run this last as it un-tasksets
            _set_cpu_affinity(SMP_TASKSET_MASK)
            self._profile_smp(d, feature_to_profile, f_name, n_samples)

        if self.ctx.use_ray() and profile_boundaries:
            self._profile_boundary_model(
                d,
                PipeVariantType.RAY,
                RAY_PROFILE_N_ACTORS,
            )
        if profile_boundaries:
            self._profile_boundary_model(
                d,
                PipeVariantType.SMP,
                SMP_PROFILE_N_PROCS,
            )

        if os.environ.get("CEDAR_PROFILE_FILTER_SELECTIVITY") == "1":
            _consolidate_filter_selectivity(d)

        # TF fusion is a separate whole-pipeline candidate validation. It is
        # deliberately excluded from the isolated-cost layer.
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
            _set_cpu_affinity(SMP_TASKSET_MASK)
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

    def _profile_compute_scaling_incremental(
        self,
        f_name: str,
        feature_to_profile: Feature,
        n_samples: Optional[int],
        output_file: Optional[str],
        existing_profile: str,
    ) -> Dict[str, Any]:
        """Add operator-level scaling semantics to an existing profile.

        Only a fresh in-process baseline observation is executed. All numeric
        measurements consumed by Cedar/DJ/Pecan and the existing DP cost
        coefficients remain untouched, so prior optimizer results remain a
        valid comparison set.
        """
        with open(existing_profile, "r") as stream:
            profile = yaml.safe_load(stream)
        if not isinstance(profile, dict):
            raise RuntimeError(
                f"Incremental profile is not a mapping: {existing_profile}"
            )
        baseline = profile.get("baseline")
        if not isinstance(baseline, dict):
            raise RuntimeError("Existing profile has no baseline mapping.")

        old_setting = os.environ.get(
            "CEDAR_PROFILE_INFER_COMPUTE_SCALING"
        )
        os.environ["CEDAR_PROFILE_INFER_COMPUTE_SCALING"] = "1"
        try:
            fresh_baseline = self._profile_feature(
                f_name, feature_to_profile, n_samples, None
            )
        finally:
            if old_setting is None:
                os.environ.pop(
                    "CEDAR_PROFILE_INFER_COMPUTE_SCALING", None
                )
            else:
                os.environ[
                    "CEDAR_PROFILE_INFER_COMPUTE_SCALING"
                ] = old_setting

        inference = fresh_baseline.get("compute_scaling_inference")
        if not isinstance(inference, dict):
            raise RuntimeError(
                "Incremental operator scaling collected no inference data."
            )
        baseline_pipe_ids = set(baseline.get("latencies", {}))
        unknown_pipe_ids = set(inference) - baseline_pipe_ids
        if unknown_pipe_ids:
            raise RuntimeError(
                "Operator scaling observation contains pipes absent from "
                f"the existing profile: {sorted(unknown_pipe_ids)}"
            )

        metadata = {}
        for p_id, pipe in feature_to_profile.logical_pipes.items():
            explicit = bool(
                getattr(pipe, "compute_scaling_explicit", False)
            )
            observation = inference.get(
                p_id, inference.get(str(p_id))
            )
            if explicit:
                scaling = pipe.compute_scaling.value
                mode = "explicit"
            elif isinstance(observation, dict):
                scaling = observation.get("scaling", "per_data")
                mode = (
                    "inferred"
                    if observation.get("reason") == "classified"
                    else "default"
                )
            else:
                scaling = "per_data"
                mode = "default"
            entry = {"scaling": scaling, "mode": mode}
            if isinstance(observation, dict):
                entry["inference"] = observation
            metadata[p_id] = entry
        profile["operator_compute_scaling"] = metadata
        profile["incremental_compute_scaling"] = {
            "schema_version": 1,
            "source_profile": os.path.abspath(existing_profile),
            "method": "legal_input_size_latency_stratification",
            "observed_operators": len(inference),
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
        _set_cpu_affinity(SMP_TASKSET_MASK)
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
    ) -> Dict[str, Any]:
        """Measure one backend with fixed inputs until confidence converges."""
        replay = _ProfileReplayPipeVariant(snapshots, 1)
        predecessor = pipe.input_pipes[0]
        replay.p_id = predecessor.id
        old_predecessor_variant = predecessor.pipe_variant
        predecessor.pipe_variant = replay
        if variant_type == PipeVariantType.RAY:
            variant_ctx = RayPipeVariantContext(
                n_actors=width,
                max_inflight=max(RAY_PROFILE_INFLIGHT, width * 5),
                max_prefetch=RAY_PROFILE_PREFETCH,
                use_threads=True,
                submit_batch_size=RAY_PROFILE_SUBMIT_BATCH_SIZE,
                profile_backend_compute=True,
            )
        elif variant_type == PipeVariantType.TF_RAY:
            variant_ctx = TFRayPipeVariantContext(
                n_actors=width,
                max_inflight=max(RAY_PROFILE_INFLIGHT, width * 5),
                max_prefetch=RAY_PROFILE_PREFETCH,
                use_threads=True,
                submit_batch_size=RAY_PROFILE_SUBMIT_BATCH_SIZE,
                profile_backend_compute=True,
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
            warm_started = time.perf_counter()
            replay.record_count = 1
            for _ in variant:
                pass
            warm_elapsed = max(time.perf_counter() - warm_started, 1e-6)
            reset_stats = getattr(
                service, "reset_backend_compute_stats", None
            )
            if reset_stats is None:
                raise RuntimeError(
                    f"{variant_type.name} service cannot reset timing stats"
                )
            reset_stats()

            # Aim for roughly half-second epochs. This bounds stopping
            # overshoot without paying iterator reset overhead per record.
            epoch_records = max(1, min(256, int(0.5 / warm_elapsed)))
            started = time.perf_counter()
            converged = False
            rse = math.inf
            stats = None
            while True:
                replay.record_count = epoch_records
                for _ in variant:
                    pass
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
            stats["adaptive_profile"] = {
                "width": width,
                "elapsed_sec": elapsed,
                "warmup_sec": warm_elapsed,
                "epoch_records": epoch_records,
                "unique_input_records": len(snapshots),
                "target_rse": target_rse,
                "observed_rse": rse,
                "min_duration_sec": min_duration,
                "max_duration_sec": max_duration,
                "min_observations": min_observations,
                "converged": converged,
                "stop_reason": "confidence" if converged else "max_duration",
            }
            return stats
        finally:
            variant.shutdown()

    def _profile_layered_backends(
        self,
        profile: Dict[str, Any],
        feature: Feature,
        reservoir: ProfileInputReservoir,
    ) -> None:
        """Collect isolated adaptive costs and targeted width calibration."""
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

        profile["offloads"] = {
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
                _set_cpu_affinity(SMP_TASKSET_MASK)
            for p_id, pipe in feature.logical_pipes.items():
                if pipe.pipe_spec is None or len(pipe.input_pipes) != 1:
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
                profile["offloads"][effective_variant.name][p_id] = (
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
        scaling = physical.setdefault("scaling", {})
        top_k = int(os.environ.get("CEDAR_PROFILE_SCALING_TOP_K", "2"))
        scaling_width = int(
            os.environ.get("CEDAR_PROFILE_SCALING_WIDTH", "8")
        )
        for family, entries in candidates.items():
            scaling[family.name] = {}
            for _, p_id, pipe, snapshots, effective_variant in sorted(
                entries, reverse=True, key=lambda item: item[0]
            )[:top_k]:
                timing = self._adaptive_operator_benchmark(
                    pipe,
                    snapshots,
                    effective_variant,
                    scaling_width,
                    min(1.0, min_duration),
                    min(10.0, max_duration),
                    min(0.15, max(target_rse, 0.01)),
                    min_observations,
                )
                scaling[family.name][p_id] = timing

        profile["layered_profile"] = {
            "schema_version": 1,
            "method": "fixed_legal_input_adaptive_microbenchmark",
            "input_pool": reservoir.metadata(),
            "isolated_operator_costs": isolated,
            "component_layers": {
                "operator_compute": "isolated_replay",
                "stage_boundary": "physical_model.boundary",
                "parallel_scaling": "physical_model.scaling",
                "contention_and_fusion": "deferred_to_selected_plan_validation",
            },
            "compatibility_throughput": "component_substitution",
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
            compute_scaling_inference = None
            if os.environ.get(
                "CEDAR_PROFILE_INFER_COMPUTE_SCALING"
            ) == "1":
                compute_scaling_inference = profiler.infer_compute_scaling()

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
        if compute_scaling_inference is not None:
            result["compute_scaling_inference"] = compute_scaling_inference
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


def _set_cpu_affinity(mask):
    pid = os.getpid()  # Get current process ID
    command = f"taskset -p {mask} {pid}"
    os.system(command)  # Execute the taskset command
