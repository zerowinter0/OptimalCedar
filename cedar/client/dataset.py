import logging
import os
import pathlib
import threading
import multiprocessing as mp
import math
import tempfile
import time
import yaml
import sys
from typing import Dict, Optional, Iterable, Any, List, Union, Tuple
from queue import Queue, Empty

from cedar.config import CedarContext
from cedar.compose import Feature, OptimizerOptions, PhysicalPlan
from cedar.pipes import (
    Pipe,
    PipeVariant,
    DataSample,
    PipeVariantType,
    PipeVariantContext,
    RayPipeVariantContext,
    TFRayPipeVariantContext,
    SMPPipeVariantContext,
)
from .profiler import FeatureProfiler
from .controller import FeatureController
from .logger import DataSetLogger
from .utils import multiprocess_worker_loop, Sentinel, unpack_feature_map
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
PROFILE_TIME_SEC = 10


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
        self._result_queue = mp.Queue(maxsize=MP_QUEUE_MAX_SIZE)
        self._plans = plans

        self._workers = {}
        self._features = features
        self._done = mp.Event()
        self._num_done = 0
        self._enable_controller = enable_controller

        self._worker_epoch_start = {}

        self._init_workers()

    def _init_workers(self):
        idx = 0

        ray_parallelism = math.ceil(AVAILABLE_RAY_SCALE / len(self._features))
        for f_name, feature in self._features.items():
            if self._plans is not None:
                plan = self._plans[f_name]
            else:
                plan = None
            epoch_start = mp.Event()
            self._worker_epoch_start[idx] = epoch_start
            worker = mp.Process(
                target=multiprocess_worker_loop,
                args=(
                    idx,
                    self._ctx,
                    self._result_queue,
                    feature,
                    f_name,
                    plan,
                    self._done,
                    epoch_start,
                    self._enable_controller,
                    {PipeVariantType.RAY: ray_parallelism},
                ),
            )
            idx += 1
            worker.daemon = True
            worker.start()
            self._workers[idx] = worker

    def __iter__(self):
        logger.info("New epoch for MP iter...")
        for _, event in self._worker_epoch_start.items():
            # Signal all workers to start next epoch
            event.set()
        self._num_done = 0
        return self

    def __next__(self):
        while (
            self._num_done < len(self._workers)
            or not self._result_queue.empty()
        ):
            try:
                data = self._result_queue.get(timeout=0.1)
                if isinstance(data, Sentinel):
                    self._num_done += 1
                else:
                    return data
            except Empty:
                continue
        else:
            logger.info("Finished fetching from workers...")
            raise StopIteration

    def _shutdown(self):
        self._done.set()
        for _, event in self._worker_epoch_start.items():
            # Need to signal start for workers to check for done signal
            event.set()
        # Force shutdown procs if necessary
        for _, w in self._workers.items():
            w.join(5)
        for idx, w in self._workers.items():
            if w.is_alive():
                logger.info(f"Terminating worker {idx}...")
        self._workers.clear()

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

        # Optionally swap in MyOptimizer as a drop-in replacement for the
        # default Optimizer, controlled via OptimizerOptions.use_my_optimizer.
        if (
            self.optimizer_options is not None
            and getattr(self.optimizer_options, "use_my_optimizer", False)
        ):
            from cedar.compose.my_optimizer import MyOptimizer

            for _, feature in self.features.items():
                feature.set_optimizer(MyOptimizer())

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

        # Enable profiling for the feature
        logger.info(
            "Profiling feature {}, output to {}...".format(f_name, output_file)
        )
        feature_to_profile = self.features[f_name]

        d = {}

        baseline_profile = self._profile_feature(
            f_name, feature_to_profile, n_samples, None
        )
        d["baseline"] = baseline_profile

        # If using ray, profile each op
        if self.ctx.use_ray():
            self._profile_ray(d, feature_to_profile, f_name, n_samples)

        # NOTE: Run this last as it un-tasksets
        _set_cpu_affinity(SMP_TASKSET_MASK)

        self._profile_smp(d, feature_to_profile, f_name, n_samples)

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
        iter = _DataSetIter(
            loaded_features={f_name: loaded_feature},
            return_datasample=False,
            source_pipes={f_name: source_pipe},
        )
        b_sz = feature_to_profile.get_batch_size()

        n_batches = 0
        for x in iter:
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

        # Reset the feature and init
        feature_to_profile.reset()
        time.sleep(5)  # Sleep in case we need some time to shutdown

        d["tf_fuse"] = {
            "throughput": throughput_samples_per_sec,
        }

    def _profile_smp(
        self,
        d: Dict[str, Any],
        feature_to_profile: Feature,
        f_name: str,
        n_samples: Optional[int],
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
        loaded_feature = feature_to_profile.profile(self.ctx, mutation_dict)
        source_pipe = feature_to_profile.get_source_pipes()

        # Create a profiler
        profiler = FeatureProfiler(feature_to_profile, profile_mode=True)
        b_sz = profiler.get_batch_size()

        # Create an Iterable
        iter = _DataSetIter(
            loaded_features={f_name: loaded_feature},
            profilers={f_name: profiler},
            return_datasample=False,
            source_pipes={f_name: source_pipe},
        )

        n_batches = 0
        for x in iter:
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
        input_sizes, output_sizes = profiler.calculate_avg_data_size()

        # Reset the feature and init
        feature_to_profile.reset()
        time.sleep(5)  # Sleep in case we need some time to shutdown

        return {
            "latencies": pipe_latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
            "throughput": throughput_samples_per_sec,
        }

    def _exit(self):
        # For test cases, force terminate dataset
        if self._mp_iter is not None:
            self._mp_iter._shutdown()


def _set_cpu_affinity(mask):
    pid = os.getpid()  # Get current process ID
    command = f"taskset -p {mask} {pid}"
    os.system(command)  # Execute the taskset command
