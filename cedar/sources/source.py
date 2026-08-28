from __future__ import annotations

import abc
import logging
import time
import tempfile
import pathlib
import pickle
from typing import Any, Dict, Iterator, Tuple, Optional

from cedar.pipes import Pipe, DataSample, Partition

from cedar.utils.frameworks import tensorflow

TRACE_FREQUENCY_SEC = 0.1
DISABLE_DATASAMPLES = False  # for benchmarking only
logger = logging.getLogger(__name__)


class Source(abc.ABC):
    """
    A Source represents a collection of raw input data,
    which is the input into the DAG of ``Pipes``.

    Calling ``to_pipe()`` on each Source will return
    a ``Pipe`` representing a read on the source.
    """

    @abc.abstractmethod
    def to_pipe(self) -> Pipe:
        """
        NOTE: This function must create a NEW pipe object each time it is
        called.
        """
        pass


class SourcePipeVariantMixin:
    """
    Mixin to provide additional state management for source pipe variants
    """

    def __init__(
        self,
        num_samples_in_partition: int = 1000,
        rank_spec: Optional[Tuple[int, int]] = None,
    ):
        # Number of samples in each partition
        self.num_samples_in_partition = num_samples_in_partition

        # NOTE: To reduce closed partition size, we could also use a
        # system resembling TCP SYN / ACK --> then we would not have to
        # store all closed partitions, but rather the last closed
        # (or the next partition we expect to be closed).

        # partition ID of partition to assign the next DataSample to
        self.current_partition_idx = 0

        # Partitions that have been succesfully processed.
        # Maps partition ID to partition.
        self.sealed_partitions = {}

        # Partitions of which at least one datasample is in flight.
        # Maps partition ID to partition.
        self.in_flight_partitions = {}

        # Partitions of which all DataSamples have been put in flight.
        # Maps partition ID to partition.
        self.fully_sent_partitions = {}

        # number of samples yielded in current epoch
        self.num_yielded = 0

        # partitions to be replayed if replay mode is on
        self.partitions_to_replay = {}

        # flag indicating whether to replay partitions
        self.replay = False

        # Indicates whether any replay state needs to be initialized
        self.initialized_replay = False

        # True if this source should enable profile mode
        self.profile_mode = False

        # True if this source is sharded
        if rank_spec is not None:
            self.sharded = True
            self.world_size = rank_spec[0]
            self.rank = rank_spec[1]
        else:
            self.sharded = False
            self.world_size = -1
            self.rank = -1

    def get_in_flight_partitions(self) -> Dict[int, Partition]:
        return self.in_flight_partitions

    def get_sealed_partitions(self) -> Dict[int, Partition]:
        return self.sealed_partitions

    def get_fully_sent_partitions(self) -> Dict[int, Partition]:
        return self.fully_sent_partitions

    def get_num_samples_in_partition(self) -> int:
        return self.num_samples_in_partition

    def seal_last_partition(self) -> None:
        curr_partition_id = self.num_yielded // self.num_samples_in_partition
        end_sample_idx = self.num_yielded - 1
        on_edge = self.num_yielded % self.num_samples_in_partition == 0

        if on_edge:
            curr_partition_id -= 1
        else:
            self._mark_partition_as_fully_sent(
                curr_partition_id, end_sample_idx
            )

        self.seal_partition(curr_partition_id)

    def seal_partition(self, partition_id: int) -> None:
        """
        Marks partition with partition_id as completed.
        """
        if partition_id not in self.fully_sent_partitions:
            raise RuntimeError(
                f"Attempting to seal partition ID\
                               {partition_id}.However, ID was not marked as\
                                fully sent. Can't seal partition that hasn't\
                                been fully sent."
            )

        sealed_partition = self.fully_sent_partitions.pop(partition_id)

        if partition_id in self.fully_sent_partitions:
            raise RuntimeError(
                f"Did not succefully remove partition at\
                               partition ID {partition_id}"
            )

        if sealed_partition.partition_id != partition_id:
            raise RuntimeError(
                f"Found partition ID\
                               {sealed_partition.partition_id} instead of\
                                {partition_id} in fully sent partitions dict."
            )

        if partition_id in self.sealed_partitions:
            raise RuntimeError(
                f"Partition ID {partition_id} is already\
                               sealed. Cannot seal again."
            )

        self.sealed_partitions[partition_id] = sealed_partition

    def reset_for_new_epoch(self, checkpoint: bool = False) -> None:
        """
        Resets the state of the source, resetting all partition information.
        Should be used at the start of a new epoch.
        If checkpoint is true, then current partition information is
        saved to disk.
        """
        logging.warning("Resetting epoch information.")

        if len(self.in_flight_partitions) > 0:
            logging.warning(
                "The number of in flight partitions is greater than 0."
            )

        if len(self.fully_sent_partitions) > 0:
            logging.warning(
                "The number of fully sent partitions is greater than 0."
            )

        if checkpoint:
            self.checkpoint_partitions()

        self.current_partition_idx = 0
        self.sealed_partitions = {}
        self.in_flight_partitions = {}
        self.fully_sent_partitions = {}
        self.num_yielded = 0
        self._reset_source_iterator_for_epoch()

    def _reset_source_iterator_for_epoch(self):
        """
        Resets iterator to its base state at the
        start of a new epoch. Should be overwritten
        by each source variant implementation.
        """
        raise NotImplementedError("Cannot reset source iterator.")

    def _create_new_in_flight_partition(
        self, partition_id: int, start_sample_id: int
    ) -> None:
        """
        Creates new in-flight partition with indicated partition_id and
        start_sample_id. end_sample_id of new partition is None.
        """
        if partition_id in self.in_flight_partitions:
            raise RuntimeError(
                f"Can't add to in flight partitions. Partition\
                               with ID {partition_id} already in flight."
            )
        elif partition_id in self.fully_sent_partitions:
            raise RuntimeError(
                f"Can't add to in flight partitions. Partition\
                               with ID {partition_id} already fully sent."
            )
        elif partition_id in self.sealed_partitions:
            raise RuntimeError(
                f"Can't add to in flight partitions. Partition\
                               with ID {partition_id} already sealed."
            )

        new_partition = Partition(
            partition_id=partition_id, starting_sample_id=start_sample_id
        )
        self.in_flight_partitions[partition_id] = new_partition

    def _mark_partition_as_fully_sent(
        self, partition_id: int, end_sample_id: int
    ) -> None:
        """
        Marks partition with partition_id as fully sent.
        Sets end_sample_id of partition to be provided end_sample_id.
        """
        if partition_id not in self.in_flight_partitions:
            raise RuntimeError(
                f"Cannot mark partition with ID {partition_id}\
                               as fully sent. Partition is not in flight."
            )

        fully_sent_partition = self.in_flight_partitions.pop(partition_id)

        if partition_id in self.in_flight_partitions:
            raise RuntimeError(
                f"Did not succefully remove partition with\
                partition ID {partition_id} from in flight partitions."
            )

        if fully_sent_partition.partition_id != partition_id:
            raise RuntimeError(
                f"Found partition ID\
                {fully_sent_partition.partition_id} instead of\
                {partition_id} in fully sent partitions dict."
            )

        fully_sent_partition.end_sample_id = end_sample_id
        self.fully_sent_partitions[partition_id] = fully_sent_partition

    def checkpoint_partitions(
        self, checkpoint_only_closed: bool = True
    ) -> None:
        """
        Checkpoints partitions by saving partitions data to pkl file.
        If this file ("/tmp/cedar_checkpointing/cedar_checkpoint.pkl")
        already exists, then the contents of the file are overwritten.
        By default, only checkpoints closed partitions.
        If checkpoint_only_closed is false, then also checkpoints
        in-flight and fully-sent partitions.
        """
        checkpoint_data = {"sealed": self.sealed_partitions}

        if not checkpoint_only_closed:
            checkpoint_data["in-flight"] = self.in_flight_partitions
            checkpoint_data["fully-sent"] = self.fully_sent_partitions

        temp_dir = tempfile.gettempdir()
        checkpoint_dir = pathlib.Path(temp_dir) / pathlib.Path(
            "cedar_checkpointing"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / pathlib.Path("cedar_checkpoint.pkl")
        with checkpoint_file.open("wb") as file:
            pickle.dump(checkpoint_data, file)

    def disable_replay(self) -> None:
        """
        Updates SourcePipeVariantMix state, indicating that
        replay has completed.
        """
        self.replay = False
        self.initialized_replay = False
        self.partitions_to_replay = {}

    def enable_replay(self) -> None:
        """
        Sets replay mode to True. On iteration, replay mode
        sends DataSamples from partitions that have been sent
        (either in flight or fully sent), but that have not
        been marked as sealed. No new DataSamples or Partitions
        will be generated in this process.

        """
        self.replay = True
        if len(self.partitions_to_replay) > 0:
            raise RuntimeError(
                "There are still previous partitions that have\
                               not been replayed."
            )

        if len(self.in_flight_partitions) > 1:
            raise RuntimeError(
                "More than 1 partition in fligh. Cannot enable replay."
            )

        for partition_id in self.in_flight_partitions:
            self.in_flight_partitions[partition_id].end_sample_id = (
                self.num_yielded - 1
            )

        self.partitions_to_replay.update(self.in_flight_partitions)

        self.in_flight_partitions = {}

        if bool(
            set(self.partitions_to_replay.keys())
            & set(self.fully_sent_partitions.keys())
        ):
            raise RuntimeError(
                "The same partitions are present in in flight\
                                partitions and fully sent partitions."
            )

        self.partitions_to_replay.update(self.fully_sent_partitions)
        self.fully_sent_partitions = {}

    def _should_be_replayed(self, sample_id) -> bool:
        """
        Checks whether given sample ID should be replayed given.
        """
        partition_id = sample_id // self.num_samples_in_partition

        if partition_id not in self.partitions_to_replay:
            return False

        start_id = self.partitions_to_replay[partition_id].starting_sample_id
        end_id = self.partitions_to_replay[partition_id].end_sample_id

        # handle in-flight case
        if end_id is None:
            valid_end = True
        else:
            valid_end = sample_id <= end_id

        valid_start = sample_id >= start_id

        return valid_start and valid_end

    def _update_partition_state(self, sample_id):
        """
        Updates the partition state before yielding DataSample.
        Marks partition as either in flight if it is new.
        Marks partition as as fully sent if sample_id is the
        last sample in the partition.
        """
        # create new partition if necessary
        curr_partition_idx = sample_id // self.num_samples_in_partition
        if sample_id % self.num_samples_in_partition == 0:
            self._create_new_in_flight_partition(curr_partition_idx, sample_id)

        # mark partition as completed if necessary
        if (
            sample_id % self.num_samples_in_partition
            == self.num_samples_in_partition - 1
        ):
            self._mark_partition_as_fully_sent(curr_partition_idx, sample_id)
            if not self.replay:
                self.current_partition_idx += 1

    def create_datasamples(self, it: Iterator[Any], size: int = 1):
        """
        Given an iterable over the source data, yields DataSamples
        wrapping each source sample.

        Args:
            it: Iterator over source data
            size: Number of samples generated per iteration. Default = 1


        Raises:
            StopIteration when source is exhausted
        """
        if self.sharded:
            curr_idx = self.world_size - self.rank - 1
        while True:
            try:
                wall_source_start = time.perf_counter_ns()
                x = next(it)
            except StopIteration:
                return

            if self.sharded:
                curr_idx += 1
                if curr_idx % self.world_size != 0:
                    continue

            if DISABLE_DATASAMPLES:
                yield x
            else:
                ds = DataSample(x)
                trace = self._should_trace()
                if trace:
                    start_time = time.process_time_ns()
                    ds.do_trace = True
                    # Use -1 to designate dummy source op
                    ds.trace_dict = {-1: start_time}
                    ds.wall_trace_dict = {-1: wall_source_start}
                    ds.wall_trace_resume_dict = {-1: wall_source_start}
                    ds.trace_order = [-1]
                    # Set size of source op and dummy op
                    ds.set_size(-1, size)
                    ds.set_size(self.p_id, size)
                    ds.buffer_size_dict = {}
                    if self.profile_mode:
                        ds.data_size_dict = {}
                        ds.trace_data_size = True

                yield ds

    def create_tf_datasamples(self, it: Iterator[Any], size: int = 1):
        """
        Given an iterable over the source data, yields DataSamples
        wrapping each source sample.

        Args:
            it: Iterator over source data
            size: Number of samples generated per iteration. Default = 1


        Raises:
            StopIteration when source is exhausted
        """
        tf = tensorflow()
        if self.sharded:
            curr_idx = self.world_size - self.rank - 1
        while True:
            try:
                wall_source_start = time.perf_counter_ns()
                x = it.get_next()
            except tf.errors.OutOfRangeError:
                return

            if self.sharded:
                curr_idx += 1
                if curr_idx % self.world_size != 0:
                    continue

            ds = DataSample(x)
            trace = self._should_trace()
            if trace:
                start_time = time.process_time_ns()
                ds.do_trace = True
                # Use -1 to designate dummy source op
                ds.trace_dict = {-1: start_time}
                ds.wall_trace_dict = {-1: wall_source_start}
                ds.wall_trace_resume_dict = {-1: wall_source_start}
                ds.trace_order = [-1]
                # Set size of source op and dummy op
                ds.set_size(-1, size)
                ds.set_size(self.p_id, size)
                ds.buffer_size_dict = {}
                if self.profile_mode:
                    ds.data_size_dict = {}
                    ds.trace_data_size = True

            yield ds

    def create_datasample(self, data: Any, creation_time: int, size: int = 1):
        """
        NOTE: Deprecated. Use create_datasamples instead.
        """
        ds = DataSample(data)
        if self._should_trace():
            ds.do_trace = True
            ds.set_size(self.p_id, size)
        return ds

    def _should_trace(self) -> bool:
        # Do not trace if pid is not assigned
        if self.p_id is None:
            return False
        try:
            curr_time = time.time()
            if curr_time - self._trace_time >= TRACE_FREQUENCY_SEC:
                self._trace_time = curr_time
                return True
            else:
                return False
        except AttributeError:
            # if it's our first time tracing
            self._trace_time = time.time()
            return True

    def enable_profiling(self) -> None:
        self.profile_mode = True

    def disable_profiling(self) -> None:
        self.profile_mode = False
