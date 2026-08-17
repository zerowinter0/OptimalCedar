from __future__ import annotations

import abc
import concurrent.futures
import logging
import multiprocessing as mp
import threading
import queue
from collections import deque
from typing import Any, Optional, Iterator, Callable, Dict

from cedar.service import (
    MultiprocessService,
    MultiprocessTask,
    MultithreadedTask,
    SMPActor,
)
from .context import (
    PipeVariantContext,
    MultithreadedPipeVariantContext,
    SMPPipeVariantContext,
)
from .common import (
    DataSample,
    MutationError,
    CedarPipeSpec,
    capture_profile_input,
)

logger = logging.getLogger(__name__)

MAX_INFLIGHT_SCALING = 3


class PipeVariant(abc.ABC):
    """
    A PipeVariant represents a *physical* implementation of an abstract Pipe.

    At runtime, Pipes must be mutated to a PipeVariant, which coordinates the
    physical execution of the Pipe itself.
    """

    def __init__(self, input_pipe_variant: Optional[PipeVariant]) -> None:
        self.p_id: Optional[int] = None
        self.variant_ctx: Optional[PipeVariantContext] = None
        self.pipe_spec: Optional[CedarPipeSpec] = None
        self._dynamic_mutate_flag = False
        self.input_pipe_variant = input_pipe_variant

        self.mutation_event = None

        # To lock the pipe variant during mutations
        self.lock = threading.RLock()

    @abc.abstractmethod
    def _iter_impl(self) -> Iterator[DataSample]:
        """
        Abstract generator that yields items (datasamples)
        processed by the pipe. Must be defined by the child class.
        """
        pass

    def __iter__(self):
        logger.info(
            "Creating new PipeVariant Iterator for {}".format(self.p_id)
        )
        logger.info(self.variant_ctx.variant_type)
        self._create_input_iter()
        self._output_iter = self._iter_impl()
        if self.mutation_event is not None:
            logger.info("Setting mutation event")
            self.mutation_event.set()

        while True:
            try:
                with self.lock:
                    x = next(self._output_iter)

                # Get the buffer size if available
                try:
                    if isinstance(x, DataSample) and not x.dummy:
                        capture_profile_input(self.p_id, x.data)
                    buf_size = self.get_buffer_size() if x.do_trace else None
                    x.trace(self.p_id, buf_size)
                    yield x
                except AttributeError:
                    yield x
            except StopIteration:
                # Clear the input iterator in case of a new epoch
                # logger.info(f"Reached StopIteration for pipe {self.p_id}")
                self._input_iter = None
                return

    def get_buffer_size(self) -> Optional[int]:
        """
        For asynchronous pipes, returns the current buffer size.
        Pipe variants are expected to override this function.
        """
        return None

    def get_input_iter(self) -> Iterator[DataSample]:
        """
        Returns the current input iterator for this pipe variant.

        Raises:
            MutationError if self is not mutable
        """
        if self.pipe_spec is None:
            raise MutationError

        if not hasattr(self, "_input_iter"):
            logger.error(
                "Pipe {} does not have input iter. Creating one now..".format(
                    self.p_id
                )
            )
            self._create_input_iter()

        return self._input_iter

    def set_input_iter(
        self, it: Iterator[DataSample], dynamic_mutate: bool = False
    ):
        """
        Sets the input iterator, which generates data samples from the
        preceding pipe, to `it`

        Set dynamic_mutate to true during dynamic mutation to ensure that
        this input iter is not overriden

        Raises:
            MutationError if self is not mutable
        """
        if self.pipe_spec is None:
            logger.error(
                "Could not set input iter for pipe {}".format(self.p_id)
            )
            raise MutationError
        if dynamic_mutate:
            self._dynamic_mutate_flag = True
        self._input_iter = it

    def set_mutation_event(self, event: threading.Event):
        self.mutation_event = event

    def _create_input_iter(self):
        # if hasattr(self, "_input_iter") and self._input_iter is not None:
        #     logger.info("Already has iter")
        #     return
        if self._dynamic_mutate_flag:
            logger.info(
                "Skipping input iter reset for pipe {} "
                "during dynamic mutation".format(self.p_id)
            )
        elif not hasattr(self, "source"):
            logger.info(
                "Resetting input iter for {}, input {}".format(
                    self.p_id, self.input_pipe_variant.p_id
                )
            )
            # Only wrap for non-sources
            self._input_iter = iter(self.input_pipe_variant)
        self._dynamic_mutate_flag = False

    def set_scale(self, resource_count: int):
        """
        When called, scale the amount of resources provided to this
        pipe variant to `resource_count`

        Args:
            resource_count (int): Resource count to set the pipe variant to
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_scale(self) -> int:
        """
        Returns the current scale of this pipe variant
        """
        pass

    @abc.abstractmethod
    def is_scalable(self) -> bool:
        """
        Returns whether or not this pipe variant can be scaled
        """
        pass

    def serialize(self) -> Dict[str, Any]:
        """
        Returns a Dict serializing this variant's context
        """
        return self.variant_ctx.serialize()

    @abc.abstractmethod
    def shutdown(self):
        pass


class InProcessPipeVariant(PipeVariant):
    """
    A PipeVariant which exposes an iterator interface,
    allowing for local execution within the main Python
    process.
    """

    def __init__(self, input_pipe_variant: Optional[PipeVariant]) -> None:
        super().__init__(input_pipe_variant)

    def set_scale(self, resource_count: int):
        logger.warning(
            "Cannot scale InProcessVariant for {}".format(self.p_id)
        )

    def get_scale(self) -> int:
        # In-process doesn't count towards scale
        return 0

    def is_scalable(self) -> bool:
        return False

    def shutdown(self):
        self.input_pipe_variant = None
        self.variant_ctx = None


class _AsyncPipeVariant(PipeVariant):
    """
    A PipeVariant which represents a superclass for both
    Multiprocess, SMP, Multithreaded Services
    """

    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        max_inflight: int,
        max_prefetch: int = 10,
        use_threads: bool = True,
    ):
        super().__init__(input_pipe_variant)
        self.max_inflight = max_inflight
        self.max_prefetch = max_prefetch

        self.issued_tasks = 0
        self.completed_tasks = 0

        self.inflight_tasks = deque()
        self.buffer = deque()

        # Threading state
        self._shutdown_event = threading.Event()
        self._inflight_lock = threading.Lock()
        self._inflight_not_empty = threading.Condition(self._inflight_lock)
        self._inflight_not_full = threading.Condition(self._inflight_lock)
        self._producer_exhausted = threading.Event()
        self._consumer_exhausted = threading.Event()
        self._producer_thread = None
        self._consumer_thread = None
        self._buffer_lock = threading.Lock()
        self._buffer_not_full = threading.Condition(self._buffer_lock)
        self._buffer_not_empty = threading.Condition(self._buffer_lock)
        self._reset_iter_event = threading.Event()
        self._force_shutdown = threading.Event()
        self._use_threads = use_threads

    def _produce(self):
        logger.info("Starting producer thread")
        skip = False

        while not self._shutdown_event.is_set():
            # get the next item
            try:
                if not skip:
                    sample = next(self._input_iter)
                    if isinstance(sample, DataSample) and sample.dummy:
                        continue
            except StopIteration:
                logger.info("Finished producing items...")
                with self._inflight_not_empty:
                    self._producer_exhausted.set()
                    self._inflight_not_empty.notify()
                break

            with self._inflight_not_full:
                while not self._can_submit():
                    # logger.info("Waiting for inflight")
                    self._inflight_not_full.wait()
                try:
                    self._submit(sample)
                    # sucessfully submitted, ok to get the next sample
                    skip = False
                    self._inflight_not_empty.notify()
                except queue.Full:
                    logger.error("Queue full when attempting to submit...")
                    skip = True
        self._producer_exhausted.set()
        self._finalize()
        logger.info("Exiting producer thread")

    def _consume(self):
        logger.info("Starting consumer thread")
        while not self._producer_exhausted.is_set():
            with self._inflight_not_empty:
                while not self._inflight_tasks_remaining():
                    if self._producer_exhausted.is_set():
                        with self._buffer_not_empty:
                            # logger.info("Finished consuming items...")
                            self._consumer_exhausted.set()
                            self._buffer_not_empty.notify()
                        return
                    else:
                        self._inflight_not_empty.wait()

            # Inflight items exist, try to get result
            try:
                sample = self._get_next_result()
            except queue.Empty:
                continue

            with self._inflight_not_full:
                self._inflight_not_full.notify()

            with self._buffer_not_full:
                # If prefetch buffer full, wait for downstream
                # operators to consume the data
                while len(self.buffer) >= self.max_prefetch:
                    self._buffer_not_full.wait()

            self.buffer.append(sample)

            with self._buffer_not_empty:
                self._buffer_not_empty.notify()

        # In the event of early shutdown, finish consuming inflight samples
        while (
            self._inflight_tasks_remaining()
            and not self._reset_iter_event.is_set()
            and not self._force_shutdown.is_set()
        ):
            try:
                sample = self._get_next_result()
            except queue.Empty:
                continue
            with self._buffer_not_full:
                while len(self.buffer) >= self.max_prefetch:
                    self._buffer_not_full.wait()
            self.buffer.append(sample)
            with self._buffer_not_empty:
                self._buffer_not_empty.notify()

        self._consumer_exhausted.set()
        logger.info("Exiting consumer thread")
        # If no samples were ever generated, notify the main thread.
        with self._buffer_not_empty:
            self._buffer_not_empty.notify()

    def _iter_impl(self):
        if self._use_threads:
            self._reset_iter()
            self._producer_thread = threading.Thread(
                target=self._produce, daemon=True
            )
            self._producer_thread.start()
            self._consumer_thread = threading.Thread(
                target=self._consume, daemon=True
            )
            self._consumer_thread.start()

            while (
                not self._consumer_exhausted.is_set() or len(self.buffer) > 0
            ):
                with self._buffer_not_empty:
                    # If buffer is empty, wait for consumer to fill it
                    while len(self.buffer) == 0:
                        if self._consumer_exhausted.is_set():
                            yield from self._check_callback()
                            return
                        else:
                            # logger.info("Waiting for data...")
                            self._buffer_not_empty.wait()
                    data = self.buffer.popleft()
                    self._buffer_not_full.notify()
                yield data

            yield from self._check_callback()
            logger.info(self.variant_ctx.variant_type)
        else:
            logger.info("Not using threads for pipe {}".format(self.p_id))

            # Still processing when no more up
            while True:
                # Prioritize returning ready results
                while True:
                    try:
                        result = self._get_next_result(timeout=0)
                        yield result
                    except queue.Empty:
                        break

                # Can we submit a task? If not, stall while service processes
                while not self._can_submit():
                    try:
                        result = self._get_next_result(timeout=0)
                        yield result
                    except queue.Empty:
                        continue

                # Should be able to submit now
                try:
                    sample = next(self._input_iter)
                    if sample.dummy:
                        continue
                except StopIteration:
                    logger.info("Upstream pipe exhausted")
                    break
                self._submit(sample)

            # No more upstream data, but still tasks inflight
            while self._inflight_tasks_remaining():
                try:
                    result = self._get_next_result(timeout=5)
                    yield result
                except queue.Empty:
                    # If we time out, just exit
                    logger.warning("Timed-out waiting for final result...")
                    break
            # Handle callback
            yield from self._check_callback()

    def _reset_iter(self):
        self._reset_iter_event.set()
        if (
            self._producer_thread is not None
            and self._consumer_thread is not None
        ):
            self.buffer.clear()
            if self.inflight_tasks is not None:
                self.inflight_tasks.clear()
            self._shutdown_event.set()
            with self._inflight_not_full:
                self._inflight_not_full.notify()
            with self._inflight_not_empty:
                self._inflight_not_empty.notify()
            with self._buffer_not_full:
                self._buffer_not_full.notify()
            self._producer_thread.join()
            self._consumer_thread.join()
        self._reset_iter_event.clear()
        self._consumer_exhausted.clear()
        self._producer_exhausted.clear()
        self._shutdown_event.clear()
        self.buffer.clear()
        if self.inflight_tasks is not None:
            self.inflight_tasks.clear()
        self._producer_thread = None
        self._consumer_thread = None

        # If new iter created after callback registered, but before executed
        # Likely because StopIteration was not raised at end of prev epoch
        if hasattr(self, "_drain_callback"):
            raise RuntimeError("Resetting with non-empty callback.")

    def _check_callback(self):
        if hasattr(self, "_drain_callback"):
            logger.info(
                "Early termination... Calling "
                "drain callback with dummy sample"
            )
            self._drain_callback()
            del self._drain_callback
            # Create a dummy datasample
            ds = DataSample(None)
            ds.dummy = True
            yield ds

    @abc.abstractmethod
    def _submit(self, data: Any):
        """
        Submit data for processing.

        The caller is guaranteed to call _can_submit before calling this.
        """
        pass

    def _get_next_result(self, timeout: float = 1.0) -> DataSample:
        """
        Returns the next result.

        This call is allowed to wait for futures with a timeout.
        Raises queue.Empty if no data is fetched after timeout.
        """
        if len(self.inflight_tasks) == 0:
            raise queue.Empty

        # Wait for head of queue to be ready
        done, _ = concurrent.futures.wait(
            [self.inflight_tasks[0].data],
            timeout=timeout,
        )

        # Check if we timed out
        if len(done) == 0:
            # Just raise the queue empty exception since we catch it already
            raise queue.Empty

        sample = self.inflight_tasks.popleft()
        sample.data = sample.data.result()
        self.completed_tasks += 1
        return sample

    def _can_submit(self):
        res = (
            len(self.inflight_tasks) < self.max_inflight
            or self.max_inflight == -1
        )
        return res

    def _inflight_tasks_remaining(self):
        """
        Returns true if there are still inflight tasks remaing
        """
        return len(self.inflight_tasks) > 0

    def signal_shutdown(self):
        logger.info("Shutting down variant for pipe {}".format(self.p_id))
        self._shutdown_event.set()
        with self._inflight_not_full:
            self._inflight_not_full.notify()

    def shutdown(self) -> None:
        self._force_shutdown.set()
        self._shutdown_event.set()
        self.buffer.clear()
        if self.inflight_tasks is not None:
            self.inflight_tasks.clear()
        with self._inflight_not_full:
            self._inflight_not_full.notify()
        with self._inflight_not_empty:
            self._inflight_not_empty.notify()
        with self._buffer_not_full:
            self._buffer_not_full.notify()

        # The Ray consumer can be inside ray.get() for up to the poll timeout.
        # Do not destroy its actors or clear variant_ctx while that thread may
        # still dereference the service; doing so races with reset() and turns
        # intentional actor cleanup into RayActorError/AttributeError.
        threads = (
            ("consumer", self._consumer_thread),
            ("producer", self._producer_thread),
        )
        alive_threads = []
        for name, thread in threads:
            if thread is None:
                continue
            thread.join(timeout=5.0)
            if thread.is_alive():
                alive_threads.append(name)
            else:
                if name == "consumer":
                    self._consumer_thread = None
                else:
                    self._producer_thread = None

        if alive_threads:
            logger.warning(
                "Deferring shutdown for pipe %s; async thread(s) still alive: %s",
                self.p_id,
                ", ".join(alive_threads),
            )
            return

        if self.variant_ctx is not None:
            self.variant_ctx.shutdown()
        self.variant_ctx = None
        logger.info("Shut down variant for pipe {}".format(self.p_id))

    def __del__(self):
        if not self._force_shutdown.is_set():
            self.shutdown()

    def get_buffer_size(self) -> Optional[int]:
        return len(self.buffer)

    def register_drain_callback(self, callback: Callable):
        """
        Shutdown the consuming thread, wait until downstream pipes have
        finished draining the buffer, and call the callback
        """
        logger.info("Registering drain callback")
        self.signal_shutdown()
        self._drain_callback = callback

    def is_scalable(self) -> bool:
        return True

    def _finalize(self) -> None:
        """
        Called by the producer thread when there is no more upstream data
        """
        pass


class MultiprocessPipeVariant(_AsyncPipeVariant):
    """
    NOTE: DEPRECATED. Use SMP instead!

    A PipeVariant which executes tasks within a process pool on the local
    instance.

    To implement a specific pipe's multiprocess variant,
    inherit from this class and define the `_create_task` function

    Args:
        input: Input pipevariant
        service: MultiprocessService instance to use as the executor
        max_inflight: Maximum number of inflight tasks. Defaults to 100.
            -1 to set unlimited inflight
    """

    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        service: MultiprocessService,
        max_inflight: int = 10,
    ):
        logger.warning(
            "Using MultiprocessPipes is deprecated. Use SMP instead."
        )
        super().__init__(input_pipe_variant, max_inflight)
        self.service = service

    def _submit(self, sample: DataSample):
        task = self._create_task(sample.data)
        self.issued_tasks += 1
        future = self.service.submit(task)
        sample.data = future
        self.inflight_tasks.append(sample)

    @abc.abstractmethod
    def _create_task(self, input_data: Any) -> MultiprocessTask:
        pass

    def get_scale(self) -> int:
        raise NotImplementedError


class MultithreadedPipeVariant(_AsyncPipeVariant):
    """
    A PipeVariant which executes tasks within a thread pool on the local
    instance.

    To implement a specific pipe's multiprocess variant,
    inherit from this class and define the `_create_task` function.
    The _create_task function operates on the data contained in
    the input_pipe_variant (e.g., on integers if the input_pipe_variant
    is a pipe containing integers).


    Args:
        input: Input pipevariant
        service: MultiprocessService instance to use as the executor
        max_inflight: Maximum number of inflight tasks. Defaults to 100.
            -1 to set unlimited inflight
    """

    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: MultithreadedPipeVariantContext,
    ):
        max_inflight = max(
            variant_ctx.max_inflight,
            variant_ctx.service.n_threads * MAX_INFLIGHT_SCALING,
        )
        super().__init__(
            input_pipe_variant,
            max_inflight=max_inflight,
            max_prefetch=variant_ctx.max_prefetch,
            use_threads=variant_ctx.use_threads,
        )
        self.service = variant_ctx.service

    def _submit(self, sample: DataSample):
        task = self._create_task(sample.data)
        self.issued_tasks += 1
        future = self.service.submit(task)
        sample.data = future
        self.inflight_tasks.append(sample)

    @abc.abstractmethod
    def _create_task(self, input_data: Any) -> MultithreadedTask:
        pass

    def get_scale(self) -> int:
        return self.variant_ctx.n_threads

    def set_scale(self, resource_count: int) -> None:
        logger.info("Setting scale for {}".format(self.p_id))
        if resource_count <= 0:
            logger.warning(
                "Cannot scale resource to {}".format(resource_count)
            )
            return

        logger.info(
            "Scaling Pipe {} to {} threads".format(self.p_id, resource_count)
        )

        self.variant_ctx.service.resize(resource_count)
        self.variant_ctx.n_threads = resource_count

        self.max_inflight = max(
            self.max_inflight,
            self.variant_ctx.n_threads * MAX_INFLIGHT_SCALING,
        )


class SMPPipeVariant(_AsyncPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: SMPPipeVariantContext,
    ):
        variant_ctx.max_inflight = max(
            variant_ctx.max_inflight,
            variant_ctx.n_procs * MAX_INFLIGHT_SCALING,
        )
        super().__init__(
            input_pipe_variant,
            max_inflight=variant_ctx.max_inflight,
            max_prefetch=variant_ctx.max_prefetch,
            use_threads=variant_ctx.use_threads,
        )
        self.variant_ctx = variant_ctx
        self.inflight_tasks = None
        self.name = name
        # create and register
        req_q = mp.Queue()
        resp_q = mp.Queue()
        procs = []
        for i in range(variant_ctx.n_procs):
            actor = self._create_actor()
            actor.profile_backend_compute = getattr(
                variant_ctx, "profile_backend_compute", False
            )
            actor.register(req_q, resp_q)
            procs.append(actor)
        self.service = variant_ctx.service
        self.service.register(name, procs, req_q, resp_q)

    @abc.abstractmethod
    def _create_actor(self) -> SMPActor:
        """
        Creates a *stateful* actor, which continously processes requests
        """
        pass

    def _submit(self, data: DataSample):
        """
        Raises:
            queue.Full if the request could not be submitted
        """
        self.service.submit(data)
        self.issued_tasks += 1

    def _get_next_result(self, timeout: float = 1.0) -> DataSample:
        """
        Returns the next result. Blocking.
        Raises:
            queue.Empty if the no result could be retrieved
        """
        result = self.service.next(timeout=timeout)
        self.completed_tasks += 1

        return result

    def _can_submit(self):
        """
        Returns true if a task can be submitted.
        NOTE: This is not reliable for SMP
        """
        return self.service.can_submit() and (
            (self.issued_tasks - self.completed_tasks) < self.max_inflight
            or self.max_inflight == -1
        )

    def set_scale(self, resource_count: int) -> None:
        logger.info("Setting scale for {}".format(self.p_id))
        if resource_count <= 0:
            logger.warning(
                "Cannot scale resource to {}".format(resource_count)
            )
            return

        logger.info(
            "Scaling Pipe {} to {} resources".format(self.p_id, resource_count)
        )

        curr_cnt = self.variant_ctx.n_procs

        if resource_count > curr_cnt:
            # Scale up
            for _ in range(resource_count - curr_cnt):
                actor = self._create_actor()
                self.variant_ctx.service.register_and_start_actor(actor)
        elif resource_count < curr_cnt:
            # Scale down
            self.variant_ctx.service.deregister(curr_cnt - resource_count)

        self.variant_ctx.n_procs = resource_count

        self.variant_ctx.max_inflight = max(
            self.variant_ctx.max_inflight,
            self.variant_ctx.n_procs * MAX_INFLIGHT_SCALING,
        )
        self.max_inflight = self.variant_ctx.max_inflight

    def get_scale(self) -> int:
        return self.variant_ctx.n_procs

    def _inflight_tasks_remaining(self):
        """
        Returns true if there are still inflight tasks remaing
        """
        return self.completed_tasks < self.issued_tasks


class TFPipeVariant(PipeVariant):
    """
    A InProcess pipe variant that is specialized for tensorflow operations.
    Enables the use of TF graph mode.
    """

    def __init__(self, input_pipe_variant: Optional[PipeVariant]) -> None:
        super().__init__(input_pipe_variant)

    def set_scale(self, resource_count: int):
        logger.warning("Cannot scale TFPipeVariant for {}".format(self.p_id))

    def get_scale(self) -> int:
        # In-process doesn't count towards scale
        return 0

    def is_scalable(self) -> bool:
        return False

    def shutdown(self):
        self.input_pipe_variant = None
        self.variant_ctx = None


class RayDSPipeVariant(PipeVariant):
    """
    A pipe variant that is specialized for ray.
    Wraps ray dataset api.
    """

    def __init__(self, input_pipe_variant: Optional[PipeVariant]) -> None:
        super().__init__(input_pipe_variant)

    def set_scale(self, resource_count: int):
        logger.warning("Cannot scale TFPipeVariant for {}".format(self.p_id))

    def get_scale(self) -> int:
        # In-process doesn't count towards scale
        return 0

    def is_scalable(self) -> bool:
        return False

    def shutdown(self):
        self.input_pipe_variant = None
        self.variant_ctx = None
