import abc
import random
import logging
import ray
import queue
import threading
import sys
from collections import deque
from typing import Any, List

logger = logging.getLogger(__name__)


class RayActor:
    """
    A RayActor is a Ray Actor that executes processing via ray.remote

    Pipe Variants should subclass this and define their own RayActor.
    Make sure to wrap your class with `@ray.remote`
    """

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process and return data.
        """
        pass

    def exit(self):
        ray.actor.exit_actor()


class SampleBatch:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.submit_batch_samples = deque()
        self.future = None
        self.has_result = False

    def append(self, x: Any) -> bool:
        """
        Returns if the buffered samples equals batch size
        """
        self.submit_batch_samples.append(x)
        return len(self.submit_batch_samples) == self.batch_size

    def submit(self, actor: RayActor) -> None:
        """
        Submit the current buffer for processing.
        """
        batch = []
        for x in self.submit_batch_samples:
            batch.append(x.data)
            x.data = None

        self.future = actor.process.remote(batch)

    def next(self) -> Any:
        if not self.has_result:
            result = ray.get(self.future)
            self.has_result = True
            if len(result) != len(self.submit_batch_samples):
                raise RuntimeError("Retrieved fewer samples than submitted")

            for i, sample in enumerate(self.submit_batch_samples):
                sample.data = result[i]

        return self.submit_batch_samples.popleft()

    def exhausted(self) -> bool:
        return len(self.submit_batch_samples) == 0

    def __len__(self) -> int:
        return len(self.submit_batch_samples)


class RayService:
    """
    RayService manages a pool of RayActors for a given PipeVariant.
    """

    def __init__(self, submit_batch_size: int = 1):
        self.name = None
        if submit_batch_size < 1:
            raise ValueError("Submit batch size cannot be <1")
        self.submit_batch_size = submit_batch_size

        self._actors = None
        self._inflight_tasks = None
        self._submit_actor_idx = 0
        self._num_inflight_tasks = 0

        self._lock = threading.Lock()
        self._submitted = threading.Condition(self._lock)
        self._submit_batch = SampleBatch(self.submit_batch_size)
        self._receive_batch = None

        self._curr_scale = 0

        self._retired_actors = None
        self._retired_inflight_tasks = None

    def register(self, name: str, actors: List[RayActor]):
        """
        Register RayActors to this service
        """
        with self._lock:
            logger.info(
                f"Registering RayService for {name} with {len(actors)} actors."
            )
            self._actors = actors
            self._inflight_tasks = []
            self.name = name

            self._retired_actors = []
            self._retired_inflight_tasks = []

            for _ in range(len(self._actors)):
                self._inflight_tasks.append(deque())
            self._curr_scale = len(self._actors)

    def submit(self, sample: Any) -> None:
        """
        Submit a request to a RayActor.
        NOTE: Only a single thread should call submit.
        """
        with self._lock:
            self._num_inflight_tasks += 1

            # Check if we've filled up the batch
            if self._submit_batch.append(sample):
                self._submit_batch_to_actor()

    def next(self, timeout: float = 1.0) -> Any:
        """
        Returns a DataSample containing a processed result.

        NOTE: Only a single thread should call next.

        Raises:
            queue.Empty if timeout is exceeded
        """
        # Do we have an active sample batch?
        # Only consumer thread should call next, no need to lock receive
        # batch
        if (
            self._receive_batch is not None
            and not self._receive_batch.exhausted()
        ):
            with self._lock:
                self._num_inflight_tasks -= 1
            return self._receive_batch.next()

        # Otherwise, need to fetch a new batch
        self._receive_batch = None
        futures = []
        futures_map = {}  # map from future to queue itself

        # Need to lock accesses to _inflight_tasks
        with self._lock:
            for idx, q in enumerate(self._retired_inflight_tasks):
                if len(q) > 0:
                    f = q[0].future
                    futures.append(f)
                    futures_map[f] = q
                else:
                    logger.info("Retiring actor...")
                    self._retired_inflight_tasks.pop(idx)
                    actor_to_retire = self._retired_actors.pop(idx)
                    ray.kill(actor_to_retire)
                    # actor_to_retire.exit()
            for q in self._inflight_tasks:
                if len(q) > 0:
                    f = q[0].future
                    futures.append(f)
                    futures_map[f] = q

            # A partially assembled batch contributes to the inflight count,
            # but it has no Ray future yet. Wait for the producer to submit a
            # complete (or final) batch instead of spinning on ray.wait([]).
            if not futures:
                self._submitted.wait(timeout=timeout)
                raise queue.Empty

        # Get the first future that returns
        ready, _ = ray.wait(futures, num_returns=1, timeout=timeout)

        if ready:
            [future] = ready
        else:
            # logger.info(f"Ray svc {self.name} timed out waiting for data.")
            raise queue.Empty

        # Which queue did we return from?
        q = futures_map[future]
        self._receive_batch = q.popleft()

        with self._lock:
            self._num_inflight_tasks -= 1
        return self._receive_batch.next()

    def get_num_inflight_tasks(self) -> int:
        """
        Returns the number of currently inflight tasks
        """
        return self._num_inflight_tasks

    def shutdown(self) -> None:
        logger.info(f"Shutting down RayService for {self.name}")

        if "pytest" in sys.modules:
            # bypass issue with ray in pytest modules
            self._actors = None
        elif self._actors is not None:
            for actor in self._actors:
                try:
                    ray.kill(actor)
                except Exception as e:
                    logger.error(f"Failed to kill ray actor {e}")
            self._actors = None

    def finalize(self):
        """
        Called by the submitter thread to signal that there will be no
        further submissions.
        """
        with self._lock:
            if len(self._submit_batch) > 0:
                self._submit_batch_to_actor()

    def _submit_batch_to_actor(self):
        # Caller holds lock
        # Load balance to the smallest queue
        # idx, q = min(enumerate(self._inflight_tasks),
        #   key=lambda x: len(x[1]))

        # Load balance to random queue
        idx = random.randrange(0, len(self._actors))
        q = self._inflight_tasks[idx]
        actor = self._actors[idx]

        self._submit_batch.submit(actor)

        q.append(self._submit_batch)
        self._submit_batch = SampleBatch(self.submit_batch_size)
        self._submitted.notify_all()

    def get_num_actors(self) -> int:
        with self._lock:
            return self._curr_scale

    def register_and_start_actor(self, actor: RayActor):
        logger.info("Registering actor for {}".format(self.name))
        with self._lock:
            self._inflight_tasks.append(deque())
            self._actors.append(actor)
            self._curr_scale += 1

    def deregister(self, n_actors: int):
        """
        Deregisters n_actors
        """
        logger.info(
            "Deregistering {} actors for {}".format(n_actors, self.name)
        )
        if n_actors < 1:
            return
        with self._lock:
            if n_actors >= self._curr_scale:
                raise RuntimeError(
                    "Cannot deregister {} actors of {} alive".format(
                        n_actors, self._curr_scale
                    )
                )

            # Ok, pop off n_actors from queues and retire them
            for _ in range(n_actors):
                self._retired_actors.append(self._actors.pop())
                self._retired_inflight_tasks.append(self._inflight_tasks.pop())

            self._curr_scale -= n_actors
