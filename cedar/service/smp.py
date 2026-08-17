import multiprocessing as mp
import logging
import math

from typing import Any, List

from .actor import SMPActor

logger = logging.getLogger(__name__)


class SMPRequest:
    def __init__(self, data: Any):
        self.data = data


class SMPResponse:
    def __init__(self, data: Any):
        self.data = data


class SMPService:
    """
    A multiprocess service that manages a pool of
    stateful multiprocess workers.
    """

    def __init__(self):
        self.req_q = None
        self.resp_q = None
        self.procs = None
        self._backend_compute_count = 0
        self._backend_compute_sum_ns = 0.0
        self._backend_compute_sum_sq_ns = 0.0

    def register(
        self,
        name: str,
        procs: List[SMPActor],
        req_q: mp.Queue,
        resp_q: mp.Queue,
    ):
        logger.info(f"Registering SMPService for {name}")

        self.procs = procs
        self.req_q = req_q
        self.resp_q = resp_q

        logger.info(f"Starting {str(len(procs))} SMP actors for {name}.")

        for proc in self.procs:
            proc.daemon = True
            proc.start()

    def submit(self, req) -> None:
        """
        Submits a request to the appropriate process(es).

        Raises:
            queue.Full if the queue is full.
        Args:
            req: SMPRequest to submit
        """
        self.req_q.put(req, block=False)

    def next(self, timeout: float = 1.0) -> Any:
        """
        Returns a DataSample containing the next element.

        Raises:
            queue.Empty if the queue is empty.
        Args:
            name: Name of the queue
        """
        data = self.resp_q.get(block=True, timeout=timeout)
        elapsed_ns = getattr(data, "backend_compute_ns", None)
        if elapsed_ns is not None:
            value = float(elapsed_ns)
            self._backend_compute_count += 1
            self._backend_compute_sum_ns += value
            self._backend_compute_sum_sq_ns += value * value
            data.backend_compute_ns = None
        return data

    def get_backend_compute_stats(self):
        count = self._backend_compute_count
        if count == 0:
            return None
        mean = self._backend_compute_sum_ns / count
        variance = 0.0
        if count > 1:
            variance = max(
                0.0,
                (
                    self._backend_compute_sum_sq_ns
                    - count * mean * mean
                )
                / (count - 1),
            )
        stddev = math.sqrt(variance)
        return {
            "method": "worker_wall_clock",
            "count": count,
            "mean_ms_per_sample": mean / 1e6,
            "stddev_ms_per_sample": stddev / 1e6,
            "stderr_ms_per_sample": stddev / math.sqrt(count) / 1e6,
        }

    def reset_backend_compute_stats(self) -> None:
        self._backend_compute_count = 0
        self._backend_compute_sum_ns = 0.0
        self._backend_compute_sum_sq_ns = 0.0

    def can_submit(self):
        """
        Returns true if the service can take more tasks.

        NOTE: This is not reliable

        Args:
            name: Name of the queue
        """
        return not self.req_q.full()

    def result_ready(self):
        """
        Returns true if the next result for the appropriate queue is ready
        NOTE: This is not reliable

        Args:
            name: Name of the queue
        """
        return not self.resp_q.empty()

    def shutdown(self):
        logger.info("Shutting down SMP Procs")
        if self.procs is not None:
            for proc in self.procs:
                if getattr(proc, "_popen", None) is not None:
                    proc.terminate()
            self.procs = []
        if self.req_q is not None:
            self.req_q.close()
            self.req_q.cancel_join_thread()
            self.req_q.join_thread()
            self.req_q = None
        if self.resp_q is not None:
            self.resp_q.close()
            self.resp_q.cancel_join_thread()
            self.resp_q.join_thread()
            self.resp_q = None

    def deregister(self, n_procs: int):
        """
        Gracefully shuts down n_procs

        Args:
            n_procs (int): Number of processes to deregister

        Raises:
            RuntimeError if n_procs not available to deregister
        """
        if n_procs > len(self.procs):
            raise RuntimeError(
                "Tried to deregister more processes than are running"
            )

        for _ in range(n_procs):
            proc = self.procs.pop()
            proc.stop()

    def register_and_start_actor(self, actor: SMPActor):
        """
        Register a single actor, used for scaling up the service
        Args:
            name: Name of the service to register the actor to
            actor: Actor to register
        """
        actor.register(self.req_q, self.resp_q)
        actor.daemon = True
        actor.start()

        self.procs.append(actor)
