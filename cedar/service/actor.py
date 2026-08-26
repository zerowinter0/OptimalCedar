import abc
import logging
import queue
import multiprocessing as mp
import torch
import time
from typing import Any

from cedar.utils.threading import limit_native_threadpools

logger = logging.getLogger(__name__)


class SMPActor(mp.Process):
    def __init__(self, name: str, disable_torch_parallelism: bool = True):
        super().__init__()
        self.req_q = None
        self.resp_q = None
        self.name = name
        self.shutdown_event = mp.Event()
        self.disable_torch_parallelism = disable_torch_parallelism
        self.profile_backend_compute = False

    def register(self, req_q: mp.Queue, resp_q: mp.Queue):
        logger.info(f"Registered SMPActor for {self.name}.")
        self.req_q = req_q
        self.resp_q = resp_q

    def run(self):
        # Need to set this to reduce contention in torch threads...
        if self.disable_torch_parallelism:
            native_threadpool_limiter = limit_native_threadpools(1)  # noqa: F841
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        logger.info(f"Running SMPActor for {self.name}.")
        if self.req_q is None or self.resp_q is None:
            logger.error("SMPActor not registered!")
            raise AssertionError("SMPActor not registered.")

        while not self.shutdown_event.is_set():
            try:
                sample = self.req_q.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if hasattr(sample, "data"):
                if self.profile_backend_compute:
                    started = time.perf_counter_ns()
                    sample.data = self.process(sample.data)
                    sample.backend_compute_ns = (
                        time.perf_counter_ns() - started
                    )
                else:
                    sample.data = self.process(sample.data)
            else:
                sample = self.process(sample)
            self.resp_q.put(sample, block=True)

    @abc.abstractmethod
    def process(self, data: Any) -> None:
        pass

    def stop(self) -> None:
        """
        Gracefully shuts down this process
        """
        logger.info(f"Stopping SMPActor for {self.name}.")
        self.shutdown_event.set()
