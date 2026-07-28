import abc
import logging
from enum import Enum
from typing import Optional, Any, Dict
from cedar.service import (
    MultiprocessService,
    MultithreadedService,
    SMPService,
    RayService,
)

logger = logging.getLogger(__name__)


class PipeVariantType(Enum):
    INPROCESS = 1
    MULTIPROCESS = 2
    MULTITHREADED = 3
    SMP = 4
    RAY = 5
    TF = 6
    TF_RAY = 7
    RAY_DS = 8


class PipeVariantContext(abc.ABC):
    """
    A wrapper class for any parameterizable variables used by each pipe
    variant (e.g., knobs). Pipe variants are expected to subclass
    this base class and implement their unique params.
    """

    variant_type: Optional[PipeVariantType] = None

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serializes this pipe context
        """
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        pass

    def set_submit_batch_size(self, size: int) -> None:
        raise NotImplementedError


class InProcessPipeVariantContext(PipeVariantContext):
    def __init__(self):
        self.variant_type = PipeVariantType.INPROCESS

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        return d

    def shutdown() -> None:
        return


class MultiprocessPipeVariantContext(PipeVariantContext):
    def __init__(self, n_procs: Optional[int] = None):
        if n_procs is None:
            self.n_procs = 1
        else:
            self.n_procs = n_procs
        self.service = MultiprocessService(self.n_procs)
        self.variant_type = PipeVariantType.MULTIPROCESS

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        d["n_procs"] = self.n_procs
        return d

    def shutdown(self) -> None:
        self.service.shutdown()


class MultithreadedPipeVariantContext(PipeVariantContext):
    def __init__(
        self,
        n_threads: int = 1,
        max_inflight: int = 10,
        max_prefetch: int = 10,
        use_threads: bool = True,
    ):
        self.n_threads = n_threads
        self.service = MultithreadedService(self.n_threads)
        self.variant_type = PipeVariantType.MULTITHREADED
        self.max_inflight = max_inflight
        self.max_prefetch = max_prefetch
        self.use_threads = use_threads

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        d["n_threads"] = self.n_threads
        d["max_inflight"] = self.max_inflight
        d["max_prefetch"] = self.max_prefetch
        d["use_threads"] = self.use_threads
        return d

    def shutdown(self) -> None:
        self.service.shutdown()


class SMPPipeVariantContext(PipeVariantContext):
    def __init__(
        self,
        n_procs: int = 1,
        max_inflight: int = 10,
        max_prefetch: int = 10,
        use_threads: bool = True,
        disable_torch_parallelism: bool = True,
    ):
        self.n_procs = n_procs
        self.max_inflight = max_inflight
        self.max_prefetch = max_prefetch
        self.use_threads = use_threads
        self.disable_torch_parallelism = disable_torch_parallelism

        self.service = SMPService()
        self.variant_type = PipeVariantType.SMP

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        d["n_procs"] = self.n_procs
        d["max_inflight"] = self.max_inflight
        d["max_prefetch"] = self.max_prefetch
        d["use_threads"] = self.use_threads
        d["disable_torch_parallelism"] = self.disable_torch_parallelism
        return d

    def shutdown(self) -> None:
        self.service.shutdown()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("service", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.service = SMPService()


class RayPipeVariantContext(PipeVariantContext):
    def __init__(
        self,
        n_actors: int = 1,
        max_inflight: int = 10,
        max_prefetch: int = 10,
        use_threads: bool = True,
        submit_batch_size: int = 1,
    ):
        self.max_inflight = max_inflight
        self.max_prefetch = max_prefetch
        self.use_threads = use_threads
        self.n_actors = n_actors
        self.service = RayService(submit_batch_size=submit_batch_size)
        self.variant_type = PipeVariantType.RAY
        self.submit_batch_size = submit_batch_size

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        d["n_actors"] = self.n_actors
        d["max_inflight"] = self.max_inflight
        d["max_prefetch"] = self.max_prefetch
        d["use_threads"] = self.use_threads
        d["submit_batch_size"] = self.submit_batch_size
        return d

    def shutdown(self) -> None:
        self.service.shutdown()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("service", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.service = RayService(submit_batch_size=self.submit_batch_size)

    def set_submit_batch_size(self, size: int) -> None:
        self.submit_batch_size = size

        self.max_inflight = max(
            self.max_inflight,
            self.submit_batch_size * self.n_actors * 3,
        )

        self.max_prefetch = self.max_inflight


class TFPipeVariantContext(PipeVariantContext):
    def __init__(self, num_parallel_calls: Optional[int] = None):
        self.variant_type = PipeVariantType.TF
        self.num_parallel_calls = num_parallel_calls

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        d["num_parallel_calls"] = self.num_parallel_calls
        return d

    def shutdown() -> None:
        return


class RayDSPipeVariantContext(PipeVariantContext):
    def __init__(self):
        self.variant_type = PipeVariantType.RAY_DS

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        return d

    def shutdown() -> None:
        return


class TFRayPipeVariantContext(PipeVariantContext):
    def __init__(
        self,
        n_actors: int = 1,
        max_inflight: int = 10,
        max_prefetch: int = 10,
        use_threads: bool = True,
        submit_batch_size: int = 1,
        num_parallel_calls: Optional[int] = None,
    ):
        self.max_inflight = max_inflight
        self.max_prefetch = max_prefetch
        self.use_threads = use_threads
        self.n_actors = n_actors
        self.service = RayService(submit_batch_size=submit_batch_size)
        self.variant_type = PipeVariantType.TF_RAY
        self.submit_batch_size = submit_batch_size
        self.num_parallel_calls = num_parallel_calls

    def serialize(self) -> Dict[str, Any]:
        d = {}
        d["variant_type"] = self.variant_type.name
        d["n_actors"] = self.n_actors
        d["max_inflight"] = self.max_inflight
        d["max_prefetch"] = self.max_prefetch
        d["use_threads"] = self.use_threads
        d["submit_batch_size"] = self.submit_batch_size
        d["num_parallel_calls"] = self.num_parallel_calls
        return d

    def shutdown(self) -> None:
        self.service.shutdown()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("service", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.service = RayService(submit_batch_size=self.submit_batch_size)

    def set_submit_batch_size(self, size: int) -> None:
        self.submit_batch_size = size

        self.max_inflight = max(
            self.max_inflight,
            self.submit_batch_size * self.n_actors * 3,
        )

        self.max_prefetch = self.max_inflight


class PipeVariantContextFactory:
    """
    Factory class that creates a pipe variant context from a given YAML spec
    """

    @staticmethod
    def create_context(
        variant_type: PipeVariantType, spec: Optional[Dict[str, Any]] = None
    ) -> PipeVariantContext:
        if variant_type == PipeVariantType.INPROCESS:
            return InProcessPipeVariantContext()
        elif variant_type == PipeVariantType.MULTIPROCESS:
            if spec is None:
                return MultiprocessPipeVariantContext()
            else:
                n_procs = spec.get("n_procs", 1)
                return MultiprocessPipeVariantContext(n_procs=n_procs)
        elif variant_type == PipeVariantType.MULTITHREADED:
            if spec is None:
                return MultithreadedPipeVariantContext()
            else:
                n_threads = spec.get("n_threads", 1)
                max_inflight = spec.get("max_inflight", 10)
                max_prefetch = spec.get("max_prefetch", 10)
                use_threads = spec.get("use_threads", True)
                return MultithreadedPipeVariantContext(
                    n_threads=n_threads,
                    max_inflight=max_inflight,
                    max_prefetch=max_prefetch,
                    use_threads=use_threads,
                )
        elif variant_type == PipeVariantType.SMP:
            if spec is None:
                return SMPPipeVariantContext()
            else:
                n_procs = spec.get("n_procs", 1)
                max_inflight = spec.get("max_inflight", 10)
                max_prefetch = spec.get("max_prefetch", 10)
                use_threads = spec.get("use_threads", True)
                disable_torch_parallelism = spec.get(
                    "disable_torch_parallelism", True
                )
                return SMPPipeVariantContext(
                    n_procs=n_procs,
                    max_inflight=max_inflight,
                    max_prefetch=max_prefetch,
                    use_threads=use_threads,
                    disable_torch_parallelism=disable_torch_parallelism,
                )
        elif variant_type == PipeVariantType.RAY:
            if spec is None:
                return RayPipeVariantContext()
            else:
                n_actors = spec.get("n_actors", 1)
                max_inflight = spec.get("max_inflight", 20)
                max_prefetch = spec.get("max_prefetch", 20)
                use_threads = spec.get("use_threads", True)
                submit_batch_size = spec.get("submit_batch_size", 10)
                return RayPipeVariantContext(
                    n_actors=n_actors,
                    max_inflight=max_inflight,
                    max_prefetch=max_prefetch,
                    use_threads=use_threads,
                    submit_batch_size=submit_batch_size,
                )
        elif variant_type == PipeVariantType.TF:
            if spec is None:
                return TFPipeVariantContext()
            else:
                num_parallel_calls = spec.get("num_parallel_calls", None)
                return TFPipeVariantContext(
                    num_parallel_calls=num_parallel_calls
                )
        elif variant_type == PipeVariantType.TF_RAY:
            if spec is None:
                return TFRayPipeVariantContext()
            else:
                n_actors = spec.get("n_actors", 1)
                max_inflight = spec.get("max_inflight", 20)
                max_prefetch = spec.get("max_prefetch", 20)
                use_threads = spec.get("use_threads", True)
                submit_batch_size = spec.get("submit_batch_size", 10)
                num_parallel_calls = spec.get("num_parallel_calls", None)
                return TFRayPipeVariantContext(
                    n_actors=n_actors,
                    max_inflight=max_inflight,
                    max_prefetch=max_prefetch,
                    use_threads=use_threads,
                    submit_batch_size=submit_batch_size,
                    num_parallel_calls=num_parallel_calls,
                )
        elif variant_type == PipeVariantType.RAY_DS:
            return RayDSPipeVariantContext()
        else:
            raise ValueError("Invalid Pipe Variant Type")
