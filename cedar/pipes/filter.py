import logging
import ray
from typing import Any, Callable, Optional

from cedar.service import (
    MultiprocessService,
    MultiprocessTask,
    MultithreadedTask,
    SMPActor,
    RayActor,
)

from .pipe import Pipe
from .variant import (
    InProcessPipeVariant,
    MultithreadedPipeVariant,
    PipeVariant,
    MultiprocessPipeVariant,
    SMPPipeVariant,
)
from .ray_variant import RayPipeVariant
from .context import (
    InProcessPipeVariantContext,
    MultiprocessPipeVariantContext,
    MultithreadedPipeVariantContext,
    SMPPipeVariantContext,
    PipeVariantType,
    RayPipeVariantContext,
)
from .common import cedar_pipe, CedarPipeSpec, DataSample

logger = logging.getLogger(__name__)


def _get_callable_name(fn: Callable) -> str:
    """Return a stable callable name that is consistent across processes."""
    try:
        name = getattr(fn, "__qualname__", None) or getattr(
            fn, "__name__", None
        )
        if name:
            return name

        inner_fn = getattr(fn, "func", None)
        if inner_fn is not None:
            inner_name = getattr(inner_fn, "__qualname__", None) or getattr(
                inner_fn, "__name__", None
            )
            if inner_name:
                return inner_name

        # Callable objects are process-local instances; use class name
        # instead of repr(object), which embeds a memory address.
        cls = getattr(fn, "__class__", None)
        if cls is not None:
            cls_name = getattr(cls, "__qualname__", None) or getattr(
                cls, "__name__", None
            )
            if cls_name:
                return cls_name
    except Exception:
        pass
    return str(fn)


class DroppedSample:
    """Sentinel returned by async filter workers for filtered-out samples."""


def is_dropped_sample(data: Any) -> bool:
    return isinstance(data, DroppedSample)


class FusedFilterCallable:
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, data: Any) -> Any:
        return data if self.fn(data) else DroppedSample()


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[
            PipeVariantType.INPROCESS,
            PipeVariantType.MULTITHREADED,
            PipeVariantType.SMP,
            PipeVariantType.RAY,
        ],
        is_fusable=True,
    )
)
class FilterPipe(Pipe):
    """
    A filter pipe that drops samples when ``fn(sample)`` is falsy.
    """

    def __init__(
        self,
        input_pipe: Pipe,
        fn: Callable,
        tag: Optional[str] = None,
        is_random: bool = False,
    ):
        try:
            name = "FilterPipe_" + _get_callable_name(fn)
        except Exception:
            logger.warning("Unable to parse FilterPipe func name")
            name = "FilterPipe_" + str(fn)

        super().__init__(name, [input_pipe], tag=tag, is_random=is_random)
        self.fn = fn

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("FilterPipe only accepts one input.")
        return InProcessFilterPipeVariant(
            self.input_pipes[0].pipe_variant, self.fn
        )

    def _to_multiprocess(
        self, variant_ctx: MultiprocessPipeVariantContext
    ) -> MultiprocessPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("FilterPipe only accepts one input.")
        return MultiprocessFilterPipeVariant(
            self.input_pipes[0].pipe_variant, self.fn, variant_ctx.service
        )

    def _to_multithreaded(
        self, variant_ctx: MultithreadedPipeVariantContext
    ) -> MultithreadedPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("FilterPipe only accepts one input.")
        return MultithreadedFilterPipeVariant(
            self.input_pipes[0].pipe_variant, self.fn, variant_ctx
        )

    def _to_smp(self, variant_ctx: SMPPipeVariantContext) -> SMPPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("FilterPipe only accepts one input.")
        return SMPFilterPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            self.fn,
            variant_ctx,
        )

    def _to_ray(self, variant_ctx: RayPipeVariantContext) -> RayPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("FilterPipe only accepts one input.")
        return RayFilterPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            self.fn,
            variant_ctx,
        )

    def get_fused_callable(self) -> Callable:
        return FusedFilterCallable(self.fn)


class InProcessFilterPipeVariant(InProcessPipeVariant):
    def __init__(
        self, input_pipe_variant: Optional[PipeVariant], fn: Callable
    ):
        super().__init__(input_pipe_variant)
        self.fn = fn

    def _iter_impl(self):
        while True:
            try:
                x = next(self._input_iter)
                if isinstance(x, DataSample):
                    if x.dummy:
                        yield x
                        continue
                    if self.fn(x.data):
                        yield x
                else:
                    if self.fn(x):
                        yield x
            except StopIteration:
                return


class MultiprocessFilterTask(MultiprocessTask):
    def __init__(self, input_data: Any, fn: Callable):
        super().__init__(input_data)
        self.fn = fn

    def process(self) -> Any:
        return self.input_data if self.fn(self.input_data) else DroppedSample()


class MultiprocessFilterPipeVariant(MultiprocessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        service: MultiprocessService,
    ):
        super().__init__(input_pipe_variant, service)
        self.fn = fn

    def _create_task(self, input_data: Any) -> MultiprocessTask:
        return MultiprocessFilterTask(input_data, self.fn)

    def _get_next_result(self, timeout: float = 1.0) -> DataSample:
        while True:
            sample = super()._get_next_result(timeout=timeout)
            if not is_dropped_sample(sample.data):
                return sample


class MultithreadedFilterTask(MultithreadedTask):
    def __init__(self, input_data: Any, fn: Callable):
        super().__init__(input_data)
        self.fn = fn

    def process(self) -> Any:
        return self.input_data if self.fn(self.input_data) else DroppedSample()


class MultithreadedFilterPipeVariant(MultithreadedPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        variant_ctx: MultithreadedPipeVariantContext,
    ):
        super().__init__(input_pipe_variant, variant_ctx=variant_ctx)
        self.fn = fn

    def _create_task(self, input_data: Any) -> MultithreadedTask:
        return MultithreadedFilterTask(input_data, self.fn)

    def _get_next_result(self, timeout: float = 1.0) -> DataSample:
        while True:
            sample = super()._get_next_result(timeout=timeout)
            if not is_dropped_sample(sample.data):
                return sample


class SMPActorFilterPipeVariant(SMPActor):
    def __init__(
        self, name: str, fn: Callable, disable_torch_parallelism: bool = True
    ) -> None:
        super().__init__(
            name, disable_torch_parallelism=disable_torch_parallelism
        )
        self.fn = fn

    def process(self, data: Any) -> Any:
        return data if self.fn(data) else DroppedSample()


class SMPFilterPipeVariant(SMPPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        variant_ctx: SMPPipeVariantContext,
    ) -> None:
        self.fn = fn
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> SMPActor:
        return SMPActorFilterPipeVariant(
            self.name, self.fn, self.variant_ctx.disable_torch_parallelism
        )

    def _get_next_result(self, timeout: float = 1.0) -> DataSample:
        while True:
            sample = super()._get_next_result(timeout=timeout)
            if not is_dropped_sample(sample.data):
                return sample


@ray.remote(num_cpus=0)
class RayActorFilterPipeVariant(RayActor):
    def __init__(self, name: str, fn: Callable):
        super().__init__(name)
        self.fn = fn

    def process(self, data: Any) -> Any:
        return [x if self.fn(x) else DroppedSample() for x in data]


class RayFilterPipeVariant(RayPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        variant_ctx: RayPipeVariantContext,
    ):
        self.fn = fn
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> ray.actor.ActorClass:
        return RayActorFilterPipeVariant.options(
            num_gpus=self.variant_ctx.num_gpus
        ).remote(self.name, self.fn)

    def _get_next_result(self, timeout: float = 1.0) -> DataSample:
        while True:
            sample = super()._get_next_result(timeout=timeout)
            if not is_dropped_sample(sample.data):
                return sample
