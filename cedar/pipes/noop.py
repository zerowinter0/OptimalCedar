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

from .pipe import (
    Pipe,
)
from .context import (
    InProcessPipeVariantContext,
    MultithreadedPipeVariantContext,
    MultiprocessPipeVariantContext,
    SMPPipeVariantContext,
    RayPipeVariantContext,
    PipeVariantType,
)
from .variant import (
    InProcessPipeVariant,
    MultithreadedPipeVariant,
    PipeVariant,
    MultiprocessPipeVariant,
    SMPPipeVariant,
)
from .ray_variant import RayPipeVariant, get_ray_actor_options
from .common import cedar_pipe, CedarPipeSpec


logger = logging.getLogger(__name__)


def _noop_callable(data: Any) -> Any:
    return data


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[
            PipeVariantType.INPROCESS,
            PipeVariantType.MULTITHREADED,
            PipeVariantType.SMP,
            PipeVariantType.RAY,
        ],
    )
)
class NoopPipe(Pipe):
    """
    A noop pipe, that effectively just forwards the output of the input pipe.

    Primarily intenteded for testing.
    """

    def __init__(
        self,
        input_pipe: Pipe,
        tag: Optional[str] = None,
        is_random: bool = False,
    ):
        super().__init__(
            "NoopPipe", [input_pipe], tag=tag, is_random=is_random
        )

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        variant = InProcessNoopPipeVariant(self.input_pipes[0].get_variant())
        return variant

    def _to_multiprocess(
        self, variant_ctx: MultiprocessPipeVariantContext
    ) -> MultiprocessPipeVariant:
        variant = MultiprocessNoopPipeVariant(
            self.input_pipes[0].get_variant(), variant_ctx.service
        )
        return variant

    def _to_multithreaded(
        self,
        variant_ctx: MultithreadedPipeVariantContext,
    ) -> MultithreadedPipeVariant:
        return MultithreadedNoopPipeVariant(
            self.input_pipes[0].get_variant(), variant_ctx
        )

    def _to_smp(self, variant_ctx: SMPPipeVariantContext) -> SMPPipeVariant:
        return SMPNoopPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].get_variant(),
            variant_ctx,
        )

    def _to_ray(self, variant_ctx: RayPipeVariantContext) -> RayPipeVariant:
        return RayNoopPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].get_variant(),
            variant_ctx,
        )

    def _check_mutation(self) -> None:
        super()._check_mutation()

        if len(self.input_pipes) != 1:
            raise RuntimeError("NoopPipe only accepts one input.")

    def get_fused_callable(self) -> Callable:
        return _noop_callable


class InProcessNoopPipeVariant(InProcessPipeVariant):
    def __init__(self, input_pipe_variant: Optional[PipeVariant]):
        super().__init__(input_pipe_variant)

    def _iter_impl(self):
        while True:
            try:
                yield next(self._input_iter)
            except StopIteration:
                return


class MultiprocessNoopTask(MultiprocessTask):
    def __init__(self, input_data: Any):
        super().__init__(input_data)

    def process(self) -> Any:
        return self.input_data


class MultiprocessNoopPipeVariant(MultiprocessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        service: MultiprocessService,
    ):
        super().__init__(input_pipe_variant, service)

    def _create_task(self, input_data: Any) -> MultiprocessTask:
        return MultiprocessNoopTask(input_data)


class MultithreadedNoopTask(MultithreadedTask):
    def __init__(self, input_data: Any):
        super().__init__(input_data)

    def process(self) -> Any:
        return self.input_data


class MultithreadedNoopPipeVariant(MultithreadedPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: MultithreadedPipeVariantContext,
    ):
        super().__init__(input_pipe_variant, variant_ctx)

    def _create_task(self, input_data: Any) -> MultithreadedTask:
        return MultithreadedNoopTask(input_data)


class SMPActorNoopPipeVariant(SMPActor):
    def __init__(self, name, disable_torch_parallelism: bool = True) -> None:
        super().__init__(
            name, disable_torch_parallelism=disable_torch_parallelism
        )

    def process(self, data: Any) -> None:
        return data


class SMPNoopPipeVariant(SMPPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: SMPPipeVariantContext,
    ) -> None:
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> SMPActor:
        return SMPActorNoopPipeVariant(
            self.name, self.variant_ctx.disable_torch_parallelism
        )


@ray.remote(num_cpus=1)
class RayActorNoopPipeVariant(RayActor):
    def __init__(self, name: str):
        super().__init__(name)
        # for testing purposes
        self._n_proc = 0

    def process(self, data: Any) -> None:
        self._n_proc += 1
        return data

    def _get_n_proc(self) -> int:
        # for testing
        return self._n_proc


class RayNoopPipeVariant(RayPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: RayPipeVariantContext,
    ):
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> ray.actor.ActorClass:
        return RayActorNoopPipeVariant.options(
            **get_ray_actor_options()
        ).remote(self.name)
