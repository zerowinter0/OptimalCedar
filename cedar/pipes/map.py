from __future__ import annotations

import logging
import ray
import time
from typing import TYPE_CHECKING, Any, Callable, Optional
from collections import deque

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
from .variant import (
    InProcessPipeVariant,
    MultithreadedPipeVariant,
    PipeVariant,
    MultiprocessPipeVariant,
    SMPPipeVariant,
    TFPipeVariant,
)
from .ray_variant import RayPipeVariant, get_ray_actor_options
from .context import (
    InProcessPipeVariantContext,
    MultiprocessPipeVariantContext,
    MultithreadedPipeVariantContext,
    SMPPipeVariantContext,
    PipeVariantType,
    RayPipeVariantContext,
    TFPipeVariantContext,
    TFRayPipeVariantContext,
)
from .common import cedar_pipe, CedarPipeSpec, DataSample
from .tf import TFOutputHint

from cedar.utils.frameworks import tensorflow

if TYPE_CHECKING:
    import tensorflow as tf

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


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[
            PipeVariantType.INPROCESS,
            PipeVariantType.MULTITHREADED,
            PipeVariantType.SMP,
            PipeVariantType.RAY,
            PipeVariantType.TF,
            PipeVariantType.TF_RAY,
        ],
        is_fusable=True,
    )
)
class MapperPipe(Pipe):
    """
    A Map pipe that applies a function to each item from the preceding Pipe.

    Args:
        input: the preceding Pipe
        fn: An arbitrary callable function to be applied to each item
        tf_spec: If provided, designate the mapped function as a tensorflow
            function. tf_spec should specify the input tensorflow TensorSpec
    """

    def __init__(
        self,
        input_pipe: Pipe,
        fn: Callable,
        tag: Optional[str] = None,
        input_tf_spec: Optional[tf.TensorSpec] = None,
        output_tf_hint: Optional[TFOutputHint] = None,
        is_random: bool = False,
    ):
        try:
            name = "MapperPipe_" + _get_callable_name(fn)
        except Exception:
            logger.warning("Unable to parse MapPipe func name")
            name = "MapperPipe_" + str(fn)
        super().__init__(
            name,
            [input_pipe],
            tag=tag,
            input_tf_spec=input_tf_spec,
            output_tf_hint=output_tf_hint,
            is_random=is_random,
        )

        self.fn = fn

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("MapPipe only accepts one input.")

        variant = InProcessMapperPipeVariant(
            self.input_pipes[0].pipe_variant, self.fn
        )
        return variant

    def _to_multiprocess(
        self, variant_ctx: MultiprocessPipeVariantContext
    ) -> MultiprocessPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("Map Pipe only accepts one input.")

        variant = MultiprocessMapperPipeVariant(
            self.input_pipes[0].pipe_variant, self.fn, variant_ctx.service
        )
        return variant

    def _to_multithreaded(
        self, variant_ctx: MultithreadedPipeVariantContext
    ) -> MultithreadedPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("Map Pipe only accepts one input.")

        variant = MultithreadedMapperPipeVariant(
            self.input_pipes[0].pipe_variant, self.fn, variant_ctx
        )
        return variant

    def _to_smp(self, variant_ctx: SMPPipeVariantContext) -> SMPPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("Map Pipe only accepts one input.")

        variant = SMPMapperPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            self.fn,
            variant_ctx,
        )
        return variant

    def _to_ray(self, variant_ctx: RayPipeVariantContext) -> RayPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("Map Pipe only accepts one input.")

        variant = RayMapperPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            self.fn,
            variant_ctx,
        )
        return variant

    def _to_tf(self, variant_ctx: TFPipeVariantContext) -> TFPipeVariant:
        logger.info(f"Mutating pipe {self.id} to TENSORFLOW.")

        if len(self.input_pipes) != 1:
            raise RuntimeError("MapPipe only accepts one input.")

        variant = TFMapperPipeVariant(
            self.input_pipes[0].pipe_variant,
            self.fn,
            self._input_tf_spec,
            variant_ctx,
        )
        return variant

    def _to_tf_ray(
        self, variant_ctx: TFRayPipeVariantContext
    ) -> RayPipeVariant:
        logger.info(f"Mutating pipe {self.id} to TENSORFLOW on RAY.")

        if len(self.input_pipes) != 1:
            raise RuntimeError("Map Pipe only accepts one input.")

        variant = TFRayMapperPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            self.fn,
            variant_ctx,
            self._input_tf_spec,
        )
        return variant

    def get_fused_callable(self) -> Callable:
        return self.fn


class InProcessMapperPipeVariant(InProcessPipeVariant):
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
                    if not x.dummy:
                        if isinstance(x.data, tuple):
                            x.data = self.fn(*x.data)
                        else:
                            x.data = self.fn(x.data)
                    yield x
                else:
                    yield self.fn(x)
            except StopIteration:
                return


class MultiprocessMapperTask(MultiprocessTask):
    def __init__(self, input_data: Any, fn: Callable):
        super().__init__(input_data)
        self.fn = fn

    def process(self) -> Any:
        return self.fn(self.input_data)


class MultiprocessMapperPipeVariant(MultiprocessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        service: MultiprocessService,
    ):
        super().__init__(input_pipe_variant, service)
        self.fn = fn

    def _create_task(self, input_data: Any) -> MultiprocessTask:
        return MultiprocessMapperTask(input_data, self.fn)


class MultithreadedMapperTask(MultithreadedTask):
    def __init__(self, input_data: Any, fn: Callable):
        super().__init__(input_data)
        self.fn = fn

    def process(self) -> Any:
        return self.fn(self.input_data)


class MultithreadedMapperPipeVariant(MultithreadedPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        variant_ctx: MultithreadedPipeVariantContext,
    ):
        super().__init__(input_pipe_variant, variant_ctx=variant_ctx)
        self.fn = fn

    def _create_task(self, input_data: Any) -> MultithreadedTask:
        return MultithreadedMapperTask(input_data, self.fn)


class SMPActorMapperPipeVariant(SMPActor):
    def __init__(
        self, name: str, fn: Callable, disable_torch_parallelism: bool = True
    ) -> None:
        super().__init__(
            name, disable_torch_parallelism=disable_torch_parallelism
        )
        self.fn = fn

    def process(self, data: Any) -> None:
        # if data is a tuple, pass in
        if isinstance(data, tuple):
            return self.fn(*data)
        else:
            return self.fn(data)


class SMPMapperPipeVariant(SMPPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        variant_ctx: SMPPipeVariantContext,
    ) -> None:
        self.fn = fn

        # call super constructor last
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> SMPActor:
        return SMPActorMapperPipeVariant(
            self.name, self.fn, self.variant_ctx.disable_torch_parallelism
        )


@ray.remote(num_cpus=0)  # Let the controller manage parallelism
class RayActorMapperPipeVariant(RayActor):
    def __init__(self, name: str, fn: Callable):
        super().__init__(name)
        self.fn = fn

    def process(self, data: Any) -> None:
        return [self.fn(x) for x in data]


class RayMapperPipeVariant(RayPipeVariant):
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
        return RayActorMapperPipeVariant.options(
            **get_ray_actor_options(self.variant_ctx.num_gpus)
        ).remote(self.name, self.fn)


class TFMapperPipeVariant(TFPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        tf_spec: tf.TensorSpec,
        variant_ctx: TFPipeVariantContext,
    ):
        tf = tensorflow()
        super().__init__(input_pipe_variant)
        self.fn = fn

        self.dataset = tf.data.Dataset.from_generator(
            self._input_generator,
            output_signature=tf_spec,
        )
        self.dataset = self.dataset.map(
            fn, num_parallel_calls=variant_ctx.num_parallel_calls
        )
        self.dataset_iter = iter(self.dataset)

        self._buffer = deque()

    def _input_generator(self):
        while True:
            try:
                x = next(self._input_iter)
                if isinstance(x, DataSample):
                    if not x.dummy:
                        data = x.data
                        x.data = None
                        self._buffer.append(x)
                        yield data
                    else:
                        self._buffer.append(x)
                else:
                    raise NotImplementedError

            except StopIteration:
                return

    def _iter_impl(self):
        tf = tensorflow()
        while True:
            try:
                data = self.dataset_iter.get_next()

                while True:
                    x = self._buffer.popleft()
                    if isinstance(x, DataSample):
                        if x.dummy:
                            continue
                        x.data = data
                        break
                    else:
                        raise NotImplementedError

                yield x

            except tf.errors.OutOfRangeError:
                self.dataset_iter = iter(self.dataset)
                return


@ray.remote(num_cpus=0)  # Let the controller manage parallelism
class TFRayActorMapperPipeVariant(RayActor):
    def __init__(
        self,
        name: str,
        fn: Callable,
        tf_spec: tf.TensorSpec,
        num_parallel_calls: Optional[int],
    ):
        tf = tensorflow()
        print("Started TF on RAY actor for {}".format(name))
        super().__init__(name)
        self.fn = fn

        self.dataset = tf.data.Dataset.from_generator(
            self._input_generator, output_signature=tf_spec
        )
        self.dataset = self.dataset.map(fn, num_parallel_calls)
        self.dataset_iter = iter(self.dataset)

        self._buffer = deque()

    def _input_generator(self):
        while True:
            try:
                data = self._buffer.popleft()
                for x in data:
                    yield x
            except IndexError:
                # Wait for new data
                time.sleep(0.001)

    def process(self, data: Any) -> None:
        # push all data to the deque
        n_elems = len(data)
        out = []

        self._buffer.append(data)

        while n_elems > 0:
            out.append(self.dataset_iter.get_next())
            n_elems -= 1

        return out


class TFRayMapperPipeVariant(RayPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        fn: Callable,
        variant_ctx: TFRayPipeVariantContext,
        input_tf_spec: tf.TensorSpec,
    ):
        self.fn = fn
        self.tf_spec = input_tf_spec
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> ray.actor.ActorClass:
        return TFRayActorMapperPipeVariant.options(
            **get_ray_actor_options(self.variant_ctx.num_gpus)
        ).remote(
            self.name,
            self.fn,
            self.tf_spec,
            self.variant_ctx.num_parallel_calls,
        )
