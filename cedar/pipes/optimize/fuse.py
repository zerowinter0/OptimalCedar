import logging
import ray
import time
from collections import deque
from typing import Optional, List, Any, Callable
from .registry import register_optimizer_pipe
from ..pipe import (
    Pipe,
    TFTensorDontCare,
)
from ..variant import (
    PipeVariant,
    InProcessPipeVariant,
    SMPPipeVariant,
    TFPipeVariant,
)
from ..ray_variant import RayPipeVariant
from ..context import (
    PipeVariantType,
    InProcessPipeVariantContext,
    SMPPipeVariantContext,
    RayPipeVariantContext,
    TFPipeVariantContext,
    TFRayPipeVariantContext,
)
from ..common import (
    MutationError,
    CedarPipeSpec,
    PipeExecutionResource,
    cedar_pipe,
)
from ..filter import is_dropped_sample
from cedar.service import SMPActor, RayActor

import tensorflow as tf

logger = logging.getLogger(__name__)


class Compose:
    def __init__(self, fns: List[Callable]):
        self.fns = fns

    def __call__(self, data: Any):
        for fn in self.fns:
            data = fn(data)
            if is_dropped_sample(data):
                break
        return data


@register_optimizer_pipe("FusedPipe")
@cedar_pipe(
    CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[
            PipeVariantType.INPROCESS,
            PipeVariantType.SMP,
            PipeVariantType.RAY,
            PipeVariantType.TF,
            PipeVariantType.TF_RAY,
        ],
    )
)
class FusedOptimizerPipe(Pipe):
    """
    This pipe represents a fused pipe, which executes the operations
    of multiple pipes within the same context (i.e., pipevariant),
    without needing to transfer data between pipe variants.

    Args:
        pipes (List[Pipe]): Ordered list of pipes to fuse together
    """

    def __init__(self, pipes: List[Pipe], is_random: bool = False):
        if len(pipes) < 2:
            raise MutationError("Cannot fuse fewer than 2 pipes.")
        fused_name = "|".join([p.get_logical_name() for p in pipes])
        fused_name = "FusedOptimizerPipe(" + fused_name + ")"
        logger.info(f"Created fused pipe {fused_name}")
        super().__init__(fused_name, pipes[0].input_pipes, is_random=is_random)

        if any(
            pipe.execution_resource == PipeExecutionResource.CUDA
            for pipe in pipes
        ):
            self.set_execution_resource(PipeExecutionResource.CUDA)

        self.pipes = pipes

        # Check if we're fusing TF pipes
        tf_pipes = [p.is_tf() for p in pipes]
        if not all(tf_pipes) and any(tf_pipes):
            raise RuntimeError("Cannot fuse non-TF and TF pipes")

        if all(tf_pipes):
            self._is_tf = True
            head_pipe = pipes[0]
            if head_pipe._fix_input_tf_spec:
                self._input_tf_spec = head_pipe._input_tf_spec
                self._fix_input_tf_spec = True

            # Merge output hints
            self._output_tf_hint = pipes[0]._output_tf_hint
            for p in pipes[1:]:
                hint = p._output_tf_hint
                if isinstance(hint, tuple):
                    for i, h in enumerate(hint):
                        if not isinstance(h.shape, TFTensorDontCare):
                            new_shape = []
                            for j, dim in enumerate(h.shape):
                                if isinstance(dim, TFTensorDontCare):
                                    try:
                                        input_dim = self._output_tf_hint[
                                            i
                                        ].shape[j]
                                    except IndexError:
                                        input_dim = None
                                    except TypeError:
                                        input_dim = None
                                    new_shape.append(input_dim)
                                else:
                                    new_shape.append(dim)
                            self._output_tf_hint[i].shape = new_shape
                        if not isinstance(h.dtype, TFTensorDontCare):
                            self._output_tf_hint.dtype = h.dtype
                else:
                    if not isinstance(hint.shape, TFTensorDontCare):
                        new_shape = []
                        for i, dim in enumerate(hint.shape):
                            if isinstance(dim, TFTensorDontCare):
                                try:
                                    input_dim = self._output_tf_hint.shape[i]
                                except IndexError:
                                    input_dim = None
                                except TypeError:
                                    input_dim = None
                                new_shape.append(input_dim)
                            else:
                                new_shape.append(dim)
                        self._output_tf_hint.shape = new_shape
                    if not isinstance(hint.dtype, TFTensorDontCare):
                        self._output_tf_hint.dtype = hint.dtype

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        try:
            fns = [p.get_fused_callable() for p in self.pipes]
            composed_fn = Compose(fns)
        except NotImplementedError:
            logger.error("Cannot fuse pipes, get_callable not implemented")
            raise MutationError
        variant = InProcessFusedOptimizerPipeVariant(
            self.input_pipes[0].pipe_variant,
            composed_fn,
        )
        return variant

    def _to_smp(self, variant_ctx: SMPPipeVariantContext) -> SMPPipeVariant:
        # Get the callable from each pipe
        try:
            fns = [p.get_fused_callable() for p in self.pipes]
            composed_fn = Compose(fns)

        except NotImplementedError:
            logger.error("Cannot fuse pipes, get_callable not implemented")
            raise MutationError

        variant = SMPFusedOptimizerPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            composed_fn,
            variant_ctx,
        )
        return variant

    def _to_ray(self, variant_ctx: RayPipeVariantContext) -> RayPipeVariant:
        try:
            fns = [p.get_fused_callable() for p in self.pipes]
            composed_fn = Compose(fns)
        except NotImplementedError:
            logger.error("Cannot fuse pipes, get_callable not implemented...")
            raise MutationError

        variant = RayFusedOptimizerPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            composed_fn,
            variant_ctx,
        )
        return variant

    def _to_tf(self, variant_ctx: TFPipeVariantContext) -> TFPipeVariant:
        for p in self.pipes:
            if not p.is_tf():
                raise RuntimeError("Cannot fuse non-tf pipe to tf.")

        fns = [p.get_fused_callable() for p in self.pipes]
        variant = TFFusedOptimizerPipeVariant(
            self.input_pipes[0].pipe_variant,
            fns,
            variant_ctx,
            self._input_tf_spec,
        )
        return variant

    def _to_tf_ray(
        self, variant_ctx: TFRayPipeVariantContext
    ) -> RayPipeVariant:
        for p in self.pipes:
            if not p.is_tf():
                raise RuntimeError("Cannot fuse non-tf pipe to tf.")

        fns = [p.get_fused_callable() for p in self.pipes]

        variant = TFRayFusedOptimizerPipeVariant(
            self.get_logical_uname(),
            self.input_pipes[0].pipe_variant,
            fns,
            variant_ctx,
            self._input_tf_spec,
        )
        return variant


class InProcessFusedOptimizerPipeVariant(InProcessPipeVariant):
    def __init__(
        self, input_pipe_variant: Optional[PipeVariant], fn: Callable
    ):
        super().__init__(input_pipe_variant)
        self.fn = fn

    def _iter_impl(self):
        while True:
            try:
                x = next(self._input_iter)
                if not x.dummy:
                    x.data = self.fn(x.data)
                    if is_dropped_sample(x.data):
                        continue
                yield x
            except StopIteration:
                return


class SMPActorFusedOptimizerPipeVariant(SMPActor):
    def __init__(self, name: str, fn: Callable) -> None:
        super().__init__(name)
        self.fn = fn

    def process(self, data: Any) -> None:
        return self.fn(data)


class SMPFusedOptimizerPipeVariant(SMPPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        composed_fn: Callable,
        variant_ctx: SMPPipeVariantContext,
    ) -> None:
        self.fn = composed_fn

        # call this last
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> SMPActor:
        return SMPActorFusedOptimizerPipeVariant(self.name, self.fn)

    def _get_next_result(self, timeout: float = 1.0):
        while True:
            sample = super()._get_next_result(timeout=timeout)
            if not is_dropped_sample(sample.data):
                return sample


@ray.remote(num_cpus=0)
class RayActorFusedOptimizerPipeVariant(RayActor):
    def __init__(self, name: str, fn: Callable) -> None:
        super().__init__(name)
        self.fn = fn

    def process(self, data: Any) -> None:
        return [self.fn(x) for x in data]


class RayFusedOptimizerPipeVariant(RayPipeVariant):
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
        return RayActorFusedOptimizerPipeVariant.options(
            num_gpus=self.variant_ctx.num_gpus
        ).remote(self.name, self.fn)

    def _get_next_result(self, timeout: float = 1.0):
        while True:
            sample = super()._get_next_result(timeout=timeout)
            if not is_dropped_sample(sample.data):
                return sample


class TFFusedOptimizerPipeVariant(TFPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        fns: List[Callable],
        variant_ctx: TFPipeVariantContext,
        tf_spec: tf.TensorSpec,
    ):
        super().__init__(input_pipe_variant)
        self.variant_ctx = variant_ctx
        self.dataset = tf.data.Dataset.from_generator(
            self._input_generator, output_signature=tf_spec
        )

        for fn in fns:
            self.dataset = self.dataset.map(fn, variant_ctx.num_parallel_calls)

        self.dataset_iter = iter(self.dataset)

        self._buffer = deque()

    def _input_generator(self):
        while True:
            try:
                x = next(self._input_iter)
                if not x.dummy:
                    data = x.data
                    x.data = None
                    self._buffer.append(x)
                    yield data
                else:
                    self._buffer.append(x)
            except StopIteration:
                return

    def _iter_impl(self):
        while True:
            try:
                data = self.dataset_iter.get_next()

                while True:
                    x = self._buffer.popleft()
                    if x.dummy:
                        continue
                    x.data = data
                    break
                yield x

            except tf.errors.OutOfRangeError:
                self.dataset_iter = iter(self.dataset)
                return


@ray.remote(num_cpus=0)
class TFRayActorFusedOptimizerPipeVariant(RayActor):
    def __init__(
        self,
        name: str,
        fns: List[Callable],
        tf_spec: tf.TensorSpec,
        num_parallel_calls: Optional[int],
    ) -> None:
        print("Started TF on RAY actor for {}".format(name))
        super().__init__(name)

        self.dataset = tf.data.Dataset.from_generator(
            self._input_generator, output_signature=tf_spec
        )
        for fn in fns:
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


class TFRayFusedOptimizerPipeVariant(RayPipeVariant):
    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        fns: List[Callable],
        variant_ctx: TFRayPipeVariantContext,
        tf_spec: tf.TensorSpec,
    ):
        self.fns = fns
        self.tf_spec = tf_spec
        super().__init__(name, input_pipe_variant, variant_ctx)

    def _create_actor(self) -> ray.actor.ActorClass:
        return TFRayActorFusedOptimizerPipeVariant.options(
            num_gpus=self.variant_ctx.num_gpus
        ).remote(
            self.name,
            self.fns,
            self.tf_spec,
            self.variant_ctx.num_parallel_calls,
        )
