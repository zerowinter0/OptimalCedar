from __future__ import annotations

import abc
import logging
import threading
import functools
from typing import List, Optional, Callable, Tuple

from cedar.config import CedarContext
from .context import (
    PipeVariantType,
    PipeVariantContext,
    PipeVariantContextFactory,
    InProcessPipeVariantContext,
    MultiprocessPipeVariantContext,
    MultithreadedPipeVariantContext,
    SMPPipeVariantContext,
    RayPipeVariantContext,
    RayDSPipeVariantContext,
    TFPipeVariantContext,
    TFRayPipeVariantContext,
)
from .common import (
    MutationError,
    CedarPipeSpec,
    PipeComputeScaling,
    PipeExecutionResource,
)
from .variant import (
    PipeVariant,
    InProcessPipeVariant,
    MultiprocessPipeVariant,
    MultithreadedPipeVariant,
    SMPPipeVariant,
    TFPipeVariant,
    RayDSPipeVariant,
)
from .ray_variant import RayPipeVariant
from .tf import TFTensorDontCare, TFOutputHint

import tensorflow as tf

logger = logging.getLogger(__name__)


class Pipe(abc.ABC):
    """
    A Pipe represents a logical, composable operation that processes
    data from the preceding pipe.

    A Pipe operates on either a preceding Pipe, or a Source.
    Pipes can then be composed with each other to create a DAG
    (referred to as a ``Feature``) that represents dataflow
    from raw data to preprocessed tensors. Pipes lazily preprocess and
    materialize data, transparent to the application.

    To implement a specific Pipe, subclass this and overwrite.

    Args:
        name (string): Name of this pipe.
        input_pipes (List[Pipe]): A list of Pipes which represent inputs
        to this Pipe.
        tag (Optional[str]): If provided, a tag used to denote the pipe within
            within the reordering API.
    """

    pipe_variant: Optional[PipeVariant]

    def __init__(
        self,
        name: str,
        input_pipes: List[Pipe],
        tag: Optional[str] = None,
        input_tf_spec: Optional[tf.TensorSpec] = None,
        output_tf_hint: Optional[TFOutputHint] = None,
        is_random: Optional[bool] = False,
    ):
        self.name = name
        self.input_pipes = input_pipes
        self.pipe_variant = None
        self.pipe_variant_type = None
        self.id = None  # for use by feature
        self.pipe_spec: Optional[CedarPipeSpec] = None
        self.tag = tag
        # Optimizer-facing cost semantics. An unannotated operator defaults to
        # data-scaled compute, but remains distinguishable from an explicit
        # annotation so the optional profiler can infer its scaling mode.
        self.compute_scaling = PipeComputeScaling.PER_DATA
        self.compute_scaling_explicit = False
        self.execution_resource = PipeExecutionResource.CPU

        self.ok_to_mutate = threading.Event()
        self.ok_to_mutate.set()

        # Reordering API
        self._fix_order = False
        self._depends_on_tags = None

        self.rank_spec = None

        # Caching API
        self._is_random = is_random

        # TF-specific API
        self._is_tf = False
        self._input_tf_spec = None
        self._fix_input_tf_spec = False
        self._output_tf_spec = None
        self._output_tf_hint = None
        self._is_tf_py_func = False

        if input_tf_spec and not output_tf_hint:
            raise ValueError("Must provide output_tf_spec for TF funcs")

        if output_tf_hint:
            self._is_tf = True
            self._output_tf_hint = output_tf_hint
            # If an input spec is provided, always use it
            if input_tf_spec:
                self._fix_input_tf_spec = True
                self._input_tf_spec = input_tf_spec

        self._pipes_to_fuse = None

    def set_compute_scaling(
        self, scaling: PipeComputeScaling
    ) -> "Pipe":
        self.compute_scaling = PipeComputeScaling(scaling)
        self.compute_scaling_explicit = True
        return self

    def set_execution_resource(
        self, resource: PipeExecutionResource
    ) -> "Pipe":
        self.execution_resource = PipeExecutionResource(resource)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        event = state.pop("ok_to_mutate", None)
        state["_ok_to_mutate_is_set"] = event is None or event.is_set()
        return state

    def __setstate__(self, state):
        is_set = state.pop("_ok_to_mutate_is_set", True)
        self.__dict__.update(state)
        self.ok_to_mutate = threading.Event()
        if is_set:
            self.ok_to_mutate.set()

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        """
        Materializes this Pipe into a InProcessPipeVariant
        """
        raise NotImplementedError

    def _to_multiprocess(
        self, variant_ctx: MultiprocessPipeVariantContext
    ) -> MultiprocessPipeVariant:
        """
        Materialize this Pipe into a MultiProcessPipeVariant
        """
        raise NotImplementedError

    def _to_multithreaded(
        self, variant_ctx: MultithreadedPipeVariantContext
    ) -> MultithreadedPipeVariant:
        """
        Materialize this pipe to a multithreaded pipe variant
        """
        raise NotImplementedError

    def _to_smp(self, variant_ctx: SMPPipeVariantContext) -> SMPPipeVariant:
        """
        Materialize this pipe to a SMP variant
        """
        raise NotImplementedError

    def _to_ray(self, variant_ctx: RayPipeVariantContext) -> RayPipeVariant:
        """
        Materialize this pipe to a SMP variant
        """
        raise NotImplementedError

    def _to_tf(self, variant_ctx: TFPipeVariantContext) -> TFPipeVariant:
        """
        Materialize this pipe to a local TF variant
        """
        raise NotImplementedError

    def _to_tf_ray(
        self, variant_ctx: TFRayPipeVariantContext
    ) -> RayPipeVariant:
        """
        Materialize this pipe to a Ray TF variant
        """
        raise NotImplementedError

    def _to_ray_ds(
        self, variant_ctx: RayDSPipeVariantContext
    ) -> RayDSPipeVariant:
        """
        Materialize this pipe to a Ray Data pipe variant
        """
        raise NotImplementedError

    def mutate(
        self,
        ctx: CedarContext,
        variant_type: PipeVariantType,
        variant_ctx: Optional[PipeVariantContext] = None,
    ) -> None:
        """
        Assigns this Pipe a specific PipeVariant

        Raises:
            MutationError if mutation fails
        """
        self._check_mutation()

        # Assign TF
        if self.is_tf():
            if len(self.input_pipes) > 1:
                raise NotImplementedError

            if len(self.input_pipes) == 0:
                if self._output_tf_spec is None:
                    raise RuntimeError("TF Source pipes need TF output spec")
            else:
                # Does a fixed input spec exist?
                if self._fix_input_tf_spec:
                    input_spec = self._input_tf_spec
                else:
                    input_pipe = self.input_pipes[0]
                    input_spec = input_pipe.generate_output_tf_spec()
                    self._input_tf_spec = (
                        input_spec  # temporarily set input spec
                    )
                logger.info(f"Pipe {self.id} has input TF spec:")
                logger.info(self._input_tf_spec)

        self.pipe_variant = self._create_pipe_variant(
            variant_type, variant_ctx
        )
        self.pipe_variant.p_id = self.id
        if hasattr(self, "pipe_spec"):
            self.pipe_variant.pipe_spec = self.pipe_spec
        self.pipe_variant_type = variant_type

    def dynamic_mutate(
        self,
        variant_type: PipeVariantType,
        downstream_pipe: Optional[Pipe],
        variant_ctx: Optional[PipeVariantContext] = None,
    ) -> None:
        """
        When called, freeze the execution of this pipe, allow outputs to drain,
        and mutate this pipe to a new variant

        Args:
            variant_type: The PipeVariantType to mutate to
            downstream_pipe: the next Pipe in the dataflow graph
            variant_ctx: If provided, the PipeVariantContext to use
                for the new pipe variant. If not provided, uses
                the default ctx generated by the PipeVariantContextFactory

        Raises:
            MutationError if mutation is not possible
        """
        logger.info(
            "Dynamically mutating pipe {} to {}".format(
                self.get_logical_uname(), variant_type
            )
        )
        self.ok_to_mutate.clear()

        if self.pipe_variant is None:
            self.ok_to_mutate.set()
            raise MutationError

        # Handle draining pipes if necessary
        new_pipe_variant = self._create_pipe_variant(variant_type, variant_ctx)
        if self.pipe_variant_type == PipeVariantType.INPROCESS:
            self._dynamic_mutate_callback(
                new_pipe_variant, variant_type, downstream_pipe
            )
        elif (
            self.pipe_variant_type == PipeVariantType.MULTIPROCESS
            or self.pipe_variant_type == PipeVariantType.MULTITHREADED
            or self.pipe_variant_type == PipeVariantType.SMP
            or self.pipe_variant_type == PipeVariantType.RAY
        ):
            # register callback to drain
            if not hasattr(self.pipe_variant, "register_drain_callback"):
                raise MutationError("Unable to register drain callback")

            cb = functools.partial(
                self._dynamic_mutate_callback,
                new_pipe_variant=new_pipe_variant,
                variant_type=variant_type,
                downstream_pipe=downstream_pipe,
            )
            self.pipe_variant.register_drain_callback(cb)
        else:
            raise NotImplementedError

    def is_source(self) -> bool:
        """
        True if this pipe is a source pipe
        """
        return len(self.input_pipes) == 0

    def get_input_pipe_variant(self, idx: int = 0) -> PipeVariant:
        """
        Returns the input pipe variant of the idx-th input pipe. Defaults to
        pipe at index 0"""
        if not self.input_pipes[idx].pipe_variant:
            raise RuntimeError(
                f"Input pipe {self.input_pipes[idx].name} is not mutated."
            )
        return self.input_pipes[idx].pipe_variant

    def get_input_pipe(self, idx: int = 0) -> Pipe:
        """
        Returns the input pipe idx-th input pipe. Defaults to
        pipe at index 0"""
        return self.input_pipes[idx]

    def get_variant(self) -> PipeVariant:
        """
        Returns the pipe variant of this pipe
        Raises:
            RuntimeError if pipe is not mutated
        """
        if not self.pipe_variant:
            raise RuntimeError(f"Pipe {self.name} is not mutated.")

        return self.pipe_variant

    def get_variant_type(self) -> PipeVariantType:
        """
        Returns the pipe variant type of this pipe
        Raises:
            RuntimeError if pipe is not mutated
        """
        if not self.pipe_variant_type:
            raise RuntimeError(f"Pipe {self.name} is not mutated.")

        return self.pipe_variant_type

    def get_spec(self) -> CedarPipeSpec:
        """
        Returns the pipe spec for this pipe
        Raises:
            RuntimeError if pipe spec is not set
        """
        if self.pipe_spec is None:
            raise RuntimeError(f"Pipe {self.name} has no spec")

        return self.pipe_spec

    def _check_mutation(self) -> None:
        """
        Checks pipe prior to mutation.

        Raises:
            MutationError if mutation checks fail
        """
        for input in self.input_pipes:
            if input.pipe_variant is None:
                raise MutationError(
                    "Input pipe must be mutated for Pipe {}".format(self.name)
                )

        if self.pipe_variant is not None:
            raise MutationError("{} is already mutated.".format(self.name))

    def get_logical_name(self) -> str:
        return self.name

    def get_physical_name(self) -> str:
        if not self.pipe_variant_type:
            raise RuntimeError(
                f"Cannot get physical name of {self.name}; not mutated"
            )
        return self.name + "_" + self.pipe_variant_type.name

    def get_logical_uname(self) -> str:
        """
        Returns a unique name for this pipe within the graph
        """
        if self.id is None:
            logger.warning("ID not assigned.")
            return self.get_logical_name()
        return self.get_logical_name() + "_" + str(self.id)

    def get_physical_uname(self) -> str:
        """
        Returns a unique name for this pipe within the graph
        """
        if self.id is None:
            logger.warning("ID not assigned.")
            return self.get_physical_name()
        return self.get_physical_name() + "_" + str(self.id)

    def set_input_pipes(self, input_pipes: List[Pipe]):
        """
        Overwrites this pipe's current set of input pipe(s) with
        the provided list.
        """
        self.input_pipes = input_pipes

    def reset(self) -> None:
        """
        Reset the mutation for this pipe
        """
        logger.info(f"Restting pipe {self.id}")
        if self.pipe_variant is not None:
            self.pipe_variant.shutdown()
        self.pipe_variant = None
        self.pipe_variant_type = None

    def _create_pipe_variant(
        self, variant_type: PipeVariantType, variant_ctx: PipeVariantContext
    ):
        if variant_ctx is None:
            variant_ctx = PipeVariantContextFactory.create_context(
                variant_type
            )
        if variant_type == PipeVariantType.INPROCESS:
            pipe_variant = self._to_inprocess(variant_ctx)
        elif variant_type == PipeVariantType.MULTIPROCESS:
            pipe_variant = self._to_multiprocess(variant_ctx)
        elif variant_type == PipeVariantType.MULTITHREADED:
            pipe_variant = self._to_multithreaded(variant_ctx)
        elif variant_type == PipeVariantType.SMP:
            pipe_variant = self._to_smp(variant_ctx)
        elif variant_type == PipeVariantType.RAY:
            pipe_variant = self._to_ray(variant_ctx)
        elif variant_type == PipeVariantType.TF:
            pipe_variant = self._to_tf(variant_ctx)
        elif variant_type == PipeVariantType.TF_RAY:
            pipe_variant = self._to_tf_ray(variant_ctx)
        elif variant_type == PipeVariantType.RAY_DS:
            pipe_variant = self._to_ray_ds(variant_ctx)
        else:
            raise MutationError(
                "Invalid PipeVariantType {}".format(variant_type)
            )
        pipe_variant.variant_ctx = variant_ctx
        return pipe_variant

    def _dynamic_mutate_callback(
        self,
        new_pipe_variant: PipeVariant,
        variant_type: PipeVariantType,
        downstream_pipe: Pipe,
    ):
        """
        Callback to mutate the pipe
        """
        logger.info("Dynamic mutate callback for {}".format(self.id))

        if downstream_pipe is None:
            raise MutationError("Cannot mutate output pipe.")

        with downstream_pipe.pipe_variant.lock:
            input_it = self.pipe_variant.get_input_iter()
            old_pipe_variant = self.pipe_variant

            self.pipe_variant = new_pipe_variant
            self.pipe_variant.p_id = self.id
            # Need to set spec before setting input iter
            if hasattr(self, "pipe_spec"):
                self.pipe_variant.pipe_spec = self.pipe_spec
            self.pipe_variant.set_input_iter(input_it, True)
            self.pipe_variant.set_mutation_event(self.ok_to_mutate)
            self.pipe_variant_type = variant_type
            # This calls this pipe variant's __iter__
            # In this case, we don't want to reset the input iter
            output_it = iter(self.pipe_variant)

            downstream_pipe.pipe_variant.input_pipe_variant = self.pipe_variant
            downstream_pipe.pipe_variant.set_input_iter(output_it)

        logger.info("Finished dynamic mutation for {}".format(self.id))
        # self.ok_to_mutate.set()

        # Explicitly delete the old variant to free resources
        old_pipe_variant.shutdown()

    def is_mutable(self) -> bool:
        if self.pipe_spec is None:
            return False
        return self.pipe_spec.mutable

    def get_fused_callable(self) -> Callable:
        """
        Returns a callable which may be chained with other callables to
        implement pipe fusion
        """
        raise NotImplementedError

    def dynamic_insert_pipe(
        self,
        pipe_variant_type: PipeVariantType,
        head_pipe: Pipe,
        downstream_pipe: Pipe,
    ) -> None:
        """
        Dynamically inserts this pipe into the dataflow graph.

        NOTE: Assumes that this pipe's input pipes are set,
        AND that the downstream pipe's pipe variant is locked.

        Args:
            pipe_variant_type: Pipe Variant to mutate this pipe into
            head_pipe: The first Pipe that we want to replace.
                For example, if we want to replace B, C in [A->B->C->D],
                this should be B.
            downstream_pipe: The first downstream pipe. In the example above,
                this should be D.
        """
        logger.info("Dynamically inserting pipe {}".format(self.id))
        self.ok_to_mutate.clear()
        # Create pipe variant
        self.pipe_variant = self._create_pipe_variant(pipe_variant_type, None)
        self.pipe_variant.p_id = self.id
        self.pipe_variant.input_pipe_variant = (
            head_pipe.pipe_variant.input_pipe_variant
        )

        if hasattr(self, "pipe_spec"):
            self.pipe_variant.pipe_spec = self.pipe_spec

        with downstream_pipe.pipe_variant.lock:
            input_it = head_pipe.pipe_variant.get_input_iter()
            self.pipe_variant.set_input_iter(input_it, True)
            self.pipe_variant.set_mutation_event(self.ok_to_mutate)
            self.pipe_variant_type = pipe_variant_type
            output_it = iter(self.pipe_variant)

            downstream_pipe.pipe_variant.input_pipe_variant = self.pipe_variant
            downstream_pipe.pipe_variant.set_input_iter(output_it)

        # self.ok_to_mutate.set()

    def shutdown_variant(self):
        """
        Shuts down and cleans up this pipe's variant
        """
        self.pipe_variant.shutdown()
        self.pipe_variant = None

    def wait_until_mutate_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks until this pipe is ready to mutate

        If timeout is set, return False if wait timed out
        """
        if timeout is not None:
            return self.ok_to_mutate.wait(timeout)
        else:
            self.ok_to_mutate.wait(timeout)
            return True

    def mark_mutate_in_progress(self) -> None:
        """
        Designates that this pipe is about to undergo mutation.
        Subsequent calls to `wait_until_mutate_ready` will block until
        the pipe has mutated.
        """
        self.ok_to_mutate.clear()

    def is_fusable(self, pipe_variant_type: PipeVariantType) -> bool:
        """
        Returns true if this pipe is fusable
        """
        if self.pipe_spec is None:
            return False
        return (
            self.pipe_spec.is_fusable
            and pipe_variant_type in self.pipe_spec.mutable_variants
        )

    def can_mutate_to(self, pipe_variant_type: PipeVariantType) -> bool:
        """
        True if this can mutate to pipe_variant_type
        """
        if self.pipe_spec is None:
            return False
        if self.is_tf():
            return (
                self.pipe_spec.mutable
                and (
                    pipe_variant_type == PipeVariantType.TF
                    or pipe_variant_type == PipeVariantType.TF_RAY
                )
                and pipe_variant_type in self.pipe_spec.mutable_variants
            )
        return (
            self.pipe_spec.mutable
            and pipe_variant_type in self.pipe_spec.mutable_variants
        )

    def dynamic_split(
        self, replacement_pipes: List[Pipe], downstream_pipe: Pipe
    ) -> None:
        """
        Replaces this pipe in the dataflow graph with the ordered
        list of pipes in `replacement_pipes`.

        """
        logger.info(f"Dynamically splitting pipe {self.get_logical_uname()}")
        self.ok_to_mutate.clear()

        if self.pipe_variant is None:
            self.ok_to_mutate.set()
            raise MutationError

        # Create a pipe variant for each pipe
        new_pipe_variants = []
        for p in replacement_pipes:
            new_pipe_variant = p._create_pipe_variant(
                PipeVariantType.INPROCESS, None
            )
            new_pipe_variants.append(new_pipe_variant)
            if hasattr(p, "pipe_spec"):
                new_pipe_variant.pipe_spec = p.pipe_spec
            p.pipe_variant = new_pipe_variant
            p.pipe_variant_type = PipeVariantType.INPROCESS
            p.pipe_variant.p_id = p.id

        if self.pipe_variant_type == PipeVariantType.INPROCESS:
            self._dynamic_split_callback(new_pipe_variants, downstream_pipe)
        elif (
            self.pipe_variant_type == PipeVariantType.MULTIPROCESS
            or self.pipe_variant_type == PipeVariantType.MULTITHREADED
            or self.pipe_variant_type == PipeVariantType.SMP
            or self.pipe_variant_type == PipeVariantType.RAY
        ):
            # register callback to drain
            if not hasattr(self.pipe_variant, "register_drain_callback"):
                raise MutationError("Unable to register drain callback")

            cb = functools.partial(
                self._dynamic_split_callback,
                new_pipe_variants=new_pipe_variants,
                downstream_pipe=downstream_pipe,
            )
            self.pipe_variant.register_drain_callback(cb)
        else:
            raise NotImplementedError

    def _dynamic_split_callback(
        self, new_pipe_variants: List[PipeVariant], downstream_pipe: Pipe
    ):
        logger.info("Dynamic split callback for {}".format(self.id))

        if downstream_pipe is None:
            raise MutationError("Cannot split output pipe.")

        if len(new_pipe_variants) < 2:
            raise MutationError("Cannot split single pipe.")

        with downstream_pipe.pipe_variant.lock:
            # Set the first pipe variant's input iter
            input_it = self.pipe_variant.get_input_iter()
            new_pipe_variants[0].set_input_iter(input_it, True)

            # Connect the chain
            curr_pipe_variant = new_pipe_variants[0]
            for pipe_variant in new_pipe_variants[1:]:
                output_it = iter(curr_pipe_variant)
                pipe_variant.input_pipe_variant = curr_pipe_variant
                pipe_variant.set_input_iter(output_it)
                curr_pipe_variant = pipe_variant

            output_it = iter(curr_pipe_variant)
            downstream_pipe.pipe_variant.input_pipe_variant = curr_pipe_variant
            downstream_pipe.pipe_variant.set_input_iter(output_it)

        logger.info("Finished dynamic split for {}".format(self.id))
        self.shutdown_variant()

        self.ok_to_mutate.set()

    def fix(self) -> Pipe:
        """
        End-user Reordering API. Affix this pipe within the feature order.

        TODO: Make end-user vs dev APIs more apparent.
        """
        self._fix_order = True
        return self

    def depends_on(self, tags: List[str]) -> Pipe:
        """
        End-user Reordering API. Designates that this pipe depends on
        the set of pipes corresponding to the input tags.

        Args:
            tags: List of tags for pipes that must precede this pipe

        TODO: Make end-user vs dev APIs more apparent.
        """
        self._depends_on_tags = tags
        return self

    def tf_py_func(self) -> Pipe:
        """
        Designates this function as a TF py_function
        """
        self._is_tf_py_func = True
        return self

    def shard(self, rank_spec: Tuple[int, int]) -> None:
        if not self.is_source():
            raise RuntimeError("Cannot shard non-source pipe.")
        if self.pipe_spec is None:
            raise RuntimeError(f"Pipe {self.name} has no spec")
        if not self.pipe_spec.is_shardable:
            raise RuntimeError(f"Pipe {self.name} is not shardable")

        self.rank_spec = rank_spec

    def is_tf(self) -> bool:
        """
        Returns true if this pipe operates on tensorflow tensors
        """
        return self._is_tf

    def is_tf_py_func(self) -> bool:
        """
        Returns true if this pipe uses tf.py_function
        """
        return self._is_tf_py_func

    def generate_output_tf_spec(self) -> tf.TensorSpec:
        """
        Generates the tf.TensorSpec that this pipe will output.
        """
        if not self.is_tf() and self._output_tf_spec is None:
            raise RuntimeError(f"{self.id} is not a TF pipe.")

        if self._output_tf_spec:
            return self._output_tf_spec

        # If a concrete spec not set, infer from hint
        if not self._input_tf_spec and not self._output_tf_hint:
            raise RuntimeError(f"{self.id}'s output cannot be inferred")

        # infer the shape
        # check if tuple
        if isinstance(self._output_tf_hint, tuple):
            output_shapes = []
            output_dtypes = []
            output_spec = ()

            hint_output_shapes = (hint.shape for hint in self._output_tf_hint)
            hint_output_dtypes = (hint.dtype for hint in self._output_tf_hint)

            for idx, hint_output_shape in enumerate(hint_output_shapes):
                # If tuple, should be same size
                if isinstance(hint_output_shape, TFTensorDontCare):
                    output_shape = self._input_tf_spec[idx].shape
                else:
                    list_output_shape = []
                    for i, dim in enumerate(hint_output_shape):
                        if isinstance(dim, TFTensorDontCare):
                            try:
                                input_dim = self._input_tf_spec[
                                    idx
                                ].shape.as_list()[i]
                            except IndexError:
                                input_dim = None

                            list_output_shape.append(input_dim)
                        else:
                            list_output_shape.append(dim)
                    output_shape = tf.TensorShape(list_output_shape)
                output_shapes.append(output_shape)

            for idx, hint_output_dtype in enumerate(hint_output_dtypes):
                if not isinstance(hint_output_dtype, TFTensorDontCare):
                    output_dtypes.append(hint_output_dtype)
                else:
                    output_dtypes.append(self._input_tf_spec[idx].dtype)

            # Create a tuple of tensorspecs
            for i in range(len(output_shapes)):
                output_spec += (
                    tf.TensorSpec(output_shapes[i], output_dtypes[i]),
                )
            return output_spec
        else:
            hint_output_shape = self._output_tf_hint.shape
            if isinstance(hint_output_shape, TFTensorDontCare):
                output_shape = self._input_tf_spec.shape
            else:
                list_output_shape = []
                for i, dim in enumerate(hint_output_shape):
                    if isinstance(dim, TFTensorDontCare):
                        try:
                            input_dim = self._input_tf_spec.shape.as_list()[i]
                        except IndexError:
                            input_dim = None

                        list_output_shape.append(input_dim)
                    else:
                        list_output_shape.append(dim)
                output_shape = tf.TensorShape(list_output_shape)

            output_dtype = self._input_tf_spec.dtype
            if not isinstance(self._output_tf_hint.dtype, TFTensorDontCare):
                output_dtype = self._output_tf_hint.dtype

            return tf.TensorSpec(output_shape, output_dtype)

    def set_output_tf_spec(self, spec: tf.TensorSpec) -> None:
        """
        Concretely set the output tf spec for this pipe.
        NOTE: This should only be used for source pipes.
        """
        self._output_tf_spec = spec

    def fuse_into(self, pipes: List[Pipe]) -> None:
        """
        Fuses the immediate downstream pipe into this pipe.

        Args:
            pipe: List of the adjacent downstream pipe to fuse
        """
        raise NotImplementedError

    def is_random(self) -> bool:
        """
        Checks whether is_random flag is set.
        """
        return self._is_random
