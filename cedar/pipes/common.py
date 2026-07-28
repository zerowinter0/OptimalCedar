from typing import Any, Optional, Dict, List
import logging
import time
import torch
import tensorflow as tf
import PIL
from pympler import asizeof

from .context import PipeVariantType


logger = logging.getLogger(__name__)


class Partition:
    def __init__(
        self,
        partition_id: int,
        starting_sample_id: int,
        end_sample_id: int = None,
        contains_cached_sample: bool = False,
    ):
        self.partition_id = partition_id  # ID of partition
        self.starting_sample_id = (
            starting_sample_id  # First sample ID included in partition
        )
        self.end_sample_id = (
            end_sample_id  # Last sample ID included in partition
        )
        self.contains_cached_sample = contains_cached_sample


class DataSample:
    def __init__(
        self,
        data: Any,
        do_trace: bool = False,
        sample_id: int = None,
        read_from_cache: bool = False,
        trace_data_size: bool = False,
    ):
        """
        A wrapper for any data type, used to tag samples with extra metadata
        as it passes through the feature.
        """
        self.data = data
        self.do_trace = do_trace
        self.sample_id = sample_id
        self.trace_dict: Optional[Dict[int, int]] = None
        # Trace of the number of raw samples contained in data,
        # For example, a batch of 10 samples would correspond to a size of 10
        self.size_dict: Optional[Dict[int, int]] = None
        # If traced, the order of pipes observed by this sample
        self.trace_order: Optional[List[int]] = None
        self.buffer_size_dict: Optional[Dict[int, int]] = None
        self.dummy = False
        self.read_from_cache = read_from_cache

        self.trace_data_size = trace_data_size
        # Size of the output "data" of each pipe, in bytes
        self.data_size_dict: Optional[Dict[int, int]] = None

    def trace(self, p_id: Optional[int], buf_size: Optional[int]) -> None:
        """
        By convention, trace is called on the *output* of a given pipe
        """
        if not self.do_trace:
            return

        if p_id is None:
            logger.warning("Attempting to trace pipe without assigned ID.")
        self.trace_dict[p_id] = time.process_time_ns()

        if p_id not in self.size_dict:
            # Not set by set_size, automatically use previous size
            if len(self.trace_order) == 0:
                raise AssertionError(
                    "Unable to trace source pipe without size."
                )
            self.size_dict[p_id] = self.size_dict[self.trace_order[-1]]

        self.trace_order.append(p_id)

        # Trace the buffer size if available
        if buf_size is not None:
            self.buffer_size_dict[p_id] = buf_size

        if self.trace_data_size:
            try:
                self.data_size_dict[p_id] = get_sizeof_data(self.data)
            except Exception as e:
                logger.warning(f"Failed to get size of data: {e}")

    def set_size(self, p_id: int, size: int) -> None:
        """
        Marks the size of this datasample, at pipe p_id. If not set,
        the trace will use the size of the previous sample
        """
        if self.size_dict is None:
            self.size_dict = {}
        self.size_dict[p_id] = size

    def copy_metadata_from(self, ds: "DataSample") -> None:
        self.do_trace = ds.do_trace
        self.trace_dict = ds.trace_dict
        self.size_dict = ds.size_dict
        self.trace_order = ds.trace_order
        self.buffer_size_dict = ds.buffer_size_dict
        self.trace_data_size = ds.trace_data_size
        self.data_size_dict = ds.data_size_dict


class MutationError(Exception):
    """
    Error raised when unable to mutate pipe variants
    """

    pass


class CedarPipeSpec:
    """
    A specification used to describe the properties of cedar Pipes

    Args:
        is_mutable: Denote that a given Pipe class is mutable
            NOTE: To be mutable, all pipe variant's _iter_impl method should
            explicitly call `next()` on `self._input_iter`,
            as opposed to using a for loop.
            NOTE: Does not work for classes with multiple inheritance
        mutable_variants: List of PipeVariantTypes available for the pipe.
            The list should be ordered by preference of mutation.
    """

    def __init__(
        self,
        is_mutable: bool,
        mutable_variants: List[PipeVariantType],
        is_fusable: bool = False,
        is_shardable: bool = False,
        is_fusable_source: bool = False,
        fusable_source_variants: Optional[List[PipeVariantType]] = None,
    ):
        self.mutable = is_mutable
        self.mutable_variants = mutable_variants
        self.is_fusable = is_fusable
        self.is_shardable = is_shardable
        self.is_fusable_source = is_fusable_source
        self.fusable_source_variants = fusable_source_variants


def cedar_pipe(spec: CedarPipeSpec) -> Any:
    """
    Decorator for cedar Pipes. Used to provide an `CedarPipeSpec` to describe
    properties of the pipe
    """

    def pipe_decorator(cls):
        class WrappedPipe(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pipe_spec = spec

        # Make decorator-generated pipe classes importable by spawn workers.
        WrappedPipe.__name__ = cls.__name__
        WrappedPipe.__qualname__ = cls.__qualname__
        WrappedPipe.__module__ = cls.__module__
        return WrappedPipe

    return pipe_decorator


def get_sizeof_data(x: Any) -> int:
    """
    Returns the size of data.
    """
    if x is None:
        return 0
    elif torch.is_tensor(x):
        return x.untyped_storage().nbytes()
    elif tf.is_tensor(x):
        if x.dtype == tf.string:
            byte_tensor = tf.io.decode_raw(x, tf.uint8)
            size = tf.size(byte_tensor, out_type=tf.int32).numpy()
            return int(size)
        else:
            num_elements = tf.size(x).numpy()
            dtype_size = tf.dtypes.as_dtype(x.dtype).size
            tensor_size = num_elements * dtype_size
            return int(tensor_size)
    # If list, tuple, set, dict then recurse
    elif isinstance(x, (list, tuple, set)):
        return sum(get_sizeof_data(item) for item in x)
    elif isinstance(x, dict):
        return sum(
            get_sizeof_data(k) + get_sizeof_data(v) for k, v in x.items()
        )
    elif isinstance(x, PIL.Image.Image):
        # get the size in bytes of the image
        return asizeof.asizeof(x.tobytes())
    # asizeof should work with np arrays...
    else:
        return asizeof.asizeof(x)
