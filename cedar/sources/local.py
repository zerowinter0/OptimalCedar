from typing import List, Union, Iterator, Tuple, Optional, Callable
import itertools
import glob
import ray
import logging

from torchdata.datapipes.iter import FileLister
from torchdata.datapipes.utils import StreamWrapper

from cedar.pipes import (
    Pipe,
    InProcessPipeVariant,
    InProcessPipeVariantContext,
    PipeVariantType,
    CedarPipeSpec,
    RayDSPipeVariantContext,
    RayDSPipeVariant,
    cedar_pipe,
    DataSample,
)
from .source import Source, SourcePipeVariantMixin

logger = logging.getLogger(__name__)


class InProcessLocalFSListerPipeVariant(
    InProcessPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self,
        source: Union[List[str], InProcessPipeVariant],
        recursive: bool,
        rank_spec: Optional[Tuple[int, int]],
        max_samples: Optional[int] = None,
    ) -> None:
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source
        self.recursive = recursive
        self.max_samples = max_samples
        self.num_yielded = 0

        # evaluate all paths upon instantiation
        self.root = []
        for x in source:
            self.root.append(x)

        self.dp = FileLister(root=self.root, recursive=recursive)
        self.all_datasamples = None

    def _reset_source_iterator_for_epoch(self):
        it = iter(self.dp)
        if self.max_samples is not None:
            it = itertools.islice(it, self.max_samples)
        self.all_datasamples = self.create_datasamples(it, size=1)

    def _iter_impl(self):
        # self.num_yielded = 0
        it = iter(self.dp)
        if self.max_samples is not None:
            it = itertools.islice(it, self.max_samples)
        all_datasamples = self.create_datasamples(it, size=1)

        while True:
            # We are not in replay mode
            if not self.replay:
                try:
                    if self.all_datasamples:
                        ds = next(self.all_datasamples)
                    else:
                        ds = next(all_datasamples)

                    if isinstance(ds, DataSample):
                        self._update_partition_state(self.num_yielded)
                        ds.sample_id = self.num_yielded
                        self.num_yielded += 1
                    yield ds
                except StopIteration:
                    return

            else:  # We are in replay mode
                if self.sharded:
                    raise NotImplementedError("Can't shard with replay")
                if not self.initialized_replay:
                    new_dp = FileLister(
                        root=self.root, recursive=self.recursive
                    )
                    replay_it = iter(new_dp)
                    if self.max_samples is not None:
                        replay_it = itertools.islice(
                            replay_it, self.max_samples
                        )
                    replay_data = iter(
                        self.create_datasamples(replay_it, size=1)
                    )
                    self.initialized_replay = True
                    curr_ds_id = 0
                try:
                    if not self.partitions_to_replay:
                        self.disable_replay()
                        continue
                    ds = next(replay_data)
                    if not self._should_be_replayed(curr_ds_id):
                        curr_ds_id += 1
                        continue
                    else:
                        self._update_partition_state(curr_ds_id)
                        ds.sample_id = curr_ds_id
                        curr_ds_id += 1
                        yield ds
                except StopIteration:
                    self.disable_replay()
                    continue


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
        is_fusable=False,
        is_shardable=True,
        is_fusable_source=True,
        fusable_source_variants=[PipeVariantType.RAY_DS],
    )
)
class LocalFSListerPipe(Pipe):
    """
    Pipe that iterates over files within a local filesystem.

    Args:
        source: Sequence of filesystem paths, representing
        files or directories.
    """

    source: List[str]

    def __init__(
        self,
        source: Union[Pipe, List[str]],
        recursive: bool,
        rank_spec: Optional[Tuple[int, int]],
        max_samples: Optional[int] = None,
        is_random: bool = False,
    ) -> None:
        if isinstance(source, Pipe):
            super().__init__(
                "LocalFSListerPipe", [source], is_random=is_random
            )
            self.source = []
        else:
            # empty inputs denotes source pipe
            super().__init__("LocalFSListerPipe", [])
            self.source = source
        self.recursive = recursive
        self.rank_spec = rank_spec
        self.max_samples = max_samples

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if self.is_source():
            variant = InProcessLocalFSListerPipeVariant(
                self.source,
                self.recursive,
                self.rank_spec,
                self.max_samples,
            )
        else:
            if (
                not self.input_pipes[0].pipe_variant_type
                != PipeVariantType.INPROCESS
            ):
                raise NotImplementedError
            variant = InProcessLocalFSListerPipeVariant(
                self.input_pipes[0].pipe_variant,
                self.recursive,
                self.rank_spec,
                self.max_samples,
            )
        return variant

    def _to_ray_ds(
        self, variant_ctx: RayDSPipeVariantContext
    ) -> RayDSPipeVariant:
        if self._pipes_to_fuse is None:
            raise RuntimeError(
                "Cannot use LocalFSListerPipe RayDS without Fuse"
            )
        if not isinstance(self.source, list):
            raise NotImplementedError
        if not self.is_source():
            raise NotImplementedError

        fns = [p.get_fused_callable() for p in self._pipes_to_fuse]
        logger.info("Creating Ray DS Source!!!")
        return RayDSLocalFileListerPipeVariant(
            self.source, fns, self.rank_spec
        )

    def fuse_into(self, pipes: List[Pipe]) -> None:
        self._pipes_to_fuse = pipes


class RayDSLocalFileListerPipeVariant(
    RayDSPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self,
        source: List[str],
        fused_fns: List[Callable],
        rank_spec: Optional[Tuple[int, int]],
    ) -> None:
        RayDSPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source

        if len(source) != 1:
            raise NotImplementedError

        files = glob.glob(f"{source[0]}/*")

        self.dataset = ray.data.from_items(files)
        for fn in fused_fns:
            print(fn)
            self.dataset = self.dataset.map(fn)

    def _iter_impl(self):
        self.dataset_iter = iter(self.dataset.iter_rows())
        datasamples = self.create_datasamples(self.dataset_iter, size=1)
        while True:
            try:
                ds = next(datasamples)
                yield ds
            except StopIteration:
                return

    def _reset_source_iterator_for_epoch(self):
        pass


class LocalFSSource(Source):
    """
    A source that represents a collection of files within
    a local filesystem.

    Upon load, generates a Pipe that iterates over the path of
    each file parsed from the input path(s)

    Args:
        source: A string or collection of strings representing
            a path to the input data in the local filesystem.
        recursive: To recursively glob all files
        rank_spec: If provided, specify [World Size, Rank] to shard
            pipe across features.
    """

    def __init__(
        self,
        source: Union[str, List[str]],
        recursive: bool = False,
        rank_spec: Optional[Tuple[int, int]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if isinstance(source, str):
            self.source = [
                source,
            ]
        elif isinstance(source, list):
            self.source = source
        else:
            raise TypeError("Invalid LocalFSSource input.")
        self.recursive = recursive
        self.rank_spec = rank_spec
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        self.max_samples = max_samples

    def to_pipe(self) -> Pipe:
        pipe = LocalFSListerPipe(
            self.source,
            self.recursive,
            rank_spec=self.rank_spec,
            max_samples=self.max_samples,
        )
        pipe.fix()
        return pipe


class LocalLineSource(Source):
    """
    A local source that represents a collection of lines
    separated by \n within one file.

    Upon load, generates a Pipe that iterates over the
    line of the indicated input file

    Args:
        source: String representing a file that can
        be partitioned along \n.
    """

    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("Invalid LocalLineSource input.")

        self.source = source
        self.rank_spec = rank_spec
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        self.max_samples = max_samples

    def to_pipe(self) -> Pipe:
        pipe = LocalLinePipe(
            self.source,
            rank_spec=self.rank_spec,
            max_samples=self.max_samples,
        )
        pipe.fix()
        return pipe


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
        is_fusable=False,
        is_shardable=True,
    )
)
class LocalLinePipe(Pipe):
    """
    Pipe that iterates over \n separated lines in a file.

    Args:
        source: String representing a file that can
        be partitioned along \n.
    """

    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]],
        max_samples: Optional[int] = None,
        is_random: bool = False,
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("LocalLinePipe only supports sources of type str.")

        super().__init__("LocalLinePipe", [], is_random=is_random)
        self.source = source
        self.rank_spec = rank_spec
        self.max_samples = max_samples

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        variant = InProcessLocalLinePipeVariant(
            self.source, self.rank_spec, self.max_samples
        )
        return variant


class InProcessLocalLinePipeVariant(
    InProcessPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]],
        max_samples: Optional[int] = None,
    ) -> None:
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source
        self.max_samples = max_samples
        self.num_yielded = 0
        length = self._calculate_length()
        self.length = min(length, max_samples) if max_samples else length
        self.all_datasamples = None
        self.file = None

    def _reset_source_iterator_for_epoch(self):
        self.file = open(self.source, mode="rb")
        stream = self._read_lines(self.file)
        stream = self._decode(stream)
        stream = self._strip(stream)
        it = iter(stream)
        if self.max_samples is not None:
            it = itertools.islice(it, self.max_samples)
        self.all_datasamples = iter(self.create_datasamples(it, size=1))

    def _iter_impl(self):
        with open(self.source, mode="rb") as file:
            stream = self._read_lines(file)
            stream = self._decode(stream)
            stream = self._strip(stream)

            # Create a new datasample for each line
            it = iter(stream)
            if self.max_samples is not None:
                it = itertools.islice(it, self.max_samples)
            all_datasamples = iter(self.create_datasamples(it, size=1))

            while True:
                # We are not in replay mode
                if not self.replay:
                    try:
                        if self.all_datasamples:
                            ds = next(self.all_datasamples)
                        else:
                            ds = next(all_datasamples)
                        if isinstance(ds, DataSample):
                            self._update_partition_state(self.num_yielded)
                            ds.sample_id = self.num_yielded
                            self.num_yielded += 1
                            yield ds
                        else:
                            yield ds
                    except StopIteration:
                        if self.file:
                            self.file.close()
                        return
                else:  # We are in replay mode
                    if not self.initialized_replay:
                        replay_file = open(self.source, mode="rb")
                        replay_stream = self._read_lines(replay_file)
                        replay_stream = self._decode(replay_stream)
                        replay_stream = self._strip(replay_stream)

                        replay_it = iter(replay_stream)
                        if self.max_samples is not None:
                            replay_it = itertools.islice(
                                replay_it, self.max_samples
                            )
                        replay_data = iter(
                            self.create_datasamples(replay_it, size=1)
                        )
                        self.initialized_replay = True
                        curr_ds_id = 0
                    try:
                        if not self.partitions_to_replay:
                            replay_file.close()
                            self.disable_replay()
                            continue
                        ds = next(replay_data)
                        if not self._should_be_replayed(curr_ds_id):
                            curr_ds_id += 1
                            continue
                        else:
                            self._update_partition_state(curr_ds_id)
                            ds.sample_id = curr_ds_id
                            curr_ds_id += 1
                            yield ds
                    except StopIteration:
                        replay_file.close()
                        self.disable_replay()
                        continue

    def _calculate_length(self):
        """
        Calculates the amount of newlines in a file.
        Assumes that file ends with a newline.
        NOTE: If there is a newline followed by another newline
        (i.e., an empty line), then this line is counted.
        This empty line will also be generated as one
        of the data samples.
        """

        def blocks(files, size=65536):
            while True:
                b = files.read(size)
                if not b:
                    break
                yield b

        with open(self.source, "r") as f:
            return sum(bl.count("\n") for bl in blocks(f))

    @staticmethod
    def _read_lines(
        file: StreamWrapper,
    ) -> Union[Iterator[str], Iterator[bytes]]:
        try:
            yield from file
        finally:
            file.close()

    @staticmethod
    def _decode(
        stream: Union[Iterator[str], Iterator[bytes]]
    ) -> Iterator[str]:
        for line in stream:
            if isinstance(line, bytes):
                yield line.decode("utf-8")
            else:
                yield line

    @staticmethod
    def _strip(stream: Iterator[str]):
        for line in stream:
            yield line.strip("\r\n")
