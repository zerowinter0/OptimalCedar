import logging
from .pipe import (
    Pipe,
)
from .variant import (
    InProcessPipeVariant,
    PipeVariant,
)
from .context import (
    PipeVariantType,
    InProcessPipeVariantContext,
)
from cedar.utils.frameworks import (
    is_tensorflow_tensor,
    is_torch_tensor,
    tensorflow,
    torch,
)
from .common import DataSample, cedar_pipe, CedarPipeSpec
from typing import Optional

logger = logging.getLogger(__name__)


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[PipeVariantType.INPROCESS],
    )
)
class BatcherPipe(Pipe):
    """
    A pipe that groups together input samples into batches

    Args:
        input_pipe: Upstream pipe
        batch_size: integer batch size
        drop_last: True to drop the final batch if not full. Default to False
    """

    def __init__(
        self,
        input_pipe: Pipe,
        batch_size: int,
        drop_last: bool = False,
        tag: Optional[str] = None,
        is_random: bool = False,
    ) -> None:
        super().__init__(
            "BatcherPipe(batch_size={})".format(batch_size),
            [input_pipe],
            tag=tag,
            is_random=is_random,
        )
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        return InProcessBatcherPipeVariant(
            self.get_input_pipe_variant(),
            self.batch_size,
            self.drop_last,
        )


class InProcessBatcherPipeVariant(InProcessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: PipeVariant,
        batch_size: int,
        drop_last: bool,
    ):
        super().__init__(input_pipe_variant)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _iter_impl(self):
        batch_ds = DataSample([])
        while True:
            if self.batch_size == 1:
                try:
                    x = next(self._input_iter)
                    if isinstance(x, DataSample):
                        if x.dummy:
                            continue
                        yield x
                    else:
                        yield x
                except StopIteration:
                    break
            else:
                try:
                    x = next(self._input_iter)
                    if isinstance(x, DataSample):
                        if x.dummy:
                            continue
                        batch_ds.data.append(x.data)
                        # If the incoming ds is tracing data, propagate it
                        # Use the last observed trace...
                        if x.do_trace:
                            batch_ds.copy_metadata_from(x)

                        if len(batch_ds.data) == self.batch_size:
                            # Set the new batch size
                            if batch_ds.do_trace:
                                batch_ds.set_size(self.p_id, len(batch_ds.data))

                            if is_torch_tensor(batch_ds.data[0]):
                                try:
                                    batch_ds.data = torch().stack(
                                        batch_ds.data, dim=0
                                    )
                                except RuntimeError:
                                    logger.warning("Could not batch data")
                            yield batch_ds
                            batch_ds = DataSample([])
                    else:
                        raise NotImplementedError

                except StopIteration:
                    # Check last batch
                    if len(batch_ds.data) > 0 and not self.drop_last:
                        if batch_ds.do_trace:
                            batch_ds.set_size(self.p_id, len(batch_ds.data))
                        if is_torch_tensor(batch_ds.data[0]):
                            try:
                                batch_ds.data = torch().stack(
                                    batch_ds.data, dim=0
                                )
                            except RuntimeError:
                                logger.warning("Could not batch torch data")
                        elif is_tensorflow_tensor(batch_ds.data[0]):
                            try:
                                batch_ds.data = tensorflow().stack(
                                    batch_ds.data, axis=0
                                )
                            except RuntimeError:
                                logger.warning("Could not batch tf data")
                        yield batch_ds
                    break
