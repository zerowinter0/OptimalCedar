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
        last_sample = None
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
                        last_sample = x
                        batch_ds.data.append(x.data)
                        if len(batch_ds.data) == self.batch_size:
                            batch_ds = self._emit_batch(batch_ds, x)
                            yield batch_ds
                            batch_ds = DataSample([])
                    else:
                        raise NotImplementedError

                except StopIteration:
                    # Check last batch
                    if len(batch_ds.data) > 0 and not self.drop_last:
                        batch_ds = self._emit_batch(batch_ds, last_sample)
                        yield batch_ds
                    break

    def _emit_batch(self, batch_ds, last_input):
        """Stack a completed batch, anchored on the *last* input's trace.

        The batch envelope historically copied the trace of every incoming
        sample while it was appended, so the emitted batch could carry the trace
        of an earlier record: its reported latency then covered the whole
        batch-assembly span (batch_size - 1 record chains) instead of the
        batcher's own stacking work.  Anchoring on the last input makes the
        batcher's window what it claims to be.
        """
        if last_input is not None and last_input.do_trace:
            batch_ds.copy_metadata_from(last_input)
        if batch_ds.do_trace:
            batch_ds.set_size(self.p_id, len(batch_ds.data))
        if is_torch_tensor(batch_ds.data[0]):
            try:
                batch_ds.data = torch().stack(batch_ds.data, dim=0)
            except RuntimeError:
                logger.warning("Could not batch torch data")
        elif is_tensorflow_tensor(batch_ds.data[0]):
            try:
                batch_ds.data = tensorflow().stack(batch_ds.data, axis=0)
            except RuntimeError:
                logger.warning("Could not batch tf data")
        return batch_ds
