import abc
import ray
import logging
from typing import Optional

from .variant import PipeVariant, _AsyncPipeVariant
from .common import DataSample
from .context import RayPipeVariantContext

logger = logging.getLogger(__name__)
MAX_INFLIGHT_SCALING = 5


class RayPipeVariant(_AsyncPipeVariant):
    """
    A PipeVariant which represents execution on a remote Ray cluster.
    """

    def __init__(
        self,
        name: str,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: RayPipeVariantContext,
    ):
        if not ray.is_initialized():
            raise RuntimeError("Ray runtime is not initialized!")

        variant_ctx.max_inflight = max(
            variant_ctx.max_inflight,
            variant_ctx.n_actors * MAX_INFLIGHT_SCALING,
            variant_ctx.submit_batch_size + 1,  # to avoid deadlock due to tail
        )
        super().__init__(
            input_pipe_variant,
            max_inflight=variant_ctx.max_inflight,
            max_prefetch=variant_ctx.max_prefetch,
            use_threads=variant_ctx.use_threads,
        )
        self.variant_ctx = variant_ctx
        self.name = name

        # Create actors
        actors = []
        for _ in range(variant_ctx.n_actors):
            actor = self._create_actor()
            actors.append(actor)

        # Actor handles are returned before their processes necessarily finish
        # starting. A fixed-width experiment must not begin with only a subset
        # of a stage's actors ready.
        try:
            ray.get(
                [actor.__ray_ready__.remote() for actor in actors],
                timeout=60,
            )
        except Exception:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            raise

        self.variant_ctx.service.register(name, actors)
        actual_actors = self.variant_ctx.service.get_num_actors()
        if actual_actors != variant_ctx.n_actors:
            raise RuntimeError(
                f"RayService for {name} registered {actual_actors} actors; "
                f"expected {variant_ctx.n_actors}"
            )

    @abc.abstractmethod
    def _create_actor(self) -> ray.actor.ActorClass:
        """
        Creates a Ray Actor, which is a remote process that processes requests

        Returns:
            a handle to the underlying Ray ActorClass
        """
        pass

    def _submit(self, sample: DataSample):
        """
        Submits a datasample for processing.
        """
        self.variant_ctx.service.submit(sample)
        self.issued_tasks += 1

    def _get_next_result(self, timeout: float = 1) -> DataSample:
        """
        Returns the next result, or raise queue.Empty if no result is ready.
        """
        res = self.variant_ctx.service.next(timeout)
        self.completed_tasks += 1
        return res

    def set_scale(self, resource_count: int) -> None:
        """
        Set the parallelism of this pipe variant to resource_count.
        """
        if resource_count <= 0:
            logger.warning(
                "Cannot scale resource to {}".format(resource_count)
            )
            return
        logger.info(
            "Scaling Pipe {} to {} resources".format(self.p_id, resource_count)
        )

        curr_scale = self.get_scale()

        if resource_count > curr_scale:
            # Scale up
            for _ in range(resource_count - curr_scale):
                actor = self._create_actor()
                self.variant_ctx.service.register_and_start_actor(actor)
        elif resource_count < curr_scale:
            # Scale down
            self.variant_ctx.service.deregister(curr_scale - resource_count)

        self.variant_ctx.max_inflight = max(
            self.variant_ctx.max_inflight,
            resource_count * MAX_INFLIGHT_SCALING,
        )
        self.max_inflight = self.variant_ctx.max_inflight

    def get_scale(self) -> int:
        """
        Returns the current parallelism of this pipe variant
        """
        return self.variant_ctx.service.get_num_actors()

    def _inflight_tasks_remaining(self):
        """
        Returns true if there are any inflight tasks
        """
        return self.variant_ctx.service.get_num_inflight_tasks() > 0

    def _can_submit(self) -> bool:
        return (
            self.max_inflight == -1
            or self.variant_ctx.service.get_num_inflight_tasks()
            < self.max_inflight
        )

    def _finalize(self) -> None:
        self.variant_ctx.service.finalize()
