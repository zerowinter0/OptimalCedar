import abc
import math
import os
import ray
import logging
from typing import Any, Dict, Optional

from .variant import PipeVariant, _AsyncPipeVariant
from .common import DataSample
from .context import RayPipeVariantContext

logger = logging.getLogger(__name__)
MAX_INFLIGHT_SCALING = 5
RAY_PLACEMENT_RESOURCE_ENV = "CEDAR_RAY_PLACEMENT_RESOURCE"
RAY_PLACEMENT_RESOURCE_FRACTION_ENV = (
    "CEDAR_RAY_PLACEMENT_RESOURCE_FRACTION"
)
DEFAULT_RAY_PLACEMENT_RESOURCE_FRACTION = 0.001


def get_ray_actor_options(
    num_gpus: float = 0.0, num_cpus: float = 1.0
) -> Dict[str, Any]:
    """Return common Ray actor options, including an optional node pin.

    Every actor performs real CPU work, including actors whose primary model
    runs on a GPU.  Requesting one Ray CPU therefore makes the remote node's
    advertised CPU capacity an enforceable resource limit instead of relying
    only on the optimizer's bookkeeping.  Remote-only experiments additionally
    advertise a tiny custom resource exclusively on the remote worker; that
    resource constrains placement without pretending to represent CPU capacity.
    """
    parsed_cpus = float(num_cpus)
    if not math.isfinite(parsed_cpus) or parsed_cpus <= 0:
        raise ValueError("num_cpus must be finite and > 0")
    options: Dict[str, Any] = {
        "num_cpus": parsed_cpus,
        "num_gpus": float(num_gpus),
    }
    resource_name = os.environ.get(RAY_PLACEMENT_RESOURCE_ENV, "").strip()
    if not resource_name:
        return options

    raw_fraction = os.environ.get(
        RAY_PLACEMENT_RESOURCE_FRACTION_ENV,
        str(DEFAULT_RAY_PLACEMENT_RESOURCE_FRACTION),
    )
    try:
        fraction = float(raw_fraction)
    except ValueError as exc:
        raise ValueError(
            f"{RAY_PLACEMENT_RESOURCE_FRACTION_ENV} must be a number: "
            f"{raw_fraction!r}"
        ) from exc
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError(
            f"{RAY_PLACEMENT_RESOURCE_FRACTION_ENV} must be finite and > 0: "
            f"{raw_fraction!r}"
        )

    options["resources"] = {resource_name: fraction}
    return options


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
