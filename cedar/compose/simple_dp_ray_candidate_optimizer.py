"""Experimental Simple-DP plan with one fixed backend substitution.

This optimizer is used only by the multimodal motivation experiment.  It
materializes Simple-DP's plan unchanged, then moves the fused Gaussian blur,
color jitter, and horizontal flip stage from INPROCESS to RAY.
"""

import logging
from typing import Any, Dict

from cedar.pipes import (
    PipeExecutionResource,
    PipeVariantContextFactory,
    PipeVariantType,
)

from . import constants
from .optimizer import PhysicalPlan
from .simple_dp_optimizer import SimpleDpOptimizer


logger = logging.getLogger(__name__)

_TARGET_TAGS = frozenset(
    {"gaussian_blur", "color_jitter", "random_flip"}
)


def move_simple_dp_augmentation_fusion_to_ray(
    plan: PhysicalPlan, logical_pipes: Dict[int, Any]
) -> int:
    """Move exactly the blur+jitter+flip fused stage to the Ray CPU backend."""

    matches = []
    for p_id in plan.graph:
        desc = plan.pipe_descs[p_id]
        members = desc.fused_pipes or []
        tags = {
            logical_pipes[member].tag
            for member in members
            if member in logical_pipes
        }
        if tags == _TARGET_TAGS and len(members) == len(_TARGET_TAGS):
            matches.append(p_id)

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one fused blur+jitter+flip stage in the "
            f"Simple-DP plan, found {matches}"
        )

    p_id = matches[0]
    desc = plan.pipe_descs[p_id]
    if desc.variant_type != PipeVariantType.INPROCESS:
        raise RuntimeError(
            "The Simple-DP source stage must be INPROCESS before the fixed "
            f"Ray substitution, got {desc.variant_type}"
        )
    desc.variant_type = PipeVariantType.RAY
    desc.variant_ctx = PipeVariantContextFactory.create_context(
        PipeVariantType.RAY,
        spec={
            "n_actors": 1,
            "max_inflight": 100,
            "max_prefetch": 100,
            "use_threads": True,
            "submit_batch_size": constants.RAY_SUBMIT_BATCH_SIZE,
            "num_gpus": 0.0,
        },
    )
    desc.execution_resource = PipeExecutionResource.CPU
    return p_id


class SimpleDpRayCandidateOptimizer(SimpleDpOptimizer):
    """Simple-DP topology with blur+jitter+flip fixed on Ray."""

    def run(self, profiled_data, options):
        plan = super().run(profiled_data, options)
        changed_pid = move_simple_dp_augmentation_fusion_to_ray(
            plan, self.logical_pipes
        )
        # Reuse the same item-size-aware Ray batching policy as the other DP
        # optimizers. Feature-level resource matching subsequently assigns the
        # shared 64-CPU budget to all materialized Ray stages.
        self._tune_final_ray_stage_contexts()
        logger.info(
            "[SimpleDpRayCandidateOptimizer] Moved fused stage %s to RAY",
            changed_pid,
        )
        return plan


__all__ = [
    "SimpleDpRayCandidateOptimizer",
    "move_simple_dp_augmentation_fusion_to_ray",
]
