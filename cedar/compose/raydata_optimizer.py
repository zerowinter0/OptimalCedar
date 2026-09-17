"""A Ray Data-style planner expressed as a Cedar optimizer.

Ray Data executes a data pipeline as a streaming block graph: every
transformation (or a fused run of map transformations) runs in its own stage
on the Ray actor pool, the pool is sized from the cluster's CPU budget, blocks
are bounded by ``target_max_block_size``, and the executor relies on
backpressure between stages instead of reordering the user's graph.

Translated into Cedar's plan space, that policy is:

  * keep the declared operator order (Ray Data does not reorder or cache);
  * fuse each maximal run of mappable operators into one stage, when compatible with Cedar mutation and fusion constraints;
  * give every stage an *equal* share of the per-worker Ray CPU budget --
    this static Cedar adapter approximates Ray Data's dynamic scheduling and
    backpressure with a shared, bounded actor allocation;
  * run those stages on the Ray pool (RAY variant), leaving sources and sinks
    in the worker process.

The planner therefore has no notion of stage boundaries, cross-host transport,
width response, or a worker-count decision, which is exactly what makes it a
useful comparison point for a cost model that prices those terms.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Set, Tuple

from .optimizer import (
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
    PipeVariantType,
)
from .utils import find_all_paths, get_fixed_pipes
from cedar.pipes import PipeExecutionResource, PipeVariantContextFactory


logger = logging.getLogger(__name__)


class RayDataOptimizer(Optimizer):
    """Declared order, map fusion, even actor split over the Ray pool."""

    preserve_optimizer_widths = True

    # ---------------------------------------------------------------- helpers
    def _logical_opt(self) -> None:
        """Ray Data keeps the user's order; only prefetching is retained."""
        if self.options.enable_prefetch:
            logger.info("*Prefetching Pass*")
            self._insert_prefetch()

    def _core_budget(self) -> Tuple[int, int]:
        """(cores per worker, Ray actors available per worker)."""
        ray_budget_raw = os.environ.get("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET")
        local_budget_raw = os.environ.get("CEDAR_PROFILE_MATCH_CPU_BUDGET")
        if local_budget_raw is None:
            budget = int(getattr(self.options, "available_local_cpus", 0) or 0)
        else:
            budget = int(local_budget_raw)
        ray_budget = int(ray_budget_raw) if ray_budget_raw else budget
        fixed_workers = os.environ.get(
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
        )
        if fixed_workers is not None:
            workers = max(1, int(fixed_workers))
        else:
            workers = max(1, int(self.physical_plan.n_local_workers or 1))
        reserve_raw = os.environ.get("CEDAR_DP_RAY_CPU_RESERVE_PER_WORKER", "1")
        try:
            reserve = max(0, int(reserve_raw))
        except ValueError:
            reserve = 1
        cores_per_worker = max(1, budget // workers)
        ray_per_worker = max(0, ray_budget // workers - reserve)
        return cores_per_worker, ray_per_worker

    def _parallelizable(self, p_id: int) -> bool:
        pipe = self.logical_pipes[p_id]
        if pipe.is_source() or pipe.pipe_spec is None:
            return False
        if PipeVariantType.RAY not in pipe.pipe_spec.mutable_variants:
            return False
        return True

    def _stage_groups(self) -> List[List[int]]:
        """Maximal runs of consecutive parallelizable operators (map fusion)."""
        # Fusion must follow the physical chain order: ``_fuse_pipe`` rewires
        # the graph from the first to the last member.
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)
        chains = find_all_paths(self.physical_plan.graph, source_p_id, output_p_id)
        linear_order = chains[0] if len(chains) == 1 else list(self.logical_pipes)
        groups: List[List[int]] = []
        current: List[int] = []
        for p_id in linear_order:
            if p_id not in self.logical_pipes:
                # Sources/sinks such as the lister and the prefetcher are
                # physical-only nodes; they break a fusion run.
                if current:
                    groups.append(current)
                    current = []
                continue
            if self._parallelizable(p_id):
                if self.logical_pipes[p_id].is_fusable(PipeVariantType.RAY):
                    current.append(p_id)
                else:
                    if current:
                        groups.append(current)
                        current = []
                    groups.append([p_id])
                continue
            if current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------- optimizer
    def _physical_opt(self) -> None:
        # A single streaming executor owns the distributed stage pool.
        self.physical_plan.set_local_workers(1)

        cores_per_worker, ray_per_worker = self._core_budget()
        groups = self._stage_groups()
        if not groups:
            logger.info("[RayData] No parallelizable stages; local plan.")
            return
        share = max(1, ray_per_worker // len(groups))
        logger.info(
            "[RayData] cores/worker=%s ray actors/worker=%s stages=%s "
            "even share=%s",
            cores_per_worker,
            ray_per_worker,
            len(groups),
            share,
        )

        staged: Set[int] = set()
        fused_ids: Set[int] = set()
        for group in groups:
            # One CUDA actor owns one model replica and one profiled GPU.
            # Giving the CPU equal-share width to a CUDA group invents dozens
            # of GPU replicas and is rejected by resource matching.
            cuda_group = any(
                self.logical_pipes[p_id].execution_resource
                == PipeExecutionResource.CUDA
                for p_id in group
            )
            stage_width = 1 if cuda_group else share
            context = PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.RAY,
                spec={
                    "n_actors": stage_width,
                    "max_inflight": 100,
                    "max_prefetch": 100,
                    "use_threads": True,
                    "submit_batch_size": 30,
                },
            )
            if len(group) == 1:
                desc = self.physical_plan.pipe_descs[group[0]]
                desc.variant_type = PipeVariantType.RAY
                desc.variant_ctx = context
            else:
                # Map fusion: one stage for the whole run of mappable
                # transformations, exactly as Ray Data's executor fuses them.
                fused_ids.add(self._fuse_pipe(group, PipeVariantType.RAY, context))
            staged.update(group)

        for p_id, desc in self.physical_plan.pipe_descs.items():
            if p_id in staged or p_id in fused_ids:
                continue
            desc.variant_type = PipeVariantType.INPROCESS
            desc.variant_ctx = PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.INPROCESS
            )

        logger.info(
            "[RayData] Allocated plan: %s",
            {
                p_id: (
                    getattr(desc.variant_type, "name", None),
                    getattr(desc.variant_ctx, "n_actors", None),
                )
                for p_id, desc in sorted(self.physical_plan.pipe_descs.items())
            },
        )
