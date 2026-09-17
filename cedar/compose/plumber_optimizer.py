"""A Plumber-style cost model expressed as a Cedar optimizer.

Plumber (MLSys'22) models an input pipeline as a set of stages with measured
per-core rates, allocates one shared pool of cores across those stages, caps
sequential stages at one core, and maximizes the minimum stage rate

    max_{theta}  X = min_i (theta_i * R_i)
    s.t.         sum_i theta_i <= n_c,   theta_i >= 0,
                 theta_i <= 1 for sequential stages.

This class keeps the pipeline's declared operator order and does not reorder,
fuse, cache, or place operators on a second backend: its only decision is the
per-stage width from that analytical model. It therefore isolates what a
single-host, rate-based model can achieve inside Cedar's runtime, which is the
comparison point for a model that prices heterogeneous backends, stage widths,
and cross-backend boundaries.
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
from .utils import get_fixed_pipes
from cedar.pipes import PipeExecutionResource, PipeVariantContextFactory


logger = logging.getLogger(__name__)


class PlumberOptimizer(Optimizer):
    """Per-stage width allocation using Plumber's bottleneck-rate model."""

    # Keep the widths this optimizer selects when Cedar later applies the
    # profile-matched resource signature.
    preserve_optimizer_widths = True

    # ---------------------------------------------------------------- helpers
    def _logical_opt(self) -> None:
        """Plumber does not reorder or cache; only prefetching is retained."""
        if self.options.enable_prefetch:
            logger.info("*Prefetching Pass*")
            self._insert_prefetch()

    def _core_budget(self) -> Tuple[int, int]:
        """Return (cores per local worker, cores available to stage widths)."""
        # The formal protocol fixes the local worker count; use that value so the
        # per-worker core budget matches the resources the plan will actually get.
        fixed_workers = os.environ.get("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS")
        if fixed_workers is not None:
            workers = max(1, int(fixed_workers))
        else:
            workers = max(1, int(self.physical_plan.n_local_workers or 1))
        return self._core_budget_for(workers)

    def _core_budget_for(self, workers: int) -> Tuple[int, int]:
        """Per-worker budget for a candidate local worker count.

        Returns ``(cores_per_worker, parallel_budget)``.  The parallel budget
        is zero when the worker count leaves no core for extra processes, which
        keeps every stage INPROCESS instead of oversubscribing the host.
        """
        budget_raw = os.environ.get("CEDAR_PROFILE_MATCH_CPU_BUDGET")
        if budget_raw is None:
            budget = int(getattr(self.options, "available_local_cpus", 0) or 0)
        else:
            budget = int(budget_raw)
        workers = max(1, int(workers))
        if budget <= 0:
            return 1, 1
        cores_per_worker = max(1, budget // workers)
        # Mirror Cedar's profile-matched accounting exactly: one core for the
        # worker process and one runtime reserve, so the baseline is charged the
        # same per-worker CPU budget as every other optimizer.
        reserve_raw = os.environ.get(
            "CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER", "1"
        )
        try:
            reserve = int(reserve_raw)
        except ValueError:
            reserve = 1
        parallel = max(0, cores_per_worker - 1 - max(0, reserve))
        return cores_per_worker, parallel

    def _stage_effective_rates(
        self,
        budget: int,
        rates: Dict[int, float],
        sequential: Set[int],
    ) -> Tuple[Dict[int, int], float]:
        """Allocation and predicted per-worker pipeline rate (records/s).

        ``theta = 0`` keeps a stage INPROCESS (served by the worker's own core),
        ``theta = k >= 1`` gives it ``k`` dedicated processors.  The pipeline
        rate of one worker is the slowest stage rate, exactly Plumber's
        bottleneck-rate objective.
        """
        if not rates:
            return {}, float("inf")
        allocation = self._allocate_widths(
            budget, rates, sequential, max_width=max(1, budget)
        )
        per_worker_rate = min(
            max(1, allocation.get(p_id, 0)) * rate
            for p_id, rate in rates.items()
        )
        return allocation, per_worker_rate

    def _stage_rates(self) -> Dict[int, float]:
        """Return the measured per-core rate (records/second) of each stage."""
        baseline = self.profiled_stats.get("baseline", {})
        latencies = baseline.get("latencies", {})
        rates: Dict[int, float] = {}
        for p_id in self.logical_pipes:
            raw = latencies.get(p_id, latencies.get(str(p_id)))
            try:
                latency_ns = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(latency_ns) or latency_ns <= 0.0:
                continue
            # Per-core rate in records/second (a stage is limited by its
            # measured per-record latency when it keeps its own worker core).
            rates[p_id] = 1e9 / latency_ns
        return rates

    def _sequential_pipes(self) -> Set[int]:
        """Pipes Plumber would cap at one core.

        Sources and operators without SMP support remain sequential. A fixed
        graph position constrains reordering, not execution parallelism.
        """
        sequential: Set[int] = set()
        for p_id, pipe in self.logical_pipes.items():
            if pipe.is_source() or pipe.pipe_spec is None:
                sequential.add(p_id)
                continue
            if PipeVariantType.SMP not in pipe.pipe_spec.mutable_variants:
                sequential.add(p_id)
        return sequential

    def _allocate_widths(
        self,
        budget: int,
        rates: Dict[int, float],
        sequential: Set[int],
        max_width: int,
    ) -> Dict[int, int]:
        """Maximize the minimum stage rate over a shared core pool.

        Cedar realizes a stage's width as dedicated SMP processes (or Ray
        actors) charged against the per-worker parallel budget, while INPROCESS
        stages are served by the worker's own core at no parallel cost. The
        mapping of Plumber's ``theta_i`` is therefore: ``theta_i = 0`` keeps the
        stage INPROCESS, and ``theta_i = k >= 1`` gives it ``k`` dedicated
        processors with rate ``k * R_i``. Greedy max-min hands one processor at a
        time to the stage with the lowest current rate, which is optimal for
        this objective and stays non-degenerate when a pipeline has more stages
        than cores.
        """
        widths: Dict[int, int] = {p_id: 0 for p_id in rates}
        spend = 0
        while spend < budget:
            candidates = [
                p_id
                for p_id in widths
                if widths[p_id] < (0 if p_id in sequential else max_width)
            ]
            if not candidates:
                break
            target = min(
                candidates,
                key=lambda p_id: max(1, widths[p_id]) * rates[p_id],
            )
            widths[target] += 1
            spend += 1
        return widths

    # ------------------------------------------------------------- optimizer
    def _physical_opt(self) -> None:
        # Plumber allocates one host's stage pool, without replicated drivers.
        self.physical_plan.set_local_workers(1)
        _, parallel_budget = self._core_budget()
        allocation = self._allocate_widths(
            parallel_budget, self._stage_rates(), self._sequential_pipes(),
            max_width=max(1, parallel_budget),
        )

        for p_id, desc in self.physical_plan.pipe_descs.items():
            width = int(allocation.get(p_id, 0))
            # An accelerator-backed operator owns a model instance.  An SMP
            # stage runs its operator in `width` separate processes, so
            # assigning one to a CUDA pipe multiplies the model footprint by
            # the stage width and exhausts the device; the joint DP forbids
            # the same placement, so keep the operator in the worker process
            # and let the width model decide everything else.
            logical_pipe = self.logical_pipes.get(p_id)
            if (
                logical_pipe is not None
                and logical_pipe.execution_resource
                == PipeExecutionResource.CUDA
            ):
                width = 0
            if width >= 1:
                desc.variant_type = PipeVariantType.SMP
                desc.variant_ctx = PipeVariantContextFactory.create_context(
                    variant_type=PipeVariantType.SMP,
                    spec={
                        "n_procs": width,
                        "max_inflight": 10,
                        "max_prefetch": 10,
                        "use_threads": True,
                        "disable_torch_parallelism": True,
                    },
                )
            else:
                desc.variant_type = PipeVariantType.INPROCESS
                desc.variant_ctx = PipeVariantContextFactory.create_context(
                    variant_type=PipeVariantType.INPROCESS
                )

        logger.info(
            "[Plumber] Allocated plan: %s",
            {
                p_id: (
                    desc.variant_type.name,
                    getattr(desc.variant_ctx, "n_procs", None),
                )
                for p_id, desc in sorted(self.physical_plan.pipe_descs.items())
            },
        )
