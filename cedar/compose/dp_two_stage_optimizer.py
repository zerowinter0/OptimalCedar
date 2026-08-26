import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from cedar.pipes import Pipe

from .dp_optimizer import DpOptimizer
from .my_optimizer import MyOptimizer
from .optimizer import OptimizerOptions, PhysicalPlan, PipeDesc, PipeVariantType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FixedBlockCandidate:
    start: int
    end: int
    variant: PipeVariantType
    cost: float


@dataclass(frozen=True)
class _FixedBackPointer:
    prev_pos: int
    prev_cache_state: int
    block: _FixedBlockCandidate
    cache_after_idx: Optional[int]


class DpTwoStageOptimizer(DpOptimizer):
    """
    Two-stage DP optimizer.

    Stage 1 uses DP only for logical reordering. Stage 2 keeps that order fixed
    and jointly optimizes contiguous fusion blocks, offload variants, and cache
    placement.
    """

    joint_actor_allocation = False

    def _allocate_final_remote_stage_resources(self) -> None:
        MyOptimizer._allocate_final_remote_stage_resources(self)

    def run(
        self, profiled_data: Union[str, Dict[str, Any]], options: OptimizerOptions
    ) -> PhysicalPlan:
        if not self._initialized:
            raise RuntimeError("Must initialize optimizer before running.")

        from yaml import safe_load

        if isinstance(profiled_data, dict):
            self.profiled_stats = profiled_data
        else:
            import pathlib

            path = pathlib.Path(profiled_data)
            try:
                with path.open("r") as f:
                    self.profiled_stats = safe_load(f)
            except Exception as e:
                logger.error("An error occurred %s", e)
                raise RuntimeError(f"Failed to read profiled stats {profiled_data}")

        self.options = options
        self._validate_stats()
        self._init_stats()

        logger.info("[DpTwoStageOptimizer] Running two-stage DP optimization pass...")
        self._logical_opt()
        self._physical_opt()

        if self.options.enable_prefetch:
            logger.info("*Prefetching Pass*")
            self._insert_prefetch()

        try:
            caching_on = self._get_cache_pid(self.physical_plan) is not None
            fused_blocks = [
                list(desc.fused_pipes)
                for desc in self.physical_plan.pipe_descs.values()
                if getattr(desc, "fused_pipes", None)
                and len(getattr(desc, "fused_pipes", [])) > 1
            ]
            fused_pipes = fused_blocks if fused_blocks else None
            optimized_cost = self.calculate_cost(
                self.physical_plan.graph,
                physical_specs=self.physical_plan.pipe_descs,
                fused_pipes=fused_pipes,
                caching_on=caching_on,
                plan=self.physical_plan,
            )
            logger.info(
                "[DpTwoStageOptimizer] Optimized plan cost = %s",
                optimized_cost,
            )
        except Exception as e:
            logger.info(
                "[DpTwoStageOptimizer] Failed to calculate optimized plan cost: %s",
                e,
            )

        self._log_optimized_pipeline(tag="DpTwoStageOptimizer")
        return self.physical_plan

    def _dp_reorder_offload_cache_fusion(
        self,
        inner_ops: List[int],
    ) -> Tuple[List[int], Optional[int]]:
        if not inner_ops:
            return [], None
        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")

        reorder_order = self._dp_reorder_only(inner_ops)
        reordered_inner_ops = [inner_ops[idx] for idx in reorder_order]
        logger.info(
            "[DpTwoStageOptimizer] Stage-1 reorder order: %s",
            reordered_inner_ops,
        )

        # Recompute DP metadata in the fixed reordered order. Stage 2 then uses
        # a prefix DP, so only contiguous blocks in this order may be fused.
        self._prepare_dp_metadata(reordered_inner_ops)
        cache_p_id = self._dp_fixed_order_offload_cache_fusion(reordered_inner_ops)
        return reordered_inner_ops, cache_p_id

    def _dp_reorder_only(self, inner_ops: List[int]) -> List[int]:
        if self._dp_r_prod is None or self._dp_pred_indices is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")

        n = len(inner_ops)
        full_mask = 1 << n
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)
        source_output_size = self.profiled_stats["baseline"]["output_sizes"][
            source_p_id
        ]

        dp = [float("inf")] * full_mask
        prev_mask = [-1] * full_mask
        prev_idx = [-1] * full_mask
        dp[0] = 0.0

        for mask in range(1, full_mask):
            sub = mask
            while sub:
                lsb = sub & -sub
                i = lsb.bit_length() - 1
                prev = mask ^ lsb
                sub ^= lsb

                if inner_ops[i] == output_p_id and mask != full_mask - 1:
                    continue
                if not self._dp_valid_single_last(prev, i):
                    continue

                input_size = source_output_size * self._dp_r_prod[prev]
                add_cost = self._calculate_pipe_cost(inner_ops[i], input_size, None)
                cand = dp[prev] + add_cost
                if cand < dp[mask]:
                    dp[mask] = cand
                    prev_mask[mask] = prev
                    prev_idx[mask] = i

        final_mask = full_mask - 1
        if dp[final_mask] == float("inf"):
            raise RuntimeError("Reorder-only DP failed: no feasible final order.")

        order_rev: List[int] = []
        mask = final_mask
        while mask:
            i = prev_idx[mask]
            if i < 0:
                raise RuntimeError("Reorder-only DP backtracking hit illegal state.")
            order_rev.append(i)
            mask = prev_mask[mask]
        order_rev.reverse()

        logger.info(
            "[DpTwoStageOptimizer] Stage-1 reorder cost (inner ops only): %s",
            dp[final_mask],
        )
        return order_rev

    def _dp_fixed_order_offload_cache_fusion(
        self,
        inner_ops: List[int],
    ) -> Optional[int]:
        if self.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")
        if self._dp_r_prod is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")

        n = len(inner_ops)
        use_cache = bool(self.options.enable_caching)
        use_fusion = bool(self.options.enable_fusion)
        source_p_id = self._get_source_p_id()
        source_output_size = self.profiled_stats["baseline"]["output_sizes"][
            source_p_id
        ]

        variant_costs = self._build_variant_costs(inner_ops)
        variants = list(variant_costs.keys())
        prefix_ratio = [1.0] * (n + 1)
        for i, p_id in enumerate(inner_ops):
            prefix_ratio[i + 1] = prefix_ratio[i] * self._data_size_ratio_map[p_id]

        all_non_random = [True] * (n + 1)
        for i, p_id in enumerate(inner_ops):
            pipe = self.logical_pipes.get(p_id)
            all_non_random[i + 1] = all_non_random[i] and not bool(
                pipe is not None and pipe.is_random()
            )

        candidates: List[List[_FixedBlockCandidate]] = [[] for _ in range(n)]
        for start in range(n):
            for end in range(start + 1, n + 1):
                block_len = end - start
                if block_len > 1 and not use_fusion:
                    break
                for vt in variants:
                    cand = self._make_fixed_block_candidate(
                        inner_ops,
                        variant_costs,
                        source_output_size,
                        start,
                        end,
                        vt,
                    )
                    if cand is not None:
                        candidates[start].append(cand)

        dp = [[float("inf"), float("inf")] for _ in range(n + 1)]
        back: List[List[Optional[_FixedBackPointer]]] = [
            [None, None] for _ in range(n + 1)
        ]
        dp[0][0] = 0.0

        read_time_per_byte = 0.0
        if use_cache:
            read_time_per_byte = self.profiled_stats["disk_info"]["read_latency"]
        cache_cost = read_time_per_byte * 1000 * source_output_size
        cache_benefit = -self._base_cost_map[source_p_id]

        for start in range(n):
            for cache_state in (0, 1):
                if dp[start][cache_state] == float("inf"):
                    continue
                for block in candidates[start]:
                    regular = dp[start][cache_state] + (
                        prefix_ratio[start] * block.cost
                    )
                    if regular < dp[block.end][cache_state]:
                        dp[block.end][cache_state] = regular
                        back[block.end][cache_state] = _FixedBackPointer(
                            prev_pos=start,
                            prev_cache_state=cache_state,
                            block=block,
                            cache_after_idx=None,
                        )

                    if (
                        use_cache
                        and cache_state == 0
                        and block.end == start + 1
                        and all_non_random[block.end]
                    ):
                        cache_candidate = cache_benefit + (
                            cache_cost * prefix_ratio[start]
                        )
                        if cache_candidate < dp[block.end][1]:
                            dp[block.end][1] = cache_candidate
                            back[block.end][1] = _FixedBackPointer(
                                prev_pos=start,
                                prev_cache_state=0,
                                block=block,
                                cache_after_idx=start,
                            )

        final_state = 1 if use_cache and dp[n][1] < dp[n][0] else 0
        if dp[n][final_state] == float("inf"):
            raise RuntimeError("Fixed-order strategy DP failed: no feasible plan.")

        blocks_rev: List[List[int]] = []
        chosen_variant_by_idx: Dict[int, PipeVariantType] = {}
        cache_after_idx: Optional[int] = None
        pos = n
        state = final_state
        while pos > 0:
            pointer = back[pos][state]
            if pointer is None:
                raise RuntimeError("Fixed-order strategy DP backtracking failed.")
            idx_block = list(range(pointer.block.start, pointer.block.end))
            blocks_rev.append(idx_block)
            for idx in idx_block:
                chosen_variant_by_idx[idx] = pointer.block.variant
            if pointer.cache_after_idx is not None:
                cache_after_idx = pointer.cache_after_idx
            pos = pointer.prev_pos
            state = pointer.prev_cache_state

        blocks_rev.reverse()
        self._store_pending_fusions_from_blocks(
            blocks_in_idx_order=blocks_rev,
            inner_ops=inner_ops,
            chosen_variant_by_idx=chosen_variant_by_idx,
        )

        for idx, p_id in enumerate(inner_ops):
            vt = chosen_variant_by_idx.get(idx, PipeVariantType.INPROCESS)
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = vt

        logger.info(
            "[DpTwoStageOptimizer] Stage-2 fixed-order DP cost (inner ops only): %s",
            dp[n][final_state],
        )

        if cache_after_idx is None:
            return None
        return inner_ops[cache_after_idx]

    def _build_variant_costs(
        self, inner_ops: List[int]
    ) -> Dict[PipeVariantType, List[float]]:
        variant_costs: Dict[PipeVariantType, List[float]] = {
            PipeVariantType.INPROCESS: [self._base_cost_map[p_id] for p_id in inner_ops]
        }

        for vt, backend_stats in self._iter_candidate_backend_stats():
            costs_v = [float("inf")] * len(inner_ops)
            for i, p_id in enumerate(inner_ops):
                pipe: Optional[Pipe] = self.logical_pipes.get(p_id)
                if pipe is None or not pipe.can_mutate_to(vt):
                    continue
                if p_id not in backend_stats:
                    continue

                base_input_size = self.profiled_stats["baseline"]["input_sizes"][p_id]
                desc = PipeDesc(name=None, variant_type=vt, variant_ctx=None)
                try:
                    costs_v[i] = self._calculate_pipe_cost(
                        p_id,
                        base_input_size,
                        desc,
                    )
                except Exception:
                    continue

            if any(c != float("inf") for c in costs_v):
                variant_costs[vt] = costs_v
        return variant_costs

    def _make_fixed_block_candidate(
        self,
        inner_ops: List[int],
        variant_costs: Dict[PipeVariantType, List[float]],
        source_output_size: float,
        start: int,
        end: int,
        vt: PipeVariantType,
    ) -> Optional[_FixedBlockCandidate]:
        block_len = end - start
        if block_len > 1:
            for p_id in inner_ops[start:end]:
                if not self._pipe_can_materialize_fusion(p_id, vt):
                    return None

        costs = variant_costs[vt]
        local_ratio = 1.0
        normalized_cost = 0.0
        baseline_io = 1.0
        for pos in range(start, end):
            p_id = inner_ops[pos]
            cost = costs[pos]
            if cost == float("inf"):
                return None

            base_input_size = self.profiled_stats["baseline"]["input_sizes"][p_id]
            if base_input_size <= 0:
                return None
            normalized_cost += local_ratio * cost / base_input_size
            if pos > start:
                baseline_io += 2.0 * local_ratio
            local_ratio *= self._data_size_ratio_map[p_id]

        baseline_io += local_ratio
        fused_io = 1.0 + local_ratio
        io_ratio = fused_io / baseline_io if baseline_io > 0 else 1.0
        block_cost = normalized_cost * io_ratio * source_output_size
        return _FixedBlockCandidate(
            start=start,
            end=end,
            variant=vt,
            cost=block_cost,
        )


__all__ = ["DpTwoStageOptimizer"]
