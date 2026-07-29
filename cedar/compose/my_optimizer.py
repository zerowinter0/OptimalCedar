import logging
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .optimizer import (
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeVariantType,
    PipeDesc,
)
from . import constants
from .utils import (
    find_all_paths,
    derive_constraint_graph,
    get_fixed_pipes,
    flip_adj_list,
)
from cedar.pipes import Pipe, PipeVariantContextFactory


logger = logging.getLogger(__name__)


class MyOptimizer(Optimizer):
    """
    一个完全独立于原 `Optimizer` 重排/缓存/offload 逻辑的优化器实现。

    目标：
    - 不再调用基类的 reorder / offload / fusion / caching 这些「枚举 + 启发式」策略；
    - 使用一个统一的 DP，将 **重排 + per‑op offload + cache** 融入同一个状态机中；
    - 保持接口与整体生命周期不变（`init` → `run`），以方便在现有测试上 A/B。
    """

    def __init__(self) -> None:
        super().__init__()

        # DP 公共元数据（在一次 _physical_opt 中针对 inner_ops 预计算一次）
        self._dp_inner_ops: List[int] = []
        self._dp_ratios: List[float] = []
        self._dp_pred_indices: List[List[int]] = []
        self._dp_costs: List[float] = []
        self._dp_r_prod: List[float] = []
        self._dp_selectivities: List[float] = []
        self._dp_cardinality_prod: List[float] = []
        self._dp_volume_prod: List[float] = []
        self._pending_fusion_blocks: List[List[int]] = []
        self._pending_fusion_variants: Dict[Tuple[int, ...], PipeVariantType] = {}
        self._invalid_cost_warnings: Set[Tuple[int, PipeVariantType]] = set()
        self._transport_rejection_logs: Set[Tuple[Any, ...]] = set()
        self._dp_wall_latency_scale: Optional[float] = None

    #复用父类的init和_init_stats
    def init(self, logical_pipes: Dict[int, Pipe], logical_adj_list: Dict[int, Set[int]]) -> None:
        # 这里**不能**提前调用 `_init_stats()`：
        # profiled_stats 只有在 `run(profiled_data, ...)` 解析 YAML 后才会被赋值。
        # 保持与基类一致：init 只做 logical graph/plan 初始化。
        super().init(logical_pipes, logical_adj_list)

    def _init_stats(self) -> None:
        super()._init_stats()
        wall_latencies = self.profiled_stats["baseline"].get(
            "wall_latencies"
        )
        if not isinstance(wall_latencies, dict):
            return
        # Filter pipelines require the isolated end-to-end offload signal:
        # dropped records do not reach the output profiler, so a wall-clock
        # sum over surviving samples is not an unbiased local cost breakdown.
        # The selectivity-aware outer DP still models their cardinality.
        has_filter = self.logical_pipes is not None and any(
            "filterpipe" in pipe.__class__.__name__.lower()
            or "filterpipe" in pipe.get_logical_name().lower()
            for pipe in self.logical_pipes.values()
        )
        if has_filter:
            logger.info(
                "[MyOptimizer] Retaining end-to-end cost attribution for a "
                "pipeline containing filters."
            )
            return
        expected_ids = set(self._base_cost_map)
        if set(wall_latencies) != expected_ids:
            logger.warning(
                "[MyOptimizer] Ignoring incomplete baseline wall latencies: "
                "expected=%s observed=%s",
                sorted(expected_ids),
                sorted(wall_latencies),
            )
            return
        try:
            wall_ns = {
                p_id: float(wall_latencies[p_id]) for p_id in expected_ids
            }
        except (TypeError, ValueError):
            return
        total_wall_ns = sum(wall_ns.values())
        if not math.isfinite(total_wall_ns) or total_wall_ns <= 0:
            return

        total_cost_ms = 1000.0 / self.profiled_stats["baseline"]["throughput"]
        self._dp_wall_latency_scale = total_cost_ms / (total_wall_ns / 1e6)
        self._fractional_latencies = {
            p_id: latency / total_wall_ns for p_id, latency in wall_ns.items()
        }
        self._base_cost_map = {
            p_id: fraction * total_cost_ms
            for p_id, fraction in self._fractional_latencies.items()
        }
        logger.info(
            "[MyOptimizer] Using baseline wall-clock operator costs "
            "(normalization scale %.6f).",
            self._dp_wall_latency_scale,
        )
    # ========= 外部入口 =========

    def run(
        self, profiled_data: Union[str, Dict[str, Any]], options: OptimizerOptions
    ) -> PhysicalPlan:
        """
        仅重写 run，以便：
        - 复用基类的 profile 解析 / 统计量初始化；
        - 但在逻辑/物理优化阶段完全走自定义实现。
        """
        if not self._initialized:
            raise RuntimeError("Must initialize optimizer before running.")

        # 直接照抄基类的 run 前半段逻辑
        from yaml import safe_load

        if isinstance(profiled_data, dict):
            self.profiled_stats = profiled_data
        else:
            import pathlib

            path = pathlib.Path(profiled_data)
            try:
                with path.open("r") as f:
                    self.profiled_stats = safe_load(f)
            except Exception as e:  # pragma: no cover - 防御性
                logger.error("An error occurred %s", e)
                raise RuntimeError(f"Failed to read profiled stats {profiled_data}")

        self.options = options
        self._validate_stats()
        self._init_stats()

        logger.info("[MyOptimizer] Running custom optimization pass...")
        logger.info("======Using profiled stats========")
        logger.info(self.profiled_stats)

        logger.info("========= Logical Pass (noop) ===========")
        self._logical_opt()

        logger.info("========= Physical Pass (DP) ===========")
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
            if fused_pipes:
                logger.info(
                    "[%s] Found %d fused blocks for calculate_cost fused_pipes=%s",
                    self.__class__.__name__,
                    len(fused_blocks),
                    fused_pipes,
                )
            optimized_cost = self.calculate_cost(
                self.physical_plan.graph,
                physical_specs=self.physical_plan.pipe_descs,
                fused_pipes=fused_pipes,
                caching_on=caching_on,
                plan=self.physical_plan,
            )
            logger.info(
                "[%s] Optimized plan cost (calculate_cost) = %s",
                self.__class__.__name__,
                optimized_cost,
            )
        except Exception as e:
            logger.info(
                "[%s] Failed to calculate optimized plan cost: %s",
                self.__class__.__name__,
                e,
            )

        self._log_optimized_pipeline(tag="MyOptimizer")
        return self.physical_plan

    # ========= 逻辑优化：这里不做任何事 =========

    def _logical_opt(self) -> None:
        """
        逻辑优化阶段留空：所有重排/缓存/offload 统统放到物理阶段的 DP 中做。
        """
        return

    def _smp_only_when_offload_disabled(self) -> bool:
        """
        当用户关闭 enable_offload 时，仍允许走“offload DP”，
        但后端候选仅限 SMP（外加默认 INPROCESS）。
        """
        return self.options is not None and (not self.options.enable_offload)

    def _iter_candidate_backend_stats(self):
        """
        枚举可参与 offload 代价比较的后端统计。
        - enable_offload=True: 使用 profile 中所有合法后端
        - enable_offload=False: 仅允许 SMP
        """
        offloads = self.profiled_stats.get("offloads", {})
        smp_only = self._smp_only_when_offload_disabled()
        for backend_name, backend_stats in offloads.items():
            if backend_name not in PipeVariantType.__members__:
                continue
            vt = PipeVariantType[backend_name]
            if smp_only and vt != PipeVariantType.SMP:
                continue
            yield vt, backend_stats

    def _calculate_pipe_cost(
        self, p_id: int, input_size: float, desc: Optional[PipeDesc]
    ) -> float:
        """Use a conservative fallback for an unidentifiable Amdahl inverse.

        The shared Cedar estimator returns zero when an observed end-to-end
        speedup exceeds the maximum attributable to one profiled operator.
        Such a measurement lies at the singularity of the inverse; treating
        it as a free operator makes the joint DP aggressively combine invalid
        candidates, while falling all the way back to baseline reverses the
        ranking against slower identifiable backends.  Use a finite speedup
        cap so the strong measured signal survives without introducing zero
        costs.
        """
        cost = super()._calculate_pipe_cost(p_id, input_size, desc)
        if desc is None or desc.variant_type in (None, PipeVariantType.INPROCESS):
            return cost
        baseline_input = self.profiled_stats["baseline"]["input_sizes"][p_id]
        baseline_cost = (
            input_size / baseline_input * self._base_cost_map[p_id]
            if baseline_input > 0
            else self._base_cost_map[p_id]
        )
        regularized_cost = max(
            baseline_cost / constants.MAX_UNIDENTIFIABLE_OPERATOR_SPEEDUP,
            1e-12,
        )
        if cost > 0 and math.isfinite(cost):
            # Close to Amdahl's asymptote, even a technically identifiable
            # inverse can imply an arbitrarily large and unstable operator
            # speedup. Apply the same finite cap on both sides of the
            # singularity so backend rankings do not flip discontinuously.
            return max(cost, regularized_cost)
        warning_key = (p_id, desc.variant_type)
        if warning_key not in self._invalid_cost_warnings:
            self._invalid_cost_warnings.add(warning_key)
            logger.warning(
                "[MyOptimizer] Invalid inferred %s cost for pipe %s; "
                "using finite capped-speedup cost (first value %s).",
                desc.variant_type.name,
                p_id,
                regularized_cost,
            )
        return regularized_cost

    def _dp_has_supported_fusion_cost(
        self, variant_type: PipeVariantType
    ) -> bool:
        """Return whether the current profile supports a fused-block estimate.

        The current block model estimates a serialized parallel-stage
        boundary.  Multi-operator INPROCESS fusion has no corresponding
        profile coefficient, so ranking it with the same IO discount is not
        supported.
        """
        if variant_type == PipeVariantType.INPROCESS:
            # Preserve local-only Cedar behavior when no offload search is in
            # scope.  In the joint physical search, do not let an unprofiled
            # INPROCESS discount outrank measured parallel candidates.
            return not self.options.enable_offload
        return variant_type in (
            PipeVariantType.SMP,
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
        )

    def _dp_fusion_transport_feasible(self, prev_mask: int, block) -> bool:
        """Reject SMP blocks whose profiled input rate exceeds Cedar's limit.

        A fused SMP stage still serializes each input from its upstream worker.
        Its aggregate input byte rate is determined by the block placement,
        which is only known during the outer DP transition.  Reuse Cedar's
        existing local-serialization threshold instead of assigning every SMP
        block the same optimistic IO discount.
        """
        if block.variant not in (
            PipeVariantType.SMP,
            PipeVariantType.RAY,
            PipeVariantType.TF_RAY,
        ):
            return True

        # Ray has a measured stage-boundary coefficient, and operators whose
        # Amdahl inversion is not identifiable already receive a conservative
        # in-process compute fallback in the candidate provider.  Requiring an
        # identifiable *exit* cost here would discard otherwise legal large
        # Ray fusion blocks and force many resource-contending singleton
        # stages.  SMP retains the stricter check below because its local
        # serialization feasibility depends directly on the stage exit.
        if block.variant != PipeVariantType.SMP:
            return True

        # The exit operator controls the stage's output service rate.  If its
        # backend Amdahl inversion is outside the identifiable range, a fused
        # block has no defensible parallel throughput estimate.  Interior
        # operators may still use conservative baseline fallbacks because
        # their boundaries are removed by fusion.
        exit_p_id = self._dp_inner_ops[block.order[-1]]
        exit_input = self.profiled_stats["baseline"]["input_sizes"][exit_p_id]
        exit_desc = PipeDesc(
            name=None,
            variant_type=block.variant,
            variant_ctx=None,
        )
        raw_exit_cost = super()._calculate_pipe_cost(
            exit_p_id, exit_input, exit_desc
        )
        if raw_exit_cost <= 0 or not math.isfinite(raw_exit_cost):
            log_key = (
                "unidentifiable_exit",
                block.variant,
                exit_p_id,
            )
            if log_key not in self._transport_rejection_logs:
                self._transport_rejection_logs.add(log_key)
                logger.info(
                    "[MyOptimizer] Rejecting %s fusion block %s: exit pipe %s "
                    "has no identifiable backend cost.",
                    block.variant.name,
                    block.order,
                    exit_p_id,
                )
            return False

        source_size = self.profiled_stats["baseline"]["output_sizes"][
            self._get_source_p_id()
        ]
        modeled_input_size = source_size * self._dp_r_prod[prev_mask]
        first_p_id = self._dp_inner_ops[block.order[0]]
        profiled_input_size = self.profiled_stats["baseline"]["input_sizes"][
            first_p_id
        ]
        # Multiplicative size ratios can under-predict fixed-shape transforms
        # after reordering (for example, a resize has nearly constant output
        # bytes).  A transport feasibility check must not be optimistic, so
        # retain the profiled input of the block's first operator as a lower
        # bound until profiles expose an explicit size function.
        input_size = max(modeled_input_size, profiled_input_size)
        next_mask = prev_mask | block.mask
        modeled_output_size = source_size * self._dp_r_prod[next_mask]
        profiled_output_size = self.profiled_stats["baseline"]["output_sizes"][
            exit_p_id
        ]
        output_size = max(modeled_output_size, profiled_output_size)
        if max(input_size, output_size) > constants.SMP_MAX_SERIALIZED_SAMPLE_SIZE:
            log_key = ("item_size", first_p_id, exit_p_id)
            if log_key not in self._transport_rejection_logs:
                self._transport_rejection_logs.add(log_key)
                logger.info(
                    "[MyOptimizer] Rejecting SMP fusion block %s (first "
                    "placement mask %s): serialized boundary input=%.3f MB "
                    "output=%.3f MB exceeds the %.3f MB per-item limit.",
                    block.order,
                    prev_mask,
                    input_size / 1e6,
                    output_size / 1e6,
                    constants.SMP_MAX_SERIALIZED_SAMPLE_SIZE / 1e6,
                )
            return False
        fixed_workers = os.environ.get("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS")
        if fixed_workers is not None:
            try:
                workers = max(1, int(fixed_workers))
            except ValueError:
                workers = max(1, self.physical_plan.n_local_workers)
        else:
            workers = max(1, self.physical_plan.n_local_workers)
        profile_throughput = self.profiled_stats["baseline"]["throughput"]
        aggregate_input_rate = input_size * profile_throughput * workers
        transport_limit = self._dp_boundary_throughput(PipeVariantType.SMP)
        if transport_limit is None:
            transport_limit = constants.LOCAL_PARALLELISM_THRESHOLD
        feasible = aggregate_input_rate <= transport_limit
        if not feasible:
            log_key = ("aggregate_rate", first_p_id)
            if log_key not in self._transport_rejection_logs:
                self._transport_rejection_logs.add(log_key)
                logger.info(
                    "[MyOptimizer] Rejecting SMP fusion block %s (first "
                    "placement mask %s): estimated input rate %.3f MB/s "
                    "exceeds %.3f MB/s.",
                    block.order,
                    prev_mask,
                    aggregate_input_rate / 1e6,
                    transport_limit / 1e6,
                )
        return feasible

    def _dp_boundary_profile(
        self, variant: PipeVariantType
    ) -> Optional[Dict[str, Any]]:
        physical_model = self.profiled_stats.get("physical_model", {})
        boundaries = (
            physical_model.get("boundary", {})
            if isinstance(physical_model, dict)
            else {}
        )
        if not isinstance(boundaries, dict):
            return None
        key = (
            PipeVariantType.RAY.name
            if variant == PipeVariantType.TF_RAY
            else variant.name
        )
        model = boundaries.get(key)
        return model if isinstance(model, dict) else None

    def _dp_boundary_throughput(self, variant: PipeVariantType) -> Optional[float]:
        model = self._dp_boundary_profile(variant)
        if model is not None:
            try:
                throughput = float(model["throughput_bytes_per_sec"])
            except (KeyError, TypeError, ValueError):
                throughput = float("nan")
            if math.isfinite(throughput) and throughput > 0:
                return throughput
        if variant == PipeVariantType.SMP:
            return constants.LOCAL_PARALLELISM_THRESHOLD
        if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            return constants.RAY_STAGE_BOUNDARY_THROUGHPUT
        return None

    def _dp_boundary_fixed_latency_ms(
        self, variant: PipeVariantType
    ) -> float:
        model = self._dp_boundary_profile(variant)
        if model is None:
            return 0.0
        try:
            latency = float(model.get("fixed_latency_ms", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return latency if math.isfinite(latency) and latency >= 0 else 0.0

    def _dp_boundary_cost_ms(
        self,
        variant: PipeVariantType,
        input_size: float,
        output_size: float,
    ) -> float:
        """Return steady-state boundary service time per sample.

        Boundary calibration is deliberately synchronous, while Cedar keeps
        many stage requests in flight.  Charging its fitted round-trip latency
        once per sample makes latency the throughput bottleneck even though it
        is overlapped in the actual pipeline.  Bandwidth remains a per-sample
        service cost; amortize only the fixed latency by the profiled in-flight
        width.
        """
        throughput = self._dp_boundary_throughput(variant)
        if throughput is None:
            return 0.0
        config = self.profiled_stats.get("resource_config", {})
        key = (
            "ray_max_inflight"
            if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
            else "smp_max_inflight"
        )
        try:
            max_inflight = int(
                config.get(key, constants.PROFILE_STAGE_MAX_INFLIGHT)
            )
        except (TypeError, ValueError):
            max_inflight = constants.PROFILE_STAGE_MAX_INFLIGHT
        max_inflight = max(1, max_inflight)
        return self._dp_boundary_fixed_latency_ms(variant) / max_inflight + (
            (input_size + output_size) / throughput * 1000.0
        )

    def _dp_profiled_operator_compute_cost(
        self,
        p_id: int,
        variant: PipeVariantType,
        profiled_total_cost: float,
    ) -> float:
        """Split a profiled offload cost into compute and stage-boundary work.

        ``profiled_total_cost`` is inferred from the measured end-to-end
        throughput after changing exactly this operator's backend.  It is the
        only observation that includes integration effects on the surrounding
        pipeline, so a singleton DP stage must reproduce that observation:
        compute = profiled total - boundary.  Worker-side timings are useful
        diagnostics, but are not directly comparable with Cedar's local costs,
        which are apportioned from cumulative baseline latencies.  Preferring
        the worker timer would therefore mix two measurement domains and can
        reverse an observed end-to-end speedup.

        Direct backend timing remains a fallback for profiles without a usable
        boundary model.  Its one-sided 95% confidence bound avoids selecting a
        backend merely because a short profile happened to be optimistic.
        """
        profile = self.profiled_stats.get("offloads", {}).get(
            variant.name, {}
        ).get(p_id, {})
        direct = profile.get("backend_compute")

        direct_upper_bound: Optional[float] = None
        if isinstance(direct, dict):
            try:
                mean = float(direct["mean_ms_per_sample"])
                stderr = float(direct.get("stderr_ms_per_sample", 0.0))
                count = int(direct["count"])
            except (KeyError, TypeError, ValueError):
                mean = float("nan")
                stderr = float("nan")
                count = 0
            if (
                count > 0
                and math.isfinite(mean)
                and mean >= 0
                and math.isfinite(stderr)
                and stderr >= 0
            ):
                direct_upper_bound = mean + 1.645 * stderr

        # New profiles provide local and backend compute in the same
        # wall-clock domain. Prefer the direct worker observation: unlike an
        # Amdahl inverse, it remains identifiable near the whole-pipeline
        # speedup asymptote and composes additively inside a fused block.
        if (
            self._dp_wall_latency_scale is not None
            and direct_upper_bound is not None
        ):
            return direct_upper_bound * self._dp_wall_latency_scale

        throughput = self._dp_boundary_throughput(variant)
        baseline = self.profiled_stats.get("baseline", {})
        input_sizes = baseline.get("input_sizes", {})
        output_sizes = baseline.get("output_sizes", {})
        if (
            throughput is not None
            and p_id in input_sizes
            and p_id in output_sizes
        ):
            input_size = input_sizes[p_id]
            output_size = output_sizes[p_id]
            boundary_cost = self._dp_boundary_cost_ms(
                variant, input_size, output_size
            )
            compute_cost = profiled_total_cost - boundary_cost
            if math.isfinite(compute_cost):
                return max(0.0, compute_cost)

        if direct_upper_bound is not None:
            return direct_upper_bound

        return profiled_total_cost

    def _dp_stage_boundary_cost(self, prev_mask: int, block) -> float:
        """Model each parallel stage boundary separately from operator work.

        The term is placement-dependent and therefore belongs in the outer DP
        transition rather than the per-mask candidate provider.  A fused block
        pays one input and one output boundary, while every operator's compute
        cost remains undiscounted.
        """
        throughput = self._dp_boundary_throughput(block.variant)
        if throughput is None:
            return 0.0

        source_size = self.profiled_stats["baseline"]["output_sizes"][
            self._get_source_p_id()
        ]
        next_mask = prev_mask | block.mask
        input_size = source_size * self._dp_work_prod(prev_mask)
        output_size = source_size * self._dp_work_prod(next_mask)
        return self._dp_boundary_cost_ms(
            block.variant, input_size, output_size
        )

    def _dp_work_prod(self, mask: int) -> float:
        """Aggregate byte-volume multiplier for one source record.

        Old profiles have no filter selectivity and retain the exact historical
        size-only product. New profiles multiply per-item size by surviving
        record cardinality.
        """
        volume_prod = getattr(self, "_dp_volume_prod", [])
        if len(volume_prod) == len(self._dp_r_prod):
            return volume_prod[mask]
        return self._dp_r_prod[mask]

    def _allowed_fusion(self, p_id: int):
        # Tensor conversion changes representation and has backend-specific
        # storage semantics; keep it as an explicit stage. Match both Cedar's
        # class-style ``ToTensor`` and function-style ``to_tensor`` names.
        logical_name = self.logical_pipes[p_id].get_logical_name().lower()
        if "totensor" in logical_name or "to_tensor" in logical_name:
            logger.info(f"Forbidding fusion {p_id} due to ToTensor")
            return False

        return True

    def _pipe_can_materialize_fusion(
        self, p_id: int, variant_type: PipeVariantType
    ) -> bool:
        pipe = self.logical_pipes.get(p_id) if self.logical_pipes is not None else None
        return bool(
            pipe is not None
            and self._allowed_fusion(p_id)
            and pipe.is_fusable(variant_type)
        )

    def _clear_pending_fusions(self) -> None:
        self._pending_fusion_blocks = []
        self._pending_fusion_variants = {}

    def _store_pending_fusions_from_blocks(
        self,
        blocks_in_idx_order: List[List[int]],
        inner_ops: List[int],
        chosen_variant_by_idx: Optional[Dict[int, PipeVariantType]] = None,
    ) -> None:
        """
        根据 DP 回溯得到的块序列，记录待 materialize 的 fusion 信息。
        这里只记录长度>=2 的块，单算子块无需生成 FusedPipe。
        """
        self._clear_pending_fusions()
        for idx_block in blocks_in_idx_order:
            if len(idx_block) < 2:
                continue
            pid_block = [inner_ops[i] for i in idx_block]
            self._pending_fusion_blocks.append(pid_block)
            if chosen_variant_by_idx is None:
                vt = PipeVariantType.INPROCESS
            else:
                vt = chosen_variant_by_idx.get(idx_block[0], PipeVariantType.INPROCESS)
            self._pending_fusion_variants[tuple(pid_block)] = vt

    def _materialize_pending_fusions(self) -> None:
        """
        将已记录的 fusion 块写入物理图，生成 FusedPipe。
        若块在重排/cache后不再是连续单链，则跳过该块。
        """
        if not self._pending_fusion_blocks:
            return

        input_graph = flip_adj_list(self.physical_plan.graph)
        for pid_block in self._pending_fusion_blocks:
            if len(pid_block) < 2:
                continue

            # 必须所有节点还在图中，且形成单链
            if any(p not in self.physical_plan.graph for p in pid_block):
                logger.warning(
                    "[MyOptimizer] Skip fusion block %s because some pipes are missing in graph.",
                    pid_block,
                )
                continue

            block_ok = True
            for i in range(len(pid_block) - 1):
                u = pid_block[i]
                v = pid_block[i + 1]
                if self.physical_plan.graph.get(u) != {v}:
                    block_ok = False
                    break
            if block_ok:
                first = pid_block[0]
                if len(input_graph.get(first, set())) > 1:
                    block_ok = False

            if not block_ok:
                logger.warning(
                    "[MyOptimizer] Skip fusion block %s because it is no longer a linear chain.",
                    pid_block,
                )
                continue

            vt = self._pending_fusion_variants.get(
                tuple(pid_block), PipeVariantType.INPROCESS
            )
            if any(
                not self._pipe_can_materialize_fusion(p_id, vt)
                for p_id in pid_block
            ):
                logger.warning(
                    "[MyOptimizer] Skip fusion block %s for variant %s because at least one pipe is not fusable.",
                    pid_block,
                    vt,
                )
                continue
            variant_ctx = PipeVariantContextFactory.create_context(variant_type=vt)
            self._fuse_pipe(pid_block, vt, variant_ctx)

            # 图变化后刷新一次反向图，供后续块检查使用
            input_graph = flip_adj_list(self.physical_plan.graph)

    # ========= DP 所需的一些基础工具 =========

    def _get_linear_inner_ops(self) -> Optional[List[int]]:
        """
        提取当前物理图中 source -> ... -> output 的线性路径中，除去 source 后的算子列表（**包含 output**）。

        若不是线性单路径图，则返回 None。
        """
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)

        all_paths = find_all_paths(self.physical_plan.graph, source_p_id, output_p_id)
        if len(all_paths) != 1:
            # 暂时只支持单路径上的重排
            return None

        path = all_paths[0]
        if len(path) <= 1:
            return []

        # NOTE:
        # - 基类 Optimizer 会把 pipeline 的最后一个逻辑算子（output 节点）也纳入 offload/fusion 等优化；
        # - MyOptimizer 之前用 path[1:-1] 把 output 排除掉，导致最后一位算子完全不参与任何优化。
        # 这里改为包含 output（path[1:]），并在 DP 约束里强制 output 必须排在最后，从而做到
        # “参与所有优化，但不被重排到中间”。
        return path[1:]

    def _compute_best_per_op_costs(
        self, ops: List[int]
    ) -> Tuple[Dict[int, float], Dict[int, PipeVariantType]]:
        """
        计算每个算子最适合的variant（不考虑融合）

        对每个算子 i：
        - 枚举 INPROCESS 以及 `profiled_stats["offloads"]` 中存在、且算子可变换到的
          backend（如 RAY / SMP / TF_RAY 等）；
        - 通过 `_calculate_pipe_cost` 在 baseline 输入 size 下估算 cost；
        - 选出全局最小 cost 以及对应 variant_type。

        返回：
        - best_costs:  p_id -> 最小 cost
        - best_variant: p_id -> 选择的 PipeVariantType（若为 INPROCESS，则为 PipeVariantType.INPROCESS）
        """
        if self.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")

        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")

        best_costs: Dict[int, float] = {}
        best_variant: Dict[int, PipeVariantType] = {}

        for p_id in ops:
            if (
                "baseline" not in self.profiled_stats
                or "input_sizes" not in self.profiled_stats["baseline"]
                or p_id not in self.profiled_stats["baseline"]["input_sizes"]
            ):
                raise RuntimeError(
                    f"Missing baseline input size for pipe {p_id}, cannot compute cost."
                )

            base_input_size = self.profiled_stats["baseline"]["input_sizes"][p_id]

            # 先假设 INPROCESS
            best = self._calculate_pipe_cost(p_id, base_input_size, None)
            best_vt = PipeVariantType.INPROCESS

            # 再遍历所有允许的 backend（可按配置限制为 SMP-only）
            for vt, backend_stats in self._iter_candidate_backend_stats():

                # 该算子是否支持变为该 variant
                lp: Pipe = self.logical_pipes.get(p_id)  # type: ignore[assignment]
                if lp is None or not lp.can_mutate_to(vt):
                    continue

                if p_id not in backend_stats:
                    continue

                desc = PipeDesc(
                    name=None,
                    variant_type=vt,
                    variant_ctx=None,
                )
                try:
                    cost_v = self._calculate_pipe_cost(p_id, base_input_size, desc)
                except Exception:
                    # profile 不完整或者算子不支持该 backend，直接跳过
                    continue

                if cost_v < best:
                    best = cost_v
                    best_vt = vt

            best_costs[p_id] = best
            best_variant[p_id] = best_vt

        return best_costs, best_variant

    def _prepare_dp_metadata(self, inner_ops: List[int]) -> None:
        """
        基于当前 inner_ops 预计算：
        - self._dp_inner_ops
        - self._dp_ratios
        - self._dp_pred_indices：仅考虑 inner_ops 内部的依赖约束
        """
        if self._data_size_ratio_map is None:
            raise RuntimeError("Data size ratio map is not initialized.")
        if self.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")

        self._dp_inner_ops = inner_ops

        self._read_time_per_byte = self.profiled_stats["disk_info"][
            "read_latency"
        ]

        # 1) ratios
        ratios: List[float] = []
        for p_id in inner_ops:
            if p_id not in self._data_size_ratio_map:
                raise RuntimeError(
                    f"Missing data size ratio for pipe {p_id}, cannot run DP."
                )
            ratio = self._data_size_ratio_map[p_id]
            if ratio is None:
                ratio = 1.0
            ratios.append(ratio)
        self._dp_ratios = ratios

        # 0) costs: 使用 _base_cost_map 直接得到 reorder/base cost 数组（offload/cache/fusion 变体可在此基础上替换）
        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")
        self._dp_costs = [self._base_cost_map[p_id] for p_id in inner_ops]

        # 3) 预计算 P[S]：r_prod[mask] = ∏_{i∈mask} ratio[i]
        source_p_id = self._get_source_p_id()
        if self.profiled_stats["baseline"]["input_sizes"][source_p_id] <= 0:
            self.profiled_stats["baseline"]["input_sizes"][source_p_id]=self.profiled_stats["baseline"]["output_sizes"][source_p_id]
        n = len(inner_ops)
        full_mask = 1 << n
        r_prod = [1.0] * full_mask
        for mask in range(1, full_mask):
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            prev = mask ^ lsb
            r_prod[mask] = r_prod[prev] * ratios[i]
        self._dp_r_prod = r_prod

        # Cardinality is distinct from serialized size per surviving item.
        # Profiles predating schema v1 have no selection counts and therefore
        # preserve the original model exactly with selectivity 1.0.
        raw_selectivities = self.profiled_stats["baseline"].get(
            "selectivities", {}
        )
        selectivities: List[float] = []
        for p_id in inner_ops:
            raw = raw_selectivities.get(
                p_id, raw_selectivities.get(str(p_id), 1.0)
            )
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid filter selectivity for pipe {p_id}: {raw!r}"
                ) from exc
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise RuntimeError(
                    f"Filter selectivity for pipe {p_id} must be in [0, 1], "
                    f"got {value!r}"
                )
            selectivities.append(value)
        self._dp_selectivities = selectivities

        cardinality_prod = [1.0] * full_mask
        volume_prod = [1.0] * full_mask
        for mask in range(1, full_mask):
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            prev = mask ^ lsb
            cardinality_prod[mask] = (
                cardinality_prod[prev] * selectivities[i]
            )
            volume_prod[mask] = (
                r_prod[mask] * cardinality_prod[mask]
            )
        self._dp_cardinality_prod = cardinality_prod
        self._dp_volume_prod = volume_prod

        # 2) inner_ops 内部依赖：edge u -> v 表示 v 依赖 u，u 必须在 v 之前
        constraint_graph = derive_constraint_graph(self.logical_pipes)
        idx_of: Dict[int, int] = {p_id: idx for idx, p_id in enumerate(inner_ops)}
        n = len(inner_ops)
        pred_indices: List[List[int]] = [[] for _ in range(n)]
        for u, succs in constraint_graph.items():
            if u not in idx_of:
                continue
            u_idx = idx_of[u]
            for v in succs:
                if v in idx_of:
                    v_idx = idx_of[v]
                    pred_indices[v_idx].append(u_idx)

        # 固定算子约束：对 `.fix()` 的算子，禁止其与其他算子交换位置。
        #
        # 语义：fixed pipe 的位置不可移动，等价于：
        # - 它依赖于线性路径中位于它之前的所有节点
        # - 并且线性路径中位于它之后的所有节点都依赖它
        #
        # 这样仍允许“非 fixed”算子在 fixed 片段之间做重排，但不会跨越 fixed 边界。
        fixed_pipes = get_fixed_pipes(self.logical_pipes)
        fixed_in_inner = [p_id for p_id in inner_ops if p_id in fixed_pipes]
        if fixed_in_inner:
            # Determine the current linear execution order on the physical graph.
            # NOTE:
            # Cedar 的 `physical_plan.graph` 在本项目里是「出边邻接表」：
            #   graph[u] = {v} 表示 u -> v（u 的输出流向 v）。
            # 因此不能通过“从 output 往回走”来重建顺序；这里直接用
            # source->...->output 的唯一路径作为当前执行顺序。
            graph = self.physical_plan.graph
            try:
                output_p_id = self._get_output_p_id(graph)
                paths = find_all_paths(graph, source_p_id, output_p_id)
                exec_order = paths[0] if len(paths) == 1 else inner_ops
            except Exception:
                exec_order = inner_ops

            order_idx: Dict[int, int] = {p_id: i for i, p_id in enumerate(exec_order)}

            for f_id in fixed_in_inner:
                f_i = idx_of[f_id]
                f_pos = order_idx.get(f_id, f_i)
                for p_id in inner_ops:
                    if p_id == f_id:
                        continue
                    p_i = idx_of[p_id]
                    p_pos = order_idx.get(p_id, p_i)
                    if p_pos < f_pos:
                        # p_id must precede fixed
                        if p_i not in pred_indices[f_i]:
                            pred_indices[f_i].append(p_i)
                    else:
                        # fixed must precede p_id
                        if f_i not in pred_indices[p_i]:
                            pred_indices[p_i].append(f_i)

        self._dp_pred_indices = pred_indices

    # ========= Fusion DP 公共辅助函数（依赖 _dp_pred_indices） =========

    def _dp_valid_single_last(self, prev_mask: int, i: int) -> bool:
        """
        若 i 作为当前集合的最后一个加入，则 i 的所有前驱必须都在 prev_mask 中。
        """
        if self._dp_pred_indices is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")
        pred_indices = self._dp_pred_indices
        for p_idx in pred_indices[i]:
            if not (prev_mask & (1 << p_idx)):
                return False
        return True

    def _dp_topo_order_in_mask(self, mask: int) -> List[int]:
        """
        返回 mask 诱导子图的一个拓扑序；若无环不存在完整拓扑序则返回 []。
        """
        if self._dp_pred_indices is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")
        pred_indices = self._dp_pred_indices

        if mask == 0:
            return []

        nodes: List[int] = []
        sub = mask
        while sub:
            lsb = sub & -sub
            nodes.append(lsb.bit_length() - 1)
            sub ^= lsb

        in_deg: Dict[int, int] = {i: 0 for i in nodes}
        for j in nodes:
            for p_idx in pred_indices[j]:
                if p_idx in in_deg:
                    in_deg[j] += 1

        q: deque[int] = deque(sorted([i for i in nodes if in_deg[i] == 0]))
        order: List[int] = []

        while q:
            u = q.popleft()
            order.append(u)
            for v in nodes:
                if u in pred_indices[v]:
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        q.append(v)

        return order if len(order) == len(nodes) else []

    def _dp_fusion_block_valid(self, t_mask: int, c_mask: int) -> bool:
        """
        C 接在 T 之后是否可行：
        - C 内每个结点的外部前驱只能来自 T（不能来自 block 外部）
        - C 的诱导子图必须无环（存在完整拓扑序）
        """
        if self._dp_pred_indices is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")
        pred_indices = self._dp_pred_indices

        sub_c = c_mask
        while sub_c:
            lsb = sub_c & -sub_c
            j = lsb.bit_length() - 1
            for p_idx in pred_indices[j]:
                if not (c_mask & (1 << p_idx)) and not (t_mask & (1 << p_idx)):
                    return False
            sub_c ^= lsb

        return len(self._dp_topo_order_in_mask(c_mask)) > 0

    @staticmethod
    def _prune_reorder_frontier(
        states: List[tuple[float, float, int, int]],
    ) -> List[tuple[float, float, int, int]]:
        """Keep exact Pareto states that can minimize an IO ratio."""
        if len(states) <= 1:
            return states

        # Lower cost and larger IO dominate. Visit the largest IO first for an
        # equal cost so the other equal-cost states are discarded immediately.
        pareto: List[tuple[float, float, int, int]] = []
        best_io = float("-inf")
        for state in sorted(states, key=lambda x: (x[0], -x[1])):
            if state[1] > best_io:
                pareto.append(state)
                best_io = state[1]

        # Only the lower convex hull in (IO, cost) can minimize
        # cost - lambda * IO for lambda >= 0. Collinear interior points cannot
        # be uniquely optimal and are safe to remove as well.
        hull: List[tuple[float, float, int, int]] = []
        for state in pareto:
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                cross = (b[1] - a[1]) * (state[0] - a[0]) - (
                    b[0] - a[0]
                ) * (state[1] - a[1])
                if cross > 0.0:
                    break
                hull.pop()
            hull.append(state)
        return hull

    def _dp_naive_reorder_cost_per_variant(
        self,
        inner_ops: List[int],
        n: int,
        candidate_variants: List[PipeVariantType],
        variant_costs: Dict[PipeVariantType, List[float]],
    ) -> Dict[PipeVariantType, List[float]]:
        """
        对每个候选 variant 做与 `_dp_reorder_only`（见 tmp.txt）相同的朴素 DP：
        仅重排，不计 cache / fusion / offload 的额外项。

        公式：DP[S] = min_i ( DP[S\\{i}] + P[S\\{i}] * Cost_v[i] )，其中 Cost_v 为该 variant 下各槽位代价。

        返回：`variant -> dp`，`dp[mask]` 表示在该 variant 固定时，子集 `mask` 对应算子集合的最优调度代价。
        可与 `_dp_reorder_offload_cache_fusion` 中已构造的 `n`、`candidate_variants`、`variant_costs` 直接对接。
        """
        if self._dp_pred_indices is None or self._dp_r_prod is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")
        pred_indices = self._dp_pred_indices
        r_prod = self._dp_r_prod
        logger.debug(
            "[MyOptimizer] Prepared r_prod for %d DP masks; first values=%s",
            len(r_prod),
            r_prod[: min(8, len(r_prod))],
        )

        full_mask = 1 << n
        out: Dict[PipeVariantType, List[float]] = {}
        self.fusion_track: Dict[PipeVariantType, List[int]] = {}

        # 用于在外层（fusion_cost 预计算）重建每个 mask 的最优顺序。
        # 每个 variant 完成后立即物化最佳顺序，不跨 variant 保留庞大 frontier。
        #
        # 说明：
        # - cost 仍按原 DP 的“baseline compute cost”累计：prev_cost + r_prod[prev] * cost_i / base_input_size_i
        # - io_base_partial 为 baseline IO 中“除最后一次输出写入”之外的部分：
        #   first_input + 2*sum_{non-first ops} input_size(op)
        #   最终 baseline IO = io_base_partial + r_prod[mask]
        # - fused IO 只记 first_input + final_output = 1 + r_prod[mask]
        self._naive_reorder_best_orders: Dict[
            PipeVariantType, List[tuple[int, ...]]
        ] = {}

        for vt in candidate_variants:
            costs_v = variant_costs.get(vt)
            if costs_v is None or len(costs_v) != n:
                raise ValueError(
                    f"variant_costs[{vt}] must be a list of length n={n}"
                )

            base_input_sizes = [
                float(self.profiled_stats["baseline"]["input_sizes"][p_id])
                for p_id in inner_ops
            ]

            # A single frontier per mask is sufficient: future legality and
            # transition costs depend on the mask, not on the previous last op.
            # State: (cost, io_base_partial, appended_op, prev_state_idx).
            frontiers: List[List[tuple[float, float, int, int]]] = [
                [] for _ in range(full_mask)
            ]
            best_end: List[int] = [-1 for _ in range(full_mask)]
            retained_states = 0
            max_frontier = 0
            progress_interval = max(1, (full_mask - 1) // 10)

            for mask in range(1, full_mask):
                candidates: List[tuple[float, float, int, int]] = []
                for i in range(n):
                    lsb = 1 << i
                    if not (mask & lsb):
                        continue
                    if costs_v[i] == float("inf"):
                        continue
                    prev = mask ^ lsb
                    if prev == 0:
                        denom = base_input_sizes[i]
                        if denom > 0:
                            candidates.append(
                                (costs_v[i] / denom, 1.0, i, -1)
                            )
                        continue

                    # 合法性：i 作为 last，需要保证 prev 内不存在依赖 i 的结点（即 i 不能是它们的前驱）
                    valid = True
                    sub_prev = prev
                    while sub_prev:
                        lsb_j = sub_prev & -sub_prev
                        j = lsb_j.bit_length() - 1
                        if i in pred_indices[j]:
                            valid = False
                            break
                        sub_prev ^= lsb_j
                    if not valid:
                        continue

                    denom = base_input_sizes[i]
                    if denom <= 0:
                        continue
                    add_cost = r_prod[prev] * costs_v[i] / denom
                    add_io_base = 2.0 * r_prod[prev]  # baseline: non-first op 的 input 记两次

                    for prev_idx, st in enumerate(frontiers[prev]):
                        prev_cost, prev_io_base, _, _ = st
                        candidates.append(
                            (
                                prev_cost + add_cost,
                                prev_io_base + add_io_base,
                                i,
                                prev_idx,
                            )
                        )

                if candidates:
                    frontiers[mask] = self._prune_reorder_frontier(candidates)
                    retained_states += len(frontiers[mask])
                    max_frontier = max(max_frontier, len(frontiers[mask]))

                # 计算这个 mask 的最终最优（考虑 io_ratio）
                best_cost = float("inf")
                fused_io = 1.0 + r_prod[mask]
                for idx, st in enumerate(frontiers[mask]):
                    cost, io_base_partial, _, _ = st
                    baseline_io = io_base_partial + r_prod[mask]
                    if baseline_io <= 0:
                        continue
                    final_cost = cost * (fused_io / baseline_io)
                    if final_cost < best_cost:
                        best_cost = final_cost
                        best_end[mask] = idx

                if mask % progress_interval == 0 or mask == full_mask - 1:
                    logger.info(
                        "[MyOptimizer] Reorder frontier variant=%s progress=%d/%d retained=%d max_frontier=%d",
                        vt,
                        mask,
                        full_mask - 1,
                        retained_states,
                        max_frontier,
                    )

            # out[vt][mask] = 已把 io_ratio 融合进来的“block baseline cost”（不含 output_sizes[source_p_id]）
            dp_final = [float("inf")] * full_mask
            dp_final[0] = 0.0
            best_orders: List[tuple[int, ...]] = [tuple() for _ in range(full_mask)]
            for mask in range(1, full_mask):
                idx = best_end[mask]
                if idx < 0:
                    continue
                cost, io_base_partial, _, _ = frontiers[mask][idx]
                fused_io = 1.0 + r_prod[mask]
                baseline_io = io_base_partial + r_prod[mask]
                if baseline_io <= 0:
                    continue
                dp_final[mask] = cost * (fused_io / baseline_io)

                order_rev: List[int] = []
                cur_mask = mask
                cur_idx = idx
                while cur_mask:
                    _, _, last, prev_idx = frontiers[cur_mask][cur_idx]
                    order_rev.append(last)
                    cur_mask ^= 1 << last
                    if cur_mask:
                        if prev_idx < 0:
                            raise RuntimeError(
                                "Reorder frontier backtracking hit illegal state."
                            )
                        cur_idx = prev_idx
                order_rev.reverse()
                best_orders[mask] = tuple(order_rev)

            out[vt] = dp_final
            self._naive_reorder_best_orders[vt] = best_orders

        return out

    def _reconstruct_naive_reorder_order(self, vt: PipeVariantType, mask: int) -> List[int]:
        """
        返回 `_dp_naive_reorder_cost_per_variant` 已物化的 mask 最优顺序。
        返回的顺序元素是 0..n-1 的“inner_ops 槽位索引”（与外层的 ratios/costs_v 对齐）。
        """
        best_orders = getattr(self, "_naive_reorder_best_orders", {}).get(vt)
        if best_orders is None:
            return []
        if mask < 0 or mask >= len(best_orders):
            return []
        return list(best_orders[mask])

    # ========= 统一 DP：重排 + offload/cache/fusion 的 8 种组合 =========

    def _dp_reorder_offload_cache_fusion(
        self,
        inner_ops: List[int],
    ) -> Tuple[List[int], Optional[int]]:
        """
        返回：
        - best_order: 重排后的中间算子顺序
        - cache_p_id: 若存在收益更好的 cache 位置，则返回该算子 ID，否则为 None
        """
        if not inner_ops:
            return [], None

        if self.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")
        if self._dp_ratios is None or self._dp_pred_indices is None or self._dp_costs is None:
            raise RuntimeError("DP metadata not prepared (call _prepare_dp_metadata).")
        if self._base_cost_map is None:
            raise RuntimeError("Base cost map is not initialized.")

        use_cache = self.options.enable_caching
        use_fusion = self.options.enable_fusion

        n = len(inner_ops)
        ratios = self._dp_ratios
        pred_indices = self._dp_pred_indices
        base_costs = self._dp_costs
        full_mask = 1 << n

        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)

        # 1) cache 可开启条件：cache 之前必须全部非随机
        is_random_flags: List[bool] = []
        for p_id in inner_ops:
            op_pipe = self.logical_pipes.get(p_id)
            is_random_flags.append(bool(op_pipe is not None and op_pipe.is_random()))
        fusion_allowed_flags = [self._allowed_fusion(p_id) for p_id in inner_ops]

        all_non_random = [False] * full_mask
        all_non_random[0] = True
        for mask in range(1, full_mask):
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            prev = mask ^ lsb
            all_non_random[mask] = all_non_random[prev] and (not is_random_flags[i])
        all_fusion_allowed = [False] * full_mask
        all_fusion_allowed[0] = True
        for mask in range(1, full_mask):
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            prev = mask ^ lsb
            all_fusion_allowed[mask] = all_fusion_allowed[prev] and fusion_allowed_flags[i]

        # 2) 依赖合法性工具
        def _valid_single_last(prev: int, i: int) -> bool:
            return self._dp_valid_single_last(prev, i)

        def _topo_order_in_mask(mask: int) -> List[int]:
            return self._dp_topo_order_in_mask(mask)

        # 3) 变体 cost：每个算子可在不同 backend 下运行
        variant_costs: Dict[PipeVariantType, List[float]] = {
            PipeVariantType.INPROCESS: base_costs[:]
        }
        

        candidate_variants: List[PipeVariantType] = [PipeVariantType.INPROCESS]
        for vt, backend_stats in self._iter_candidate_backend_stats():

            costs_v = [float("inf")] * n
            for i, p_id in enumerate(inner_ops):
                lp: Pipe = self.logical_pipes.get(p_id)  # type: ignore[assignment]
                if lp is None or not lp.can_mutate_to(vt):
                    continue
                if p_id not in backend_stats:
                    continue

                base_input_size = self.profiled_stats["baseline"]["input_sizes"][p_id]
                desc = PipeDesc(name=None, variant_type=vt, variant_ctx=None)
                try:
                    costs_v[i] = self._calculate_pipe_cost(p_id, base_input_size, desc)
                except Exception:
                    continue

            if any(c != float("inf") for c in costs_v):
                variant_costs[vt] = costs_v
                candidate_variants.append(vt)
        
        variant_final_costs=self._dp_naive_reorder_cost_per_variant(inner_ops, n, candidate_variants, variant_costs)

        # 4) 预计算 P[S]（已在 _prepare_dp_metadata 里完成）
        r_prod = self._dp_r_prod

        # 5) 预计算 fusion_cost[S] 以及最佳 variant（fusion 在 cache 后才会用到，但单点也需要 offload）
        fusion_cost: List[float] = [float("inf")] * full_mask
        best_variant_for_mask: List[PipeVariantType] = [
            PipeVariantType.INPROCESS
        ] * full_mask
        topo_order_cache: List[List[int]] = [[] for _ in range(full_mask)]
        fusion_cost[0] = 0.0

        for mask in range(1, full_mask):
            if (not use_fusion) and mask.bit_count() > 1:
                continue
            if mask.bit_count() > 1 and not all_fusion_allowed[mask]:
                continue

            for vt in candidate_variants:
                if mask.bit_count() > 1 and not self._dp_has_supported_fusion_cost(vt):
                    continue
                dp_by_mask = variant_final_costs[vt]
                if mask.bit_count() > 1 and any(
                    not self._pipe_can_materialize_fusion(inner_ops[i], vt)
                    for i in range(n)
                    if mask & (1 << i)
                ):
                    continue

                block_baseline = dp_by_mask[mask]
                if block_baseline == float("inf"):
                    continue

                # 这里的 block_baseline 已经在 _dp_naive_reorder_cost_per_variant 里把 io_ratio 融入进去了；
                # 仅需乘以 source 的 output_size 做尺度还原。
                c = (
                    block_baseline
                    * self.profiled_stats["baseline"]["output_sizes"][source_p_id]
                )
                if c < fusion_cost[mask]:
                    fusion_cost[mask] = c
                    best_variant_for_mask[mask] = vt
                    # 用 Pareto frontier 的 backpointer 重建该 mask 的最优顺序（用于后续回溯/可视化/调试）
                    topo_order_cache[mask] = self._reconstruct_naive_reorder_order(vt, mask)
                    logger.debug(
                        "[MyOptimizer] Fusion candidate mask=%s order=%s variant=%s cost=%s",
                        mask,
                        topo_order_cache[mask],
                        vt,
                        c,
                    )

        def _fusion_block_valid(t_mask: int, c_mask: int) -> bool:
            return self._dp_fusion_block_valid(t_mask, c_mask)

        # 6) 2D DP：dp[mask][0] 不用 cache；dp[mask][1] 已开启 cache
        dp = [[float("inf"), float("inf")] for _ in range(full_mask)]
        prev_mask = [[-1, -1] for _ in range(full_mask)]
        taken_mask = [[0, 0] for _ in range(full_mask)]
        prev_flag = [[0, 0] for _ in range(full_mask)]  # 回溯时的上一维度 flag

        dp[0][0] = 0.0
        cache_cost = self._read_time_per_byte * 1000 * self.profiled_stats["baseline"]["output_sizes"][source_p_id]
        logger.debug("[MyOptimizer] Cache read latency: %s", self._read_time_per_byte)

        for mask in range(1, full_mask):
            # 6.1 单算子追加（两种维度都可以）
            sub = mask
            while sub:
                lsb = sub & -sub
                i = lsb.bit_length() - 1
                t = mask ^ lsb

                if _valid_single_last(t, i) and fusion_cost[lsb] < float("inf"):
                    # dp[mask][0]：无 cache，且本近似不做 fusion ****important****
                    cand0 = dp[t][0] + r_prod[t] * fusion_cost[lsb] \
                        # * self.profiled_stats["baseline"]["output_sizes"][source_p_id] \
                            #/ self.profiled_stats["baseline"]["input_sizes"][inner_ops[i]]
                    if cand0 < dp[mask][0]:
                        dp[mask][0] = cand0
                        prev_mask[mask][0] = t
                        taken_mask[mask][0] = lsb
                        prev_flag[mask][0] = 0

                    if use_cache:
                        # dp[mask][1]：cache 已开启后的延续，或在 i 位置新开启 cache
                        cand1_continue = dp[t][1] + r_prod[t] * fusion_cost[lsb] \
                        # * self.profiled_stats["baseline"]["output_sizes"][source_p_id] 
                        best_cand1 = cand1_continue
                        best_prev_flag = 1
                        if all_non_random[mask]:
                            # 新开启 cache：忽略 cache 之前的算子代价；只计 cache 读开销
                            # cache 位于当前算子之后，因此按当前 mask 的输出尺寸计费。
                            # DP 状态不包含 source cost，不应再减去 source cost。
                            cand1_new_cache = cache_cost * r_prod[mask]
                            logger.debug(
                                "[MyOptimizer] Cache transition cost=%s r_prod=%s",
                                cache_cost,
                                r_prod[mask],
                            )
                            if cand1_new_cache < best_cand1:
                                best_cand1 = cand1_new_cache
                                best_prev_flag = 0

                        if best_cand1 < dp[mask][1]:
                            dp[mask][1] = best_cand1
                            prev_mask[mask][1] = t
                            taken_mask[mask][1] = lsb
                            prev_flag[mask][1] = best_prev_flag

                sub ^= lsb

            # 6.2 fusion 块追加（无 cache 维度）
            if use_fusion:
                t = mask
                # NOTE: 需要枚举到 t==0（这样 c==mask 的“把所有算子作为同一 fusion 块”
                # 方案才不会被遗漏）。
                while True:
                    c = mask ^ t
                    if (
                        c != 0
                        and (c & (c - 1)) != 0
                        and all_fusion_allowed[c]
                        and _fusion_block_valid(t, c)
                    ):
                        if dp[t][0] < float("inf") and fusion_cost[c] < float("inf"):
                            cand = dp[t][0] + r_prod[t] * fusion_cost[c] \
                        # * self.profiled_stats["baseline"]["output_sizes"][source_p_id] 
                            if cand < dp[mask][0]:
                                dp[mask][0] = cand
                                prev_mask[mask][0] = t
                                taken_mask[mask][0] = c
                                prev_flag[mask][0] = 0
                        if (
                            use_cache
                            and dp[t][0] < float("inf")
                            and all_non_random[mask]
                            and fusion_cost[c] < float("inf")
                        ):
                            # Open cache after this fused block. The physical
                            # insertion uses the block's last real pipe; after
                            # fusion materialization the cache sits after the
                            # resulting FusedPipe.
                            cache_candidate = cache_cost * r_prod[mask]
                            if cache_candidate <= dp[mask][1]:
                                dp[mask][1] = cache_candidate
                                prev_mask[mask][1] = t
                                taken_mask[mask][1] = c
                                prev_flag[mask][1] = 0
                    if t == 0:
                        break
                    t = (t - 1) & mask

            # 6.3 fusion 块追加（cache 已开启维度）
            if use_cache and use_fusion:
                t = mask
                # 同理：确保枚举到 t==0。
                while True:
                    c = mask ^ t
                    if (
                        c != 0
                        and (c & (c - 1)) != 0
                        and all_fusion_allowed[c]
                        and _fusion_block_valid(t, c)
                    ):
                        if dp[t][1] < float("inf") and fusion_cost[c] < float("inf"):
                            cand = dp[t][1] + r_prod[t] * fusion_cost[c] \
                        # * self.profiled_stats["baseline"]["output_sizes"][source_p_id] 
                            if cand < dp[mask][1]:
                                dp[mask][1] = cand
                                prev_mask[mask][1] = t
                                taken_mask[mask][1] = c
                                prev_flag[mask][1] = 1
                    if t == 0:
                        break
                    t = (t - 1) & mask

        # 7) 决定最终是否使用 cache
        full_state = full_mask - 1
        use_cache_flag = 1 if (use_cache and dp[full_state][1] < dp[full_state][0]) else 0
        if prev_mask[full_state][use_cache_flag] == -1 and full_state != 0:
            raise RuntimeError("Offload+Cache+Fusion DP failed: cannot reconstruct best order.")

        # 8) 回溯：展开顺序（块内按 topo 展开），同时写回 offload variant
        blocks_rev: List[List[int]] = []
        chosen_variant_by_idx: Dict[int, PipeVariantType] = {}
        cache_p_id = None
        m = full_state
        flag = use_cache_flag
        while m:
            pm = prev_mask[m][flag]
            tk = taken_mask[m][flag]
            pf = prev_flag[m][flag]

            if pm == -1 or tk == 0:
                raise RuntimeError("Offload+Cache+Fusion DP backtracking hit illegal state.")

            indices_in_block = topo_order_cache[tk]
            if not indices_in_block:
                raise RuntimeError("Offload+Cache+Fusion block has no valid topo order.")

            if flag == 1 and pf == 0:
                cache_p_id = inner_ops[indices_in_block[-1]]

            vt = best_variant_for_mask[tk]
            for idx in indices_in_block:
                chosen_variant_by_idx[idx] = vt

            blocks_rev.append(indices_in_block)
            m = pm
            flag = pf

        blocks_rev.reverse()
        flat: List[int] = []
        for blk in blocks_rev:
            flat.extend(blk)
        best_order = [inner_ops[i] for i in flat]
        self._store_pending_fusions_from_blocks(
            blocks_in_idx_order=blocks_rev,
            inner_ops=inner_ops,
            chosen_variant_by_idx=chosen_variant_by_idx,
        )

        # 写回 inner ops 的 offload variant（cache 前后的 cost 忽略不会影响我们仍需给出一个可执行 variant）
        for i, p_id in enumerate(inner_ops):
            vt = chosen_variant_by_idx.get(i, PipeVariantType.INPROCESS)
            if p_id in self.physical_plan.pipe_descs:
                self.physical_plan.pipe_descs[p_id].variant_type = vt

        dp_inner_cost = dp[full_state][use_cache_flag]
        logger.info("[MyOptimizer] DP state cost (inner ops only): %s", dp_inner_cost)

        # Hard-calculate an "aligned" full critical-path cost without calling
        # Optimizer.calculate_cost.
        #
        # This matches the cost walk logic for the (currently common) case where
        # no materialized fusion nodes / cache pipe are present at this moment.
        # `best_order` 现在包含 output（最后一位算子），不要再重复拼接。
        critical_path = [source_p_id] + best_order

        curr_size = 1
        # dp_inner_cost 已包含 inner_ops（含 output）的代价；这里只需要加上 source 的代价。
        full_path_cost = self._base_cost_map[critical_path[0]] + dp_inner_cost

        # # Walk from source to output; apply per-pipe cost with the current curr_size.
        # for p_id in critical_path[1:]:
        #     if(p_id == output_p_id):
        #         full_path_cost += curr_size*self._base_cost_map[p_id]
        #     curr_size = curr_size * self._data_size_ratio_map[p_id]

        logger.info(
            "[MyOptimizer] Full path cost (hard-walk, no calculate_cost): %s %s",
            full_path_cost,dp_inner_cost,
        )

        return best_order, cache_p_id
        

    

    # ========= 物理优化：应用 DP 结果到物理计划 =========

    def _apply_reorder_and_cache(
        self,
        new_inner_order: List[int],
        cache_p_id: Optional[int],
    ) -> None:
        """
        根据 DP 的输出：
        - 将线性路径中的中间算子重排为 `new_inner_order`；
        - 若 cache_p_id 不为 None，则在该算子后面插入缓存算子（近似实现，规则参考原 Optimizer）。
        """
        source_p_id = self._get_source_p_id()

        # 1. 重建线性 graph
        # `new_inner_order` 已包含 output（最后一位算子）
        new_path = [source_p_id] + new_inner_order
        new_graph: Dict[int, Set[int]] = {}
        for p_id in self.physical_plan.graph.keys():
            new_graph[p_id] = set()
        for u, v in zip(new_path[:-1], new_path[1:]):
            new_graph[u].add(v)

        self.physical_plan.graph = new_graph

        # 2. 近似插入 cache：只在「可插且有收益」的情况下，调用基类的 cache 插入逻辑
        if cache_p_id is not None and self.options.enable_caching:
            logger.info("[MyOptimizer] Decided to insert cache after pipe %s", cache_p_id)
            # 这里我们直接复用基类的缓存插入工具，避免重复造轮子
            # 思路：调用 _calculate_caching_plans + _find_optimal_caching_plan，
            #       但强制只考虑在 cache_p_id 后插入 cache 的方案。
            try:
                self._insert_cache_at(cache_p_id)
            except Exception as e:  # pragma: no cover - 防御性兜底
                logger.warning(
                    "[MyOptimizer] Failed to insert cache at %s, fallback to no cache. Error: %s",
                    cache_p_id,
                    e,
                )

    def _insert_cache_at(self, pipe_id_before_cache: int) -> None:
        """
        参考原 Optimizer._calculate_caching_plans 的做法，在指定算子后插入 cache。
        - 只构造一个包含单一 cache 位置的 PhysicalPlan；
        - 使用 `calculate_cost(..., caching_on=True)` 估算是否优于「无 cache」；
        - 若优于，则替换当前 physical_plan。
        """
        from cedar.pipes.optimize import OptimizerPipeRegistry  # 延迟导入，避免循环

        # 1. 基线计划成本（无 cache）
        base_cost = self.calculate_cost(self.physical_plan.graph, caching_on=False, plan=self.physical_plan)

        # 2. 构造含 cache 的 plan 副本
        cache_pipe_cls = OptimizerPipeRegistry.get_pipe("ObjectDiskCachePipe")
        cache_pipe = cache_pipe_cls()
        cache_p_id = self._get_new_p_id()

        cache_pipe_desc = PipeDesc(
            name=cache_pipe.get_logical_name(),
            variant_type=PipeVariantType.INPROCESS,
            variant_ctx=None,
        )

        plan = PhysicalPlan(
            graph={k: v.copy() for k, v in self.physical_plan.graph.items()},
            pipe_descs=self.physical_plan.pipe_descs.copy(),
            n_local_workers=self.physical_plan.n_local_workers,
        )
        plan.pipe_descs[cache_p_id] = cache_pipe_desc

        # 插在 pipe_id_before_cache 后面
        if pipe_id_before_cache not in plan.graph:
            raise RuntimeError(f"Pipe {pipe_id_before_cache} is not in current graph.")

        next_nodes = plan.graph[pipe_id_before_cache]
        plan.graph[pipe_id_before_cache] = {cache_p_id}
        plan.graph[cache_p_id] = next_nodes.copy()

        # 3. 计算带 cache 的成本
        cached_cost = self.calculate_cost(plan.graph, caching_on=True, plan=plan)

        if cached_cost < base_cost:
            logger.info(
                "[MyOptimizer] Cache at %s is beneficial: base=%.4f, cached=%.4f",
                pipe_id_before_cache,
                base_cost,
                cached_cost,
            )
            self.physical_plan = plan
        else:
            logger.info(
                "[MyOptimizer] Cache at %s not beneficial: base=%.4f, cached=%.4f",
                pipe_id_before_cache,
                base_cost,
                cached_cost,
            )

    # ========= 最终物理优化入口 =========

    def _physical_opt(self) -> None:
        """
        使用一个统一 DP 完成：
        - 中间算子重排；
        - 每个算子选择自己最优的 offload backend；
        - 近似地决定是否插入 cache 以及插在谁后面。

        然后：
        - 不再调用基类的 offload / fusion / caching 逻辑；
        - 仅复用基类的本地并行度调优（local parallelism），因为该部分与顺序无关。
        """
        # 1. 抽取线性路径上的中间算子
        inner_ops = self._get_linear_inner_ops()
        if inner_ops is None:
            logger.info(
                "[MyOptimizer] Graph is not linear, falling back to base Optimizer._physical_opt."
            )
            # 对复杂图暂时沿用原物理优化，以保证行为可预期
            return super()._physical_opt()

        if len(inner_ops) == 0:
            logger.info(
                "[MyOptimizer] No inner ops to reorder, delegating to base Optimizer._physical_opt."
            )
            return super()._physical_opt()

        use_offload = self.options.enable_offload
        use_cache = self.options.enable_caching
        use_fusion = self.options.enable_fusion
        use_reorder = self.options.enable_reorder

        # 2.1 预计算与 inner_ops 相关的 DP 公共元数据（ratio / 依赖关系）
        self._prepare_dp_metadata(inner_ops)

        logger.info(
            "[MyOptimizer] Physical DP flags (offload=%s, cache=%s, fusion=%s)",
            use_offload,
            use_cache,
            use_fusion,
        )
        if not use_offload:
            logger.info(
                "[MyOptimizer] enable_offload is False: using offload-DP with backend candidates {INPROCESS, SMP}."
            )

        # 3. 按 8 种组合调用对应的 DP 函数
        #    若显式关闭 reorder，则保持原顺序（作为 drop-in 替换更安全）
        best_order: List[int]
        cache_p_id: Optional[int] = None
        self._clear_pending_fusions()

        if not use_reorder:
            best_order = inner_ops
            _, best_variant = self._compute_best_per_op_costs(inner_ops)
            for p_id in inner_ops:
                if p_id in self.physical_plan.pipe_descs and p_id in best_variant:
                    self.physical_plan.pipe_descs[p_id].variant_type = best_variant[p_id]
        else:
            best_order, cache_p_id = self._dp_reorder_offload_cache_fusion(inner_ops)

        logger.info("[MyOptimizer] Best inner order (DP): %s", best_order)
        if cache_p_id is not None:
            logger.info("[MyOptimizer] DP suggests inserting cache after pipe %s", cache_p_id)
        if use_fusion:
            if self._pending_fusion_blocks:
                logger.info(
                    "[MyOptimizer] DP selected %d fusion block(s).",
                    len(self._pending_fusion_blocks),
                )
                for idx, block in enumerate(self._pending_fusion_blocks, start=1):
                    block_variant = self._pending_fusion_variants.get(tuple(block))
                    logger.info(
                        "[MyOptimizer] Fusion block %d: ops=%s, variant=%s",
                        idx,
                        block,
                        block_variant,
                    )
            else:
                logger.info(
                    "[MyOptimizer] Fusion enabled but DP produced no multi-op fusion block."
                )

        # 4. 应用顺序与 cache 到物理图
        self._apply_reorder_and_cache(best_order, cache_p_id)

        # 4.1 若 fusion 开启，则将 DP 选中的 fusion 块 materialize 为 FusedPipe
        if use_fusion:
            self._materialize_pending_fusions()

        # 5. 仅复用基类的 local parallelism 调优逻辑
        if self.options.enable_local_parallelism:
            num_local_workers = self._calculate_local_parallelism(
                self.physical_plan.graph, self.options
            )
            logger.info(
                "[MyOptimizer] Using %d local workers (from base parallelism logic).",
                num_local_workers,
            )
            self.physical_plan.set_local_workers(num_local_workers)

        # 6. 为每个算子补齐 physical plan 所需字段：
        # - variant_type：若 DP 未设置则回退到 (TF / INPROCESS)
        # - variant_ctx：必须设置，否则 PhysicalPlan.validate() 会失败
        if self.logical_pipes is None:
            raise RuntimeError("logical_pipes is not initialized.")

        for p_id, desc in self.physical_plan.pipe_descs.items():
            if desc.variant_type is None:
                if p_id in self.logical_pipes and self.logical_pipes[p_id].is_tf():
                    desc.variant_type = PipeVariantType.TF
                else:
                    desc.variant_type = PipeVariantType.INPROCESS

            # 为 TF 设置 num_parallel_calls 语义与基类一致
            if desc.variant_type == PipeVariantType.TF:
                spec = {
                    "num_parallel_calls": (
                        -1 if self.physical_plan.n_local_workers == 1 else None
                    )
                }
                desc.variant_ctx = PipeVariantContextFactory.create_context(
                    variant_type=PipeVariantType.TF, spec=spec
                )
            else:
                spec = None
                if desc.variant_type in (
                    PipeVariantType.RAY,
                    PipeVariantType.TF_RAY,
                ):
                    # Match Optimizer._offload_and_fuse.  Constructing a Ray
                    # context with its generic defaults leaves only ten
                    # requests in flight for batch-size-one, which starves
                    # large-item stages even when the DP selected exactly the
                    # same physical topology as Cedar's staged optimizer.
                    spec = {
                        "n_actors": 1,
                        "max_inflight": 100,
                        "max_prefetch": 100,
                        "use_threads": True,
                        "submit_batch_size": constants.RAY_SUBMIT_BATCH_SIZE,
                    }
                desc.variant_ctx = PipeVariantContextFactory.create_context(
                    variant_type=desc.variant_type,
                    spec=spec,
                )

        # Match Cedar's shared Ray actor budget across all offloaded stages.
        ray_contexts = [
            self.physical_plan.pipe_descs[p_id].variant_ctx
            for p_id in self.physical_plan.graph
            if self.physical_plan.pipe_descs[p_id].variant_type
            in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
        ]
        if ray_contexts:
            n_actors = math.ceil(
                constants.RAY_AVAILABLE_PARALLELISM
                / (len(ray_contexts) * self.physical_plan.n_local_workers)
            )
            for ctx in ray_contexts:
                ctx.n_actors = n_actors

        # The DP path deliberately bypasses Optimizer._offload_and_fuse(),
        # which normally tunes these queueing parameters after materializing a
        # Ray stage.  Keep the DP's plan choices unchanged, but apply the same
        # execution-parameter rule to its final active Ray stages.
        self._tune_final_ray_stage_contexts()

    def _tune_final_ray_stage_contexts(self) -> None:
        """Apply Cedar's Ray submit-batch rule to the final physical plan.

        This is intentionally a post-processing step: it neither changes the
        selected variants nor the reorder/fusion structure produced by DP.
        """
        input_size_map, output_size_map = self._calculate_size_map(
            self.physical_plan.graph
        )

        for p_id in self.physical_plan.graph:
            desc = self.physical_plan.pipe_descs[p_id]
            if desc.variant_type not in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            ):
                continue

            if desc.is_fused_pipe():
                # The original member ids are no longer graph nodes after
                # fusion, so calculate their final output from the fused
                # stage's physical input size.
                input_size = input_size_map[p_id]
                output_size = input_size
                for fused_p_id in desc.fused_pipes:
                    output_size *= self._data_size_ratio_map[fused_p_id]
            else:
                input_size = input_size_map[p_id]
                output_size = output_size_map[p_id]

            total_io_size = input_size + output_size
            submit_batch_size = min(
                max(
                    int(
                        constants.RAY_SUBMIT_BATCH_SCALING_FACTOR
                        // total_io_size
                    ),
                    1,
                ),
                500,
            )
            desc.variant_ctx.set_submit_batch_size(submit_batch_size)
            logger.info(
                "[MyOptimizer] Tuned final Ray stage %s: input=%s, output=%s, "
                "submit_batch_size=%s, max_inflight=%s, max_prefetch=%s",
                p_id,
                input_size,
                output_size,
                desc.variant_ctx.submit_batch_size,
                desc.variant_ctx.max_inflight,
                desc.variant_ctx.max_prefetch,
            )


__all__ = ["MyOptimizer"]
