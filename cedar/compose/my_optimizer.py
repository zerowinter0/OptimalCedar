import logging
import math
import os
import statistics
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
from cedar.pipes import (
    Pipe,
    PipeComputeScaling,
    PipeExecutionResource,
    PipeVariantContextFactory,
)


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
        self._dp_compute_scalings: List[PipeComputeScaling] = []
        self._dp_profiled_input_cardinality: Dict[int, float] = {}
        self._dp_compute_scaling = PipeComputeScaling.PER_DATA
        self._pending_fusion_blocks: List[List[int]] = []
        self._pending_fusion_variants: Dict[Tuple[int, ...], PipeVariantType] = {}
        self._invalid_cost_warnings: Set[Tuple[int, PipeVariantType]] = set()
        self._transport_rejection_logs: Set[Tuple[Any, ...]] = set()

    #复用父类的init和_init_stats
    def init(self, logical_pipes: Dict[int, Pipe], logical_adj_list: Dict[int, Set[int]]) -> None:
        # 这里**不能**提前调用 `_init_stats()`：
        # profiled_stats 只有在 `run(profiled_data, ...)` 解析 YAML 后才会被赋值。
        # 保持与基类一致：init 只做 logical graph/plan 初始化。
        super().init(logical_pipes, logical_adj_list)

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
        """Conservatively combine worker and end-to-end backend measurements.

        Direct worker timings isolate compute but omit IPC, queueing and
        serialization.  A valid Amdahl inverse includes those runtime costs,
        but can become unidentifiable near its singularity.  Use the larger
        of the direct-compute sample mean and a valid end-to-end inverse;
        if only one is valid, retain it.  When neither is identifiable, fall
        back to the measured in-process cost rather than inventing speedup.
        """
        if desc is None or desc.variant_type in (
            None,
            PipeVariantType.INPROCESS,
        ):
            baseline_cost = super()._calculate_pipe_cost(
                p_id, input_size, desc
            )
            # Formal execution replicates every local INPROCESS operator in
            # W worker processes. Worker-side SMP timing at width W executes
            # the same Python callable in W independent processes and excludes
            # IPC, making it a direct contention measurement for local
            # compute. Use it conservatively when the adaptive run converged.
            width_cost = self._dp_profiled_width_compute_cost(
                p_id, PipeVariantType.SMP, input_size
            )
            if width_cost is not None:
                return max(baseline_cost, width_cost)
            return baseline_cost
        baseline_input = self.profiled_stats["baseline"]["input_sizes"][p_id]
        baseline_cost = (
            input_size / baseline_input * self._base_cost_map[p_id]
            if baseline_input > 0
            else self._base_cost_map[p_id]
        )
        backend_entry = self.profiled_stats.get("offloads", {}).get(
            desc.variant_type.name, {}
        ).get(p_id)
        if backend_entry is None:
            backend_entry = self.profiled_stats.get("offloads", {}).get(
                desc.variant_type.name, {}
            ).get(str(p_id))
        direct = (
            backend_entry.get("backend_compute")
            if isinstance(backend_entry, dict)
            else None
        )
        direct_cost: Optional[float] = None
        if isinstance(direct, dict):
            try:
                mean = float(direct["mean_ms_per_sample"])
                stderr = float(direct.get("stderr_ms_per_sample", 0.0))
                count = int(direct.get("count", 0))
            except (KeyError, TypeError, ValueError):
                mean = float("nan")
                stderr = float("nan")
                count = 0
            if (
                count > 0
                and math.isfinite(mean)
                and mean >= 0.0
                and math.isfinite(stderr)
                and stderr >= 0.0
            ):
                # The objective estimates expected execution time, so use the
                # unbiased sample mean. Profiled stderr remains available for
                # accuracy/error-bar reporting; adding a per-operator 95% UCB
                # here systematically rejects useful offloads when only a few
                # actor batches were observed.
                direct_cost = mean
                scaling = self._dp_compute_scaling_for_pipe(p_id)
                if scaling == PipeComputeScaling.PER_RECORD:
                    # Worker timing is per processed record. Convert it to the
                    # profiled per-source-record total expected by
                    # _BlockCostIndex; the index then removes this original
                    # cardinality before applying the reordered cardinality.
                    direct_cost *= self._dp_profiled_input_cardinality.get(
                        p_id, 1.0
                    )
                elif baseline_input > 0:
                    direct_cost *= input_size / baseline_input
                direct_cost = max(direct_cost, 1e-12)

        width_cost = self._dp_profiled_width_compute_cost(
            p_id, desc.variant_type, input_size
        )
        if width_cost is not None:
            direct_cost = (
                width_cost
                if direct_cost is None
                else max(direct_cost, width_cost)
            )

        inferred_cost = super()._calculate_pipe_cost(p_id, input_size, desc)
        inferred_valid = inferred_cost > 0 and math.isfinite(inferred_cost)
        if direct_cost is not None and inferred_valid:
            return max(direct_cost, inferred_cost)
        if direct_cost is not None:
            return direct_cost
        if inferred_valid:
            return inferred_cost
        warning_key = (p_id, desc.variant_type)
        if warning_key not in self._invalid_cost_warnings:
            self._invalid_cost_warnings.add(warning_key)
            logger.warning(
                "[MyOptimizer] Invalid inferred %s cost for pipe %s; "
                "using conservative INPROCESS cost (first value %s).",
                desc.variant_type.name,
                p_id,
                baseline_cost,
            )
        return max(baseline_cost, 1e-12)

    def _dp_profiled_width_compute_cost(
        self,
        p_id: int,
        variant_type: PipeVariantType,
        input_size: float,
    ) -> Optional[float]:
        """Return converged worker compute measured under W-way contention.

        The isolated layer measures one worker accurately, while
        ``physical_model.scaling`` repeats selected expensive operators at the
        formal width. Only converged width>1 measurements are ranking data;
        incomplete max-duration observations remain diagnostics.
        """
        key = (
            PipeVariantType.RAY.name
            if variant_type == PipeVariantType.TF_RAY
            else variant_type.name
        )
        entries = (
            self.profiled_stats.get("physical_model", {})
            .get("scaling", {})
            .get(key, {})
        )
        if not isinstance(entries, dict):
            return None
        entry = entries.get(p_id, entries.get(str(p_id)))
        target_width = max(
            2,
            int(
                os.environ.get(
                    "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", "8"
                )
            ),
        )
        mean = self._dp_scaling_mean(entry, target_width)
        if mean is None:
            return None

        baseline_input = float(
            self.profiled_stats["baseline"]["input_sizes"][p_id]
        )
        cost = mean
        scaling = self._dp_compute_scaling_for_pipe(p_id)
        if scaling == PipeComputeScaling.PER_RECORD:
            cost *= self._dp_profiled_input_cardinality.get(p_id, 1.0)
        elif baseline_input > 0:
            cost *= input_size / baseline_input
        return max(cost, 1e-12)

    @staticmethod
    def _dp_scaling_mean(
        entry: Any, target_width: int
    ) -> Optional[float]:
        """Return an interpolated converged per-actor latency at one width."""
        if not isinstance(entry, dict) or target_width < 1:
            return None
        raw_widths = entry.get("widths")
        timings = raw_widths if isinstance(raw_widths, dict) else None
        if timings is None:
            timings = {}
            adaptive = entry.get("adaptive_profile", {})
            if isinstance(adaptive, dict) and "width" in adaptive:
                timings[adaptive["width"]] = entry
        points = []
        for raw_width, timing in timings.items():
            if not isinstance(timing, dict):
                continue
            adaptive = timing.get("adaptive_profile", {})
            if not isinstance(adaptive, dict) or adaptive.get("converged") is not True:
                continue
            try:
                width = int(raw_width)
                mean = float(timing["mean_ms_per_sample"])
            except (TypeError, ValueError, KeyError):
                continue
            if width >= 1 and math.isfinite(mean) and mean >= 0.0:
                points.append((width, mean))
        if not points:
            return None
        points.sort()
        for width, mean in points:
            if width == target_width:
                return mean
        lower = [point for point in points if point[0] < target_width]
        upper = [point for point in points if point[0] > target_width]
        if lower and upper:
            left_width, left_mean = lower[-1]
            right_width, right_mean = upper[0]
            fraction = (target_width - left_width) / (right_width - left_width)
            return left_mean + fraction * (right_mean - left_mean)
        # Extrapolation is deliberately conservative: use the nearest measured
        # per-actor latency rather than assuming continued linear speedup.
        return (lower[-1] if lower else upper[0])[1]

    def _dp_pipe_cost_at_parallelism(
        self,
        p_id: int,
        variant_type: PipeVariantType,
        parallelism: int,
        width_one_cost: float,
    ) -> float:
        """Return per-actor latency at the candidate's global concurrency.

        A formal plan is copied into ``W`` local workers.  A stage configured
        with ``a`` actors/processes per worker therefore runs ``W * a``
        workers on the same machine, not ``a``.  Looking up the curve at only
        ``a`` was systematically optimistic for multi-stage plans: every
        stage appeared to have an isolated CPU pool even though Ray and SMP
        share the same host cores, memory bandwidth, and storage path.
        """
        if parallelism < 1 or not math.isfinite(width_one_cost):
            return width_one_cost
        key = (
            PipeVariantType.RAY.name
            if variant_type == PipeVariantType.TF_RAY
            else variant_type.name
        )
        entries = (
            self.profiled_stats.get("physical_model", {})
            .get("scaling", {})
            .get(key, {})
        )
        if not isinstance(entries, dict):
            return width_one_cost
        entry = entries.get(p_id, entries.get(str(p_id)))
        formal_workers = max(1, int(self.physical_plan.n_local_workers))
        if os.environ.get("CEDAR_MATCH_PROFILE_RESOURCES") == "1":
            raw_workers = os.environ.get(
                "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
            )
            if raw_workers is not None:
                try:
                    formal_workers = int(raw_workers)
                except ValueError as exc:
                    raise RuntimeError(
                        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS must be an integer"
                    ) from exc
                if formal_workers < 1:
                    raise RuntimeError(
                        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS must be positive"
                    )
        # Price every remote block under the final plan's total concurrently
        # active remote CPU slots, since all stages share the same host.
        assumed_total = getattr(
            self, "_dp_assumed_total_parallel_stage_cpus", None
        )
        contention_parallelism = parallelism
        if assumed_total is not None:
            contention_parallelism = max(parallelism, int(assumed_total))
        global_concurrency = formal_workers * contention_parallelism
        mean = self._dp_scaling_mean(entry, global_concurrency)
        if mean is None:
            return width_one_cost
        baseline_input = float(
            self.profiled_stats["baseline"]["input_sizes"][p_id]
        )
        cost = mean
        scaling = self._dp_compute_scaling_for_pipe(p_id)
        if scaling == PipeComputeScaling.PER_RECORD:
            cost *= self._dp_profiled_input_cardinality.get(p_id, 1.0)
        elif baseline_input <= 0:
            return width_one_cost
        return max(cost, 1e-12)

    def _dp_compute_scaling_for_pipe(
        self, p_id: int
    ) -> PipeComputeScaling:
        if self.logical_pipes is not None and p_id in self.logical_pipes:
            pipe = self.logical_pipes[p_id]
            if getattr(pipe, "compute_scaling_explicit", False):
                return pipe.compute_scaling
        entry = self.profiled_stats.get("operator_compute_scaling", {}).get(
            p_id
        )
        if entry is None:
            entry = self.profiled_stats.get(
                "operator_compute_scaling", {}
            ).get(str(p_id))
        if isinstance(entry, dict):
            entry = entry.get("scaling")
        if entry is not None:
            try:
                return PipeComputeScaling(entry)
            except ValueError:
                pass
        return PipeComputeScaling.PER_DATA

    def _dp_observed_selectivities(self) -> Dict[int, float]:
        """Return baseline selectivities with offload-count fallback.

        Older frozen profiles did not persist baseline input/output counts, but
        their isolated RAY/SMP observations did. Median aggregation reuses
        those already-collected counts without changing or rerunning profile.
        """
        result: Dict[int, float] = {}
        raw = self.profiled_stats.get("baseline", {}).get(
            "selectivities", {}
        )
        for key, value in raw.items():
            try:
                p_id = int(key)
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and 0.0 <= parsed <= 1.0:
                result[p_id] = parsed

        observations: Dict[int, List[float]] = {}
        for variant_entries in self.profiled_stats.get(
            "offloads", {}
        ).values():
            if not isinstance(variant_entries, dict):
                continue
            for pipe_profile in variant_entries.values():
                if not isinstance(pipe_profile, dict):
                    continue
                for key, value in pipe_profile.get(
                    "selectivities", {}
                ).items():
                    try:
                        p_id = int(key)
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(parsed) and 0.0 <= parsed <= 1.0:
                        observations.setdefault(p_id, []).append(parsed)
        for p_id, values in observations.items():
            if p_id not in result and values:
                result[p_id] = statistics.median(values)
        return result

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

    def _dp_object_boundary_operator(
        self, variant: PipeVariantType, p_id: int
    ) -> Optional[Dict[str, Any]]:
        """Return real-object marshalling measurements for one operator."""
        physical_model = self.profiled_stats.get("physical_model", {})
        if not isinstance(physical_model, dict):
            return None
        object_boundaries = physical_model.get("object_boundary", {})
        if not isinstance(object_boundaries, dict):
            return None
        key = (
            PipeVariantType.RAY.name
            if variant == PipeVariantType.TF_RAY
            else variant.name
        )
        backend = object_boundaries.get(key)
        operators = backend.get("operators", {}) if isinstance(backend, dict) else {}
        if not isinstance(operators, dict):
            return None
        entry = operators.get(p_id, operators.get(str(p_id)))
        return entry if isinstance(entry, dict) else None

    def _dp_boundary_cost_ms(
        self,
        variant: PipeVariantType,
        input_size: float,
        output_size: float,
    ) -> float:
        """Return steady-state boundary service time per sample.

        Boundary calibration is deliberately synchronous and therefore fits
        one fixed cost per submitted task. Ray submits a batch in one actor
        call, so only that task-level term is shared by the samples in the
        batch. ``max_inflight`` is a queue/backpressure limit, not execution
        parallelism, and must not be used to divide service demand. SMP
        submits one sample per request and therefore receives no batching
        discount. Bandwidth remains a per-sample service cost.
        """
        throughput = self._dp_boundary_throughput(variant)
        if throughput is None:
            return 0.0
        fixed_cost_divisor = 1
        if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            fixed_cost_divisor = self._dp_ray_submit_batch_size(
                input_size, output_size
            )
        return (
            self._dp_boundary_fixed_latency_ms(variant) / fixed_cost_divisor
        ) + (
            (input_size + output_size) / throughput * 1000.0
        )

    @staticmethod
    def _dp_ray_submit_batch_size(
        input_size: float, output_size: float
    ) -> int:
        """Return the exact submit-batch rule used by a final Ray stage."""
        total_io_size = input_size + output_size
        if not math.isfinite(total_io_size) or total_io_size <= 0:
            return 1
        return min(
            max(
                int(
                    constants.RAY_SUBMIT_BATCH_SCALING_FACTOR
                    // total_io_size
                ),
                1,
            ),
            500,
        )

    def _dp_stage_boundary_cost(self, prev_mask: int, block) -> float:
        """Model each parallel stage boundary separately from operator work.

        The term is placement-dependent and therefore belongs in the outer DP
        transition rather than the per-mask candidate provider.  A fused block
        pays one input and one output boundary, while every operator's compute
        cost remains undiscounted.
        """
        local_cost, parallel_cost = self._dp_stage_boundary_components(
            prev_mask, block
        )
        return local_cost + parallel_cost

    def _dp_stage_boundary_components(
        self, prev_mask: int, block
    ) -> Tuple[float, float]:
        """Return local byte service and remote task service per source row.

        A Ray batch is selected from the size of one surviving item, whereas
        transport work is proportional to the expected byte volume and task
        count is proportional to the number of surviving input records.  The
        old implementation used byte volume as an item size; after a filter it
        therefore invented larger batches than the runtime could submit.

        New layered profiles measure driver marshalling on the actual legal
        objects seen by every operator.  For those profiles, submission and
        driver serialization are charged to the shared local lane while byte
        transport remains on the parallel stage.  Old profiles retain the
        previous end-to-end parallel-only interpretation.
        """
        throughput = self._dp_boundary_throughput(block.variant)
        if throughput is None:
            return 0.0, 0.0

        source_size = float(
            self.profiled_stats["baseline"]["output_sizes"][
                self._get_source_p_id()
            ]
        )
        next_mask = prev_mask | block.mask
        input_item_size = source_size * self._dp_r_prod[prev_mask]
        output_item_size = source_size * self._dp_r_prod[next_mask]

        cardinality = getattr(self, "_dp_cardinality_prod", [])
        if len(cardinality) == len(self._dp_r_prod):
            input_records = cardinality[prev_mask]
            output_records = cardinality[next_mask]
        else:
            input_records = 1.0
            output_records = 1.0

        submit_batch = 1
        if block.variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            submit_batch = self._dp_ray_submit_batch_size(
                input_item_size, output_item_size
            )
        transported_bytes_ms = (
            input_item_size * input_records
            + output_item_size * output_records
        ) / throughput * 1000.0
        fixed_ms = (
            self._dp_boundary_fixed_latency_ms(block.variant)
            * input_records
            / submit_batch
        )
        block_order = getattr(block, "order", ())
        inner_ops = getattr(self, "_dp_inner_ops", ())
        if not block_order or not inner_ops:
            return 0.0, transported_bytes_ms + fixed_ms
        first_p_id = inner_ops[block_order[0]]
        last_p_id = inner_ops[block_order[-1]]
        input_entry = self._dp_object_boundary_operator(
            block.variant, first_p_id
        )
        output_entry = self._dp_object_boundary_operator(
            block.variant, last_p_id
        )
        if input_entry is not None and output_entry is not None:
            try:
                input_identity_ms = float(
                    input_entry["input_identity_stage"][
                        "mean_ms_per_sample"
                    ]
                )
                output_identity_ms = float(
                    output_entry["output_identity_stage"][
                        "mean_ms_per_sample"
                    ]
                )
            except (KeyError, TypeError, ValueError):
                input_identity_ms = float("nan")
                output_identity_ms = float("nan")
            if (
                math.isfinite(input_identity_ms)
                and input_identity_ms >= 0.0
                and math.isfinite(output_identity_ms)
                and output_identity_ms >= 0.0
            ):
                # Each identity measurement is a complete same-object stage
                # round trip. Half of the first boundary plus half of the last
                # approximates submit(input_type) + receive(output_type), while
                # preserving one measured task/queue fixed cost. This is
                # charged to the shared local runtime lane; block compute stays
                # on its Ray/SMP parallel-stage coordinate.
                identity_boundary_ms = (
                    0.5 * input_identity_ms * input_records
                    + 0.5 * output_identity_ms * output_records
                )
                return identity_boundary_ms, 0.0
            try:
                input_marshal_ms = float(
                    input_entry["input_serialize_ms_per_sample"]
                )
                output_marshal_ms = float(
                    output_entry["output_deserialize_ms_per_sample"]
                )
            except (KeyError, TypeError, ValueError):
                input_marshal_ms = float("nan")
                output_marshal_ms = float("nan")
            if (
                math.isfinite(input_marshal_ms)
                and input_marshal_ms >= 0.0
                and math.isfinite(output_marshal_ms)
                and output_marshal_ms >= 0.0
            ):
                local_stage_ms = (
                    fixed_ms
                    + input_marshal_ms * input_records
                    + output_marshal_ms * output_records
                )
                if block.variant == PipeVariantType.SMP:
                    residuals = []
                    for entry in (input_entry, output_entry):
                        try:
                            residual = float(
                                entry[
                                    "shared_runtime_overhead_ms_per_sample"
                                ]
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                        if math.isfinite(residual) and residual >= 0.0:
                            residuals.append(residual)
                    if residuals:
                        local_stage_ms += max(residuals) * input_records
                return local_stage_ms, transported_bytes_ms
        return 0.0, transported_bytes_ms + fixed_ms

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

    def _dp_compute_scaling_for_idx(
        self, operator_idx: Optional[int]
    ) -> PipeComputeScaling:
        if (
            operator_idx is not None
            and operator_idx < len(self._dp_compute_scalings)
        ):
            return self._dp_compute_scalings[operator_idx]
        return self._dp_compute_scaling

    def _dp_compute_work_prod(
        self, mask: int, operator_idx: Optional[int] = None
    ) -> float:
        """Return the multiplier appropriate for operator compute work.

        Cedar historically scales compute by serialized byte volume.  An
        operator annotated as per-record instead scales only by the number of
        surviving records. Stage boundaries and cache I/O continue to use
        :meth:`_dp_work_prod` and therefore remain byte based.
        """
        if (
            self._dp_compute_scaling_for_idx(operator_idx)
            == PipeComputeScaling.PER_RECORD
        ):
            return self._dp_cardinality_prod[mask]
        return self._dp_work_prod(mask)

    def _dp_compute_cost_denominator(
        self,
        operator_idx: int,
        baseline_input_size: float,
        source_size: float,
    ) -> float:
        """Normalize a profiled cost without changing its baseline value.

        ``_BlockCostIndex`` stores costs per source byte and multiplies the
        final sum by ``source_size``. For per-record work, divide by the
        cardinality at the operator's original position as well as the source
        size. This makes the estimated cost at that original position exactly
        equal to its profiled cost while removing unrelated item-size growth.
        """
        if (
            self._dp_compute_scaling_for_idx(operator_idx)
            == PipeComputeScaling.PER_RECORD
        ):
            if operator_idx < len(self._dp_inner_ops):
                p_id = self._dp_inner_ops[operator_idx]
                profiled_cardinality = (
                    self._dp_profiled_input_cardinality.get(p_id)
                )
                if profiled_cardinality is not None:
                    return source_size * profiled_cardinality
            # Compatibility fallback for isolated unit-test stubs and old
            # callers that do not prepare pipe-keyed baseline metadata.
            original_prefix = (1 << operator_idx) - 1
            return source_size * self._dp_cardinality_prod[original_prefix]
        return baseline_input_size

    def _dp_smp_supported_for_pipe(self, p_id: int) -> bool:
        """Whether SMP can safely execute this operator's resource class."""
        if self.logical_pipes is None:
            return False
        pipe = self.logical_pipes.get(p_id)
        return bool(
            pipe is not None
            and pipe.execution_resource != PipeExecutionResource.CUDA
        )

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
        raw_selectivities = self._dp_observed_selectivities()

        # Anchor per-record costs at each pipe's position in the profiled
        # logical pipeline. A two-stage optimizer may call this method after
        # reordering ``inner_ops``; using that new index would silently change
        # the baseline normalization and make costs incomparable across
        # reorder policies.
        output_p_id = self._get_output_p_id(self.logical_graph)
        profiled_paths = find_all_paths(
            self.logical_graph, source_p_id, output_p_id
        )
        if len(profiled_paths) != 1:
            raise RuntimeError(
                "DP operator scaling requires one profiled logical path."
            )
        profiled_cardinality = 1.0
        self._dp_profiled_input_cardinality = {}
        for p_id in profiled_paths[0][1:]:
            self._dp_profiled_input_cardinality[p_id] = (
                profiled_cardinality
            )
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
            profiled_cardinality *= value

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

        profiled_scalings = self.profiled_stats.get(
            "operator_compute_scaling", {}
        )
        self._dp_compute_scalings = []
        for p_id in inner_ops:
            pipe = self.logical_pipes[p_id]
            scaling = pipe.compute_scaling
            if not getattr(pipe, "compute_scaling_explicit", False):
                entry = profiled_scalings.get(
                    p_id, profiled_scalings.get(str(p_id))
                )
                if isinstance(entry, dict):
                    entry = entry.get("scaling")
                if entry is not None:
                    try:
                        scaling = PipeComputeScaling(entry)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Invalid profiled compute scaling for pipe "
                            f"{p_id}: {entry!r}"
                        ) from exc
            self._dp_compute_scalings.append(scaling)

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

        # The sink Pipe is part of ``inner_ops`` so it can still receive a
        # physical Variant or participate in a legal final fusion block.  It
        # must nevertheless remain the sink after reordering, even when the
        # application did not annotate it with ``fix()``.  Without these
        # edges a cheap terminal mapper can move into the middle of the graph
        # and change which Pipe is returned to the consumer.
        output_p_id = self._get_output_p_id(self.physical_plan.graph)
        if output_p_id in idx_of:
            output_idx = idx_of[output_p_id]
            for predecessor_idx in range(n):
                if (
                    predecessor_idx != output_idx
                    and predecessor_idx not in pred_indices[output_idx]
                ):
                    pred_indices[output_idx].append(predecessor_idx)

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

        self._allocate_final_remote_stage_resources()

        # The DP path deliberately bypasses Optimizer._offload_and_fuse(),
        # which normally tunes these queueing parameters after materializing a
        # Ray stage. Keep the selected widths and tune queueing afterwards.
        self._tune_final_ray_stage_contexts()

    def _allocate_final_remote_stage_resources(self) -> None:
        """Apply Cedar's native shared actor allocation before final policy."""
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

    def _tune_final_ray_stage_contexts(self) -> None:
        """Apply Cedar's Ray submit-batch rule to the final physical plan.

        This is intentionally a post-processing step: it neither changes the
        selected variants nor the reorder/fusion structure produced by DP.
        """
        stage_sizes = self._dp_final_stage_item_size_map()
        ray_stage_count = sum(
            self.physical_plan.pipe_descs[p_id].variant_type
            in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
            for p_id in self.physical_plan.graph
        )
        finite_workload_cap = self._dp_finite_workload_ray_batch_cap(
            ray_stage_count
        )

        for p_id in self.physical_plan.graph:
            desc = self.physical_plan.pipe_descs[p_id]
            if desc.variant_type not in (
                PipeVariantType.RAY,
                PipeVariantType.TF_RAY,
            ):
                continue

            input_size, output_size = stage_sizes[p_id]

            submit_batch_size = self._dp_ray_submit_batch_size(
                input_size, output_size
            )
            submit_batch_size = min(
                submit_batch_size, finite_workload_cap
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

    def _dp_finite_workload_ray_batch_cap(
        self, ray_stage_count: int
    ) -> int:
        """Bound Ray batch size so a finite pipeline can reach steady state.

        The byte-size rule is sufficient for one Ray stage. With multiple
        stages, however, a batch of 500 can leave only a handful of batches
        per W=8 worker. The bottleneck objective assumes stage overlap, so
        require enough batches that ideal fill/drain overhead is at most 10%:
        ``n_batches >= (n_stages - 1) / 0.10``.
        """
        if ray_stage_count <= 1 or self.options is None:
            return 500
        num_samples = getattr(self.options, "num_samples", None)
        if num_samples is None:
            return 500
        try:
            num_samples = int(num_samples)
        except (TypeError, ValueError):
            return 500
        if num_samples <= 0:
            return 500
        fixed_workers = os.environ.get(
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
        )
        workers = (
            max(1, int(fixed_workers))
            if fixed_workers is not None
            else max(1, int(self.physical_plan.n_local_workers))
        )
        samples_per_worker = math.ceil(num_samples / workers)
        min_batches = max(1, math.ceil((ray_stage_count - 1) / 0.10))
        return max(1, min(500, samples_per_worker // min_batches))

    def _dp_final_stage_item_size_map(
        self,
    ) -> Dict[int, Tuple[float, float]]:
        """Propagate per-item sizes through the materialized linear plan.

        ``Optimizer._calculate_size_map`` sees a fused pipe only by its new
        synthetic id and therefore cannot propagate the fused members' size
        ratios into the next physical stage. Walk the final source-to-sink
        chain explicitly and apply every logical member ratio exactly once.
        Synthetic cache/prefetch nodes preserve the current item size.
        """
        graph = self.physical_plan.graph
        downstream_ids = {
            child for children in graph.values() for child in children
        }
        sources = [p_id for p_id in graph if p_id not in downstream_ids]
        if len(sources) != 1:
            raise RuntimeError(
                "Ray batch tuning requires one linear physical-plan source."
            )

        source_p_id = sources[0]
        current_size = float(
            self.profiled_stats["baseline"]["output_sizes"][source_p_id]
        )
        sizes: Dict[int, Tuple[float, float]] = {}
        current_p_id = source_p_id
        while True:
            children = graph[current_p_id]
            if not children:
                break
            if len(children) != 1:
                raise RuntimeError(
                    "Ray batch tuning requires a linear physical plan."
                )
            next_p_id = next(iter(children))
            desc = self.physical_plan.pipe_descs[next_p_id]
            input_size = current_size
            members = (
                desc.fused_pipes
                if desc.fused_pipes is not None
                else [next_p_id]
            )
            for member_p_id in members:
                ratio = self._data_size_ratio_map.get(member_p_id, 1.0)
                if ratio is not None:
                    current_size *= ratio
            sizes[next_p_id] = (input_size, current_size)
            current_p_id = next_p_id
        return sizes


__all__ = ["MyOptimizer"]
