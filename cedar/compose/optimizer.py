import logging
import yaml
import copy
import math
import multiprocessing
import os
import psutil
import time

from typing import Tuple, Dict, Set, Optional, List, Any, Union

from cedar.pipes import (
    CedarPipeSpec,
    Pipe,
    PipeVariantType,
    PipeVariantContext,
    PipeVariantContextFactory,
)
from cedar.pipes.optimize import OptimizerPipeRegistry
from .utils import (
    calculate_reorderings,
    flip_adj_list,
    all_slices,
    find_all_paths,
)
from .constants import (
    LOCAL_PARALLELISM_SCALING_FACTOR,
    OFFLOAD_THRESHOLD_FRAC,
    FUSED_PIPE_NAME,
    RAY_SUBMIT_BATCH_SIZE,
    RAY_SUBMIT_BATCH_SCALING_FACTOR,
    LOCAL_PARALLELISM_THRESHOLD,
    SMP_AVAILABLE_PARALLELISM,
    LOCAL_PARALLELISM_SAMPLES_PER_SEC_THRESHOLD,
)
from cedar.compose import constants

logger = logging.getLogger(__name__)


class OptimizerOptions:
    def __init__(
        self,
        enable_prefetch: bool = True,
        est_throughput: Optional[float] = None,
        available_local_cpus: int = 1,
        enable_offload: bool = True,
        enable_reorder: bool = True,
        enable_local_parallelism: bool = True,
        enable_fusion: bool = True,
        enable_caching: bool = False,
        disable_physical_opt: bool = False,
        num_samples: Optional[int] = None,
        use_my_optimizer: int = 0,
        reorder_timeout_sec: Optional[float] = None,
    ):
        self.enable_prefetch = enable_prefetch

        # Estimated throughput required, in samples per second
        # If none, then est_throughput is inf
        self.est_throughput = est_throughput

        # Number of CPUs available locally
        self.available_local_cpus = available_local_cpus

        # Enable the offload/fusion pass
        self.enable_offload = enable_offload

        # Enable the reordering pass
        self.enable_reorder = enable_reorder

        # Enable local parallelism pass
        self.enable_local_parallelism = enable_local_parallelism

        # Enable fusion pass
        self.enable_fusion = enable_fusion

        # Enable caching pass
        self.enable_caching = enable_caching

        self.disable_physical_opt = disable_physical_opt

        self.num_samples = num_samples

        # Optimizer implementation selector used by DataSet:
        # 0/default Optimizer, 1/MyOptimizer, 2/DpOptimizer, 3/DjOptimizer,
        # 4/DpTwoStageOptimizer, 5/DpCedarOptimizer, 6/CedarJointOptimizer,
        # 7/ExpOptimizer, 8/PecanOptimizer.
        self.use_my_optimizer = int(use_my_optimizer)

        # Maximum wall-clock time allowed for the original Optimizer reorder
        # pass. None disables the timeout.
        self.reorder_timeout_sec = reorder_timeout_sec


class PipeDesc:
    """
    A serializable representation of a pipe
    """

    def __init__(
        self,
        name: str,
        variant_type: Optional[PipeVariantType] = None,
        variant_ctx: Optional[PipeVariantContext] = None,
        fused_pipes: Optional[List[int]] = None,
    ) -> None:
        self.name = name
        self.variant_type = variant_type
        self.variant_ctx = variant_ctx
        self.fused_pipes = fused_pipes

    def serialize(self) -> Dict[str, Any]:
        """
        Returns a serialized (as a dict) representation of this pipe
        """
        d = {}
        d["name"] = self.name
        if self.variant_type is not None:
            d["variant"] = self.variant_type.name
        if self.variant_ctx is not None:
            d["variant_ctx"] = self.variant_ctx.serialize()
        if self.fused_pipes is not None:
            d["fused_pipes"] = self.fused_pipes

        return d

    def is_set(self) -> bool:
        if self.name == FUSED_PIPE_NAME and self.fused_pipes is None:
            return False
        return (
            self.name is not None
            and self.variant_type is not None
            and self.variant_ctx is not None
        )

    def is_inplace_fuse(self) -> bool:
        """
        True if this pipe should be fused in-place
        """
        return self.name != FUSED_PIPE_NAME and self.fused_pipes is not None

    def is_fused_pipe(self) -> bool:
        return self.fused_pipes is not None and len(self.fused_pipes) > 1


class PhysicalPlan:
    """
    Represents a physical plan
    """

    def __init__(
        self,
        graph: Dict[int, Set[int]],
        pipe_descs: Dict[int, PipeDesc],
        n_local_workers: int = 1,
    ) -> None:
        self.graph = graph
        self.pipe_descs = pipe_descs
        self.n_local_workers = n_local_workers

    def validate(self) -> bool:
        """
        Returns true if the physical plan is valid
        """
        if self.graph is None or self.pipe_descs is None:
            return False
        if not set(self.graph).issubset(set(self.pipe_descs)):
            return False
        for p_id, desc in self.pipe_descs.items():
            if not desc.is_set() and p_id in self.graph:
                return False
        return True

    def set_local_workers(self, n: int) -> None:
        self.n_local_workers = n

    @classmethod
    def from_dict(cls, d):
        graph = PhysicalPlan._parse_physical_graph(d["graph"])
        pipe_descs = {}

        for p_id, pipe_dict in d["pipes"].items():
            variant_type = PipeVariantType[pipe_dict["variant"]]
            variant_spec = pipe_dict.get("variant_ctx", None)
            variant_ctx = PipeVariantContextFactory.create_context(
                variant_type=variant_type, spec=variant_spec
            )
            pipe_name = pipe_dict["name"]
            if pipe_name == FUSED_PIPE_NAME:
                if "fused_pipes" not in pipe_dict:
                    raise ValueError(f"Fused pipes not supplied in dict {d}.")
            fused_pipes = pipe_dict.get("fused_pipes", None)
            desc = PipeDesc(
                name=pipe_name,
                variant_type=variant_type,
                variant_ctx=variant_ctx,
                fused_pipes=fused_pipes,
            )
            pipe_descs[p_id] = desc

        n_local_workers = d.get("n_local_workers", 1)
        return cls(
            graph=graph, pipe_descs=pipe_descs, n_local_workers=n_local_workers
        )

    @staticmethod
    def _parse_physical_graph(graph: Dict[int, str]):
        parsed_graph = {}
        for p_id, input_pids in graph.items():
            try:
                parsed_graph[p_id] = set(
                    [int(num) for num in input_pids.split(",") if num.strip()]
                )
            except AttributeError as e:
                logger.fatal("Input graph should be specified as strings")
                logger.info(str(e))
        return parsed_graph

    def to_dict(self):
        serialized_plan = {}
        serialized_pipes = {}

        for p_id, desc in self.pipe_descs.items():
            serialized_pipes[p_id] = desc.serialize()

        formatted_output_graph = {}
        for p_id, adj_pids in self.graph.items():
            formatted_output_graph[p_id] = ",".join(map(str, adj_pids))

        serialized_plan["pipes"] = serialized_pipes
        serialized_plan["graph"] = formatted_output_graph
        serialized_plan["n_local_workers"] = self.n_local_workers
        return serialized_plan


class Optimizer:
    """
    This class exposes a (static) optimization API for use by features.
    """

    def __init__(
        self,
    ) -> None:
        self.logical_pipes: Optional[Dict[int, Pipe]] = None
        self._initialized = False

        self._base_cost_map = None
        self._data_size_ratio_map = None

    def init(
        self,
        logical_pipes: Dict[int, Pipe],
        logical_adj_list: Dict[int, Set[int]],
    ) -> None:
        if logical_pipes is None or logical_adj_list is None:
            raise ValueError(
                "Invalid input. Features must have sources applied."
            )
        self.logical_pipes = copy.copy(logical_pipes)  # shallow copy
        self.logical_graph = copy.deepcopy(logical_adj_list)

        # Create a pipedesc for each logical pipe
        pipe_descs = {}
        for p_id, pipe in logical_pipes.items():
            desc = PipeDesc(name=pipe.get_logical_name())
            pipe_descs[p_id] = desc

        graph = copy.deepcopy(logical_adj_list)  # deep copy the adj list
        self.physical_plan = PhysicalPlan(graph, pipe_descs, 1)

        self._initialized = True

        self.forbid_local_parallelism = False

        self._tf_fuse_cost = None

    def _init_stats(self):
        # Assign a weight based on fractional latency
        total_cost = 1000.0 / self.profiled_stats["baseline"]["throughput"]
        sum_latencies = sum(
            [
                v
                for _, v in self.profiled_stats["baseline"][
                    "latencies"
                ].items()
            ]
        )
        self._fractional_latencies = {
            k: (v / sum_latencies)
            for k, v in self.profiled_stats["baseline"]["latencies"].items()
        }
        # The fractional cost of each pipe given the default graph
        self._base_cost_map = {
            k: self._fractional_latencies[k] * total_cost
            for k, v in self.profiled_stats["baseline"]["latencies"].items()
        }
        logger.debug("Base pipe cost map: %s", self._base_cost_map)

        self._data_size_ratio_map = {
            k: self._calculate_data_size_ratio(k) for k in self._base_cost_map
        }

        logger.info("Baseline cost {}".format(total_cost))

        if "tf_fuse" in self.profiled_stats:
            tf_throughput = self.profiled_stats["tf_fuse"]["throughput"]
            self._tf_fuse_cost = 1000.0 / tf_throughput
            logger.info("Fused TF cost {}".format(self._tf_fuse_cost))

    def run(
        self, profiled_data: Union[str, Dict[str, Any]], options: OptimizerOptions
    ) -> PhysicalPlan:
        """
        Runs the optimizer. Returns a physical plan for the feature.

        profiled_data: Path to a YAML file, or a dict of profiled stats
            (for testing with custom pipeline data).

        NOTE: The caller should hold a lock to ensure that
        the pipes are thread-safe.
        """
        if not self._initialized:
            raise RuntimeError("Must intiailize optimizer before running.")

        if isinstance(profiled_data, dict):
            self.profiled_stats = profiled_data
        else:
            try:
                with open(profiled_data, "r") as f:
                    self.profiled_stats = yaml.safe_load(f)
            except Exception as e:
                logger.error("An error occurred {}".format(e))
                raise RuntimeError(
                    "Failed to read profiled stats {}".format(profiled_data)
                )
        self.options = options
        self._validate_stats()
        self._init_stats()

        logger.info("Running optimization pass...")
        logger.info("======Using profiled stats========")
        logger.info(self.profiled_stats)

        logger.info("========= Logical Pass ===========")
        self._logical_opt()

        logger.info("========= Physical Pass ===========")
        # if not self.options.disable_physical_opt:
        self._physical_opt()

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

        self._log_optimized_pipeline(tag="Optimizer")
        return self.physical_plan

    def _log_optimized_pipeline(self, *, tag: str) -> None:
        """
        Print the optimized physical plan in a compact, comparable format.
        """
        plan = getattr(self, "physical_plan", None)
        if plan is None:
            logger.info("[%s] No physical plan to print.", tag)
            return

        logger.info("[%s] ===== Optimized pipeline (PhysicalPlan) =====", tag)
        logger.info("[%s] graph: %s", tag, plan.graph)

        try:
            pipe_descs = plan.pipe_descs
        except Exception:
            pipe_descs = None

        if not pipe_descs:
            logger.info("[%s] pipe_descs: <empty>", tag)
            return

        for p_id in sorted(pipe_descs.keys()):
            desc = pipe_descs[p_id]
            name = getattr(desc, "name", None)
            vt = getattr(desc, "variant_type", None)
            variant = vt.name if vt is not None else "None"
            fused = getattr(desc, "fused_pipes", None)
            logger.info(
                "[%s] pipe_id=%s name=%r variant=%s fused_pipes=%s",
                tag,
                p_id,
                name,
                variant,
                fused,
            )

    def _logical_opt(self) -> None:
        logger.info(
            "[Baseline] Baseline plan: {}".format(self.physical_plan.graph)
        )
        baseline_cost = self.calculate_cost(self.physical_plan.graph)
        logger.info(
            "[Baseline] Calculated baseline cost: {}".format(baseline_cost)
        )

        if self.options.enable_reorder:
            logger.info("*Reordering Pass*")
            self.physical_plan.graph = self._pass_reordering()

        if self.options.enable_caching:
            logger.info("*Caching Pass*")
            if not self.options.num_samples:
                logger.info(
                    "Number of samples not specified. Skipping caching pass."
                )
            else:
                self.physical_plan = self._insert_cache()
                self.pipe_desc = self.physical_plan.pipe_descs

        if self.options.enable_prefetch:
            logger.info("*Prefetching Pass*")
            self._insert_prefetch()

        logger.info(
            "[Logical Pass] Optimized graph {}".format(
                self.physical_plan.graph
            )
        )

    def _physical_opt(
        self,
    ) -> None:
        # Tune local parallelism...
        if self.options.enable_local_parallelism:
            num_local_workers = self._calculate_local_parallelism(
                self.physical_plan.graph, self.options
            )
            logger.info(
                "[Parallelism] Using {} local workers".format(
                    num_local_workers
                )
            )
            self.physical_plan.set_local_workers(num_local_workers)

        if self.options.enable_offload:
            set_pipes = self._offload_and_fuse()
        elif self.options.enable_fusion:
            # enable fusion, but no offloading
            set_pipes = self._local_fusion()
        else:
            set_pipes = set()

        pipes_set = len(set_pipes)

        # Any pipes not set?
        for p_id, pipe_desc in self.physical_plan.pipe_descs.items():
            if p_id not in set_pipes:
                if (
                    p_id in self.logical_pipes
                    and self.logical_pipes[p_id].is_tf()
                ):
                    pipe_desc.variant_type = PipeVariantType.TF
                    pipe_desc.variant_ctx = (
                        PipeVariantContextFactory.create_context(
                            variant_type=PipeVariantType.TF
                        )
                    )
                else:
                    pipe_desc.variant_type = PipeVariantType.INPROCESS
                    pipe_desc.variant_ctx = (
                        PipeVariantContextFactory.create_context(
                            variant_type=PipeVariantType.INPROCESS
                        )
                    )

        # If there are only tf pipes, let tf handle scaling
        if (
            pipes_set == 0
            and self._is_tf_graph()
            and not self._contains_tf_py_func()
        ):
            logger.info("All pipes are TF, Setting parallelism to 1")
            self.physical_plan.n_local_workers = 1

        # Check for adjacent TF pipes
        start_p_id = self._get_source_p_id()
        end_p_id = self._get_output_p_id(self.physical_plan.graph)
        all_paths = find_all_paths(
            self.physical_plan.graph, start_p_id, end_p_id
        )
        if self.options.enable_fusion:
            for path in all_paths:
                curr_fusion = []
                for p_id in path:
                    if (
                        self.physical_plan.pipe_descs[p_id].variant_type
                        == PipeVariantType.TF
                        and len(self.physical_plan.graph[p_id]) <= 1
                    ):
                        curr_fusion.append(p_id)
                    elif len(curr_fusion) >= 2:
                        self._fuse_local_tf(curr_fusion)
                        curr_fusion = []
                    else:
                        curr_fusion = []
                if len(curr_fusion) >= 2:
                    self._fuse_local_tf(curr_fusion)

        logger.info("Checking for Ray DS fusion...")
        should_fuse_ray_source = (
            not self.options.enable_offload
            and self.options.enable_fusion
            and self._check_fuse_ray_local()
        ) or (
            self.options.enable_offload
            and self.options.enable_fusion
            and self._check_fuse_ray_remote()
        )
        if should_fuse_ray_source:
            if os.environ.get("CEDAR_MATCH_PROFILE_RESOURCES") == "1":
                # RAY_DS does not expose a fixed per-stage actor width: Ray
                # Data schedules map tasks independently of the profiled
                # RAY/TF_RAY actor count. Keep the already fused Ray stage so
                # strict runs can account for exactly one actor per worker.
                logger.warning(
                    "Skipping RAY_DS source lowering under strict profile "
                    "resource matching; preserving the actor-accounted "
                    "physical fusion stage."
                )
            else:
                self._fuse_ray_source()

        logger.info(
            "[Physical Pass] Optimized graph {}".format(
                self.physical_plan.graph
            )
        )
        for p_id, desc in self.physical_plan.pipe_descs.items():
            logger.info(f"\t[Pipe {p_id}]: {desc.serialize()}")

    def _fuse_ray_source(self) -> Set[int]:
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)
        all_paths = find_all_paths(
            self.physical_plan.graph, source_p_id, output_p_id
        )

        if len(all_paths) != 1:
            raise RuntimeError("Can't fuse ray dataset with multiple paths")

        if self.options.enable_prefetch:
            path = all_paths[0][1:-1]
        else:
            path = all_paths[0][1:]
        fused_pipes = []

        for p_id in path:
            if self.physical_plan.pipe_descs[p_id].is_fused_pipe():
                fused_pipes.extend(
                    self.physical_plan.pipe_descs[p_id].fused_pipes
                )
            else:
                fused_pipes.append(p_id)
            del self.physical_plan.graph[p_id]

        source_desc = self.physical_plan.pipe_descs[source_p_id]

        desc_variant_ctx = PipeVariantContextFactory.create_context(
            variant_type=PipeVariantType.RAY_DS,
        )
        source_desc.variant_ctx = desc_variant_ctx
        source_desc.variant_type = PipeVariantType.RAY_DS
        source_desc.fused_pipes = fused_pipes

        if self.options.enable_prefetch:
            # disable the prefetch op
            del self.physical_plan.graph[output_p_id]
        self.physical_plan.graph[source_p_id] = set()

        self.physical_plan.n_local_workers = 1

        logger.info("Fusing {} into RAY_DS Source".format(fused_pipes))
        return set(fused_pipes)

    def _check_fuse_ray_remote(self) -> bool:
        # TODO: only worth it if ops are numpy-heavy (as opposed to tensors)
        # add a check for this...
        # Is the source fusable?
        source_p_id = self._get_source_p_id()
        source_pipe = self.logical_pipes[source_p_id]
        spec = source_pipe.get_spec()
        if not (
            spec.is_fusable_source
            and spec.fusable_source_variants is not None
            and PipeVariantType.RAY_DS in spec.fusable_source_variants
        ):
            logger.info("Source not fusable")
            return False

        path = self._get_full_path()[1:]
        # Did the optimizer fuse everything else (except the prefetcher)
        # into a ray variant?
        if len(path) != 1:
            return False
        p_id = path[0]
        logical_set = set(self.logical_pipes.keys())
        logical_set.remove(source_p_id)

        if (
            self.physical_plan.pipe_descs[p_id].is_fused_pipe()
            and (
                logical_set
                == set(self.physical_plan.pipe_descs[p_id].fused_pipes)
            )
            and (
                self.physical_plan.pipe_descs[p_id].variant_type
                == PipeVariantType.RAY
            )
        ):
            return True
        return False

    def _check_fuse_ray_local(self) -> bool:
        # TODO: only worth it if ops are numpy-heavy (as opposed to tensors)
        # add a check for this...

        # Is the source fusable?
        source_p_id = self._get_source_p_id()
        source_pipe = self.logical_pipes[source_p_id]
        if not (
            source_pipe.get_spec().is_fusable_source
            and PipeVariantType.RAY_DS
            in source_pipe.get_spec().fusable_source_variants
        ):
            return False

        # Did the optimizer not find a good solution?
        path = self._get_full_path()

        for p_id in path[1:]:
            if (
                self.physical_plan.pipe_descs[p_id].variant_type
                != PipeVariantType.INPROCESS
                or p_id not in self.logical_pipes
                or not self.logical_pipes[p_id].get_spec().is_fusable
            ):
                return False

        return True

    def _fuse_local_tf(self, pipes_to_fuse: List[int]) -> int:
        logger.info("\t Fusing {} into one TF pipe".format(pipes_to_fuse))

        all_pipes = set(self.logical_pipes.keys())
        if len(pipes_to_fuse) == len(all_pipes) - 1:
            logger.info("\t All pipes fused to TF - defaulting to TF dataset")
            self.physical_plan.n_local_workers = 1

        if self.physical_plan.n_local_workers == 1:
            num_parallel_calls = -1
        else:
            num_parallel_calls = None
        desc_variant_type = PipeVariantType.TF
        # If we only have 1 worker, let TF autotune, otherwise
        # disable TF autotuning
        desc_variant_ctx = PipeVariantContextFactory.create_context(
            variant_type=PipeVariantType.TF,
            spec={"num_parallel_calls": num_parallel_calls},
        )

        fused_p_id = self._fuse_pipe(
            pipes_to_fuse, desc_variant_type, desc_variant_ctx
        )

        return fused_p_id

    def _fuse_tf(self) -> Set[int]:
        set_pipes = set()
        # IF we only want to fuse pipes without offloading, likely just TF
        # pipes.
        graph = self.physical_plan.graph
        ccs = self._enumerate_connected_components(graph, PipeVariantType.TF)

        # List of lists, denoting which pipes to fuse/offload
        checked_pipes = set()

        # Rely on TF to autotune its ops
        self.physical_plan.n_local_workers = 1

        if len(ccs) == 1 and ccs[0][0] == self._get_source_p_id():
            return self._fuse_tf_source()

        for cc in ccs:
            checked_pipes.update(cc)
            logger.info("Connected component: {}".format(cc))

            desc_variant_type = PipeVariantType.TF
            # If we only have 1 worker, let TF autotune, otherwise
            # disable TF autotuning
            desc_variant_ctx = PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.TF,
                spec={"num_parallel_calls": -1},
            )

            # For TF pipes, just fuse all to TF graph and let TF do the work
            if len(cc) == 1:
                desc = self.physical_plan.pipe_descs[cc[0]]
                desc.variant_type = desc_variant_type
                desc.variant_ctx = desc_variant_ctx
            else:
                fused_p_id = self._fuse_pipe(
                    cc, desc_variant_type, desc_variant_ctx
                )
                set_pipes.add(fused_p_id)

            logger.info(f"\tFused {cc} into TF pipe")
        return set_pipes

    def _fuse_local_smp(self) -> Set[int]:
        logger.info("[Fusion] Single process... Determining best SMP fusion")
        set_pipes = set()

        offloaded_variant_ctxs = []
        offloads, _, _ = self._calculate_offloads(PipeVariantType.SMP)

        for t in offloads:
            offload, offload_variant = t
            # Check the total fractional
            total_frac = sum([self._fractional_latencies[x] for x in offload])
            if total_frac >= OFFLOAD_THRESHOLD_FRAC:
                # ok to offload
                desc_variant_type = offload_variant
                desc_variant_ctx = PipeVariantContextFactory.create_context(
                    variant_type=offload_variant,
                    spec={
                        "n_procs": 1,
                        "max_inflight": 50,
                        "max_prefetch": 50,
                        "use_threads": True,
                    },
                )
                if len(offload) == 1:
                    # Set the existing pipe desc
                    desc = self.physical_plan.pipe_descs[offload[0]]
                    desc.variant_type = desc_variant_type
                    desc.variant_ctx = desc_variant_ctx
                else:
                    # Make a fused pipe
                    fused_p_id = self._fuse_pipe(
                        offload, desc_variant_type, desc_variant_ctx
                    )
                    set_pipes.add(fused_p_id)
                offloaded_variant_ctxs.append(desc_variant_ctx)

                # Mark pipes as set
                set_pipes.update(offload)
            else:
                logger.info(f"Not offloading {offload}, below threshold.")

        # Set the offload parallelism
        n_procs = SMP_AVAILABLE_PARALLELISM // (
            len(offloaded_variant_ctxs) * self.physical_plan.n_local_workers
        )
        for ctx in offloaded_variant_ctxs:
            ctx.n_procs = n_procs

        logger.info(offloads)
        return set_pipes

    def _local_fusion(self) -> Set[int]:
        logger.info("\tFusing pipes...")
        if self._is_tf_graph() and not self._contains_tf_py_func():
            return self._fuse_tf()
        else:
            if self.forbid_local_parallelism:
                return self._fuse_local_smp()
            else:
                return set()

    def _fuse_tf_source(self) -> Set[int]:
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)
        all_paths = find_all_paths(
            self.physical_plan.graph, source_p_id, output_p_id
        )

        if len(all_paths) != 1:
            raise RuntimeError("Can't fuse tf dataset with multiple paths")

        path = all_paths[0]
        fused_pipes = []
        prefetch_inserted = False

        for idx, p_id in enumerate(path):
            if self.physical_plan.pipe_descs[p_id].name != "PrefetcherPipe":
                if not self.logical_pipes[p_id].is_tf():
                    raise RuntimeError("Attempting to fuse non-TF pipe.")
                fused_pipes.append(p_id)
            elif idx != len(path) - 1:
                raise RuntimeError("Inserted prefetch pipe in middle..")
            else:
                prefetch_inserted = True

        source_desc = self.physical_plan.pipe_descs[source_p_id]

        desc_variant_ctx = PipeVariantContextFactory.create_context(
            variant_type=PipeVariantType.TF,
            spec={"num_parallel_calls": -1},
        )
        source_desc.variant_ctx = desc_variant_ctx
        source_desc.variant_type = PipeVariantType.TF
        source_desc.fused_pipes = fused_pipes[1:]

        for p_id in fused_pipes[1:]:
            del self.physical_plan.graph[p_id]

        if prefetch_inserted:
            self.physical_plan.graph[source_p_id] = {path[-1]}
        else:
            self.physical_plan.graph[source_p_id] = set()
        return set(fused_pipes)

    def _offload_and_fuse(self) -> Set[int]:
        set_pipes = set()
        input_size_map, output_size_map = self._calculate_size_map(
            self.physical_plan.graph
        )

        cache_pipe = self._get_cache_pid(self.physical_plan)
        if cache_pipe is not None:
            logger.info("Using cache!!")
            cache_cost = self.calculate_cost(
                self.physical_plan.graph,
                caching_on=True,
                plan=self.physical_plan,
            )
            logger.info("Cache cost = {}".format(cache_cost))

        # Decide what's the best variant to offload...
        is_tf = self._is_tf_graph()
        if self.options.enable_offload and self.options.enable_fusion:
            if is_tf:
                variant = PipeVariantType.TF_RAY
            else:
                variant = PipeVariantType.RAY
        elif self.options.enable_offload:
            # if we just want to offload, see which variant is best
            profiled_offloads = self.profiled_stats["offloads"]
            baseline_throughput = self.profiled_stats["baseline"]["throughput"]
            backend_throughputs = {}
            for backend, backend_stats in profiled_offloads.items():
                if len(backend_stats) == 0:
                    continue
                backend_throughputs[backend] = 0
                for p_id, p_stats in backend_stats.items():
                    if p_stats["throughput"] > baseline_throughput:
                        backend_throughputs[backend] += (
                            p_stats["throughput"] - baseline_throughput
                        )
            logger.info(backend_throughputs)
            # Choose backend with best throughput
            variant_str = sorted(
                backend_throughputs,
                key=lambda x: backend_throughputs[x],
                reverse=True,
            )[0]
            variant = PipeVariantType[variant_str]
        else:
            raise RuntimeError

        logger.info("Selected offload variant {}".format(variant))

        offloaded_variant_ctxs = []
        offloads, base_cost, offload_cost = self._calculate_offloads(variant)

        cache_pipe = self._get_cache_pid(self.physical_plan)
        if cache_pipe is not None:
            del_cache = False

            for offload in offloads:
                logger.info(offload)
                pids = offload[0]
                desc = {p_id: PipeDesc(None, variant) for p_id in pids}
                logger.info(desc)
                offloaded_cost = self.calculate_cost(
                    self.physical_plan.graph,
                    desc,
                    fused_pipes=pids,
                    plan=self.physical_plan,
                )
                cached_cost = self.calculate_cost(
                    self.physical_plan.graph,
                    desc,
                    caching_on=True,
                    fused_pipes=pids,
                    plan=self.physical_plan,
                )
                if cached_cost > offloaded_cost:
                    del_cache = True
                    logger.info(
                        "Planning to remove cache: "
                        "Offload cost={}, cache cost = {}".format(
                            offloaded_cost, cached_cost
                        )
                    )

            if del_cache:
                # remove the cache!
                input_graph = flip_adj_list(self.physical_plan.graph)
                prev_p_id = list(input_graph[cache_pipe])[0]
                next_p_id = list(self.physical_plan.graph[cache_pipe])[0]
                self.physical_plan.graph[prev_p_id] = {next_p_id}
                del self.physical_plan.graph[cache_pipe]
                del self.physical_plan.pipe_descs[cache_pipe]

            # If the offloads are BEFORE the cache.

            # logger.info("Using cache!!")
            # cache_cost = self.calculate_cost(
            #     self.physical_plan.graph,
            #     fused_pipes=
            #     caching_on=True,
            #     plan=self.physical_plan,
            # )
            # logger.info("Cache cost = {}".format(cache_cost))
        # if cache_cost > offload_cost:
        #     # disable caching

        # Check if we should just fuse to TF
        if (
            self._tf_fuse_cost is not None
            and self._tf_fuse_cost < offload_cost
            and self.forbid_local_parallelism
            and is_tf
            and self.options.enable_fusion
        ):
            logger.info("Fusing into to full TF dataset")
            return self._fuse_tf_source()

        # Update the plan with each offload
        for t in offloads:
            offload, offload_variant = t
            # Check the total fractional
            total_frac = sum([self._fractional_latencies[x] for x in offload])
            if total_frac >= OFFLOAD_THRESHOLD_FRAC:
                # ok to offload
                desc_variant_type = offload_variant
                desc_variant_ctx = PipeVariantContextFactory.create_context(
                    variant_type=offload_variant,
                    spec={
                        "n_actors": 1,
                        "max_inflight": 100,
                        "max_prefetch": 100,
                        "use_threads": True,
                        "submit_batch_size": RAY_SUBMIT_BATCH_SIZE,
                    },
                )
                if len(offload) == 1:
                    # Set the existing pipe desc
                    desc = self.physical_plan.pipe_descs[offload[0]]
                    desc.variant_type = desc_variant_type
                    desc.variant_ctx = desc_variant_ctx
                else:
                    # Make a fused pipe
                    fused_p_id = self._fuse_pipe(
                        offload, desc_variant_type, desc_variant_ctx
                    )
                    set_pipes.add(fused_p_id)
                offloaded_variant_ctxs.append(desc_variant_ctx)

                # Mark pipes as set
                set_pipes.update(offload)
            else:
                logger.info(f"Not offloading {offload}, below threshold.")

        # Set the offload parallelism
        try:
            n_actors = math.ceil(
                constants.RAY_AVAILABLE_PARALLELISM
                / (
                    len(offloaded_variant_ctxs)
                    * self.physical_plan.n_local_workers
                )
            )
        except ZeroDivisionError:
            n_actors = 0

        # n_actors = min(n_actors, 16)
        for ctx in offloaded_variant_ctxs:
            ctx.n_actors = n_actors

        # Set the submit batch size
        logger.info("Calculating submit batch size...")
        for p_id in self.physical_plan.graph:
            pipe_desc = self.physical_plan.pipe_descs[p_id]
            if (
                pipe_desc.variant_type == PipeVariantType.RAY
                or pipe_desc.variant_type == PipeVariantType.TF_RAY
            ):
                logger.info(f"   Setting submit batch size for {p_id}")
                # is this a fused pipe?
                if pipe_desc.is_fused_pipe():
                    input_size = input_size_map[pipe_desc.fused_pipes[0]]
                    output_size = output_size_map[pipe_desc.fused_pipes[-1]]
                else:
                    input_size = input_size_map[p_id]
                    output_size = output_size_map[p_id]

                logger.info(
                    f"  Input size: {input_size}, output size: {output_size}"
                )
                total_io_size = input_size + output_size

                submit_batch_size = min(
                    max(
                        int(RAY_SUBMIT_BATCH_SCALING_FACTOR // total_io_size),
                        1,
                    ),
                    500,
                )
                logger.info(
                    f"  Input size: {input_size}, output size: {output_size}"
                    f"  submit batch size: {submit_batch_size}"
                )
                pipe_desc.variant_ctx.set_submit_batch_size(submit_batch_size)

        return set_pipes

    def _is_tf_graph(self):
        for _, pipe in self.logical_pipes.items():
            if pipe.is_tf():
                return True
        return False

    def _contains_tf_py_func(self):
        for _, pipe in self.logical_pipes.items():
            if pipe.is_tf_py_func():
                return True
        return False

    def _calculate_offloads(
        self, offload_variant: PipeVariantType
    ) -> Tuple[List[Tuple[List[int], PipeVariantType]], float]:
        assert isinstance(offload_variant, PipeVariantType)
        g = self.physical_plan.graph
        logger.info("Enumerating plans for offloading graph {}".format(g))
        baseline_cost = self.calculate_cost(g)
        logger.info("Baseline cost : {}".format(baseline_cost))

        # Get all connected fusable components in the graph
        if not self.options.enable_fusion:
            ccs = []
            for p_id in self.logical_pipes:
                if self.logical_pipes[p_id].is_fusable(offload_variant):
                    ccs.append([p_id])
        else:
            ccs = self._enumerate_connected_components(g, offload_variant)
            logger.info("Fusable connected components: {}".format(ccs))

        # List of lists, denoting which pipes to fuse/offload
        all_offloads = []
        checked_pipes = set()

        final_cost = baseline_cost

        # For each connected component, get the best fusion/offloadings
        for cc in ccs:
            checked_pipes.update(cc)
            logger.info("Connected component: {}".format(cc))
            fusions = self._enumerate_fusions(g, cc, offload_variant)

            logger.info("Found fusions: {}".format(fusions))
            for fusion in fusions:
                all_offloads.append((fusion[0], offload_variant))
                final_cost -= fusion[1]

        # Are there any pipes that weren't considered?
        for p_id in set(self.logical_pipes.keys()).difference(checked_pipes):
            logger.info(f"Checking pipe {p_id} for offload")
            if self.logical_pipes[p_id].can_mutate_to(offload_variant):
                desc = {p_id: PipeDesc(None, offload_variant)}
                offload_cost = self.calculate_cost(g, desc)
                if offload_cost < baseline_cost:
                    logger.info(f"Offloading pipe {p_id}")
                    all_offloads.append(([p_id], offload_variant))
                    cost_savings = baseline_cost - offload_cost
                    logger.info("Savings {}".format(cost_savings))
                    final_cost -= cost_savings

        logger.info("Found final offload/fusions: {}".format(all_offloads))
        logger.info(
            "Baseline cost: {}, Final Fused Cost: {}".format(
                baseline_cost, final_cost
            )
        )
        return all_offloads, baseline_cost, final_cost

    def _enumerate_connected_components(
        self, graph: Dict[int, Set[int]], variant_type: PipeVariantType
    ):
        # Get all connected components in the graph that share the
        # ability to be offloaded
        connected_components = []
        visited = set()

        def _dfs_connected_components(out_g, in_g, p_id, current_component):
            visited.add(p_id)
            # Only fuse linear regions of the dfg
            if (
                p_id in self.logical_pipes
                and self.logical_pipes[p_id].is_fusable(variant_type)
                and len(out_g[p_id]) <= 1
                and len(in_g[p_id]) <= 1
                and (
                    (
                        self.logical_pipes[p_id].is_tf()
                        and (variant_type == PipeVariantType.TF)
                    )
                    or variant_type != PipeVariantType.TF
                )
            ):
                current_component.append(p_id)
            else:
                if len(current_component) > 0:
                    connected_components.append(copy.copy(current_component))
                current_component = []

            for n in out_g[p_id]:
                if n not in visited:
                    _dfs_connected_components(
                        out_g, in_g, n, current_component
                    )

        input_graph = flip_adj_list(graph)
        source_p_id = self._get_source_p_id()
        _dfs_connected_components(graph, input_graph, source_p_id, [])

        return connected_components

    def _allowed_fusion(self, slice: List[int], variant_type: PipeVariantType):
        # Don't allow ToTensor maps to run in SMP
        for p_id in slice:
            if (
                "MapperPipe_ToTensor"
                in self.logical_pipes[p_id].get_logical_name()
            ):
                logger.info(f"Forbidding fusion {slice} due to ToTensor")
                return False

        return True

    def _enumerate_fusions(
        self,
        graph: Dict[int, Set[int]],
        connected_component: List[int],
        variant_type: PipeVariantType,
    ) -> List[Tuple[List[int], float]]:
        """
        Given a connected component, within graph, evaluate all possible
        fusions and return the best fusion
        """
        fusions = []

        # Generate the set of all slices
        slices = all_slices(connected_component)
        logger.info("Slices: {}".format(slices))
        fuse_cost_map = {}

        baseline_cost = self.calculate_cost(graph)

        for s in slices:
            if not self._allowed_fusion(s, variant_type):
                logger.info("Disallowed fusion {}".format(s))
                continue
            # Calculate the cost for offloading each slice
            desc = {k: PipeDesc(None, variant_type, None) for k in s}
            if len(s) > 1:
                fuse = s
            else:
                fuse = None

            # Only consider fusions which at least one offloaded op
            # benefits from offloading
            ok_to_fuse = False
            for p_id in s:
                offload_p_desc = {p_id: PipeDesc(None, variant_type, None)}
                if self.calculate_cost(graph, offload_p_desc) < baseline_cost:
                    ok_to_fuse = True
                    break

            if not ok_to_fuse:
                logger.info("Unbeneficial fusion {}".format(s))
                continue

            fuse_cost_map[tuple(s)] = self.calculate_cost(graph, desc, fuse)

        logger.info(fuse_cost_map)
        baseline_cost = self.calculate_cost(graph)
        logger.info(baseline_cost)

        sorted_costs = sorted(fuse_cost_map, key=lambda x: fuse_cost_map[x])
        logger.info(sorted_costs)

        while True:
            if len(sorted_costs) == 0:
                break

            # Get the lowest cost plan
            best_fusion = sorted_costs[0]
            best_cost = fuse_cost_map[best_fusion]
            logger.info("Remaining fusions: {}".format(sorted_costs))
            logger.info(
                "Best Fusion: {}, Cost: {}".format(best_fusion, best_cost)
            )

            if best_cost < baseline_cost:
                # Add this fusion
                # fusions.append(list(best_fusion))
                logger.info(
                    "Using Fusion: {}, Cost: {}, baseline: {}".format(
                        best_fusion, best_cost, baseline_cost
                    )
                )
                cost_savings = baseline_cost - best_cost
                logger.info("Savings: {}".format(cost_savings))
                fusions.append((list(best_fusion), cost_savings))

                # Remove all slices that contain fused pipes
                fused_set = set(best_fusion)
                slices_to_delete = []

                for s in fuse_cost_map.keys():
                    if len(fused_set.intersection(set(s))) > 0:
                        slices_to_delete.append(s)

                for s in slices_to_delete:
                    del fuse_cost_map[s]

                sorted_costs = sorted(
                    fuse_cost_map, key=lambda x: fuse_cost_map[x]
                )
            else:
                # Break if baseline is better than all other options
                break
        return fusions

    def _calculate_local_parallelism(
        self, graph: Dict[int, Set[int]], options: OptimizerOptions
    ) -> int:
        """
        Returns the number of parallel processes that should run, given
        a logical graph
        """
        # Calculate the expected throughput from the graph
        expected_throughput = self.calculate_throughput(graph)

        logger.info(
            "[Parallelism] Calculated throughput of logical plan to {}"
            " samples/s".format(expected_throughput)
        )

        _, output_size_map = self._calculate_size_map(graph)
        output_p_id = self._get_output_p_id(graph)
        output_sample_size = output_size_map[output_p_id]
        output_throughput = output_sample_size * expected_throughput

        logger.info(
            "[Parallelism] Calculated throughput of logical plan to {}"
            " bytes/s".format(output_throughput)
        )

        if self.options.enable_offload:
            parallelism_threshold = LOCAL_PARALLELISM_THRESHOLD * 0.95
        else:
            # Set this lower since we'll need resources for SMP
            parallelism_threshold = LOCAL_PARALLELISM_THRESHOLD * 0.8

        if (
            output_throughput > parallelism_threshold
            and expected_throughput
            > LOCAL_PARALLELISM_SAMPLES_PER_SEC_THRESHOLD
        ):
            logger.info(
                "[Parallelism] Throughput exceeds threshold "
                f"of {parallelism_threshold}... Setting local "
                "paralellism to 1."
            )
            self.forbid_local_parallelism = True

            return 1

        if options.est_throughput is None:
            # unbounded...
            logger.info("[Parallelism] Unbounded throughput requested...")
            return options.available_local_cpus
        else:
            num_workers = options.est_throughput / (
                expected_throughput * LOCAL_PARALLELISM_SCALING_FACTOR
            )
            logger.info(
                "[Parallelism] Estimated {} workers"
                " needed to meet {} samples/s".format(
                    num_workers, options.est_throughput
                )
            )

            if num_workers <= 1:
                num_workers = 1
            elif num_workers > options.available_local_cpus:
                num_workers = options.available_local_cpus
            else:
                num_workers = math.ceil(num_workers)
            return num_workers

    def _pass_reordering(self) -> Dict[int, Set[int]]:
        """
        Runs the reordering optimizer pass

        Returns: Optimized graph with ideal reordering
        """
        start_time = time.perf_counter()
        timeout_sec = getattr(self.options, "reorder_timeout_sec", None)
        candidate_plans = calculate_reorderings(
            self.logical_pipes,
            self.physical_plan.graph,
            timeout_sec=timeout_sec,
        )
        self._raise_if_reorder_timed_out(start_time, timeout_sec)
        logger.info(
            "[Reordering] Generated {} possible plans.".format(
                len(candidate_plans)
            )
        )
        optimal_reordering, reordered_cost = self._find_optimal_reordering(
            candidate_plans,
            reorder_start_time=start_time,
            reorder_timeout_sec=timeout_sec,
        )
        logger.info("[Reordering] Optimal plan: {}".format(optimal_reordering))
        logger.info(
            "[Reordering] Calculated optimal cost: {}".format(reordered_cost)
        )
        return optimal_reordering

    def _find_optimal_reordering(
        self,
        candidate_plans: List[Dict[int, Set[int]]],
        reorder_start_time: Optional[float] = None,
        reorder_timeout_sec: Optional[float] = None,
    ) -> Tuple[Dict[int, Set[int]], float]:
        # Prune plans...
        plan_costs = {}
        for idx, plan in enumerate(candidate_plans):
            self._raise_if_reorder_timed_out(
                reorder_start_time, reorder_timeout_sec
            )
            cost = self.calculate_cost(plan)
            plan_costs[idx] = cost
            self._raise_if_reorder_timed_out(
                reorder_start_time, reorder_timeout_sec
            )

        # Sort plans by cost
        ordered_plan_idxs = sorted(plan_costs, key=lambda x: plan_costs[x])
        self._raise_if_reorder_timed_out(
            reorder_start_time, reorder_timeout_sec
        )
        optimal_idx = ordered_plan_idxs[0]

        return candidate_plans[optimal_idx], plan_costs[optimal_idx]

    @staticmethod
    def _raise_if_reorder_timed_out(
        reorder_start_time: Optional[float],
        reorder_timeout_sec: Optional[float],
    ) -> None:
        if reorder_start_time is None or reorder_timeout_sec is None:
            return
        elapsed = time.perf_counter() - reorder_start_time
        if elapsed > reorder_timeout_sec:
            raise TimeoutError(
                "Optimizer reorder exceeded timeout of "
                f"{reorder_timeout_sec:.6f}s after {elapsed:.6f}s."
            )

    def calculate_throughput(self, graph: Dict[int, Set[int]]) -> float:
        """
        Given a plan, return the estimated single-core throughput of the plan
        in units of samples/second
        """
        cost = self.calculate_cost(graph)  # cost in terms of ms/sample
        return 1000.0 / cost

    def calculate_cost(
        self,
        graph: Dict[int, Set[int]],
        physical_specs: Optional[Dict[int, PipeDesc]] = None,
        fused_pipes: Optional[Union[List[int], List[List[int]]]] = None,
        caching_on: Optional[bool] = False,
        plan: PhysicalPlan = None,
    ) -> float:
        """
        Given a plan, return the estimated cost to execute the plan,
        in units of ms/sample/cpu.

        Args:
            graph: Adjacency list of graph to calculate cost for
            physical_specs: PipeDesc detailing physical plan for each
                pipe. If no physical_specs are provided, assume INPROCESS
        """
        # Adjust for reorderings...
        # Assume that each pipe consistently grows/shrinks data by a fixed
        # ratio
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(graph)

        # PipeDesc 来源：评估临时 plan（如插入 cache）时必须与 graph 一致，否则新 pipe id
        # 只存在于 plan.pipe_descs 中会触发 Invalid pipe ID。
        pipe_descs = (
            plan.pipe_descs if plan is not None else self.physical_plan.pipe_descs
        )

        # Get the critical path of this graph
        # TODO: should really be the sum of all nodes here
        critical_path, _ = self._get_critical_path(
            graph, source_p_id, output_p_id, plan
        )
        if len(critical_path) != len(graph):
            raise RuntimeError("Failed to extract critical path.")

        # Walk through the critical path.
        curr_size = self.profiled_stats["baseline"]["output_sizes"][
            critical_path[0]
        ]
        # use default cost for source pipe
        curr_cost = self._base_cost_map[critical_path[0]]

        input_size_map = {
            critical_path[0]: self.profiled_stats["baseline"]["input_sizes"][
                critical_path[0]
            ]
        }
        output_size_map = {
            critical_path[0]: self.profiled_stats["baseline"]["output_sizes"][
                critical_path[0]
            ]
        }
        pipe_cost_map = {critical_path[0]: curr_cost}

        materialized_fused_sets: List[Set[int]] = []
        for p_id in critical_path[1:]:
            if not self._is_optimizer_pipe(p_id, plan):
                # If optimizer pipe, assume zero cost and no size change
                if physical_specs is None:
                    new_cost = self._calculate_pipe_cost(p_id, curr_size, None)
                else:
                    desc = physical_specs.get(p_id, None)
                    new_cost = self._calculate_pipe_cost(p_id, curr_size, desc)

                input_size_map[p_id] = curr_size
                pipe_cost_map[p_id] = new_cost

                curr_cost += new_cost
                curr_size = curr_size * self._data_size_ratio_map[p_id]
                output_size_map[p_id] = curr_size
            else:
                desc = None
                if physical_specs is not None:
                    desc = physical_specs.get(p_id, None)
                elif p_id in pipe_descs:
                    desc = pipe_descs[p_id]

                # If this is a materialized fused node, reconstruct historical
                # per-op cost/size maps from desc.fused_pipes and variant.
                if (
                    desc is not None
                    and desc.is_fused_pipe()
                    and desc.fused_pipes is not None
                    and len(desc.fused_pipes) > 1
                ):
                    fused_hist = list(desc.fused_pipes)
                    fused_desc = PipeDesc(
                        name=None,
                        variant_type=desc.variant_type,
                        variant_ctx=desc.variant_ctx,
                    )
                    for hist_p_id in fused_hist:
                        input_size_map[hist_p_id] = curr_size
                        hist_cost = self._calculate_pipe_cost(
                            hist_p_id, curr_size, fused_desc
                        )
                        pipe_cost_map[hist_p_id] = hist_cost
                        curr_size = curr_size * self._data_size_ratio_map[hist_p_id]
                        output_size_map[hist_p_id] = curr_size

                    fused_specs = {
                        hist_p_id: fused_desc for hist_p_id in fused_hist
                    }
                    _, fused_block_cost = self._calculate_cost_fused(
                        fused_specs,
                        fused_hist,
                        input_size_map,
                        output_size_map,
                        pipe_cost_map,
                    )
                    input_size_map[p_id] = input_size_map[fused_hist[0]]
                    output_size_map[p_id] = curr_size
                    pipe_cost_map[p_id] = fused_block_cost
                    curr_cost += fused_block_cost
                    materialized_fused_sets.append(set(fused_hist))
                else:
                    logger.warning(
                        f"Assigning zero cost to optimizer pipe {p_id}"
                    )

        normalized_fusions: List[List[int]] = []
        if fused_pipes is not None:
            if len(fused_pipes) > 0 and isinstance(fused_pipes[0], list):
                normalized_fusions = [list(x) for x in fused_pipes]  # type: ignore[index]
            else:
                normalized_fusions = [list(fused_pipes)]  # type: ignore[arg-type]

        fused_block_sets: List[Set[int]] = []
        fused_block_fused_costs: List[float] = []
        for fused_block in normalized_fusions:
            if any(set(fused_block) == s for s in materialized_fused_sets):
                continue
            pipe_cost_baseline, pipe_cost_fused = self._calculate_cost_fused(
                physical_specs,
                fused_block,
                input_size_map,
                output_size_map,
                pipe_cost_map,
            )
            curr_cost = curr_cost - pipe_cost_baseline + pipe_cost_fused
            fused_block_sets.append(set(fused_block))
            fused_block_fused_costs.append(pipe_cost_fused)

        if caching_on:
            # Factor in caching (i.e. reads from disk, cache)
            # NOTE: Assumes all plans have a validly placed cache
            # (i.e., cache before first non-deterministic op)
            logger.debug("[Caching] Pipe cost map: %s", pipe_cost_map)

            if plan is None:
                raise RuntimeError(
                    "Compute cost needs plan argument if cache is active."
                )
            cache_p_id = self._get_cache_pid(plan)

            # NOTE: We could have plan where no cache gets inserted
            if cache_p_id is not None:
                cache_index = critical_path.index(cache_p_id)
                if cache_index == 0:
                    raise RuntimeError(
                        "Cache cannot be inserted at index 0 of plan."
                    )

                cost_to_subtract = 0

                # NOTE: Can be done more efficiently,
                # but separation of logic for better overview
                # 1. Subtract saved cost
                fused_to_subtract: Set[int] = set()
                for i in range(cache_index):
                    p_id = critical_path[i]
                    if fused_block_sets:
                        covered = False
                        for f_idx, fset in enumerate(fused_block_sets):
                            if p_id in fset:
                                fused_to_subtract.add(f_idx)
                                covered = True
                                break
                        if not covered:
                            cost_to_subtract += pipe_cost_map[p_id]
                    else:
                        cost_to_subtract += pipe_cost_map[p_id]

                curr_cost -= cost_to_subtract
                for f_idx in fused_to_subtract:
                    curr_cost -= fused_block_fused_costs[f_idx]

                logger.debug("[Caching] Cost after removing cached prefix: %s", curr_cost)
                # 2. Add reading overhead

                # calculate read latency for cache
                pre_cache_p_id = critical_path[cache_index - 1]
                cache_size = output_size_map.get(pre_cache_p_id)
                if cache_size is None:
                    cache_size = self.profiled_stats["baseline"]["output_sizes"][
                        pre_cache_p_id
                    ]

                read_time_per_byte = self.profiled_stats["disk_info"][
                    "read_latency"
                ]
                read_latency = cache_size * (read_time_per_byte)

                # transform latency into cost
                if read_latency <= 0:
                    # read_latency 为 0：profile 中 read_latency=0、或 cache 前输出大小为 0。
                    # 此时按「无额外读盘代价」处理，避免 1/read_latency 除零。
                    cache_cost = 0.0
                else:
                    cache_throughput_sample_per_second = 1 / read_latency
                    cache_throughput_sample_per_second_per_cpu = (
                        cache_throughput_sample_per_second
                    )
                    # cache_throughput_sample_per_second_per_cpu = cache_throughput_sample_per_second / multiprocessing.cpu_count()
                    cache_cost = 1000 / cache_throughput_sample_per_second
                curr_cost += cache_cost

        return curr_cost

    def _calculate_size_map(
        self, graph: Dict[int, Set[int]]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(graph)

        # Get the critical path of this graph
        # TODO: should really be the sum of all nodes here
        all_paths = find_all_paths(graph, source_p_id, output_p_id)
        if len(all_paths) != 1:
            raise NotImplementedError

        path = all_paths[0]
        # Walk through the critical path.
        curr_size = self.profiled_stats["baseline"]["output_sizes"][path[0]]

        input_size_map = {
            path[0]: self.profiled_stats["baseline"]["input_sizes"][path[0]]
        }
        output_size_map = {
            path[0]: self.profiled_stats["baseline"]["output_sizes"][path[0]]
        }

        for p_id in path[1:]:
            if not self._is_optimizer_pipe(p_id):
                input_size_map[p_id] = curr_size

                curr_size = curr_size * self._data_size_ratio_map[p_id]
                output_size_map[p_id] = curr_size
            else:
                input_size_map[p_id] = curr_size
                curr_size = curr_size
                output_size_map[p_id] = curr_size
        return input_size_map, output_size_map

    def _calculate_cost_fused(
        self,
        physical_specs: Optional[Dict[int, PipeDesc]],
        fused_pipes: Optional[List[int]],
        input_size_map: Dict[int, float],
        output_size_map: Dict[int, float],
        pipe_cost_map: Dict[int, float],
    ) -> float:
        """
        Returns the cost of ONLY THE FUSED PIPES, by fusing all pipes together.
        """
        if len(fused_pipes) < 1:
            raise RuntimeError("Cannot fuse fewer than 1 pipe.")

        for p_id in fused_pipes:
            logger.info(p_id)
            if (
                p_id not in physical_specs
                or p_id not in input_size_map
                or p_id not in pipe_cost_map
            ):
                raise RuntimeError(f"Fused pipe {p_id} does not contain spec.")
        logger.info(input_size_map)
        logger.info(output_size_map)
        # 1 upload
        total_io_baseline = input_size_map[fused_pipes[0]]
        total_io_fused = input_size_map[fused_pipes[0]]
        # Calculate the cost of fusing pipes together
        for p_id in fused_pipes[1:]:
            # one download, one upload for inner pipes
            total_io_baseline += input_size_map[p_id] * 2
        # 1 download for last pipe
        total_io_baseline += output_size_map[fused_pipes[-1]]
        total_io_fused += output_size_map[fused_pipes[-1]]

        logger.info(
            "Total IO of baseline: {}, fused: {}".format(
                total_io_baseline, total_io_fused
            )
        )

        # Calculate the proportion
        pipe_cost_baseline = sum([pipe_cost_map[x] for x in fused_pipes])
        pipe_cost_fused = pipe_cost_baseline * (
            total_io_fused / total_io_baseline
        )
        return pipe_cost_baseline, pipe_cost_fused

    def _calculate_pipe_cost(
        self, p_id: int, input_size: float, desc: Optional[PipeDesc]
    ):
        #input_size要注意，在dp函数调用时是初始size，所以后续dp时还需要乘以real size/initial size
        """
        Calculates the estimated cost of a pipe, given its input size and
        physical pipe desc. If no desc is provided, assume INPROCESS pipe.
        """
        # Scale cost by data size w.r.t. profiled data size
        cost = (
            input_size / self.profiled_stats["baseline"]["input_sizes"][p_id]
        ) * self._base_cost_map[p_id]
        if desc is not None:
            variant_type = desc.variant_type
            # INPROCESS uses baseline profile directly; no offload lookup needed.
            if (
                variant_type is None
                or variant_type == PipeVariantType.INPROCESS
            ):
                return cost
            if p_id not in self.profiled_stats["offloads"][variant_type.name]:
                raise RuntimeError(
                    f"Pipe {p_id} does not have a {variant_type.name} profile."
                )

            # Calculate fractional speedup based on amdahl's law
            baseline_tput = self.profiled_stats["baseline"]["throughput"]
            offload_tput = self.profiled_stats["offloads"][variant_type.name][
                p_id
            ]["throughput"]

            # Check if speedup exceeds theoretical max
            total_speedup = offload_tput / baseline_tput
            # logger.info(f"Original cost {cost}")
            # logger.info(f"Fraction: {self._fractional_latencies[p_id]}")
            # logger.info(f"Total speedup {total_speedup}")
            if total_speedup >= 1 / (1 - self._fractional_latencies[p_id]):
                # logger.info(f"Pipe {p_id} Exceeded theoretical speedup")
                # set to inf
                cost = 0
            else:
                # solve amdahl's law for the fractional speedup
                pipe_speedup = self._fractional_latencies[p_id] / (
                    (baseline_tput / offload_tput)
                    - (1 - self._fractional_latencies[p_id])
                )
                # logger.info(f"Pipe speedup {pipe_speedup}")
                # Cost is inverse to speedup
                cost = cost / pipe_speedup
            # logger.info(f"Final Cost {cost}")

        return cost

    def _calculate_data_size_ratio(self, p_id: int) -> float:
        """
        Returns the fractional data size of the output / input for pipe
        p_id
        """
        try:
            ratio = (
                self.profiled_stats["baseline"]["output_sizes"][p_id]
                / self.profiled_stats["baseline"]["input_sizes"][p_id]
            )
        except ZeroDivisionError:
            if self.logical_pipes[p_id].is_source():
                ratio = None
            else:
                raise RuntimeError(f"Pipe {p_id} has a zero input size stat.")
        return ratio

    def _validate_stats(self) -> None:
        if (
            "baseline" not in self.profiled_stats
            or "input_sizes" not in self.profiled_stats["baseline"]
            or "latencies" not in self.profiled_stats["baseline"]
            or "output_sizes" not in self.profiled_stats["baseline"]
            or "throughput" not in self.profiled_stats["baseline"]
        ):
            raise RuntimeError("Improperly formatted stats")
        for p_id in self.logical_pipes:
            if (
                p_id not in self.profiled_stats["baseline"]["input_sizes"]
                or p_id not in self.profiled_stats["baseline"]["latencies"]
                or p_id not in self.profiled_stats["baseline"]["output_sizes"]
            ):
                raise RuntimeError("Improperly formatted stats")

        # Check for variants
        if self.options.enable_offload and (
            "offloads" not in self.profiled_stats
            or PipeVariantType.RAY.name not in self.profiled_stats["offloads"]
        ):
            raise RuntimeError(
                "Profiled stats do not contain Ray data. "
                "Run with --disable_offload to not use Ray."
            )

    def _get_source_p_id(self) -> int:
        # Source node should not change across graphs
        source_p_id = None
        for p_id, pipe in self.logical_pipes.items():
            if pipe.is_source():
                if source_p_id is not None:
                    raise RuntimeError("Found multiple source pipes")
                source_p_id = p_id

        if source_p_id is not None:
            return source_p_id
        raise RuntimeError("Could not find source pipe")

    def _get_cache_pid(self, plan: PhysicalPlan) -> int:
        graph = plan.graph
        pipe_desc = plan.pipe_descs
        cache_p_id = None
        for p_id, adj in graph.items():
            if pipe_desc[p_id].name == "ObjectDiskCachePipe":
                if cache_p_id is not None:
                    raise RuntimeError("Found multiple cache pipes.")
                cache_p_id = p_id
        return cache_p_id

    def _get_output_p_id(self, graph: Dict[int, Set[int]]) -> int:
        output_p_id = None
        for p_id, adj in graph.items():
            if len(adj) == 0:
                if output_p_id is not None:
                    raise RuntimeError("Found multiple output pipes")
                output_p_id = p_id
        if output_p_id is not None:
            return output_p_id
        raise RuntimeError("Could not find output pipe")

    def _get_graph_entry_node(self, graph: Dict[int, Set[int]]) -> int:
        """
        在「出边邻接表」表示的 DAG 中，返回唯一入度为 0 的节点。
        fusion/reorder 之后物理图可能不再包含逻辑 source id，此时应用此入口而非
        ``_get_source_p_id()``。
        """
        all_nodes: Set[int] = set(graph.keys())
        for succs in graph.values():
            all_nodes |= succs
        has_incoming: Set[int] = set()
        for succs in graph.values():
            has_incoming |= succs
        entries = all_nodes - has_incoming
        if len(entries) != 1:
            raise RuntimeError(
                "Expected exactly one entry node in graph, got "
                f"{sorted(entries)!r} (graph keys: {sorted(graph)!r})"
            )
        return next(iter(entries))

    def _resolve_cost_graph_source(self, graph: Dict[int, Set[int]]) -> int:
        """用于 ``calculate_cost`` / ``_calculate_size_map`` 等与给定 graph 一致的起点。"""
        logical = self._get_source_p_id()
        if logical in graph:
            return logical
        return self._get_graph_entry_node(graph)

    def _get_critical_path(
        self,
        graph: Dict[int, Set[int]],
        start: int,
        end: int,
        plan: Optional[PhysicalPlan] = None,
    ) -> List[int]:
        """
        Returns the critical path through the graph, based on profiled
        latencies
        """
        visited = set()
        max_len = 0
        max_path = []

        # source node
        def dfs(p_id, len, path):
            nonlocal max_len, max_path

            curr_len = len + self._get_pipe_latency(p_id, plan)
            curr_path = path.copy()
            curr_path.append(p_id)

            if p_id == end:
                if curr_len > max_len or not max_path:
                    max_len = curr_len
                    max_path = curr_path.copy()

            visited.add(p_id)

            for neighbor in graph[p_id]:
                if neighbor not in visited:
                    dfs(neighbor, curr_len, curr_path)

            visited.remove(p_id)

        dfs(start, 0, [])
        return max_path, max_len

    def _get_pipe_latency(
        self, p_id, plan: Optional[PhysicalPlan] = None
    ) -> float:
        if self._is_optimizer_pipe(p_id, plan):
            return 0
        else:
            return self.profiled_stats["baseline"]["latencies"][p_id]

    def _is_optimizer_pipe(
        self, p_id: int, plan: Optional[PhysicalPlan] = None
    ) -> bool:
        """
        Checks wehther pipe is optimizer pipe of current plan.
        When ``plan`` is set (e.g. hypothetical graph in ``calculate_cost``),
        use that plan's ``pipe_descs`` so newly inserted optimizer pipes resolve.
        """
        descs = plan.pipe_descs if plan is not None else self.physical_plan.pipe_descs
        if p_id not in descs and p_id not in self.logical_pipes:
            raise ValueError(f"Invalid pipe ID {p_id}")
        return p_id in descs and p_id not in self.logical_pipes

    def _insert_prefetch(self) -> None:
        """
        Inserts a prefetch pipe at the end of the current graph
        """
        # Create the pipe desc
        prefetch_pipe_cls = OptimizerPipeRegistry.get_pipe("PrefetcherPipe")
        prefetch_pipe = prefetch_pipe_cls()
        prefetch_p_id = self._get_new_p_id()

        prefetch_pipe_desc = PipeDesc(
            name=prefetch_pipe.get_logical_name(),
            variant_type=PipeVariantType.INPROCESS,
            variant_ctx=PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.INPROCESS
            ),
        )
        self.physical_plan.pipe_descs[prefetch_p_id] = prefetch_pipe_desc

        # Update the graph
        curr_output_p_id = self._get_output_p_id(self.physical_plan.graph)
        self.physical_plan.graph[prefetch_p_id] = set()
        self.physical_plan.graph[curr_output_p_id].add(prefetch_p_id)

        logger.info(f"Inserted prefetch pipe with ID {prefetch_p_id}")

    def _fuse_pipe(
        self,
        p_ids: List[int],
        variant_type: PipeVariantType,
        variant_ctx: PipeVariantContext,
    ) -> int:
        """
        Fuses pipes specified and updates the physical plan
        Returns the p_id of the new pipe.
        """
        if len(p_ids) < 2:
            raise RuntimeError("Cannot fuse pipes with <2 elems")
        pipe_desc = PipeDesc(
            name=FUSED_PIPE_NAME,
            variant_type=variant_type,
            variant_ctx=variant_ctx,
            fused_pipes=p_ids,
        )

        new_p_id = self._get_new_p_id()
        self.physical_plan.pipe_descs[new_p_id] = pipe_desc

        # Update the graph
        start_p_id = p_ids[0]
        end_p_id = p_ids[-1]
        input_graph = flip_adj_list(self.physical_plan.graph)

        # Update upstream pipe
        if len(input_graph[start_p_id]) > 1:
            raise RuntimeError("Cannot fuse with non-single input")

        if len(input_graph[start_p_id]) == 1:
            input_p_id = list(input_graph[start_p_id])[0]
            if len(self.physical_plan.graph[input_p_id]) != 1:
                raise RuntimeError("Cannot fuse with fanout at input")
            self.physical_plan.graph[input_p_id] = {new_p_id}
        logger.debug("Graph before fusion cleanup: %s", self.physical_plan.graph)
        # Update downstream pipe
        output_p_ids = self.physical_plan.graph[end_p_id]
        if len(output_p_ids) > 1:
            raise RuntimeError("Cannot fuse with multiple outputs")
        self.physical_plan.graph[new_p_id] = output_p_ids.copy()

        # clear the old pipes
        for p_id in p_ids:
            del self.physical_plan.graph[p_id]

        
        logger.debug("Graph after fusion cleanup: %s", self.physical_plan.graph)
        logger.debug("Logical graph: %s", self.logical_graph)

        return new_p_id

    def _get_new_p_id(self) -> int:
        return max(self.physical_plan.pipe_descs.keys()) + 1

    def _insert_cache(self) -> Dict[int, Set[int]]:
        """
        Runs the caching optimizer pass.

        returns: Optimized graph with cache inserted at ideal place.
        """

        candidate_plans = self._calculate_caching_plans(
            self.logical_pipes, self.physical_plan.graph
        )

        logger.info(
            f"[Caching] Generated {len(candidate_plans)} possible plans."
        )

        optimal_reordering, reordered_cost = self._find_optimal_caching_plan(
            candidate_plans
        )

        logger.info(f"[Caching] Optimal plan: {optimal_reordering}")
        logger.info(f"[Caching] Calculated optimal cost: {reordered_cost}")

        return optimal_reordering

    def _calculate_caching_plans(
        self, pipes: Dict[int, Pipe], graph: Dict[int, Set[int]]
    ) -> List[Dict[int, Set[int]]]:
        """
        Enumerate all possible caching plans and return them in a list.

        Precondition: The graph should be fully connected

        NOTE: non-linear graphs are in general not currently supported for 
        caching

        Args:
            pipes: Dict of all pipe IDs to pipes in the graph. Pipes should
                contain attributes specifying their ordering/dependencies
            graph: The original dataflow

        Returns:
            A list of all permissible plans resulting from cache
            insertion before the first non-deterministic operation.
        """
        # Create the pipe desc
        cache_pipe_cls = OptimizerPipeRegistry.get_pipe("ObjectDiskCachePipe")
        cache_pipe = cache_pipe_cls()
        cache_p_id = self._get_new_p_id()

        cache_pipe_desc = PipeDesc(
            name=cache_pipe.get_logical_name(),
            variant_type=PipeVariantType.INPROCESS,
            variant_ctx=PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.INPROCESS
            ),
        )

        # Allow for possiblity of no cache op
        original_physical_plan = copy.deepcopy(self.physical_plan)
        cache_base_plan = copy.deepcopy(self.physical_plan)
        cache_base_plan.pipe_descs[cache_p_id] = cache_pipe_desc

        # Create list of graphs
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(graph)

        # NOTE: Assumes linear graph
        ordered_pipes, _ = self._get_critical_path(
            graph, source_p_id, output_p_id
        )

        if len(ordered_pipes) != len(graph):
            raise RuntimeError("Failed to extract critical path.")

        first_non_deterministic_index = self._find_first_random_pipe(
            pipes, ordered_pipes
        )

        all_cache_plans = [original_physical_plan]

        # Place cache after pipe at index i
        for i in range(first_non_deterministic_index):
            # check cache validity
            curr_plan = copy.deepcopy(cache_base_plan)
            pipe_id_to_rewire = ordered_pipes[i]

            if self._has_enough_disk_space(pipe_id_to_rewire):
                # insert cache pipe
                cache_next_nodes = curr_plan.graph[pipe_id_to_rewire]
                curr_plan.graph[pipe_id_to_rewire] = set()
                curr_plan.graph[pipe_id_to_rewire].add(cache_p_id)
                curr_plan.graph[cache_p_id] = cache_next_nodes
                all_cache_plans.append(curr_plan)
            else:
                logger.info(
                    f"[Caching] Did not consider inserting cache after pipe index {i} because not enough disk space is available."
                )

        logger.info(f"Generated {len(all_cache_plans)} valid cache plans.")

        return all_cache_plans

    def _find_optimal_caching_plan(
        self, candidate_plans: List[Dict[int, Set[int]]]
    ) -> Tuple[Dict[int, Set[int]], float]:
        """
        Find the optimal caching plan. Return it with its associated cost.
        """
        # Prune plans...
        plan_costs = {}
        for idx, plan in enumerate(candidate_plans):
            # NOTE: Temporarily reset physical plan
            self.physical_plan = plan
            cost = self.calculate_cost(plan.graph, caching_on=True, plan=plan)
            plan_costs[idx] = cost

        # Sort plans by cost
        ordered_plan_idxs = sorted(plan_costs, key=lambda x: plan_costs[x])
        logger.info(ordered_plan_idxs)
        optimal_idx = ordered_plan_idxs[0]
        logger.info(optimal_idx)
        logger.info(plan_costs)

        logger.info("===CACHING PLANS===")
        # for i, plan in enumerate(candidate_plans) in range(l)
        # for k, v in candidate_plans.items:
        #     logger.info(k)
        #     logger.info(v)
        logger.info(candidate_plans[optimal_idx])
        logger.info(plan_costs[optimal_idx])

        return candidate_plans[optimal_idx], plan_costs[optimal_idx]

    def _find_first_random_pipe(
        self, pipes: Dict[int, Pipe], ops: List[int]
    ) -> int:
        """
        Traverse the list of pipe ids and return index of first
        non-deterministic pipe. If no non-deterministic pipe present,
        the value returned is the length of the list.

        NOTE: Only works on linear execution graphs.
        """
        for i in range(len(ops)):
            op = ops[i]
            op_pipe = pipes[op]
            if op_pipe.is_random():
                return i
        return len(ops)

    def _get_full_path(self, include_prefetch: bool = False) -> List[int]:
        source_p_id = self._get_source_p_id()
        output_p_id = self._get_output_p_id(self.physical_plan.graph)
        all_paths = find_all_paths(
            self.physical_plan.graph, source_p_id, output_p_id
        )
        if len(all_paths) != 1:
            raise RuntimeError("Multiple paths...")
        if not include_prefetch and self.options.enable_prefetch:
            path = all_paths[0][:-1]
        else:
            path = all_paths[0]
        return path

    def _has_enough_disk_space(
        self, pipe_id_before_cache: int, path: str = "/"
    ) -> bool:
        """
        Checks whether enough local disk space is available to
        cache all samples after an index of the plan.

        Args:
            pipe_id_before_cache (int): output of this pipie is cached.
            path (str): path in which to check available disk space.

        Returns True if cacheable, False otherwise.
        """
        num_samples = self.options.num_samples
        disk_usage = psutil.disk_usage(path)
        free_bytes = disk_usage.free

        # NOTE: not perfectly accurate as the stats are averages of samples
        # TODO: Only considers the baseline.
        # Also only considers the non-reordered graph, which means
        # it leaves out potential size reductions through reordered ops.
        output_size_per_sample = self.profiled_stats["baseline"][
            "output_sizes"
        ][pipe_id_before_cache]
        total_size = num_samples * output_size_per_sample

        return total_size < free_bytes


def test_optimizer_with_custom_pipeline() -> None:
    """
    测试函数：使用自定义的简单 pipeline 和手工指定的 profile 数据
    （选择率、执行代价等）运行 optimizer，并打印优化后的 pipeline 信息，
    包括 fusion、offload、variant 等。

    运行前请在 cedar 项目根目录下激活环境：
        cd /path/to/cedar && source env/bin/activate
    然后执行：
        python -m cedar.compose.optimizer
    """
    # --- 1. 定义 Mock Pipe（仅用于 optimizer 的图结构，不实际执行）---
    class MockPipe(Pipe):
        """仅满足 optimizer 所需接口的占位 Pipe。"""

        def __init__(
            self,
            name: str,
            input_pipes: List[Pipe],
            is_fusable: bool = True,
            is_fusable_source: bool = False,
        ):
            super().__init__(name, input_pipes)
            self.pipe_spec = CedarPipeSpec(
                is_mutable=True,
                mutable_variants=[
                    PipeVariantType.INPROCESS,
                    PipeVariantType.MULTITHREADED,
                    PipeVariantType.SMP,
                    PipeVariantType.RAY,
                ],
                is_fusable=is_fusable,
                is_fusable_source=is_fusable_source,
            )

    # --- 2. 构建简单 DAG：Source(0) -> Map1(1) -> Map2(2) -> Map3(3) ---
    source = MockPipe("SourcePipe", [], is_fusable_source=True)
    map1 = MockPipe("MapperPipe_1", [source], is_fusable=True)
    map2 = MockPipe("MapperPipe_2", [map1], is_fusable=True)
    map3 = MockPipe("MapperPipe_3", [map2], is_fusable=True)

    source.id = 0
    map1.id = 1
    map2.id = 2
    map3.id = 3

    logical_pipes = {0: source, 1: map1, 2: map2, 3: map3}
    # adj_list: p_id -> 后继节点（该 pipe 输出指向的 pipe id）
    logical_adj_list = {
        0: {1},
        1: {2},
        2: {3},
        3: set(),
    }

    # --- 3. 自定义 profile 数据（选择率、延迟、吞吐、数据大小等）---
    # baseline: 本地执行时的延迟(μs)、每条样本的输入/输出字节、整体吞吐
    baseline_throughput = 100.0  # samples/sec
    profiled_stats: Dict[str, Any] = {
        "baseline": {
            "input_sizes": {0: 100.0, 1: 80.0, 2: 60.0, 3: 40.0},
            "latencies": {0: 2000.0, 1: 3000.0, 2: 2500.0, 3: 2500.0},
            "output_sizes": {0: 80.0, 1: 60.0, 2: 40.0, 3: 40.0},
            "throughput": baseline_throughput,
        },
        "offloads": {
            "RAY": {
                0: {
                    "input_sizes": {0: 100.0, 1: 80.0, 2: 60.0, 3: 40.0},
                    "latencies": {0: 2000.0, 1: 3000.0, 2: 2500.0, 3: 2500.0},
                    "output_sizes": {0: 80.0, 1: 60.0, 2: 40.0, 3: 40.0},
                    "throughput": 120.0,
                },
                1: {
                    "input_sizes": {0: 100.0, 1: 80.0, 2: 60.0, 3: 40.0},
                    "latencies": {0: 2000.0, 1: 2800.0, 2: 2500.0, 3: 2500.0},
                    "output_sizes": {0: 80.0, 1: 60.0, 2: 40.0, 3: 40.0},
                    "throughput": 150.0,
                },
                2: {
                    "input_sizes": {0: 100.0, 1: 80.0, 2: 60.0, 3: 40.0},
                    "latencies": {0: 2000.0, 1: 3000.0, 2: 2200.0, 3: 2500.0},
                    "output_sizes": {0: 80.0, 1: 60.0, 2: 40.0, 3: 40.0},
                    "throughput": 160.0,
                },
                3: {
                    "input_sizes": {0: 100.0, 1: 80.0, 2: 60.0, 3: 40.0},
                    "latencies": {0: 2000.0, 1: 3000.0, 2: 2500.0, 3: 2300.0},
                    "output_sizes": {0: 80.0, 1: 60.0, 2: 40.0, 3: 40.0},
                    "throughput": 180.0,
                },
            },
            "SMP": {},
        },
    }

    # --- 4. 创建 Optimizer 并运行 ---
    opt = Optimizer()
    opt.init(logical_pipes, logical_adj_list)
    options = OptimizerOptions(
        enable_prefetch=True,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=False,
    )
    plan = opt.run(profiled_stats, options)

    # --- 5. 打印优化后的 pipeline（含 fusion、offload、variant）---
    print("\n" + "=" * 60)
    print("优化后的 Pipeline (Physical Plan)")
    print("=" * 60)
    print("\n[图结构] graph: pipe_id -> 后继 pipe_id 集合")
    for p_id, succs in sorted(plan.graph.items()):
        print(f"  {p_id} -> {succs}")

    print("\n[算子与执行方式] pipe_descs:")
    for p_id in sorted(plan.pipe_descs.keys()):
        desc = plan.pipe_descs[p_id]
        variant = desc.variant_type.name if desc.variant_type else "None"
        ctx = ""
        if desc.variant_ctx is not None:
            ctx = desc.variant_ctx.serialize()
        fused = ""
        if desc.fused_pipes is not None and len(desc.fused_pipes) > 1:
            fused = f" [FUSED: {desc.fused_pipes}]"
        elif desc.fused_pipes is not None and len(desc.fused_pipes) == 1:
            fused = " [单算子]"
        print(f"  pipe_id={p_id}: name={desc.name!r} variant={variant} "
              f"variant_ctx={ctx}{fused}")

    print(f"\n[本地并行度] n_local_workers = {plan.n_local_workers}")

    # 汇总 fusion / offload 信息
    fused = [
        (p_id, desc.fused_pipes)
        for p_id, desc in plan.pipe_descs.items()
        if desc.fused_pipes is not None and len(desc.fused_pipes) > 1
    ]
    offloaded = [
        p_id
        for p_id, desc in plan.pipe_descs.items()
        if desc.variant_type is not None
        and desc.variant_type.name in ("RAY", "TF_RAY", "SMP")
    ]
    print("\n[Reorder 结果校验]")
    original_order = list(logical_adj_list.keys())
    print(f"  原始逻辑顺序 (按 graph 拓扑): {original_order}")
    reordered_logical = None
    if fused:
        for _p_id, orig_ids in fused:
            reordered_logical = orig_ids
            break
    if reordered_logical is not None:
        print(f"  Reorder 后融合段内的逻辑顺序: {reordered_logical}")
        assert set(reordered_logical) == set(logical_pipes.keys()), (
            "Reorder 后 fused 的 pipe id 集合应与逻辑图一致"
        )
        if reordered_logical != original_order:
            print("  -> 与原始顺序不同，reorder 已生效。")
        else:
            print("  -> 与原始顺序相同。")
    assert plan.validate(), "Physical plan 应通过 validate"

    print("\n[Fusion 汇总] 融合后的算子:")
    if fused:
        for p_id, orig_ids in fused:
            print(f"  pipe_id {p_id} 由原算子 {orig_ids} 融合")
    else:
        print("  (无多算子融合)")
    print("\n[Offload 汇总] 已 offload 的算子 (RAY/SMP/TF_RAY):")
    if offloaded:
        print(f"  pipe_ids: {offloaded}")
    else:
        print("  (无)")

    print("\n[完整序列化] plan.to_dict():")
    print(yaml.dump(plan.to_dict(), default_flow_style=False, allow_unicode=True))
    print("=" * 60)


if __name__ == "__main__":
    # 确保 optimizer 使用的 optimizer pipes（如 PrefetcherPipe）已注册
    import cedar.pipes.optimize  # noqa: F401
    test_optimizer_with_custom_pipeline()
