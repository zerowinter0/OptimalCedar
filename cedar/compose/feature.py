import abc
import logging
import yaml
import copy
import threading
import inspect
import os

from pathlib import Path
from typing import Any, Iterable, List, Dict, Tuple, Optional, Set, Union

from cedar.config import CedarContext
from cedar.pipes import (
    Pipe,
    PipeVariantType,
    PipeVariantContext,
    PipeVariantContextFactory,
    MutationError,
)
from cedar.sources import Source
from cedar.pipes.optimize import OptimizerPipeRegistry, FusedOptimizerPipe

from .optimizer import Optimizer, OptimizerOptions, PhysicalPlan
from .utils import (
    traverse_feature_graph,
    viz_graph,
    topological_sort,
    flip_adj_list,
    get_output_pipe,
)
from .constants import FUSED_PIPE_NAME

logger = logging.getLogger(__name__)


def _positive_resource_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid profile resource signature {name}={value!r}"
        ) from exc
    if parsed < 1:
        raise RuntimeError(
            f"Profile resource signature {name} must be >= 1"
        )
    return parsed


def apply_profile_matched_resources(
    plan: PhysicalPlan,
    profiled_stats: Dict[str, Any],
    cpu_budget: int,
    fixed_local_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Bind a physical plan to the exact per-stage profile width.

    Local workers replicate every active Ray/SMP stage. With one CPU for each
    local worker process, the unified single-node budget is
    W * (1 + K_ray * A_ray + K_smp * A_smp).
    """
    if cpu_budget < 1:
        raise RuntimeError(
            "CEDAR_PROFILE_MATCH_CPU_BUDGET must be >= 1, "
            f"got {cpu_budget}"
        )

    config = profiled_stats.get("resource_config")
    if not isinstance(config, dict):
        raise RuntimeError(
            "Profile resource matching requested, but profile YAML has no "
            "resource_config signature. Re-profile with explicit actor widths."
        )
    if config.get("schema_version") != 1:
        raise RuntimeError(
            "Unsupported profile resource signature schema_version="
            f"{config.get('schema_version')!r}"
        )
    if config.get("profile_scope") != "single_local_worker":
        raise RuntimeError(
            "Profile resource matching requires "
            "profile_scope=single_local_worker"
        )
    if _positive_resource_int(
        config.get("profile_local_workers"), "profile_local_workers"
    ) != 1:
        raise RuntimeError(
            "Profile must have been collected with one local worker"
        )

    ray_width = _positive_resource_int(
        config.get("ray_actors_per_stage"), "ray_actors_per_stage"
    )
    smp_width = _positive_resource_int(
        config.get("smp_procs_per_stage"), "smp_procs_per_stage"
    )
    if (
        ray_width != 1
        or smp_width != 1
        or config.get("actors_per_stage") != 1
    ):
        raise RuntimeError(
            "This experiment requires actors_per_stage=1 for both Ray and SMP; "
            f"profile has ray={ray_width}, smp={smp_width}, "
            f"actors_per_stage={config.get('actors_per_stage')!r}"
        )

    active_descs = [plan.pipe_descs[p_id] for p_id in plan.graph]
    active_ray_ds = [
        p_id
        for p_id in plan.graph
        if plan.pipe_descs[p_id].variant_type == PipeVariantType.RAY_DS
    ]
    if active_ray_ds:
        raise RuntimeError(
            "Strict profile resource matching does not support active "
            "RAY_DS stages because their Ray Data task width is not bound "
            "by ray_actors_per_stage=1; active pipe IDs: "
            f"{active_ray_ds}"
        )
    ray_descs = [
        desc
        for desc in active_descs
        if desc.variant_type
        in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
    ]
    smp_descs = [
        desc
        for desc in active_descs
        if desc.variant_type == PipeVariantType.SMP
    ]
    per_worker_cpus = (
        1 + len(ray_descs) * ray_width + len(smp_descs) * smp_width
    )
    if per_worker_cpus > cpu_budget:
        raise RuntimeError(
            "Final plan cannot fit one local worker under the unified CPU "
            f"budget: per_worker={per_worker_cpus}, budget={cpu_budget}"
        )

    max_local_workers = cpu_budget // per_worker_cpus
    if fixed_local_workers is None:
        local_workers = max_local_workers
        local_worker_policy = "max_under_cpu_budget"
    else:
        local_workers = _positive_resource_int(
            fixed_local_workers, "fixed_local_workers"
        )
        if local_workers > max_local_workers:
            raise RuntimeError(
                "Fixed local-worker ablation exceeds the unified CPU budget: "
                f"fixed={local_workers}, max={max_local_workers}"
            )
        local_worker_policy = "fixed_ablation"
    plan.set_local_workers(local_workers)
    for desc in ray_descs:
        desc.variant_ctx.n_actors = ray_width
    for desc in smp_descs:
        desc.variant_ctx.n_procs = smp_width

    signature = {
        "cpu_budget": cpu_budget,
        "local_workers": local_workers,
        "local_worker_policy": local_worker_policy,
        "ray_stages": len(ray_descs),
        "smp_stages": len(smp_descs),
        "ray_actors_per_stage_per_worker": ray_width,
        "smp_procs_per_stage_per_worker": smp_width,
        "global_ray_actors": (
            local_workers * len(ray_descs) * ray_width
        ),
        "global_smp_procs": (
            local_workers * len(smp_descs) * smp_width
        ),
        "total_accounted_cpus": local_workers * per_worker_cpus,
    }

    if signature["total_accounted_cpus"] > cpu_budget:
        raise RuntimeError(
            f"Resource policy exceeded CPU budget: {signature}"
        )
    if any(
        desc.variant_ctx.n_actors != ray_width for desc in ray_descs
    ):
        raise RuntimeError(
            "Ray plan width does not match profile resource signature"
        )
    if any(desc.variant_ctx.n_procs != smp_width for desc in smp_descs):
        raise RuntimeError(
            "SMP plan width does not match profile resource signature"
        )

    logger.warning(
        "Applied and verified profile-matched resource signature: %s",
        signature,
    )
    return signature


# make this a decorator?
class Feature(abc.ABC):
    """
    A Feature is a composition of Pipes, representing
    a dataflow graph that can be applied to some input Source.

    To define a Feature, subclass this function
    and override ``_compose()``

    Features are not associated with Sources. Calling
    ``apply()`` on a feature binds the Feature
    to Source(s), and must be done after ``_compose()``

    NOTE: If subclasses of features provide a constructor with arguments
        (other than self), the subclass *must* create an attribute
        with the **same name** as the argument. Alternatively,
        you may override the `create_copy` method to appropriately
        instantiate a copy of your subclassed feature.

        If you expect to modify your feature's state after creation
        (other than calling apply()), you **must** override the `create_copy`
        method to ensure those effects are carried over to the copy.

    Example:
        class CropFeature(Feature):
            def _compose(self, sources):
                # define transform graph

        img_source = Source(...)
        crop_feature = CropFeature(...)

        dataset = crop_feature.apply([img_source])

        for batch in dataset:
            # train
    """

    def __init__(self):
        self.output_pipe: Optional[Pipe] = None
        self.loaded: bool = False

        # Adj list represents outgoing edges
        self.logical_adj_list: Dict[int, Set[int]] = {}
        self.logical_pipes: Dict[int, Pipe] = {}
        self.source_pipes: List[Pipe] = []
        self.sources: List[Source] = []

        # Adj list represents outgoing edges
        # Physical pipes can contain additional pipes not specified by
        # the logical feature.
        self.physical_adj_list: Dict[int, Set[int]] = {}
        self.physical_pipes: Dict[int, Pipe] = {}

        self._has_changed_dataflow_flag = True

        self._lock = threading.RLock()

        # map of pipe ID to pipes that are fused within that pipe
        self.fused_pipes: Dict[int, List[int]] = {}

        self.optimizer = Optimizer()
        self._plan = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """
        Replace the optimizer implementation for this feature.

        If the logical graph has already been constructed (i.e. sources have
        been applied), re-initialize the provided optimizer with the current
        logical pipes/adjacency list so it can be used immediately.
        """
        self.optimizer = optimizer
        if self.logical_pipes and self.logical_adj_list:
            self.optimizer.init(self.logical_pipes, self.logical_adj_list)

    @abc.abstractmethod
    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        """
        Composes together a DAG of Pipes, representing
        transformations applied to a set of Sources.

        Args:
            source_pipes: List of Pipes generated from loading Sources
        """
        pass

    def apply(self, *sources: Source):
        """
        Applies a list of sources to the Feature.
        Note the list is ordered, and will be provided to the
        ``_compose()`` function in the same order.

        Args:
            sources: List of Sources to apply to this feature
        """
        with self._lock:
            if len(sources) > 1:
                raise NotImplementedError(
                    "No support for one or more sources\
                                           pipes yet."
                )
            self.sources = sources

            self.source_pipes = [source.to_pipe() for source in sources]
            self._apply()

    def load(
        self,
        ctx: CedarContext,
        prefetch: bool = False,
    ) -> Iterable[Any]:
        """
        Schedules the execution of this feature across physical substrates.

        Args:
            ctx: CedarContext for this session
            prefetch: Insert a prefetch pipe at the end of the feature.
                Defaults to false.

        Returns:
            An iterable that iterates over the feature
        """
        if self.loaded:
            raise RuntimeError("Cannot load a loaded feature...")

        with self._lock:
            pipe_order = topological_sort(self.logical_adj_list)
            for p_id in pipe_order:
                if (
                    self.logical_pipes[p_id].is_tf()
                    and PipeVariantType.TF
                    in self.logical_pipes[p_id].get_spec().mutable_variants
                ):
                    self.logical_pipes[p_id].mutate(ctx, PipeVariantType.TF)
                else:
                    self.logical_pipes[p_id].mutate(
                        ctx, PipeVariantType.INPROCESS
                    )

                self.physical_pipes[p_id] = self.logical_pipes[p_id]
                self.physical_adj_list[p_id] = self.logical_adj_list[
                    p_id
                ].copy()
            if prefetch:
                self._insert_prefetch(ctx)

            # Set the output pipe
            self.output_pipe = self._get_output_pipe()
            logger.info(f"Set output pipe to: {self.output_pipe.id}")
            self.loaded = True
            return self.output_pipe.pipe_variant

    def viz_logical_plan(self, path: str) -> None:
        if not self.logical_pipes:
            raise RuntimeError(
                "Must apply a feature first before visualization"
            )
        viz_graph(self.logical_pipes, self.logical_adj_list, path, True)

    def viz_physical_plan(self, path: str) -> None:
        # TODO (myzhao): update this to use a phys plan
        if not self.loaded:
            raise RuntimeError(
                "Must load a feature first before visualization"
            )
        viz_graph(self.physical_pipes, self.physical_adj_list, path, False)

    def serialize_plan(
        self,
    ) -> Tuple[Dict[int, Dict[str, Any]], Optional[Dict[int, Dict[str, Any]]]]:
        """
        Returns a tuple containing the logical and physical (if loaded) plan
        for this feature.

        NOTE: Serialization does not preserve input pipe ordering.

        Returns:
            Tuple(
                Dict[int, Dict]: logical plan, with each Pipe represented
                    as a Dict
                Optional[Dict[int, Dict]]: If loaded, physical plan with
                    each Pipe represented as a Dict
            )
        """
        logical_plan = self._serialize_plan(
            self.logical_adj_list, self.logical_pipes
        )

        physical_plan = None
        if self.loaded:
            if self._plan is not None:
                physical_plan = self._plan.to_dict()
            else:
                physical_plan = self._serialize_plan(
                    self.physical_adj_list, self.physical_pipes, True
                )
        return logical_plan, physical_plan

    def load_from_plan(
        self, ctx: CedarContext, plan: PhysicalPlan
    ) -> Iterable[Any]:
        """
        Given a physical plan, load this feature according to that plan.

        NOTE:
            Insertions not currently supported. Only reorderings

        Args:
            plan: PhysicalPlan specifying the physical plan of this feature

        Returns:
            The loaded iterable over the feature
        """
        if self.loaded:
            raise RuntimeError("Cannot load a loaded feature...")
        with self._lock:
            self._check_physical_plan(plan)
            self._plan = plan

            # Shallow copy pipes
            self.physical_pipes = copy.copy(self.logical_pipes)

            self.physical_adj_list = plan.graph
            sorted_plan = topological_sort(self.physical_adj_list)
            input_adj_list = flip_adj_list(self.physical_adj_list)

            # Go in topological order, re-build the logical and physical plan
            for p_id in sorted_plan:
                # Create new optimizer pipes if necessary
                if p_id not in self.physical_pipes:
                    p_name = plan.pipe_descs[p_id].name
                    new_pipe_cls = OptimizerPipeRegistry.get_pipe(p_name)

                    # Special case for fused pipes
                    if p_name == FUSED_PIPE_NAME:
                        logger.info(f"Creating new fused pipe {p_id}")
                        fused_p_ids = plan.pipe_descs[p_id].fused_pipes
                        if fused_p_ids is None or len(fused_p_ids) == 0:
                            raise RuntimeError("Fused pipe desc not found")
                        fused_pipes = [
                            self.physical_pipes[i] for i in fused_p_ids
                        ]
                        self.physical_pipes[p_id] = new_pipe_cls(fused_pipes)
                        self.physical_pipes[p_id].id = p_id
                    else:
                        logger.info(f"Created new Optimizer Pipe {p_name}.")
                        self.physical_pipes[p_id] = new_pipe_cls()
                        self.physical_pipes[p_id].id = p_id

                # Is this a fuse in place?
                if plan.pipe_descs[p_id].is_inplace_fuse():
                    fused_pipes = [
                        self.physical_pipes[i]
                        for i in plan.pipe_descs[p_id].fused_pipes
                    ]

                    # TODO: merge these apis into one
                    if (
                        not self.physical_pipes[p_id].get_spec().is_fusable
                        and not self.physical_pipes[p_id]
                        .get_spec()
                        .is_fusable_source
                    ):
                        raise RuntimeError(f"{p_id} is not fusable")

                    if (
                        plan.pipe_descs[p_id].variant_type
                        == PipeVariantType.RAY_DS
                        and ctx.use_ray()
                    ):
                        raise RuntimeError(
                            "Detected ray data fused source with "
                            " --use_ray enabled."
                            " For optimal performance, simply connect "
                            " this node to the ray cluster and load from "
                            " plan"
                        )

                    self.physical_pipes[p_id].fuse_into(fused_pipes)

                    logger.info(
                        "Fusing {} into {} in-place".format(
                            plan.pipe_descs[p_id].fused_pipes, p_id
                        )
                    )

                # NOTE: input order not preserved
                input_pipes = [
                    self.physical_pipes[i] for i in input_adj_list[p_id]
                ]
                self.physical_pipes[p_id].set_input_pipes(input_pipes)
                p_variant = plan.pipe_descs[p_id].variant_type
                variant_ctx = plan.pipe_descs[p_id].variant_ctx

                self.physical_pipes[p_id].mutate(ctx, p_variant, variant_ctx)

                logger.info(
                    f"Mutated {plan.pipe_descs[p_id].name} "
                    f"into {p_variant.name}."
                )

            self.loaded = True
            self._has_changed_dataflow_flag = True

            self.output_pipe = self._get_output_pipe()
            return self.output_pipe.pipe_variant

    def to_yaml(self, path: str) -> None:
        """
        Outputs this feature's graph to a YAML file.
        Saves both the logical plan, as well as the physical plan
        if the feature is loaded.
        """
        logical_plan, physical_plan = self.serialize_plan()
        d = {"logical_plan": logical_plan}
        if physical_plan:
            d["physical_plan"] = physical_plan
        with open(path, "w") as f:
            yaml.dump(d, f)

    def load_from_yaml(self, ctx: CedarContext, path: str) -> Iterable[Any]:
        """
        Load a physical plan for this feature from a yaml config
        """
        if not self.logical_adj_list:
            raise RuntimeError(
                "Features must have sources applied prior to loading."
            )
        with open(path, "r") as f:
            d = yaml.safe_load(f)

        if "physical_plan" not in d:
            raise RuntimeError(f"Unable to parse physical plan from {path}")

        return self.load_from_dict(ctx, d["physical_plan"])

    def load_from_dict(self, ctx: CedarContext, d: Dict):
        plan = PhysicalPlan.from_dict(d)
        return self.load_from_plan(ctx, plan)

    def reset(self) -> None:
        """
        Resets the physical plan of this feature.
        Also resets the logical plan to the original specified by _compose

        NOTE: If during mutation, source pipes were changed, this function
        does not reset the source pipes. Call apply() again before resetting
        to reset the source pipes.
        """
        with self._lock:
            logger.info("Resetting feature")
            for _, v in self.physical_pipes.items():
                v.reset()

            self._apply()
            self.physical_adj_list = {}
            self.physical_pipes = {}
            self.loaded = False
            self._has_changed_dataflow_flag = True

    def release_profile_resources(self) -> None:
        """Release workload-owned resources between profiling trials.

        Most Cedar operators own only ordinary Python objects and therefore
        need no action here. Workloads backed by process-global model caches
        may override this hook so repeated backend trials do not retain a new
        copy of every accelerator model for the lifetime of the profiler.
        The hook is deliberately profiling-only: normal execution keeps model
        caches warm for the duration of an epoch.
        """

        return None

    def _apply(self) -> None:
        self.output_pipe = self._compose(self.source_pipes)

        if self.output_pipe is None:
            raise RuntimeError("Ensure that _compose() returns a Pipe.")

        try:
            self.logical_pipes, self.logical_adj_list = traverse_feature_graph(
                self.output_pipe
            )

            # Also initialize the optimizer
            self.optimizer.init(self.logical_pipes, self.logical_adj_list)
        except AttributeError:
            raise RuntimeError("Ensure that _compose() returns a Pipe.")

    def _check_physical_plan(self, plan: PhysicalPlan):
        """
        Parse physical plan, returns dict of pipe ID to each pipe dict.
        """
        if self.loaded:
            raise RuntimeError("Cannot load plan on loaded feature.")

        if not isinstance(plan, PhysicalPlan):
            raise ValueError("Plan is formatted incorrectly.")
        if not plan.validate():
            raise ValueError("Plan is formatted incorrectly.")

        # Parse the pipes in the plan
        for p_id, desc in plan.pipe_descs.items():
            if p_id not in self.logical_pipes:
                logger.info(f"Detected extra pipe {desc.name}")
            # If the pipe is present in the logical plan, make sure it
            # has the same ID
            elif self.logical_pipes[p_id].get_logical_name() != desc.name:
                print("==================")
                print(self.logical_pipes[p_id].get_logical_name())
                print(desc.name)
                logical_name = self.logical_pipes[p_id].get_logical_name()
                raise RuntimeError(
                    f"Parsed pipe {desc.name} does not "
                    f" match logical pipe {logical_name}"
                )

        # Check that all pipes specified by the graph is a superset of
        # all logical pipes specified by the plan
        covered_pipes = set()
        for p_id in plan.graph.keys():
            if p_id not in plan.pipe_descs:
                raise RuntimeError(f"{p_id} found in graph but not desc.")
            if plan.pipe_descs[p_id].fused_pipes is not None:
                covered_pipes.update(plan.pipe_descs[p_id].fused_pipes)
            covered_pipes.add(p_id)
        if not covered_pipes.issuperset(set(self.logical_pipes.keys())):
            raise RuntimeError("Not all pipes specified in physical plan.")

    def _serialize_plan(self, adj_list, pipes, physical=False):
        serialized_plan = {}
        serialized_pipes = {}

        sorted_nodes = topological_sort(adj_list)

        for p_id in sorted_nodes:
            pipe = pipes[p_id]
            pipe_desc = {
                "name": pipe.name,
            }
            if physical:
                pipe_desc["variant"] = pipe.pipe_variant_type.name
            serialized_pipes[p_id] = pipe_desc

        formatted_output_graph = {}
        for p_id, adj_pids in adj_list.items():
            formatted_output_graph[p_id] = ",".join(map(str, adj_pids))
        serialized_plan["pipes"] = serialized_pipes
        serialized_plan["graph"] = formatted_output_graph

        return serialized_plan

    def _get_output_pipe(self):
        """
        Returns the output pipe for this feature
        NOTE: Multiple output nodes not supported
        """
        return self.physical_pipes[get_output_pipe(self.physical_adj_list)]

    def get_source_pipes(self) -> List[Pipe]:
        """
        Returns the source pipe for this feature.
        If no source pipe is detected, raises AssertionError.
        NOTE: If more than one source pipe is detected,
        issues a warning.
        NOTE: Multiple source pipes
        """
        source_pipes = [
            self.physical_pipes[pipe_id]
            for pipe_id in self.physical_pipes
            if self.physical_pipes[pipe_id].is_source()
            and self.is_active_pipe(pipe_id)
        ]

        if len(source_pipes) < 1:
            raise AssertionError("Could not find source node.")
        elif len(source_pipes) > 1:
            logging.warning(
                "CAREFUL: Detected more than 1 source pipe.\
                            Cedar currently does not offer support for more\
                            than 1 source pipe."
            )

        return source_pipes

    def _insert_prefetch(self, ctx: CedarContext) -> None:
        prefetch_pipe_cls = OptimizerPipeRegistry.get_pipe("PrefetcherPipe")
        prefetch_pipe = prefetch_pipe_cls()
        new_p_id = max(self.physical_pipes.keys()) + 1
        self.physical_pipes[new_p_id] = prefetch_pipe
        self.physical_adj_list[new_p_id] = set()
        self.physical_adj_list[self.output_pipe.id].add(new_p_id)
        prefetch_pipe.id = new_p_id
        logger.info(f"Inserted prefetch pipe with ID {new_p_id}")

        prefetch_pipe.set_input_pipes([self.output_pipe])
        prefetch_pipe.mutate(ctx, PipeVariantType.INPROCESS)

        self.output_pipe = prefetch_pipe

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

    def has_changed_dataflow(self):
        """
        Returns true if the feature has changed its dataflow graph
        since the last time this function was checked
        """
        if self._has_changed_dataflow_flag:
            self._has_changed_dataflow_flag = False
            return True
        else:
            return False

    def get_pipe(self, p_id: int) -> Pipe:
        """
        Returns the Pipe object specified by p_id
        """
        with self._lock:
            if p_id not in self.physical_pipes:
                logger.warning(
                    "Pipe {} not found as a physical pipe".format(p_id)
                )
                return self.logical_pipes[p_id]

            if p_id in self.logical_pipes:
                # p_id should be in physical_pipes as well
                if self.logical_pipes[p_id] != self.physical_pipes[p_id]:
                    raise RuntimeError(
                        "Mismatch between phys and logical pipes"
                    )

            return self.physical_pipes[p_id]

    def dynamic_mutate(
        self,
        p_id: int,
        pipe_variant_type: PipeVariantType,
        serialized_ctx: Optional[Dict[str, Any]] = None,
        wait_for_mutation: bool = True,
    ) -> bool:
        """
        Dynamically mutates p_id to a new variant type. Returns True
        on success.

        NOTE: The caller should wait for any mutations on p_id to complete
        before calling this.
        """
        if not self.is_active_pipe(p_id):
            raise RuntimeError(f"{p_id} is not an active pipe")

        if wait_for_mutation:
            self.wait_for_mutation(p_id)
        ctx = None
        if serialized_ctx is not None:
            ctx = PipeVariantContextFactory.create_context(
                pipe_variant_type, serialized_ctx
            )

        with self._lock:
            pipe = self.physical_pipes[p_id]

            # NOTE: only supports linear features for now
            downstream_pipe_ids = self.physical_adj_list[p_id]

            if len(downstream_pipe_ids) == 0:
                downstream_pipe = None
            elif len(downstream_pipe_ids) == 1:
                downstream_p_id = list(downstream_pipe_ids)[0]
                downstream_pipe = self.physical_pipes[downstream_p_id]
            else:
                logger.error("Only linear pipes supported")
                return False

        pipe.dynamic_mutate(pipe_variant_type, downstream_pipe, ctx)
        if wait_for_mutation:
            self.wait_for_mutation(p_id)
        return True

    def reset_pipe(
        self,
        p_id: int,
        serialized_ctx: Optional[Dict[str, Any]] = None,
        wait_for_mutation: bool = True,
    ) -> Dict[str, Any]:
        """
        Reset a given pipe. If serialized_ctxs is provided, reset
        the p_id to its serialized_ctx. Otherwise, reset to an INPROCESS
        variant.

        Returns a Dict mapping its pipe_id to its serialized_ctx prior
        to reset. Returns an empty dict if not active.
        """
        if not self.is_active_pipe(p_id):
            # Not an active pipe
            logger.warning(
                "Attempting to reset non-active pipe {}".format(p_id)
            )
            return {}
        with self._lock:
            ctx = self.physical_pipes[p_id].get_variant().serialize()
        # if not self.physical_pipes[p_id].wait_until_mutate_ready(10):
        #     raise MutationError(f"Timed out waiting for {p_id} to mutate.")
        if p_id in self.fused_pipes:
            if serialized_ctx is not None:
                logger.warning("Cannot reset a fused pipe to a serialized ctx")
            if not self.dynamic_split(p_id, wait_for_mutation):
                raise MutationError(f"Failed to split {p_id}")
        else:
            if serialized_ctx is not None:
                if not self.mutate_to(p_id, serialized_ctx, wait_for_mutation):
                    raise MutationError(f"Failed to mutate {p_id}")
            else:
                if not self.dynamic_mutate(
                    p_id,
                    PipeVariantType.INPROCESS,
                    serialized_ctx=None,
                    wait_for_mutation=wait_for_mutation,
                ):
                    raise MutationError(f"Failed to mutate {p_id}")
        return ctx

    def get_aggregate_scale(self) -> Dict[PipeVariantType, int]:
        """
        Returns a dict mapping the pipe variant type to the total
        amount of resources used by all pipes of that variant
        """
        d = {}
        with self._lock:
            for p_id, pipe in self.physical_pipes.items():
                # Only active pipes
                if self.is_active_pipe(p_id):
                    scale = pipe.pipe_variant.get_scale()
                    variant_type = pipe.get_variant_type()

                    if variant_type not in d:
                        d[variant_type] = scale
                    else:
                        d[variant_type] += scale

        return d

    def get_batch_size(self) -> int:
        """
        Returns the batch size of this feature
        """
        b_sz = 1
        with self._lock:
            for p_id, pipe in self.physical_pipes.items():
                # If multiple batch pipes, multiply sizes
                if pipe.name.startswith(
                    "BatcherPipe("
                ) and self.is_active_pipe(p_id):
                    b_sz = b_sz * pipe.batch_size
        return b_sz

    def get_next_free_pipe_id(self) -> int:
        if len(self.physical_pipes) == 0:
            return 0
        else:
            return max(self.physical_pipes, key=int) + 1

    def dynamic_fusion(
        self, p_ids: List[int], wait_for_mutation: bool = True
    ) -> int:
        """
        Dynamically fuse the pipes specified into one pipe.

        Args:
            p_ids: Ordered list of physical pipe ids to fuse together.
                The list must be in order, with each successive pipe
                being the next pipe in the dataflow graph.

        NOTE: p_ids must not include any fused pipes.
                    The caller must reset any fused pipe to INPROCESS
                    before calling this function
                AND wait for the reset to complete.
        Returns:
            the ID of the fused pipe
        Raises:
            MutationError on failure
        """
        if len(p_ids) < 2:
            raise MutationError("Cannot fuse fewer than 2 pipes")
        for p_id in p_ids:
            if p_id in self.fused_pipes:
                raise ValueError(f"{p_id} is a fused pipe")
            if not self.is_active_pipe(p_id):
                raise RuntimeError(f"{p_id} is not an active pipe.")
        logger.info("Calling dynamic fusion on {}".format(p_ids))
        if wait_for_mutation:
            self.wait_for_mutation(p_ids)

        with self._lock:
            leaf_pipes = [self.physical_pipes[i] for i in p_ids]
            for p in leaf_pipes:
                if p.get_variant_type() != PipeVariantType.INPROCESS:
                    raise MutationError("Cannot fuse non-inprocess pipes")

            # Fuse all of the pipes into one.
            fused_pipe = FusedOptimizerPipe(leaf_pipes)
            fused_pipe_id = self.get_next_free_pipe_id()
            self.physical_pipes[fused_pipe_id] = fused_pipe
            fused_pipe.id = fused_pipe_id

            self.fused_pipes[fused_pipe_id] = p_ids

            # Set the input pipe for the fused pipe
            head_pipe = self.physical_pipes[p_ids[0]]
            input_pipes = head_pipe.input_pipes.copy()
            fused_pipe.set_input_pipes(input_pipes)

            # Update the graph
            if len(head_pipe.input_pipes) != 1:
                raise MutationError("Cannot fuse with multiple inputs")
            input_pipe_id = head_pipe.input_pipes[0].id
            if input_pipe_id is None:
                raise MutationError("Input pipe ID is not set.")
            if len(self.physical_adj_list[input_pipe_id]) != 1:
                raise MutationError("Cannot fuse with multiple outputs")
            self.physical_adj_list[input_pipe_id] = {fused_pipe_id}

            # Get the downstream pipe
            downstream_pipe_ids = self.physical_adj_list[p_ids[-1]]
            downstream_p_id = list(downstream_pipe_ids)[0]
            if len(downstream_pipe_ids) != 1:
                raise MutationError(f"Invalid downstream pipe for {p_ids[-1]}")
            downstream_p_id = list(downstream_pipe_ids)[0]
            downstream_pipe = self.physical_pipes[downstream_p_id]
            downstream_pipe.set_input_pipes([fused_pipe])
            self._has_changed_dataflow_flag = True

            self.physical_adj_list[fused_pipe_id] = {downstream_p_id}

            # Clear the old pipes from the graph
            for p_id in p_ids:
                del self.physical_adj_list[p_id]

        # Freeze execution of the downstream pipe
        with downstream_pipe.get_variant().lock:
            # Insert the fused pipe
            # TODO: fix this. Just use SMP for now
            fused_pipe.dynamic_insert_pipe(
                PipeVariantType.SMP, head_pipe, downstream_pipe
            )
            # Clear the old pipes
            for pipe in leaf_pipes:
                pipe.shutdown_variant()
        if wait_for_mutation:
            self.wait_for_mutation(fused_pipe_id)
        return fused_pipe_id

    def wait_for_mutation(self, pipe_ids: Union[int, List[int]]) -> None:
        """
        Waits until all pipes in pipe_ids are done mutating
        """
        if isinstance(pipe_ids, list):
            for p_id in pipe_ids:
                logger.info("Waiting for {} to mutate".format(p_id))
                if not self.physical_pipes[p_id].wait_until_mutate_ready(100):
                    raise MutationError(
                        f"Timed out waiting for {p_id} to mutate."
                    )
                logger.info("Pipe {} finished mutation".format(p_id))
        else:
            logger.info("Waiting for {} to mutate".format(pipe_ids))
            if not self.physical_pipes[pipe_ids].wait_until_mutate_ready(100):
                raise MutationError(
                    f"Timed out waiting for {pipe_ids} to mutate."
                )
            logger.info("Pipe {} finished mutation".format(pipe_ids))

    def reset_pipes(
        self,
        pipe_ids: List[int],
        serialized_ctxs: Optional[Dict[int, Dict[str, Any]]] = None,
        wait_for_mutation: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Reset a given pipe. If serialized_ctxs is provided, reset
        the p_id to its serialized_ctx. Otherwise, reset to an INPROCESS
        variant.

        Returns a Dict mapping its pipe_id to its serialized_ctx prior
        to reset
        """
        ctxs = {}
        for p_id in pipe_ids:
            if p_id not in self.physical_pipes:
                raise MutationError("Pipe {} not found".format(p_id))

            # if not self.physical_pipes[p_id].wait_until_mutate_ready(10):
            #   raise MutationError(f"Timed out waiting for {p_id} to mutate.")

            if serialized_ctxs is not None and p_id in serialized_ctxs:
                ctxs[p_id] = self.reset_pipe(
                    p_id, serialized_ctxs[p_id], wait_for_mutation
                )
            else:
                ctxs[p_id] = self.reset_pipe(p_id, None, wait_for_mutation)
        return ctxs

    def get_active_pipes(self) -> List[int]:
        """
        Returns a list of all active pipe ids
        """
        with self._lock:
            return [k for k in self.physical_adj_list]

    def get_active_pipe_variants(self) -> Dict[PipeVariantType, Set[int]]:
        """
        Returns a dict of pipe variants to all active pipes of that variant
        """
        with self._lock:
            active_pipe_ids = self.get_active_pipes()
            d = {}
            for p in active_pipe_ids:
                variant = self.physical_pipes[p].get_variant_type()
                if variant not in d:
                    d[variant] = {p}
                else:
                    d[variant].add(p)

            return d

    def is_fusable(
        self, p_id: int, pipe_variant_type: PipeVariantType
    ) -> bool:
        """
        Returns if the pipe with id p_id is fusable
        """
        with self._lock:
            return self.physical_pipes[p_id].is_fusable(pipe_variant_type)

    def get_neighbor_ids(self, p_id: int) -> Tuple[int, int]:
        """
        Returns the (Upstream, Downstream) neighbors of p_id
        """
        downstream_pipe = None
        upstream_pipe = None
        with self._lock:
            if p_id not in self.physical_adj_list:
                logger.error(f"{p_id} not active.")
                raise ValueError
            downstream_pipes = self.physical_adj_list[p_id]

            if len(downstream_pipes) > 1:
                logger.error("Multiple pipes not implemetned")
                raise NotImplementedError("Only one pipe supported")
            elif len(downstream_pipes) == 1:
                downstream_pipe = list(downstream_pipes)[0]

            # get upstream pipe
            for k, v in self.physical_adj_list.items():
                # only support one item
                for n in v:
                    if n == p_id:
                        if upstream_pipe is not None:
                            logger.error("Multiple pipes not implemetned")
                            raise NotImplementedError(
                                "Only one pipe supported"
                            )
                        else:
                            upstream_pipe = k

        return (upstream_pipe, downstream_pipe)

    def dynamic_split(self, p_id: int, wait_for_mutation: bool = True) -> bool:
        """
        Splits the pipe designated by p_id to its original pipes, with
        an INPROCESS variant

        NOTE: wait_for_mutation should be True, unless you are calling
            this from the main thread
        """
        logger.info(f"Splitting fused pipe {p_id}")
        if wait_for_mutation:
            self.wait_for_mutation(p_id)

        with self._lock:
            if p_id not in self.fused_pipes:
                logger.error(f"Failed to find {p_id} in fused pipes")
                return False
            original_pipe_ids = self.fused_pipes[p_id]
            original_pipes = [
                self.physical_pipes[x] for x in original_pipe_ids
            ]
            if len(original_pipe_ids) < 2:
                logger.error("Original pipes not valid")
                return False

            upstream_pipe_id, downstream_pipe_id = self.get_neighbor_ids(p_id)

            downstream_pipe = self.physical_pipes[downstream_pipe_id]

            # Update the dataflow graph
            self.physical_adj_list[upstream_pipe_id] = {original_pipe_ids[0]}
            for idx, original_p_id in enumerate(original_pipe_ids[:-1]):
                self.physical_adj_list[original_p_id] = {
                    original_pipe_ids[idx + 1]
                }
            self.physical_adj_list[original_pipe_ids[-1]] = {
                downstream_pipe_id
            }

            # Update the pipes' pointers
            # Assumes that the internal pipes' pointers are still consistent
            pipe = self.physical_pipes[p_id]
            original_pipes[0].set_input_pipes(pipe.input_pipes.copy())
            downstream_pipe.set_input_pipes([original_pipes[-1]])
            self._has_changed_dataflow_flag = True

            # Perform the actual split
            try:
                pipe.dynamic_split(original_pipes, downstream_pipe)
                if wait_for_mutation:
                    self.wait_for_mutation(p_id)
                del self.physical_adj_list[p_id]
                return True
            except MutationError:
                # TODO: error handling
                logger.error(f"Failed to split pipe {p_id}")
                return False

    def mutate_to(
        self,
        p_id: int,
        serialized_ctx: Dict[str, Any],
        wait_for_mutation: bool = True,
    ) -> bool:
        """
        Mutates p_id into a pipe specified by serialized_ctx
        """
        variant_type = PipeVariantType[serialized_ctx["variant_type"]]
        res = self.dynamic_mutate(
            p_id, variant_type, serialized_ctx, wait_for_mutation
        )

        return res

    def profile(
        self,
        ctx: CedarContext,
        mutation_dict: Optional[Dict[int, PipeVariantContext]] = None,
    ) -> Iterable[Any]:
        """
        Configures this feature to run in profile mode.

        Returns the output pipe variant
        """
        if self.loaded:
            raise RuntimeError("Cannot profile a loaded feature.")

        # Load all logical pipes as inprocess
        with self._lock:
            pipe_order = topological_sort(self.logical_adj_list)
            for p_id in pipe_order:
                if mutation_dict is not None and p_id in mutation_dict:
                    variant_ctx = mutation_dict[p_id]
                    self.logical_pipes[p_id].mutate(
                        ctx, variant_ctx.variant_type, variant_ctx
                    )
                else:
                    if (
                        self.logical_pipes[p_id].is_tf()
                        and PipeVariantType.TF
                        in self.logical_pipes[p_id].get_spec().mutable_variants
                    ):
                        self.logical_pipes[p_id].mutate(
                            ctx, PipeVariantType.TF
                        )
                    else:
                        self.logical_pipes[p_id].mutate(
                            ctx, PipeVariantType.INPROCESS
                        )

                self.physical_pipes[p_id] = self.logical_pipes[p_id]
                self.physical_adj_list[p_id] = self.logical_adj_list[
                    p_id
                ].copy()

        # Set the source to enable tracing
        for source_pipe in self.source_pipes:
            source_pipe.get_variant().enable_profiling()

        self.loaded = True
        return self.output_pipe.pipe_variant

    def profile_tf(self, ctx: CedarContext) -> bool:
        if self.loaded:
            raise RuntimeError("Cannot profile a loaded feature.")

        if len(self.source_pipes) != 1:
            return None

        if not self.source_pipes[0].get_spec().is_fusable:
            return None

        for _, pipe in self.logical_pipes.items():
            if not pipe.is_tf():
                return None

        logger.info("Profiling TF-only...")
        source_pipe = self.source_pipes[0]
        source_p_id = source_pipe.id

        # Load all logical pipes as inprocess
        with self._lock:
            pipe_order = topological_sort(self.logical_adj_list)

            source_pipe.fuse_into(
                [self.logical_pipes[x] for x in pipe_order[1:]]
            )
            logger.info(
                "Fusing {} into {} in-place".format(
                    pipe_order[1:], source_p_id
                )
            )

            variant_ctx = PipeVariantContextFactory.create_context(
                PipeVariantType.TF, spec={"num_parallel_calls": -1}
            )

            source_pipe.mutate(ctx, PipeVariantType.TF, variant_ctx)
            self.physical_adj_list = {source_p_id: set()}
            self.physical_pipes = {source_p_id: source_pipe}

        self.loaded = True
        return source_pipe.pipe_variant

    def optimize(
        self, options: OptimizerOptions, profiled_data: str
    ) -> PhysicalPlan:
        """
        Runs an optimizer pass over this feature.

        Returns:
            Dict representing the optimized physical plan of this feature.
        """
        if not Path(profiled_data).is_file():
            raise RuntimeError("{} not found".format(profiled_data))

        plan = self.optimizer.run(profiled_data, options)
        if os.environ.get("CEDAR_MATCH_PROFILE_RESOURCES") == "1":
            cpu_budget_raw = os.environ.get(
                "CEDAR_PROFILE_MATCH_CPU_BUDGET"
            )
            if cpu_budget_raw is None:
                raise RuntimeError(
                    "CEDAR_MATCH_PROFILE_RESOURCES=1 requires "
                    "CEDAR_PROFILE_MATCH_CPU_BUDGET"
                )
            profile = getattr(self.optimizer, "profiled_stats", None)
            if not isinstance(profile, dict):
                raise RuntimeError(
                    "Optimizer did not retain loaded profile statistics"
                )
            fixed_local_workers_raw = os.environ.get(
                "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
            )
            fixed_local_workers = (
                int(fixed_local_workers_raw)
                if fixed_local_workers_raw is not None
                else None
            )
            self.profile_matched_resource_signature = (
                apply_profile_matched_resources(
                    plan, profile, int(cpu_budget_raw), fixed_local_workers
                )
            )
        return plan

    def shard_source(self, rank_spec: Tuple[int, int]):
        """
        Shards this source according to rank_spec
        """
        if len(self.source_pipes) != 1:
            raise RuntimeError("Can only shard 1 source.")
        logger.info(f"Sharding source to {rank_spec}")
        self.source_pipes[0].shard(rank_spec)

    def create_copy(self) -> "Feature":
        """
        Creates and returns a copy of this feature
        """
        logger.info(f"Creating a copy of feature {self.__class__}")
        if self.loaded:
            raise RuntimeError("Cannot create a copy of a loaded feature")
        sig = inspect.signature(self.__class__.__init__)

        # Extract arg names from signature, skip self
        arg_names = list(sig.parameters.keys())[1:]

        # Create dict with arg names to values - deep copy values
        args = {name: copy.deepcopy(getattr(self, name)) for name in arg_names}

        # Create a new object
        new_feature = self.__class__(**args)

        # Apply the original source to the new feature
        sources = self.get_sources()
        new_feature.apply(*sources)

        # Preserve optimizer choice across copies (used by sharding / MP).
        # Replacing the optimizer after apply() is safe because it only
        # re-initializes with the already-built logical graph.
        new_feature.set_optimizer(self.optimizer.__class__())

        return new_feature

    def get_sources(self) -> Tuple[Source]:
        return self.sources

    def is_active_pipe(self, p_id: int) -> bool:
        return p_id in self.physical_adj_list
