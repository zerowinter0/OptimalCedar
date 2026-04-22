from typing import Optional, Union, Dict
from cedar.config import RayConfig


class CedarEvalSpec:
    def __init__(
        self,
        batch_size: int,
        num_total_samples: Optional[int],
        num_epochs: int,
        config: Optional[Union[str, Dict[str, str]]] = None,
        kwargs: Dict[str, str] = None,
        use_ray: bool = False,
        ray_ip: str = "",
        iteration_time: Optional[float] = None,
        profiled_stats: str = "",
        run_profiling: bool = False,
        disable_optimizer: bool = False,
        disable_controller: bool = False,
        disable_prefetch: bool = False,
        disable_offload: bool = False,
        disable_parallelism: bool = False,
        disable_reorder: bool = False,
        disable_fusion: bool = False,
        disable_caching: bool = False,
        use_my_optimizer: bool = False,
        generate_plan: bool = False,
    ):
        self.batch_size = batch_size
        self.num_total_samples = num_total_samples
        self.num_epochs = num_epochs
        self.config = config
        self.kwargs = kwargs
        self.use_ray = use_ray
        self.ray_ip = ray_ip
        self.iteration_time = iteration_time
        self.profiled_stats = profiled_stats
        self.run_profiling = run_profiling
        self.disable_optimizer = disable_optimizer
        self.disable_controller = disable_controller
        self.disable_prefetch = disable_prefetch
        self.disable_offload = disable_offload
        self.disable_parallelism = disable_parallelism
        self.disable_reorder = disable_reorder
        self.disable_fusion = disable_fusion
        self.disable_caching = disable_caching
        self.use_my_optimizer = use_my_optimizer
        self.generate_plan = generate_plan

    def to_ray_config(self) -> Optional[RayConfig]:
        """
        Returns a Ray spec for the CedarContext, if specified by
        the profiler spec
        """
        if not self.use_ray:
            return None

        return RayConfig(self.ray_ip)
