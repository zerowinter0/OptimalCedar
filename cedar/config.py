"""
Config file for cedar
"""

from typing import Any, Dict, Type, TypeVar, Optional
import ray
import logging

logger = logging.getLogger(__name__)


try:
    import nvidia.dali as dali  # noqa: F401

    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

T = TypeVar("T", bound="CedarContext")


class RayConfig:
    """
    Configuration class for Ray
    """

    def __init__(
        self,
        ip: str = "",
        n_cpus: Optional[int] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
    ):
        self.ip = ip
        self.n_cpus = n_cpus
        self.runtime_env = runtime_env


class CedarContext:
    """
    Context holding necessary state for cedar services.
    """

    def __init__(self, ray_config: Optional[RayConfig] = None):
        self.ray_config = ray_config

    def init_ray(self):
        """
        Initialize the Ray runtime.
        NOTE: If calling this from a child process, ensure that the parent
        process does not call init_ray().
        """
        if self.ray_config is None:
            raise RuntimeError("Ray config not specified.")

        runtime_kwargs = (
            {"runtime_env": self.ray_config.runtime_env}
            if self.ray_config.runtime_env is not None
            else {}
        )
        if ray.is_initialized():
            logger.warning("Ray already initialized. Defaulting to it.")
        elif self.ray_config.ip != "":
            if ray.is_initialized():
                ray.shutdown()
            logger.info(f"Connecting to ray cluster at {self.ray_config.ip}")
            if ":" in self.ray_config.ip:
                # A host:port value is a native Ray GCS address.
                ray.init(address=self.ray_config.ip, **runtime_kwargs)
            else:
                ray.init(
                    f"ray://{self.ray_config.ip}:10001", **runtime_kwargs
                )
        else:
            logger.info("Launching to local ray instance")
            if self.ray_config.n_cpus is not None:
                logger.info(
                    "Using {} CPUs for local ray instance".format(
                        self.ray_config.n_cpus
                    )
                )
                ray.init(
                    num_cpus=self.ray_config.n_cpus, **runtime_kwargs
                )
            else:
                ray.init(**runtime_kwargs)

    @classmethod
    def from_yaml(cls: Type[T], config_file: str) -> T:
        # TODO (myzhao)
        raise NotImplementedError

    def __del__(self):
        if self.ray_config:
            if ray.is_initialized():
                ray.shutdown()
        pass

    def use_ray(self) -> bool:
        """
        Returns if the context should use Ray.
        """
        return self.ray_config is not None
