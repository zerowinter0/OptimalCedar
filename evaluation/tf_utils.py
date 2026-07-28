from typing import Any, Dict, Optional


class TFEvalSpec:
    def __init__(
        self,
        batch_size: int,
        num_parallel_calls: Optional[int],
        num_epochs: int = 1,
        num_total_samples: Optional[int] = None,
        iteration_time: Optional[float] = None,
        service_addr: Optional[str] = None,
        read_from_remote: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.num_total_samples = num_total_samples
        self.num_epochs = num_epochs
        self.iteration_time = iteration_time
        self.service_addr = service_addr
        self.read_from_remote = read_from_remote
        self.kwargs = dict(kwargs or {})
