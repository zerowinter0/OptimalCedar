from typing import Any, Dict, Optional


class TorchEvalSpec:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_epochs: int = 1,
        num_total_samples: Optional[int] = None,
        iteration_time: Optional[float] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_total_samples = num_total_samples
        self.num_epochs = num_epochs
        self.iteration_time = iteration_time
        self.kwargs = dict(kwargs or {})
