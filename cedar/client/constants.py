import os

from cedar.compose.constants import (
    PROFILE_STAGE_MAX_INFLIGHT,
    RAY_SUBMIT_BATCH_SIZE,
)


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value

RAY_PROFILE_N_ACTORS = _positive_int_env("CEDAR_PROFILE_RAY_ACTORS", 8)
RAY_PROFILE_INFLIGHT = PROFILE_STAGE_MAX_INFLIGHT
RAY_PROFILE_PREFETCH = 100
RAY_PROFILE_SUBMIT_BATCH_SIZE = 10
CONTROLLER_PERIOD_SEC = 3
MAX_HISTORY = CONTROLLER_PERIOD_SEC * 10
THROUGHPUT_LOG_TIME_SEC = 1
SCALE_ATTEMPTS = 3
THROUGHPUT_THRESHOLD = 1.01
EMPTY_BUFFER_THRESHOLD = 500  # set arouhd half of max buffer size
AVAILABLE_RAY_SCALE = 32
SMP_PROFILE_N_PROCS = _positive_int_env("CEDAR_PROFILE_SMP_PROCS", 8)
SMP_TASKSET_MASK = 0xFF  # should match the taskset cpu mask of smp n_procs
SMP_PROFILE_INFLIGHT = PROFILE_STAGE_MAX_INFLIGHT
SMP_PROFILE_PREFETCH = 100
CONTROLLER_SCALE_DOWN_COUNTER = 10
