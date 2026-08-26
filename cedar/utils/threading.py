"""Helpers for enforcing Cedar's one-logical-CPU worker contract."""

from __future__ import annotations

import os
from typing import Any


def limit_native_threadpools(num_threads: int = 1) -> Any:
    """Limit loaded and future native CPU pools, returning a live limiter.

    Cedar accounts each local worker, SMP process, and CPU Ray actor as one
    logical CPU.  ``torch.set_num_threads`` only constrains Torch's OpenMP
    pool; NumPy/OpenBLAS, MKL, and NumExpr can otherwise create a full-machine
    pool in every worker.  Set environment defaults for libraries loaded
    later and use threadpoolctl for libraries already inherited through
    ``fork``.  The caller must keep the returned controller alive for the
    worker lifetime.
    """
    value = str(max(1, int(num_threads)))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = value

    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(value))
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "threadpoolctl is required to enforce Cedar CPU accounting"
        ) from exc
