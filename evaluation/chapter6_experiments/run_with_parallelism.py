#!/usr/bin/env python3
"""Run one evaluation entry point with the Chapter 6 CPU budget."""

import multiprocessing as mp
from multiprocessing.spawn import freeze_support
import os
import runpy
import sys


PARALLELISM = 16


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_parallelism.py SCRIPT [ARGS ...]")

    # Keep local process workers and inherited Ray clients within the same
    # 16-core experiment budget.  MapperPipe consults multiprocessing.cpu_count.
    try:
        os.sched_setaffinity(0, set(range(PARALLELISM)))
    except (AttributeError, OSError):
        pass

    # runpy leaves this wrapper as the multiprocessing main module under the
    # spawn method. Force fork before importing Cedar so workers inherit this
    # configured process instead of recursively rerunning the wrapper.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    mp.cpu_count = lambda: PARALLELISM  # type: ignore[method-assign]

    script = sys.argv[1]
    sys.modules["__main__"].__file__ = os.path.abspath(script)
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    freeze_support()
    main()
