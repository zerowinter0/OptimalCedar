"""Add the pinned in-repository Data-Juicer checkout to ``sys.path``."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def data_juicer_root() -> Path:
    """Locate the pinned Data-Juicer checkout.

    Ray stages execute from an uploaded copy of the working directory, so the
    repository-relative path does not exist inside an actor.  The benchmark
    containers all mount the repository at ``/workspace/OptimalCedar``, which
    is why an absolute override wins over the relative default.
    """
    override = os.environ.get("CEDAR_DATA_JUICER_ROOT", "").strip()
    if override:
        candidate = Path(override)
        if (candidate / "data_juicer").is_dir():
            return candidate
    # Both benchmark containers mount the repository at this absolute path.
    # It is preferred over the relative path because a Ray actor runs from the
    # *uploaded* copy of the working directory, where ``data-juicer`` is only
    # partially replicated and the operators below are missing.
    absolute = Path("/workspace/OptimalCedar/data-juicer")
    if (absolute / "data_juicer").is_dir():
        return absolute
    relative = Path(__file__).resolve().parents[3] / "data-juicer"
    return relative


def ensure_data_juicer_path() -> None:
    root = data_juicer_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Data-Juicer's package initializers eagerly import every one of its 200+
    # operators and therefore require unrelated optional dependencies. This
    # workload intentionally loads only the seven pinned recipe operators.
    # Namespace stubs preserve normal relative imports without executing the
    # eager ``data_juicer.ops`` and ``data_juicer.ops.filter`` initializers.
    packages = {
        "data_juicer.core": root / "data_juicer/core",
        "data_juicer.ops": root / "data_juicer/ops",
        "data_juicer.ops.filter": root / "data_juicer/ops/filter",
    }
    for name, path in packages.items():
        # Overwrite any earlier import of these packages: an eager
        # ``data_juicer.ops`` import pulls 200+ operators and unrelated
        # optional dependencies, so the stubs must win regardless of order.
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module

    if "data_juicer.core.data" not in sys.modules:
        data_module = types.ModuleType("data_juicer.core.data")

        def wrap_func_with_nested_access(function):
            return function

        data_module.wrap_func_with_nested_access = wrap_func_with_nested_access
        sys.modules["data_juicer.core.data"] = data_module
