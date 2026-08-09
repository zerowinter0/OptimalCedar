"""Add the pinned in-repository Data-Juicer checkout to ``sys.path``."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_data_juicer_path() -> None:
    root = Path(__file__).resolve().parents[3] / "data-juicer"
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
        if name in sys.modules:
            continue
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
