"""Lazy optional-framework imports for CPU-only Cedar workers."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


def tensorflow() -> ModuleType:
    return importlib.import_module("tensorflow")


def torch() -> ModuleType:
    return importlib.import_module("torch")


def is_tensorflow_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("tensorflow") and tensorflow().is_tensor(value)


def is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("torch") and torch().is_tensor(value)
