#!/usr/bin/env python3
"""Validate a Cedar text feature against a pinned Data-Juicer Hub recipe."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from cedar.compose import Feature
from cedar.sources import IterSource


def _operator_name(fn: Any) -> str:
    class_name = type(fn).__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()


def _feature_chain(feature: Feature) -> list[Any]:
    feature.apply(IterSource(["{}"]),)
    pipes = list(feature.logical_pipes.values())
    referenced = {
        input_pipe
        for pipe in pipes
        for input_pipe in pipe.input_pipes
    }
    sinks = [pipe for pipe in pipes if pipe not in referenced]
    if len(sinks) != 1:
        raise RuntimeError(f"expected one feature sink, found {len(sinks)}")
    reverse_chain = []
    pipe = sinks[0]
    while not pipe.is_source():
        reverse_chain.append(pipe)
        if len(pipe.input_pipes) != 1:
            raise RuntimeError(
                f"expected a linear feature, found {len(pipe.input_pipes)} inputs"
            )
        pipe = pipe.input_pipes[0]
    return list(reversed(reverse_chain))


def _recipe_operators(
    recipe: Path, omitted_operators: set[str]
) -> list[tuple[str, dict[str, Any]]]:
    payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    process = payload.get("process")
    if not isinstance(process, list):
        raise RuntimeError(f"recipe process is not a list: {recipe}")
    operators = []
    for item in process:
        if not isinstance(item, dict) or len(item) != 1:
            raise RuntimeError(f"invalid recipe entry: {item!r}")
        name, kwargs = next(iter(item.items()))
        if name in omitted_operators:
            continue
        if kwargs is not None and not isinstance(kwargs, dict):
            raise RuntimeError(f"invalid arguments for {name}: {kwargs!r}")
        operators.append((str(name), dict(kwargs or {})))
    return operators


def validate_text_recipe(
    recipe: Path,
    feature: Feature,
    *,
    adapter_tags: Iterable[str],
    omitted_operators: Iterable[str],
) -> list[dict[str, Any]]:
    """Compare actual Cedar functions with Hub operator order and arguments."""
    adapters = set(adapter_tags)
    omitted = set(omitted_operators)
    cedar_pipes = [
        pipe for pipe in _feature_chain(feature) if pipe.tag not in adapters
    ]
    recipe_ops = _recipe_operators(recipe, omitted)
    if len(cedar_pipes) != len(recipe_ops):
        raise RuntimeError(
            f"operator count mismatch: Cedar={len(cedar_pipes)}, "
            f"Hub={len(recipe_ops)}"
        )

    observed = []
    for pipe, (recipe_name, recipe_kwargs) in zip(cedar_pipes, recipe_ops):
        cedar_name = _operator_name(pipe.fn)
        if cedar_name != recipe_name:
            raise RuntimeError(
                f"operator mismatch: Cedar={cedar_name}, Hub={recipe_name}"
            )
        actual_kwargs = {}
        for name, expected in recipe_kwargs.items():
            if not hasattr(pipe.fn, name):
                raise RuntimeError(
                    f"Cedar {cedar_name} does not expose recipe argument {name}"
                )
            actual = getattr(pipe.fn, name)
            if actual != expected:
                raise RuntimeError(
                    f"argument mismatch for {cedar_name}.{name}: "
                    f"Cedar={actual!r}, Hub={expected!r}"
                )
            actual_kwargs[name] = actual
        observed.append(
            {
                "operator": recipe_name,
                "arguments": actual_kwargs,
                "cedar_tag": pipe.tag,
            }
        )
    return observed


__all__ = ["validate_text_recipe"]
