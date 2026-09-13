#!/usr/bin/env python3
"""Measure a literal exhaustive search over PICO's joint plan dimensions.

The benchmark enumerates every logical order, contiguous fusion partition,
backend/width assignment that fits the Ray and SMP budgets, and zero-or-one
cache placement at a physical-stage boundary.  Every visited plan is scored;
the benchmark does not substitute an analytical candidate count for search.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


FAMILIES = ("LOCAL", "RAY", "SMP")


@dataclass(frozen=True)
class OperatorProfile:
    size_ratio: float
    selectivity: float
    costs: Mapping[str, float]


@dataclass(frozen=True)
class JointEnumerationConfig:
    ray_width_budget: int = 8
    smp_width_budget: int = 8
    max_width: int = 8
    cache_read_cost: float = 0.05


@dataclass(frozen=True)
class JointEnumerationResult:
    best_cost: float
    best_order: tuple[int, ...]
    visited_plans: int
    elapsed_seconds: float


def _stage_partitions(order: tuple[int, ...]):
    for boundary_mask in range(1 << (len(order) - 1)):
        stages = []
        start = 0
        for position in range(len(order) - 1):
            if boundary_mask & (1 << position):
                stages.append(order[start : position + 1])
                start = position + 1
        stages.append(order[start:])
        yield tuple(stages)


def _implementations(config: JointEnumerationConfig):
    yield ("LOCAL", 1)
    for family in ("RAY", "SMP"):
        for width in range(1, config.max_width + 1):
            yield (family, width)


def _fits_budget(
    assignments: Sequence[tuple[str, int]], config: JointEnumerationConfig
) -> bool:
    ray = sum(width for family, width in assignments if family == "RAY")
    smp = sum(width for family, width in assignments if family == "SMP")
    return ray <= config.ray_width_budget and smp <= config.smp_width_budget


def _score_plan(
    profiles: Sequence[OperatorProfile],
    stages: Sequence[Sequence[int]],
    assignments: Sequence[tuple[str, int]],
    cache_after_stage: int | None,
    cache_read_cost: float,
) -> float:
    volume = 1.0
    demands = {family: 0.0 for family in FAMILIES}
    for stage_index, (stage, (family, width)) in enumerate(
        zip(stages, assignments)
    ):
        stage_cost = 0.0
        for operator_index in stage:
            profile = profiles[operator_index]
            stage_cost += volume * profile.costs[family]
            volume *= profile.selectivity * profile.size_ratio
        demands[family] += stage_cost / width
        if cache_after_stage == stage_index:
            demands = {family_name: 0.0 for family_name in FAMILIES}
            demands["LOCAL"] = cache_read_cost
    return max(demands.values())


def exhaustive_joint_search(
    profiles: Sequence[OperatorProfile], config: JointEnumerationConfig
) -> JointEnumerationResult:
    if not profiles:
        raise ValueError("profiles must contain at least one operator")
    if config.max_width < 1:
        raise ValueError("max_width must be positive")

    started = time.perf_counter()
    implementations = tuple(_implementations(config))
    best_cost = float("inf")
    best_order: tuple[int, ...] = ()
    visited = 0
    for order in itertools.permutations(range(len(profiles))):
        for stages in _stage_partitions(order):
            for assignments in itertools.product(
                implementations, repeat=len(stages)
            ):
                if not _fits_budget(assignments, config):
                    continue
                for cache_after_stage in (None, *range(len(stages))):
                    visited += 1
                    cost = _score_plan(
                        profiles,
                        stages,
                        assignments,
                        cache_after_stage,
                        config.cache_read_cost,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_order = order
    return JointEnumerationResult(
        best_cost=best_cost,
        best_order=best_order,
        visited_plans=visited,
        elapsed_seconds=time.perf_counter() - started,
    )


def generate_profiles(num_operators: int, seed: int) -> tuple[OperatorProfile, ...]:
    rng = random.Random(seed + num_operators)
    return tuple(
        OperatorProfile(
            size_ratio=rng.uniform(0.4, 1.2),
            selectivity=rng.uniform(0.4, 1.0),
            costs={family: rng.uniform(0.25, 8.0) for family in FAMILIES},
        )
        for _ in range(num_operators)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--width-budget", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operators < 1 or args.repeat < 1:
        parser.error("--operators and --repeat must be positive")

    config = JointEnumerationConfig(
        ray_width_budget=args.width_budget,
        smp_width_budget=args.width_budget,
        max_width=args.max_width,
    )
    profiles = generate_profiles(args.operators, args.seed)
    result = exhaustive_joint_search(profiles, config)
    payload = {
        "num_operators": args.operators,
        "repeat": args.repeat,
        "seed": args.seed,
        "config": asdict(config),
        **asdict(result),
    }
    encoded = json.dumps(payload, sort_keys=True)
    print(encoded, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
