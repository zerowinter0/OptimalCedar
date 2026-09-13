"""Legacy strict-chain reference model; production entrypoints use explicit Features."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

_RNG_LOCK = threading.RLock()


@contextmanager
def random_stream(seed):
    # torchvision and upstream augmentations use process-global generators.
    # Guard them in threaded variants and restore callers' states on exit.
    with _RNG_LOCK:
        py_state, np_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            try:
                yield
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)


@dataclass(frozen=True)
class Stage:
    tag: str
    source: str
    target: str
    operation: Callable
    dependencies: tuple[str, ...] = ()
    modality: str = "image"
    scaling: str = "none"
    stochastic: bool = False

    def __call__(self, record):
        result = dict(record)
        if self.stochastic:
            seed = int.from_bytes(hashlib.blake2b(
                f"{record['_seed']}:{self.tag}".encode(), digest_size=8
            ).digest(), "big") % (2**63)
            with random_stream(seed):
                value = self.operation(record[self.source])
        else:
            value = self.operation(record[self.source])
        result[self.target] = value
        return result


@dataclass(frozen=True)
class Prepare:
    seed: int = 0
    epoch: int = 0
    image_root: str = ""
    image_kind: str = "path"

    def __call__(self, record):
        from pathlib import Path
        if isinstance(record, str):
            record = json.loads(record)
        if not isinstance(record, dict):
            raise TypeError("Expected a JSON object with image and optional caption")
        result = dict(record)
        path = Path(result["image"])
        if not path.is_absolute():
            path = Path(self.image_root) / path
        result["image"] = str(path)
        identity = result.get("id", str(path))
        result["_seed"] = f"{self.seed}:{self.epoch}:{identity}"
        if self.image_kind != "path":
            from .operators import ReadImage
            result["original"] = ReadImage(self.image_kind)(str(path))
        return result


@dataclass(frozen=True)
class Finalize:
    kind: str
    fields: tuple[str, ...]

    def __call__(self, record):
        if self.kind == "views":
            return {"views": [record[k] for k in self.fields]}
        if self.kind == "clip":
            return {"pixel_values": record["pixels"], **record["tokens"]}
        return {"pixel_values": record["pixels"], "caption": record["caption"]}


@dataclass
class Workload:
    name: str
    stages: tuple[Stage, ...]
    prepare: Prepare
    finalize: Finalize

    @property
    def tags(self):
        return [stage.tag for stage in self.stages]

    def validate_order(self, order):
        if len(order) != len(self.tags) or set(order) != set(self.tags):
            raise ValueError("Order must contain each stage exactly once")
        by_tag = {s.tag: s for s in self.stages}
        seen = set()
        for tag in order:
            if not set(by_tag[tag].dependencies) <= seen:
                raise ValueError(f"Unsatisfied dependencies for {tag}")
            seen.add(tag)

    def run(self, record, order=None):
        order = self.tags if order is None else list(order)
        self.validate_order(order)
        state = self.prepare(copy.deepcopy(record))
        by_tag = {s.tag: s for s in self.stages}
        for tag in order:
            state = by_tag[tag](state)
        return self.finalize(state)

    def sample_orders(self, count, seed=17):
        rng = random.Random(seed)
        for _ in range(count):
            done, order = set(), []
            while len(order) < len(self.stages):
                ready = [s.tag for s in self.stages
                         if s.tag not in done and set(s.dependencies) <= done]
                if not ready:
                    raise ValueError("Dependency cycle")
                tag = rng.choice(ready)
                order.append(tag)
                done.add(tag)
            yield order

    def order_count(self):
        # Exact counting for disjoint chains, the contract used by this suite.
        by_tag = {s.tag: s for s in self.stages}
        children = {tag: [] for tag in by_tag}
        for s in self.stages:
            if len(s.dependencies) > 1:
                raise ValueError("Expected independent chains")
            for parent in s.dependencies:
                children[parent].append(s.tag)
        if any(len(v) > 1 for v in children.values()):
            raise ValueError("Expected independent chains")
        lengths = []
        for stage in self.stages:
            if not stage.dependencies:
                n, tag = 1, stage.tag
                while children[tag]:
                    tag = children[tag][0]
                    n += 1
                lengths.append(n)
        if sum(lengths) != len(self.stages):
            raise ValueError("Dependency cycle")
        return math.factorial(sum(lengths)) // math.prod(
            math.factorial(n) for n in lengths
        )
