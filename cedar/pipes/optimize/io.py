from typing import Optional
from .registry import register_optimizer_pipe
from ..context import (
    PipeVariantType,
    InProcessPipeVariantContext,
)
from ..pipe import (
    Pipe,
)
from ..variant import (
    InProcessPipeVariant,
    PipeVariant,
)

from typing import List, Any

import copy
import json
import logging
import os
import pathlib
import pickle
import shutil
import tempfile
import torch


@register_optimizer_pipe("ObjectDiskCachePipe")
class ObjectDiskCachePipe(Pipe):
    """
    Given a pipe that yielding any type of object,
    saves the tensors to files in temporary directory on disk.

    Can save data to following format: .pt (tensors) and .pkl (any object).
    User should indicate the type of format in the file_type variable.
    """

    def __init__(
        self,
        input_pipe: Optional[Pipe] = None,
        file_type: str = "pkl",
        is_random: bool = False,
    ):
        if input_pipe:
            super().__init__(
                "ObjectDiskCachePipe", [input_pipe], is_random=is_random
            )
        else:
            super().__init__("ObjectDiskCachePipe", [], is_random=is_random)
        self.file_type = file_type

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        return InProcessObjectDiskCachePipeVariant(
            self.input_pipes[0].pipe_variant, file_type=self.file_type
        )


class InProcessObjectDiskCachePipeVariant(InProcessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: PipeVariant,
        file_type: str = "pkl",
        max_samples_in_cache_file: int = 1000,
    ):
        super().__init__(input_pipe_variant)
        # NOTE: Could write method that check the data type of the input pipe
        # and chooses the file_type appropriately
        self.file_type = file_type
        self.save_funcs = {"pt": self._save_pt, "pkl": self._save_pkl}
        self.load_funcs = {"pt": self._load_pt, "pkl": self._load_pkl}
        self.max_samples_in_cache_file = max_samples_in_cache_file

        self._check_file_type()
        self.save_function = self.save_funcs[file_type]
        self.load_function = self.load_funcs[file_type]
        cache_root = pathlib.Path(
            os.environ.get("CEDAR_CACHE_ROOT", tempfile.gettempdir())
        )
        namespace = os.environ.get("CEDAR_CACHE_NAMESPACE")
        if namespace:
            shard = os.environ.get("CEDAR_CACHE_SHARD", "single")
            self.cache_dir = cache_root / namespace / shard
        else:
            # Preserve Cedar's historical path for callers outside the paper
            # experiment driver.
            self.cache_dir = cache_root / f"cedar_{os.getpid()}"
        self.manifest_path = self.cache_dir / ".manifest.json"

    def _check_file_type(self) -> None:
        if (
            self.file_type not in self.save_funcs
            or self.file_type not in self.load_funcs
        ):
            raise ValueError(
                f"Specified file type {self.file_type} is not supported for\
                    caching.Supported file types: {self.save_funcs.keys()}"
            )

    def _save_pt(self, tensors: List[torch.Tensor], file_name: str) -> None:
        torch.save(tensors, f"{file_name}.pt")

    def _load_pt(self, file_name: str) -> List[torch.Tensor]:
        return torch.load(file_name)

    def _save_pkl(self, data: List[Any], file_name: str) -> None:
        with open(f"{file_name}.pkl", "wb") as file:
            pickle.dump(data, file)

    def _load_pkl(self, file_name: str) -> List[Any]:
        with open(file_name, "rb") as file:
            loaded_data = pickle.load(file)
        return loaded_data

    def _save(self, data: List[Any], file_name: str) -> None:
        self.save_function(data, file_name)

    def _load(self, file_name: str) -> List[Any]:
        return self.load_function(file_name)

    def _save_given_count(self, data: List[Any], count: int) -> None:
        if not data:
            return
        file_name = f"data_batch_{count}"
        file_path = self.cache_dir / pathlib.Path(file_name)
        temp_path = self.cache_dir / pathlib.Path(f".{file_name}.tmp")
        self._save(data, temp_path)
        suffix = ".pt" if self.file_type == "pt" else ".pkl"
        os.replace(f"{temp_path}{suffix}", f"{file_path}{suffix}")

    def _write_manifest(self, num_items: int, num_files: int) -> None:
        manifest = {
            "complete": True,
            "file_type": self.file_type,
            "num_items": num_items,
            "num_files": num_files,
        }
        temp_manifest = self.cache_dir / ".manifest.json.tmp"
        with open(temp_manifest, "w") as file:
            json.dump(manifest, file, sort_keys=True)
        os.replace(temp_manifest, self.manifest_path)

    def _load_manifest(self) -> Optional[dict]:
        try:
            with open(self.manifest_path) as file:
                manifest = json.load(file)
        except (OSError, ValueError):
            return None
        if (
            manifest.get("complete") is not True
            or manifest.get("file_type") != self.file_type
        ):
            return None
        return manifest

    def _iter_impl(self):
        # A directory alone is not a valid cache: a worker may have been
        # interrupted while materializing it. Only an atomically committed
        # manifest makes the shard readable.
        pid = os.getpid()
        logging.info(
            "Checking cache shard for pid %d at %s", pid, self.cache_dir
        )
        manifest = self._load_manifest()
        logging.info(
            "Complete cache shard exists for pid %d? %s",
            pid,
            manifest is not None,
        )

        if manifest is not None:
            suffix = ".pt" if self.file_type == "pt" else ".pkl"
            for count in range(int(manifest["num_files"])):
                file = self.cache_dir / f"data_batch_{count}{suffix}"
                items = self._load(file)
                for item in items:
                    item.read_from_cache = True
                    yield item
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=False)
            file_count = 0
            num_items = 0
            item_buffer = []
            completed = False
            try:
                for item in self.input_pipe_variant:
                    item_cache_copy = copy.deepcopy(item)
                    item_buffer.append(item_cache_copy)
                    num_items += 1
                    if len(item_buffer) == self.max_samples_in_cache_file:
                        self._save_given_count(item_buffer, file_count)
                        item_buffer = []
                        file_count += 1
                    yield item
                completed = True
            finally:
                if completed:
                    if item_buffer:
                        self._save_given_count(item_buffer, file_count)
                        file_count += 1
                    self._write_manifest(num_items, file_count)
                    logging.info(
                        "Committed complete cache shard %s with %d items in %d files",
                        self.cache_dir,
                        num_items,
                        file_count,
                    )
