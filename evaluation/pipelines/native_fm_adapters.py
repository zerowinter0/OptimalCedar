"""Native PyTorch, tf.data, and Ray Data adapters for FM workloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.pipelines.datajuicer_workloads import (
    DEFAULT_PATHS,
    build_stages,
    execute_stages,
    output_text,
)


def _options(spec: Any, workload: str) -> tuple[Path, str]:
    kwargs = getattr(spec, "kwargs", {}) or {}
    dataset_path = Path(kwargs.get("dataset_path") or DEFAULT_PATHS[workload])
    image_root = str(kwargs.get("image_root") or "")
    return dataset_path, image_root


def get_torch_dataset(spec: Any, workload: str):
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info

    dataset_path, image_root = _options(spec, workload)

    class _PipelineDataset(IterableDataset):
        def __iter__(self):
            worker = get_worker_info()
            worker_id = worker.id if worker else 0
            worker_count = worker.num_workers if worker else 1
            stages = build_stages(workload, image_root=image_root)
            with dataset_path.open("r", encoding="utf-8") as source:
                for index, line in enumerate(source):
                    if index % worker_count != worker_id:
                        continue
                    value = execute_stages(line, stages)
                    if value is not None:
                        yield output_text(value)

    return DataLoader(
        _PipelineDataset(),
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
    )


def get_tf_dataset(spec: Any, workload: str):
    import numpy as np
    import tensorflow as tf

    dataset_path, image_root = _options(spec, workload)
    stages = build_stages(workload, image_root=image_root)

    def _process(line):
        raw = line.numpy()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = execute_stages(str(raw), stages)
        if value is None:
            return b"", np.bool_(False)
        # np.bytes_(str) encodes through NumPy's ASCII path and raises on
        # legitimate non-ASCII training text.  TensorFlow's string dtype
        # accepts arbitrary byte strings, so preserve the text as UTF-8.
        return output_text(value).encode("utf-8"), np.bool_(True)

    def _tf_process(line):
        value, keep = tf.py_function(
            _process,
            [line],
            Tout=(tf.string, tf.bool),
        )
        value.set_shape(())
        keep.set_shape(())
        return value, keep

    dataset = tf.data.TextLineDataset(str(dataset_path))
    dataset = dataset.map(
        _tf_process,
        num_parallel_calls=spec.num_parallel_calls,
    )
    dataset = dataset.filter(lambda value, keep: keep)
    dataset = dataset.map(
        lambda value, keep: value,
        num_parallel_calls=spec.num_parallel_calls,
    )
    dataset = dataset.batch(spec.batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_ray_dataset(spec: Any, workload: str):
    import ray

    dataset_path, image_root = _options(spec, workload)
    dataset = ray.data.read_text(str(dataset_path))
    stages = build_stages(workload, image_root=image_root)
    for index, stage in enumerate(stages):
        if stage.kind == "map":
            operator = stage.operator
            if index == len(stages) - 1:
                operator = _RayRowOutputAdapter(operator)
            dataset = dataset.map(operator)
        else:
            dataset = dataset.filter(stage.operator)
    return dataset


class _RayRowOutputAdapter:
    """Keep the logical output while satisfying Ray Data's row contract."""

    def __init__(self, operator):
        self.operator = operator

    def __call__(self, value):
        result = self.operator(value)
        if isinstance(result, dict):
            return result
        return {"item": result}


__all__ = ["get_ray_dataset", "get_tf_dataset", "get_torch_dataset"]
