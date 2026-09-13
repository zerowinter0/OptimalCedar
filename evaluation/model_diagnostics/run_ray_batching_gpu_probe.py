"""Diagnostic-only probe for Ray batching and CUDA profile fidelity.

This script is intentionally not part of the benchmark harness.  It mimics
Cedar's Ray actor contract: each task receives a Python list and applies the
operator once per record.  It measures worker compute and pipelined end-to-end
time separately so profile errors can be attributed to device placement or
Ray submit batching.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import statistics
import time
from typing import Any, Iterable

import ray


REMOTE_RESOURCE = {"cedar_remote": 0.001}


@ray.remote
class ClipProbeActor:
    def __init__(self, force_cpu: bool) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.Image = Image
        self.device = torch.device(
            "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        )
        model_path = snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            local_files_only=True,
        )
        self.processor = CLIPProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = CLIPModel.from_pretrained(
            model_path, local_files_only=True, use_safetensors=False
        )
        self.model.to(self.device)
        self.model.eval()

    def device_info(self) -> dict[str, Any]:
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": self.torch.cuda.is_available(),
            "torch_cuda_device_count": self.torch.cuda.device_count(),
            "ray_gpu_ids": list(ray.get_gpu_ids()),
            "node_id": ray.get_runtime_context().get_node_id(),
            "selected_device": str(self.device),
        }

    def process(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        started = time.perf_counter_ns()
        scores = []
        import pico_multimodal_data

        package_root = pathlib.Path(pico_multimodal_data.__file__).parent
        for record in records:
            relative = pathlib.Path(str(record["image_path"]))
            with self.Image.open(package_root / relative) as image:
                rgb = image.convert("RGB")
            inputs = self.processor(
                text=str(record.get("caption", "")),
                images=rgb,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.text_config.max_position_embeddings,
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits_per_text / 100.0
            scores.append(float(logits.detach().cpu().reshape(-1)[0]))
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        return {
            "records": len(scores),
            "compute_ns": elapsed_ns,
            "checksum": float(sum(scores)),
        }


@ray.remote
class CropProbeActor:
    def __init__(self) -> None:
        import torch
        from torchvision import transforms

        self.torch = torch
        self.transform = transforms.RandomResizedCrop((224, 224))

    def device_info(self) -> dict[str, Any]:
        return {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": self.torch.cuda.is_available(),
            "torch_cuda_device_count": self.torch.cuda.device_count(),
            "ray_gpu_ids": list(ray.get_gpu_ids()),
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def process(self, tensors: list[Any]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        outputs = [self.transform(tensor) for tensor in tensors]
        elapsed_ns = time.perf_counter_ns() - started
        return {
            "records": len(outputs),
            "compute_ns": elapsed_ns,
            "checksum": float(sum(output.shape[-1] for output in outputs)),
        }


def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _run_epoch(actor: Any, values: list[Any], batch_size: int) -> dict[str, Any]:
    started = time.perf_counter_ns()
    futures = [actor.process.remote(batch) for batch in _batches(values, batch_size)]
    results = ray.get(futures)
    wall_ns = time.perf_counter_ns() - started
    records = sum(int(result["records"]) for result in results)
    compute_ns = sum(int(result["compute_ns"]) for result in results)
    return {
        "records": records,
        "tasks": len(results),
        "compute_ms_per_record": compute_ns / records / 1e6,
        "end_to_end_ms_per_record": wall_ns / records / 1e6,
        "throughput_records_per_sec": records / (wall_ns / 1e9),
        "checksum": sum(float(result["checksum"]) for result in results),
    }


def _summarize(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "compute_ms_per_record",
        "end_to_end_ms_per_record",
        "throughput_records_per_sec",
    ):
        values = [float(repeat[key]) for repeat in repeats]
        summary[key] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def _load_records(path: pathlib.Path, count: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
            if len(records) == count:
                break
    if len(records) < count:
        raise RuntimeError(f"Requested {count} records, found {len(records)}")
    return records


def _load_crop_inputs(
    records: list[dict[str, Any]], image_root: pathlib.Path
) -> list[Any]:
    from torchvision.io import ImageReadMode, read_image

    tensors = []
    for record in records:
        path = pathlib.Path(str(record["image_path"]))
        if not path.is_absolute():
            path = image_root / path
        tensors.append(read_image(str(path), mode=ImageReadMode.RGB))
    return tensors


def _run_config(
    *,
    operator: str,
    values: list[Any],
    image_root: pathlib.Path,
    num_gpus: float,
    batch_size: int,
    warmup_records: int,
    repeats: int,
) -> dict[str, Any]:
    actor_cls = ClipProbeActor if operator == "clip" else CropProbeActor
    args = (num_gpus == 0.0,) if operator == "clip" else ()
    actor = actor_cls.options(
        num_cpus=1,
        num_gpus=num_gpus,
        resources=REMOTE_RESOURCE,
    ).remote(*args)
    try:
        device = ray.get(actor.device_info.remote())
        warm_values = [values[index % len(values)] for index in range(warmup_records)]
        _run_epoch(actor, warm_values, batch_size)
        measured = []
        for _ in range(repeats):
            measured.append(_run_epoch(actor, values, batch_size))
        return {
            "operator": operator,
            "num_gpus": num_gpus,
            "submit_batch_size": batch_size,
            "device": device,
            "repeats": measured,
            "summary": _summarize(measured),
        }
    finally:
        ray.kill(actor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--image-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--warmup-records", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--operators", nargs="+", choices=("clip", "crop"), default=("clip", "crop")
    )
    args = parser.parse_args()
    if args.records < 10 or args.warmup_records < 1 or args.repeats < 1:
        raise ValueError("Probe sizes must be positive and records must be >= 10")

    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()[:16]
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    image_package = (
        repo_root
        / "outputs/motivation_multimodal/ray_data_packages"
        / digest
        / "pico_multimodal_data"
    )
    if not image_package.is_dir():
        raise FileNotFoundError(
            f"No staged Ray image package for dataset digest {digest}: "
            f"{image_package}"
        )
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"py_modules": [str(image_package)]},
    )
    base_records = _load_records(args.dataset, args.records)
    output: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_only": True,
        "dataset": str(args.dataset),
        "image_root": str(args.image_root),
        "records": args.records,
        "warmup_records": args.warmup_records,
        "repeats": args.repeats,
        "cluster_resources": ray.cluster_resources(),
        "configs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for operator in args.operators:
            values = (
                base_records
                if operator == "clip"
                else _load_crop_inputs(base_records, args.image_root)
            )
            gpu_options = (0.0, 1.0) if operator == "clip" else (0.0,)
            for num_gpus in gpu_options:
                for batch_size in (1, 10):
                    result = _run_config(
                        operator=operator,
                        values=values,
                        image_root=args.image_root,
                        num_gpus=num_gpus,
                        batch_size=batch_size,
                        warmup_records=args.warmup_records,
                        repeats=args.repeats,
                    )
                    output["configs"].append(result)
                    args.output.write_text(
                        json.dumps(output, indent=2), encoding="utf-8"
                    )
                    print(
                        operator,
                        f"gpu={num_gpus:g}",
                        f"batch={batch_size}",
                        json.dumps(result["summary"], sort_keys=True),
                        flush=True,
                    )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
