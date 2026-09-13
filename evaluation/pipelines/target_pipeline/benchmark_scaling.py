"""Controlled linear input-size sweeps in native Cedar worker contexts."""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.io import read_image, ImageReadMode

from cedar.compose.utils import topological_sort
from cedar.pipes import MapperPipe, BatcherPipe, ImageReaderPipe
from cedar.pipes.batch import InProcessBatcherPipeVariant
from cedar.pipes.common import DataSample
from cedar.pipes.map import SMPActorMapperPipeVariant, RayActorMapperPipeVariant
from cedar.sources import IterSource
from .fields import OnField, ReadRecord
from .simclr.cedar_dataset import SimCLRV2Feature
from .dino.cedar_dataset import DINOFeature
from .swav.cedar_dataset import SwAVFeature
from .clip.cedar_dataset import CLIPFeature
from .blip.cedar_dataset import BLIPFeature

NAMES = ("simclr", "dino", "swav", "clip", "blip")
CLASSES = (SimCLRV2Feature, DINOFeature, SwAVFeature, CLIPFeature, BLIPFeature)
ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, Image.Image):
        return x.width * x.height * len(x.getbands())
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, str):
        return len(x.encode("utf-8"))
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return sum(nbytes(v) for v in x)
    if isinstance(x, int):
        return 8
    raise TypeError(type(x))


class BatchKernel:
    """Run the real native Batcher iterator for four records, including stack."""
    def __call__(self, value):
        variant = InProcessBatcherPipeVariant(None, 4, False)
        variant._input_iter = iter([DataSample(value) for _ in range(4)])
        return list(variant._iter_impl())[0].data


class RGBReader:
    def __call__(self, path):
        return read_image(path, mode=ImageReadMode.RGB)


def inventory(records, tokenizer):
    random.seed(7341)
    np.random.seed(7341)
    torch.manual_seed(7341)
    stages = []
    for name, cls in zip(NAMES, CLASSES):
        feature = cls(4, tokenizer_path=tokenizer) if name == "clip" else cls(4)
        feature.apply(IterSource([]))
        values = [r["image"] if name == "simclr" else r for r in records]
        for position, pid in enumerate(topological_sort(feature.logical_adj_list)):
            pipe = feature.logical_pipes[pid]
            if pipe.is_source():
                continue
            if isinstance(pipe, MapperPipe):
                fn = pipe.fn
            elif isinstance(pipe, ImageReaderPipe):
                fn = RGBReader()
            elif isinstance(pipe, BatcherPipe):
                fn = BatchKernel()
            else:
                raise TypeError(f"Uncovered operator: {pipe.name}")
            stage = dict(workload=name, position=position,
                         tag=pipe.tag or f"op{position}_{type(fn).__name__}",
                         name=pipe.name, fn=fn, inputs=values,
                         field=fn.field if isinstance(fn, OnField) else None,
                         reader=isinstance(fn, (ReadRecord, RGBReader)),
                         batch=isinstance(fn, BatchKernel),
                         native_variants=[v.name for v in pipe.get_spec().mutable_variants])
            operation = fn.operation if isinstance(fn, OnField) else fn
            operation_name = type(operation).__name__
            stage["stochastic"] = (
                "Random" in operation_name or operation_name in
                ("ColorJitter", "GaussianBlur", "Solarize"))
            stages.append(stage)
            values = [fn(v) for v in values]
    return stages


def scale(value, pixels, text_bytes, tokens):
    side = round(math.sqrt(pixels))
    if isinstance(value, Image.Image):
        return value.resize((side, side), Image.Resampling.BILINEAR)
    if isinstance(value, torch.Tensor):
        if value.ndim >= 3:
            return TF.resize(value, [side, side], antialias=True)
        repeats = math.ceil(tokens / max(1, value.numel()))
        return value.repeat(repeats)[:tokens]
    if isinstance(value, str):
        raw = (value.strip() + " ").encode("utf-8")
        return (raw * math.ceil(text_bytes / len(raw)))[:text_bytes].decode("utf-8", "ignore")
    if isinstance(value, dict):
        return {k: scale(v, pixels, text_bytes, tokens) for k, v in value.items()}
    if isinstance(value, list):
        if value and isinstance(value[0], int):
            return (value * math.ceil(tokens / len(value)))[:tokens]
        return [scale(v, pixels, text_bytes, tokens) for v in value]
    return value


class Engine:
    def __init__(self, output, points):
        self.output = str(output)
        self.points = points
        self.stages = None

    def __call__(self, request):
        try:
            if self.stages is None:
                torch.set_num_threads(1)
                records = json.loads((Path(self.output) / "records.json").read_text())
                self.stages = inventory(records, str(
                    ROOT / "evaluation/datasets/target_pipeline/clip_tokenizer"))
            if request.get("inventory"):
                return [{k: v for k, v in stage.items() if k not in ("fn", "inputs")}
                        for stage in self.stages]
            index, point, repeat = request["index"], request["point"], request["round"]
            stage = self.stages[index]
            pixels = int(np.linspace(64**2, 2048**2, self.points)[point])
            text_bytes = int(np.linspace(256, 65536, self.points)[point])
            tokens = int(np.linspace(16, 4096, self.points)[point])
            inputs, sizes = [], []
            for j, original in enumerate(stage["inputs"]):
                if stage["reader"]:
                    path = str(Path(self.output) / "images" / f"{point}_{j}.png")
                    value = path if isinstance(original, str) else dict(original, image=path)
                    size = Path(path).stat().st_size
                elif stage["field"]:
                    field = stage["field"]
                    active = scale(original[field], pixels, text_bytes, tokens)
                    value = dict(original)
                    value[field] = active
                    size = nbytes(active)
                else:
                    value = scale(original, pixels, text_bytes, tokens)
                    size = nbytes(value)
                inputs.append(value)
                sizes.append(size)
            fn = stage["fn"]
            # Equal input content and random stream for every backend in a round.
            seed = 100003 * index + 101 * point + repeat
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            for value in inputs:
                fn(value)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            calls = request.get("calls", 4)
            start = time.perf_counter_ns()
            result = None
            for i in range(calls):
                result = fn(inputs[i % len(inputs)])
            elapsed = (time.perf_counter_ns() - start) / 1e9
            records_per_call = 4 if stage["batch"] else 1
            return dict(seconds=elapsed, calls=calls, input_bytes=sum(sizes)/len(sizes),
                        records_per_call=records_per_call,
                        output_bytes=nbytes(result), pid=os.getpid(),
                        torch_threads=torch.get_num_threads(),
                        target_pixels=pixels, side=round(math.sqrt(pixels)),
                        target_text_bytes=text_bytes, target_tokens=tokens)
        except Exception:
            return {"error": traceback.format_exc()}


def prepare(output, points):
    annotation = ROOT / "evaluation/datasets/target_pipeline/coco/annotations/coco_karpathy_train.json"
    rows = json.loads(annotation.read_text())
    records, seen = [], set()
    for row in rows:
        if row["image"] in seen:
            continue
        seen.add(row["image"])
        records.append({"image": str(ROOT / "evaluation/datasets/target_pipeline/coco/images" / row["image"]),
                        "caption": row["caption"]})
        if len(records) == 4:
            break
    write_json(output / "records.json", records)
    (output / "images").mkdir(exist_ok=True)
    for j, row in enumerate(records):
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            for point, area in enumerate(np.linspace(64**2, 2048**2, points)):
                side = round(math.sqrt(area))
                image.resize((side, side), Image.Resampling.BILINEAR).save(
                    output / "images" / f"{point}_{j}.png", compress_level=0)


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    data = pd.read_csv(output / "raw.csv")
    summary = data.groupby(["operator_index", "workload", "tag", "backend", "input_bytes"]).agg(
        mean_ms=("ms_per_record", "mean"), std_ms=("ms_per_record", "std"),
        mean_mib_per_sec=("mib_per_sec", "mean"), std_mib_per_sec=("mib_per_sec", "std"),
        rounds=("round", "count")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    inv = json.loads((output / "operators.json").read_text())
    (output / "figures").mkdir(exist_ok=True)
    pages = []
    for workload in NAMES:
        pages.append(f"<h2>{workload}</h2>")
        with PdfPages(output / "figures" / f"{workload}.pdf") as pdf:
            for index, op in enumerate(inv):
                if op["workload"] != workload:
                    continue
                subset = data[data["operator_index"] == index]
                if subset.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
                for backend in ("local", "smp", "ray"):
                    cells = subset[subset["backend"] == backend].groupby("input_bytes")
                    for ax, metric in zip(axes, ("ms_per_record", "mib_per_sec")):
                        stats = cells[metric].agg(["mean", "std"]).sort_index()
                        x = stats.index / 1024
                        ax.errorbar(x, stats["mean"], yerr=stats["std"].fillna(0),
                                    label=backend, marker=".", capsize=2)
                        ax.set_xlabel("Active input (KiB; linear scale)")
                        ax.grid(alpha=.25)
                axes[0].set_ylabel("Worker compute (ms / record)")
                axes[1].set_ylabel("Logical input rate (MiB / s)")
                axes[1].legend()
                fig.suptitle(f'{workload}: {op["tag"]} — mean ± SD, 3 rounds')
                fig.tight_layout()
                filename = f"{workload}_{index:03d}.png"
                fig.savefig(output / "figures" / filename, dpi=130)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(f'<p>{op["name"]}</p><img loading="lazy" width="1000" src="figures/{filename}">')
    (output / "index.html").write_text(
        '<meta charset="utf-8"><title>Operator scaling</title>'
        '<h1>Controlled operator input-size sweeps</h1>'
        '<p>Worker compute only; preparation, startup and IPC excluded. '
        'Error bars: sample SD across three rounds. Linear axes. '
        'Counterfactual sizes may exceed reachable pipeline inputs. '
        'Reader/Batcher kernels are hosted in workers; this does not add native remote variants.</p>'
        + "".join(pages))


def calibration_calls(seconds, stochastic):
    # A random transform can skip every call in a short pilot. Never infer an
    # unbounded fast rate from this; use the same bounded count in all workers.
    limit = 32 if stochastic else 8192
    return max(4, min(limit, 4 * math.ceil(.03 / max(seconds, 1e-9))))


def complete_cells(rows):
    """Keep only complete 3-resource x 3-round cells for unbiased resumption."""
    groups = {}
    for row in rows:
        key = (int(row["operator_index"]), int(row["point"]))
        groups.setdefault(key, []).append(row)
    expected = {(backend, repeat) for backend in ("local", "smp", "ray")
                for repeat in (1, 2, 3)}
    complete = set()
    for key, cell in groups.items():
        actual = {(r["backend"], int(r["round"])) for r in cell}
        if len(cell) == 9 and actual == expected:
            if len({int(r["calls"]) for r in cell}) != 1:
                raise ValueError(f"Inconsistent call counts for cell {key}")
            complete.add(key)
    return [r for r in rows if (int(r["operator_index"]), int(r["point"])) in complete], complete


def main():
    import ray
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gate-operator", type=int)
    parser.add_argument("--gate-point", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(output)
        return
    previous_rows, done_cells = [], set()
    if (output / "raw.csv").exists():
        if not args.resume:
            raise FileExistsError("Use a fresh output directory or --resume")
        metadata = json.loads((output / "metadata.json").read_text())
        if metadata["points"] != args.points or metadata["rounds"] != 3:
            raise ValueError("Cannot resume with a different size grid or round count")
        with (output / "raw.csv").open() as stream:
            previous_rows, done_cells = complete_cells(list(csv.DictReader(stream)))
        archive = output / ("before_resume_" + str(time.time_ns()))
        archive.mkdir()
        for name in ("raw.csv", "metadata.json", "status.json", "operators.json"):
            if (output / name).exists():
                shutil.copy2(output / name, archive / name)
        write_json(archive / "resume.json", {
            "retained_rows": len(previous_rows), "retained_cells": len(done_cells),
            "reason": "Random skip-only calibration caused excessive calls and SMP timeout"})

    torch.set_num_threads(1)
    write_json(output / "metadata.json", {
        "points": args.points, "rounds": 3,
        "target_pixels": np.linspace(64**2, 2048**2, args.points).astype(int).tolist(),
        "target_text_bytes": np.linspace(256, 65536, args.points).astype(int).tolist(),
        "target_tokens": np.linspace(16, 4096, args.points).astype(int).tolist(),
        "threads_per_worker": 1, "workers_per_backend": 1,
        "ray_version": ray.__version__, "torch_version": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement": "worker_compute_only",
        "stochastic_max_calls": 32, "deterministic_max_calls": 8192,
        "retained_rows_on_resume": len(previous_rows),
        "sampling": "controlled linear size interventions on four real COCO records"
    })
    write_json(output / "status.json", {"status": "running", "phase": "prepare", "pid": os.getpid()})
    if not args.resume:
        prepare(output, args.points)
    elif not (output / "records.json").is_file():
        raise FileNotFoundError("Resume requires the original input records")
    engine = Engine(output, args.points)
    operators = engine({"inventory": True})
    if isinstance(operators, dict):
        raise RuntimeError(operators)
    if args.resume:
        old = json.loads((output / "operators.json").read_text())
        for op in old:
            op.pop("stochastic", None)
        current = [{k: v for k, v in op.items() if k != "stochastic"} for op in operators]
        if old != current:
            raise ValueError("Cannot resume changed workload graphs")
    write_json(output / "operators.json", operators)
    req, resp = mp.Queue(2), mp.Queue(2)
    actor = SMPActorMapperPipeVariant("scaling-smp", Engine(output, args.points))
    actor.register(req, resp)
    actor.start()
    # Start a private Ray runtime; never stop an existing experiment cluster.
    ray.init(address="local", num_cpus=1, include_dashboard=False,
             _node_ip_address="127.0.0.1", _temp_dir=f"/tmp/scaling_ray_{os.getpid()}",
             log_to_driver=False, object_store_memory=256*1024*1024)
    remote = RayActorMapperPipeVariant.options(num_cpus=1).remote(
        "scaling-ray", Engine(output, args.points))

    def run(backend, request):
        if backend == "local":
            result = engine(request)
        elif backend == "smp":
            req.put(request)
            try:
                result = resp.get(timeout=180)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"SMP request exceeded 180 seconds: {request}; "
                    f"actor alive={actor.is_alive()}, exitcode={actor.exitcode}") from exc
        else:
            result = ray.get(remote.process.remote([request]), timeout=180)[0]
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    fields = ["operator_index", "workload", "tag", "backend", "round", "point",
              "input_bytes", "ms_per_record", "mib_per_sec", "seconds", "calls",
              "records_per_call", "output_bytes", "pid", "torch_threads",
              "target_pixels", "side", "target_text_bytes", "target_tokens"]
    completed = len(previous_rows)
    try:
        with (output / "raw.csv").open("w", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(previous_rows)
            indices = list(range(len(operators)))
            if args.gate:
                indices = [next(i for i, op in enumerate(operators) if op["workload"] == name)
                           for name in NAMES]
            if args.gate_operator is not None:
                if not args.gate:
                    raise ValueError("--gate-operator requires --gate")
                indices = [args.gate_operator]
            points = [0, args.points - 1] if args.gate else range(args.points)
            if args.gate_point is not None:
                if not args.gate:
                    raise ValueError("--gate-point requires --gate")
                points = [args.gate_point]
            for index in indices:
                op = operators[index]
                for point in points:
                    if (index, point) in done_cells:
                        continue
                    base = {"index": index, "point": point, "round": 0}
                    calibration = run("local", base)
                    calls = calibration_calls(calibration["seconds"], op["stochastic"])
                    for repeat in range(3):
                        backends = ("local", "smp", "ray")
                        for backend in backends[repeat:] + backends[:repeat]:
                            row = run(backend, dict(base, round=repeat, calls=calls))
                            count = row["calls"] * row["records_per_call"]
                            row.update(operator_index=index, workload=op["workload"],
                                       tag=op["tag"], backend=backend, round=repeat+1,
                                       point=point, ms_per_record=1000*row["seconds"]/count,
                                       mib_per_sec=row["input_bytes"]*count/row["seconds"]/2**20)
                            writer.writerow(row)
                            completed += 1
                write_json(output / "status.json",
                           {"status": "running", "completed_operators": index+1,
                            "total_operators": len(operators), "rows": completed})
                print(f"Completed {index+1}/{len(operators)} {op['workload']} {op['tag']}", flush=True)
        if not args.gate:
            plot(output)
        write_json(output / "status.json",
                   {"status": "passed", "operators": len(indices), "rows": completed,
                    "points": len(points), "rounds": 3, "backends": ["local", "smp", "ray"],
                    "gate": args.gate})
    except Exception:
        write_json(output / "status.json",
                   {"status": "failed", "rows": completed, "error": traceback.format_exc()})
        raise
    finally:
        actor.stop()
        actor.join(timeout=5)
        if actor.is_alive():
            actor.terminate()
            actor.join()
        req.close()
        resp.close()
        ray.kill(remote)
        ray.shutdown()


if __name__ == "__main__":
    main()
