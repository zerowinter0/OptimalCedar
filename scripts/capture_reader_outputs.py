"""Capture ImageReader outputs for the real-block fusion experiment.

Runs the real SimCLRv2 source + ImageReader (in-process, no fusion) and saves
the tensors that enter the first mapper of the declared order, together with
their measured serialized sizes.

Usage:
  python -u scripts/capture_reader_outputs.py --run-dir outputs/<run>/expB \
      --records 400
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import (  # noqa: E402
    PIPE_IMAGE_READER,
    PIPE_LOCAL_FS,
    build_feature,
    payload_bytes,
)
from cedar.pipes import DataSample, PipeVariantType  # noqa: E402
from cedar.pipes.context import InProcessPipeVariantContext  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=400)
    args = parser.parse_args()

    torch.set_num_threads(1)
    feature = build_feature(batch_size=4)
    pipes = feature.logical_pipes
    source = pipes[PIPE_LOCAL_FS]._create_pipe_variant(
        PipeVariantType.INPROCESS, InProcessPipeVariantContext()
    )
    pipes[PIPE_LOCAL_FS].pipe_variant = source
    pipes[PIPE_LOCAL_FS].pipe_variant_type = PipeVariantType.INPROCESS
    reader = pipes[PIPE_IMAGE_READER]._create_pipe_variant(
        PipeVariantType.INPROCESS, InProcessPipeVariantContext()
    )
    reader._input_iter = source._iter_impl()

    records = []
    sizes = []
    for sample in reader._iter_impl():
        data = sample.data if isinstance(sample, DataSample) else sample
        records.append(data.detach().clone())
        sizes.append(payload_bytes(data) or 0)
        if len(records) >= args.records:
            break

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "inputs").mkdir(exist_ok=True)
    torch.save(
        {"records": records, "shape": list(records[0].shape),
         "dtype": str(records[0].dtype)},
        args.run_dir / "inputs" / "block_inputs.pt",
    )
    meta = {
        "records": len(records),
        "shape": list(records[0].shape),
        "dtype": str(records[0].dtype),
        "bytes_mean": statistics.fmean(sizes),
        "bytes_median": statistics.median(sizes),
        "bytes_min": min(sizes),
        "bytes_max": max(sizes),
        "stage": "ImageReaderPipe(8) output (input of to_float(7))",
    }
    (args.run_dir / "inputs" / "block_inputs_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
