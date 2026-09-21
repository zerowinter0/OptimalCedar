"""Inspect what the Batcher's trace window actually contains.

Runs the declared-order plan in a single process with tracing enabled and
prints the trace_order / per-pipe deltas of the first few batched samples.
"""

import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

import yaml

from cedar.compose.optimizer import PhysicalPlan
from cedar.config import CedarContext
from cedar.sources import LocalFSSource
from evaluation.pipelines.simclrv2.cedar_dataset import SimCLRV2Feature

PLAN = ROOT / "outputs/unopt_order_transfer_20260921/plans/declared.yaml"
PIPELINE = ROOT / "evaluation/pipelines/simclrv2/cedar_dataset.py"
TRAIN = ROOT / "evaluation/datasets/imagenette2/imagenette2/train"


def main() -> int:
    payload = yaml.safe_load(PLAN.read_text())["physical_plan"]
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for desc in payload["pipes"].values():
        desc.setdefault("variant", "INPROCESS")
        desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})
    plan = PhysicalPlan.from_dict(payload)

    feature = SimCLRV2Feature(batch_size=4)
    feature.apply(LocalFSSource(str(TRAIN), recursive=True))
    ctx = CedarContext()
    iterator = iter(feature.load_from_plan(ctx, plan))
    for source_pipe in feature.source_pipes:
        source_pipe.get_variant().enable_profiling()
    for idx in range(6):
        sample = next(iterator)
        order = list(sample.trace_order or [])
        wall = sample.wall_trace_dict or {}
        resume = sample.wall_trace_resume_dict or {}
        proc = sample.trace_dict or {}
        sizes = sample.size_dict or {}
        print(f"--- sample {idx}: trace_order={order} size_dict={sizes}")
        base = wall.get(0, 0)
        for position, pid in enumerate(order):
            prev = order[position - 1] if position else None
            wall_delta = (wall.get(pid, 0) - resume.get(prev, 0)) / 1e6 if prev is not None else 0.0
            proc_delta = (proc.get(pid, 0) - proc.get(prev, 0)) / 1e6 if prev is not None else 0.0
            print(
                "    pipe %-2s wall_delta %9.3f ms | pipe_end_ts %9.3f ms before yield"
                " | resume_ts %9.3f ms before yield | process_delta %9.3f ms"
                % (
                    pid,
                    wall_delta,
                    (wall.get(pid, 0) - base) / 1e6,
                    (resume.get(pid, 0) - base) / 1e6,
                    proc_delta,
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
