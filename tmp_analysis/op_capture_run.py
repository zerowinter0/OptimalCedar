"""Run one plan and capture what its operators really receive.

Combines the reconciliation trace (per-stage wall/process service) with the
diagnostic operator capture (per-call representation, direct callable time and
real input payloads), so A/B/C can be compared on the same run.

Usage (inside the container):
  python -u tmp_analysis/op_capture_run.py <plan.yaml> <out_dir> [samples] [workers]
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

ENTRY = ROOT / "outputs/simclrv2_four_remote_20260911/entry.py"
DATA = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
ADDRESS = "172.23.166.105:6379"
DEFAULT_PROFILE = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2"
    / "profiles/shared.yaml"
)


def normalize(plan_path: Path, target: Path, workers: int) -> Path:
    data = yaml.safe_load(plan_path.read_text())
    plan = data.get("physical_plan", data)
    payload = plan.get("feature_r0", plan)
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for pipe in payload["pipes"].values():
        pipe["variant"] = pipe.get("variant") or "INPROCESS"
        pipe.setdefault("variant_ctx", {"variant_type": pipe["variant"]})
    payload["n_local_workers"] = int(workers)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump({"physical_plan": payload}))
    return target


def main() -> int:
    plan_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_file = normalize(plan_path, out_dir / "plan.yaml", workers)
    env = dict(os.environ)
    env.update(
        {
            "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
            "CEDAR_RECONCILE_DIR": str(out_dir / "trace"),
            "CEDAR_OP_CAPTURE_DIR": str(out_dir / "capture"),
            "CEDAR_OP_CAPTURE_SAMPLES": os.environ.get(
                "CEDAR_OP_CAPTURE_SAMPLES", "8"
            ),
            "CEDAR_TRACE_FREQUENCY_SEC": "0",
        }
    )
    cmd = [
        sys.executable,
        "-u",
        str(ENTRY),
        "evaluation/eval_cedar.py",
        "--dataset_file",
        DATA,
        "--batch_size",
        os.environ.get("CEDAR_CAPTURE_BATCH", "1"),
        "--num_total_samples",
        str(samples),
        "--profiled_stats",
        os.environ.get("CEDAR_CAPTURE_PROFILE", str(DEFAULT_PROFILE)),
        "--master_feature_config",
        str(plan_file),
        "--use_ray",
        "--ray_ip",
        ADDRESS,
        "--disable_controller",
        "--disable_caching",
    ]
    log = out_dir / "run.log"
    with log.open("w") as handle:
        code = subprocess.run(
            cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
        ).returncode
    text = log.read_text(errors="replace")
    match = re.search(r"Total time: ([0-9.]+)s", text)
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "out": str(out_dir),
                "exit": code,
                "total_sec": float(match.group(1)) if match else None,
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
