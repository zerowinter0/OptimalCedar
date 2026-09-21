#!/usr/bin/env bash
# simclrv2, 9,469 records: cedar's plan (fused {B,H,J} on Ray) and the same
# plan with that block local, each at W=1 and W=64.
#
# This isolates the claim from the profile: the profiled "Ray offload speeds the
# pipeline up" measurement was taken with ONE local worker and ONE actor, i.e.
# a serial worker chain.  At W=64 the local work is already parallel, so the Ray
# hop's per-record serialize/transfer cost is added instead of hidden.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

BASE=outputs/simclrv2_local_vs_ray_9469_20260920/plans
RUN=outputs/simclrv2_raylocal_w1_w64_20260921
PROFILE=outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml
DATASET=evaluation/pipelines/simclrv2/cedar_dataset.py
TRAIN=evaluation/datasets/imagenette2/imagenette2/train
RAY_IP=172.23.166.105:6379

mkdir -p "$RUN"/{plans,logs,results}

python - "$BASE" "$RUN/plans" <<'PY'
import sys, yaml
from pathlib import Path
base, out = Path(sys.argv[1]), Path(sys.argv[2])
for variant in ("ray", "local"):
    payload = yaml.safe_load((base / f"cedar_opt_{variant}_w1.yaml").read_text())
    for workers in (1, 64):
        plan = yaml.safe_load(yaml.safe_dump(payload))
        plan["physical_plan"]["n_local_workers"] = workers
        (out / f"cedar_opt_{variant}_W{workers}.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False)
        )
        print(f"wrote cedar_opt_{variant}_W{workers}.yaml")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1 \
       CEDAR_WORKER_READY_TIMEOUT_SEC=600

for variant in ray local; do
  for workers in 1 64; do
    cell="${variant}_W${workers}"
    echo "RUN $cell"
    python -u scripts/run_fixed_plan_throughput.py \
      --plan "$RUN/plans/cedar_opt_${variant}_W${workers}.yaml" \
      --label "cedar-fuse-${variant}-W${workers}" \
      --results-path "$RUN/results/${cell}.json" \
      --dataset-file "$DATASET" \
      --dataset-kwargs "dataset_path=$REPO/$TRAIN" \
      --batch-size 4 --num-epochs 1 --num-total-samples 0 \
      --profiled-stats "$PROFILE" --ray-ip "$RAY_IP" \
      > "$RUN/logs/${cell}.log" 2>&1
    tail -1 "$RUN/logs/${cell}.log"
  done
done

python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print()
print(f"{'cell':<12}{'samples':>9}{'steady s':>10}{'throughput':>12}{'setup s':>9}")
for cell in ("ray_W1", "local_W1", "ray_W64", "local_W64"):
    f = root / "results" / f"{cell}.json"
    d = json.loads(f.read_text())
    print("%-12s%9d%10.2f%12.2f%9.2f" % (
        cell, d["num_samples"], d["perf_time_sec"],
        d["throughput_samples_per_sec"], d["setup_time_sec"]))
PY
echo DONE
