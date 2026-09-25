#!/usr/bin/env bash
# C3: hold the final PICO plan's order/fusion/backend fixed and sweep W.
#
# The plan is materialised once by the final optimizer, then re-executed with
# n_local_workers forced to each value in the ladder.  This isolates the W
# effect from plan re-optimisation.
#
# Usage (inside the container):
#   bash scripts/pico_final_w_scaling_20260924.sh <workload> <plan.yaml>
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

WORKLOAD=${1:?workload required}
PLAN=${2:?plan yaml required}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
LADDER=${LADDER:-1,4,16,64}
ROUNDS=${ROUNDS:-3}
SAMPLES=${SAMPLES:-9469}
ENTRY="$OUT/entry.py"
RAY_IP=172.23.166.105:6379
DATA=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64

work="$OUT/$WORKLOAD/w_scaling"
mkdir -p "$work/logs" "$work/plans"

for w in ${LADDER//,/ }; do
  plan_w="$work/plans/${WORKLOAD}_W${w}.yaml"
  python - "$PLAN" "$plan_w" "$w" <<'PY'
import sys
from pathlib import Path

import yaml

source, target, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = yaml.safe_load(Path(source).read_text())
plan = data.get("physical_plan", data)
payload = plan.get("feature_r0", plan)
payload["n_local_workers"] = workers
# PhysicalPlan.from_dict keeps graph keys as given; the feature's logical pipe
# ids are ints, and a string-keyed graph fails the coverage check with
# "Not all pipes specified in physical plan."
payload["graph"] = {int(key): value for key, value in payload["graph"].items()}
open(target, "w").write(yaml.safe_dump({"physical_plan": payload}))
PY
  result="$work/results_W${w}.json"
  if [ -s "$result" ]; then
    echo "SKIP $WORKLOAD W=$w"
    continue
  fi
  echo "W-CELL $WORKLOAD W=$w $(date -Is)"
  python -u "$ENTRY" "$OUT/modules/evaluation/compare_optimizer_perf.py" \
    --dataset_file "$OUT/modules/evaluation/pipelines/simclrv2/cedar_dataset.py" \
    --dataset_kwargs "dataset_path=$DATA" \
    --batch_size 4 --num_epochs 1 --num_total_samples "$SAMPLES" \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "outputs/affine_repr_profile_20260924/${WORKLOAD}/shared.yaml" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --master_feature_config "$plan_w" \
    --optimizer_time_limit_sec 600 \
    --num_repeats "$ROUNDS" --skip_pico_plan_cost --disable_caching \
    --results_path "$result" > "$work/logs/W${w}.log" 2>&1
  echo "W-CELL-DONE $WORKLOAD W=$w exit=$? $(date -Is)"
done
echo "W-SCALING-FINISHED $WORKLOAD $(date -Is)"
