#!/usr/bin/env bash
# Can Cedar's per-operator cost transfer to a new plan?
#
# Four pipelines over the same 9 simclrv2 operators and the same execution
# style as the unoptimized plan (single worker, every operator INPROCESS, no
# fusion, no prefetch).  Only the operator ORDER differs, which changes the
# input size each operator sees:
#   declared : F C H J G B N   (the unoptimized/declared order)
# W=4 (not 1): the reconcile trace is written by the MP workers, and W=1 would
# run the pipeline inside the driver process where nothing is dumped.
#   pico     : C G J H B F N   (PICO's plan order)
#   cedar    : G C B H J F N   (cedar-opt's plan order)
#   old-dp   : F N B G J C H   (old-dp's fused-block order)
# Each cell is traced with CEDAR_RECONCILE_DIR, so every operator reports its
# input bytes per record and its measured wall time per record.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

CAMPAIGN=outputs/ultimate_eight_optimizers_fix_20260921
RUN=outputs/unopt_order_transfer_20260921
PROFILE=outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml
DATASET=evaluation/pipelines/simclrv2/cedar_dataset.py
TRAIN=evaluation/datasets/imagenette2/imagenette2/train
RAY_IP=172.23.166.105:6379

mkdir -p "$RUN"/{plans,logs,results}

python - "$CAMPAIGN/simclrv2/plans/round1__unopti.yaml" "$RUN/plans" <<'PY'
import sys, yaml
from pathlib import Path

source, out = Path(sys.argv[1]), Path(sys.argv[2])
payload = yaml.safe_load(source.read_text())
payload = payload.get("feature_r0") or payload.get("feature")
payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
for desc in payload["pipes"].values():
    desc.setdefault("variant", "INPROCESS")
    desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})

LETTER = {"F": 7, "N": 1, "B": 2, "G": 3, "J": 4, "H": 5, "C": 6}
ORDERS = {
    "declared": "FCHJGBN",   # unoptimized / declared order
    "pico": "CGJHBFN",       # PICO plan order
    "cedar": "GCBHJFN",      # cedar-opt plan order
    "old-dp": "FNBGJCH",     # old-dp fused-block order
}
for name, order in ORDERS.items():
    plan = yaml.safe_load(yaml.safe_dump(payload))
    graph = {9: "8"}
    previous = 8
    for letter in order:
        node = LETTER[letter]
        graph[previous] = str(node)
        previous = node
    graph[previous] = "0"      # batcher last, exactly like the unoptimized plan
    graph[0] = ""
    plan["graph"] = graph
    plan["n_local_workers"] = 4   # MP mode: the driver path does not dump traces
    (out / f"{name}.yaml").write_text(
        yaml.safe_dump({"physical_plan": plan}, sort_keys=False)
    )
    print(f"wrote {name}.yaml  order={order}")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1 \
       CEDAR_WORKER_READY_TIMEOUT_SEC=600

for cell in declared pico cedar old-dp; do
  echo "RUN $cell"
  rm -rf "$RUN/reconcile_$cell"
  mkdir -p "$RUN/reconcile_$cell"
  CEDAR_RECONCILE_DIR="$REPO/$RUN/reconcile_$cell" \
  python -u scripts/run_fixed_plan_throughput.py \
    --plan "$RUN/plans/${cell}.yaml" \
    --label "order-${cell}" \
    --results-path "$RUN/results/${cell}.json" \
    --dataset-file "$DATASET" \
    --dataset-kwargs "dataset_path=$REPO/$TRAIN" \
    --batch-size 4 --num-epochs 1 --num-total-samples 0 \
    --profiled-stats "$PROFILE" --ray-ip "$RAY_IP" \
    > "$RUN/logs/${cell}.log" 2>&1
  tail -1 "$RUN/logs/${cell}.log"
done
echo DONE
