#!/usr/bin/env bash
# SimCLRv2, 9,469 records: cedar-opt's plan with its fused augmentation block
# executed either on Ray (as cedar-opt chose) or locally (INPROCESS), with the
# rest of the plan and W=64 unchanged.  Three interleaved rounds per arm.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

RUN=outputs/simclrv2_local_vs_ray_9469_20260920
PROFILE=outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml
SOURCE_PLAN=outputs/six_workload_formal_v3_20260919/simclrv2/plans/round1__optimizer.yaml
DATASET_FILE=evaluation/pipelines/simclrv2/cedar_dataset.py
TRAIN=evaluation/datasets/imagenette2/imagenette2/train
RAY_IP=172.23.166.105:6379

mkdir -p "$RUN/plans" "$RUN/logs" "$RUN/results"

python - "$SOURCE_PLAN" "$RUN/plans" <<'PY'
import sys
import yaml
from pathlib import Path

source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
plan = yaml.safe_load(source.read_text())["feature_r0"]
plan["graph"] = {int(k): v for k, v in plan["graph"].items()}
plan["pipes"] = {int(k): v for k, v in plan["pipes"].items()}
for desc in plan["pipes"].values():
    # Operators that were fused away keep no variant in the recorded plan;
    # they must still deserialize.
    desc.setdefault("variant", "INPROCESS")
    desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})

ray = yaml.safe_load(yaml.safe_dump(plan))
(out_dir / "cedar_opt_ray_w1.yaml").write_text(
    yaml.safe_dump({"physical_plan": ray}, sort_keys=False)
)

local = yaml.safe_load(yaml.safe_dump(plan))
for p_id, desc in local["pipes"].items():
    if desc.get("variant") == "RAY":
        desc["variant"] = "INPROCESS"
        desc["variant_ctx"] = {"variant_type": "INPROCESS"}
        fused = desc.get("fused_pipes")
        name = desc["name"]
        print(f"fused stage {p_id} {name} fused={fused} RAY -> INPROCESS (local)")
(out_dir / "cedar_opt_local_w1.yaml").write_text(
    yaml.safe_dump({"physical_plan": local}, sort_keys=False)
)
print(f"n_local_workers={plan['n_local_workers']} for both arms")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1 \
       CEDAR_WORKER_READY_TIMEOUT_SEC=600

for round in 1 2 3; do
  for arm in ray local; do
    cell="round${round}__${arm}"
    echo "RUN $cell"
    python -u scripts/run_fixed_plan_throughput.py \
      --plan "$RUN/plans/cedar_opt_${arm}_w1.yaml" \
      --label "cedar-opt-fuse-${arm}" \
      --results-path "$RUN/results/${cell}.json" \
      --dataset-file "$DATASET_FILE" \
      --dataset-kwargs "dataset_path=$REPO/$TRAIN" \
      --batch-size 4 --num-epochs 1 --num-total-samples 0 \
      --profiled-stats "$PROFILE" --ray-ip "$RAY_IP" \
      > "$RUN/logs/${cell}.log" 2>&1
    tail -1 "$RUN/logs/${cell}.log"
  done
done

( cd "$RUN" && sha256sum plans/*.yaml results/*.json > results.sha256 )
echo "DONE"
