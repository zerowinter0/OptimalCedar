#!/usr/bin/env bash
# §3.1 mechanism evidence: SimCLRv2 B/H/J with four execution organisations.
#
#   experiment A  serial service time, one batch in flight (services/serial)
#   experiment B  full pipeline, W=1 and W=64, three repeats (services/pipeline)
#   section 4     Cedar's own cost model, every intermediate exported
#
# Everything lands in a fresh run directory; historical snapshots are read-only.
#
# Usage (inside the container):
#   bash scripts/run_block_mechanism_20260923.sh
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

RUN=${RUN:-outputs/simclrv2_fusion_offload_mechanism_$(date -u +%Y%m%d_%H%M%S)}
RECORDS=${RECORDS:-400}
MIN_SECONDS=${MIN_SECONDS:-30}
PIPELINE_SAMPLES=${PIPELINE_SAMPLES:-80000}
PIPELINE_REPEATS=${PIPELINE_REPEATS:-3}
PROFILE=${PROFILE:-outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml}
BASE_PLAN=${BASE_PLAN:-outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/plans/round1__optimizer.yaml}
LOCAL_CPU=${LOCAL_CPU:-12}
REMOTE_CPU_BASE=${REMOTE_CPU_BASE:-8}

mkdir -p "$RUN"/{logs,plans,inputs}
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote
export CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_RAY_PATH_TIMING=1
export CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC=${CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC:-900}
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_LOCAL_WORKERS_MAX=64
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

echo "RUN DIR $RUN" | tee "$RUN/logs/driver.log"

echo "=== capture block inputs" | tee -a "$RUN/logs/driver.log"
python -u scripts/block_service_harness.py capture \
    --run-dir "$RUN" --records "$RECORDS" --cpu "$LOCAL_CPU" \
    > "$RUN/logs/capture.log" 2>&1
tail -3 "$RUN/logs/capture.log" | tee -a "$RUN/logs/driver.log"

echo "=== plans" | tee -a "$RUN/logs/driver.log"
python -u scripts/block_pipeline_matrix.py plans --out-dir "$RUN/plans" \
    > "$RUN/logs/plans.log" 2>&1

echo "=== experiment A: serial service time" | tee -a "$RUN/logs/driver.log"
python -u scripts/block_service_harness.py service \
    --run-dir "$RUN" --batch-size 4 \
    --pilot-batches 40 --warmup-batches 120 --min-batches 120 \
    --min-seconds "$MIN_SECONDS" \
    --cpu "$LOCAL_CPU" --remote-cpu-base "$REMOTE_CPU_BASE" \
    > "$RUN/logs/service.log" 2>&1
grep -E "PILOT|MEASURE|RESULT" "$RUN/logs/service.log" | tee -a "$RUN/logs/driver.log"

echo "=== experiment B: full pipeline" | tee -a "$RUN/logs/driver.log"
python -u scripts/block_pipeline_matrix.py run \
    --run-dir "$RUN" --profile "$PROFILE" \
    --repeats "$PIPELINE_REPEATS" --num-samples "$PIPELINE_SAMPLES" \
    > "$RUN/logs/pipeline.log" 2>&1
grep -E "RUN |-> rc" "$RUN/logs/pipeline.log" | tail -20 | tee -a "$RUN/logs/driver.log"

echo "=== Cedar model breakdown" | tee -a "$RUN/logs/driver.log"
python -u scripts/cedar_block_cost_breakdown.py \
    --profile "$PROFILE" --base-plan "$BASE_PLAN" \
    --out "$RUN/cedar_cost_breakdown.json" \
    > "$RUN/logs/cedar_cost.log" 2>&1

echo "=== figure data" | tee -a "$RUN/logs/driver.log"
python -u scripts/block_mechanism_figure_data.py --run-dir "$RUN" \
    > "$RUN/logs/figure_data.log" 2>&1

python - <<PY | tee -a "$RUN/logs/driver.log"
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
run = Path("$RUN")
def sha(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
commit = subprocess.run(["git", "-C", ".", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
payload = {
    "run_dir": str(run.resolve()),
    "git_commit": commit,
    "profile_path": "$PROFILE",
    "profile_sha256": sha("$PROFILE"),
    "base_plan_path": "$BASE_PLAN",
    "base_plan_sha256": sha("$BASE_PLAN"),
    "hostname": platform.node(),
    "python": sys.version,
    "pipeline_samples": $PIPELINE_SAMPLES,
    "pipeline_repeats": $PIPELINE_REPEATS,
    "service_min_seconds": $MIN_SECONDS,
    "input_records": $RECORDS,
    "env": {k: v for k, v in os.environ.items() if k.startswith("CEDAR_")},
}
(run / "run_manifest.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY

echo "DONE $RUN" | tee -a "$RUN/logs/driver.log"
