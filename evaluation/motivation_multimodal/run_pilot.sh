#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="outputs/motivation_multimodal/pilot"
FIXTURE_ROOT="outputs/motivation_multimodal/fixture"
PILOT_NUM_SAMPLES=500
DISABLE_SMP=0
GRID="standard"
PLAN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --fixture-root) FIXTURE_ROOT="$2"; shift 2 ;;
    --pilot-num-samples) PILOT_NUM_SAMPLES="$2"; shift 2 ;;
    --disable-smp) DISABLE_SMP=1; shift ;;
    --grid) GRID="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source /workspace/OptimalCedar/env/bin/activate
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"
if [[ ! "$PILOT_NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--pilot-num-samples must be a positive integer" >&2
  exit 2
fi
if [[ "$FIXTURE_ROOT" != /* ]]; then
  FIXTURE_ROOT="$REPO_ROOT/$FIXTURE_ROOT"
fi
if [[ ! -f "$FIXTURE_ROOT/pilot.jsonl" ]]; then
  echo "Missing pilot fixture: $FIXTURE_ROOT/pilot.jsonl" >&2
  exit 2
fi
PILOT_RECORDS=$(wc -l < "$FIXTURE_ROOT/pilot.jsonl")
if (( PILOT_RECORDS < PILOT_NUM_SAMPLES )); then
  echo "Pilot fixture has $PILOT_RECORDS records; requested $PILOT_NUM_SAMPLES" >&2
  exit 2
fi
PID_FILE="$OUTPUT_ROOT/run.pid"
LOG_FILE="$OUTPUT_ROOT/run.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "Pilot is already running with PID $(<"$PID_FILE")" >&2
  exit 1
fi

export W=1
export CPU_BUDGET=64
export CEDAR_PROFILE_TIME_SEC=10
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_FILTER_SELECTIVITY=1
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET="$CPU_BUDGET"
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET="$CPU_BUDGET"
export CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS="$W"
if (( DISABLE_SMP == 1 )); then
  export PICO_MULTIMODAL_DISABLE_SMP=1
else
  unset PICO_MULTIMODAL_DISABLE_SMP
fi
{
  echo "W=$W"
  echo "CPU_BUDGET=$CPU_BUDGET"
  echo "fixture=$FIXTURE_ROOT"
  echo "pilot_num_samples=$PILOT_NUM_SAMPLES"
  echo "disable_smp=$DISABLE_SMP"
  echo "grid=$GRID"
  echo "plan_only=$PLAN_ONLY"
  echo "image_root=/workspace/OptimalCedar/datasets/coco/val2017"
  date --iso-8601=seconds
} > "$OUTPUT_ROOT/metadata.txt"

EXTRA_ARGS=()
if (( PLAN_ONLY == 1 )); then EXTRA_ARGS+=(--plan-only); fi
python -m evaluation.motivation_multimodal.pilot \
  --fixture "$FIXTURE_ROOT" \
  --image-root /workspace/OptimalCedar/datasets/coco/val2017 \
  --pilot-num-samples "$PILOT_NUM_SAMPLES" \
  --grid "$GRID" \
  "${EXTRA_ARGS[@]}" \
  --output "$OUTPUT_ROOT" > "$LOG_FILE" 2>&1 &
CHILD_PID=$!
echo "$CHILD_PID" > "$PID_FILE"
if wait "$CHILD_PID"; then
  echo success > "$OUTPUT_ROOT/SUCCESS"
else
  STATUS=$?
  echo "$STATUS" > "$OUTPUT_ROOT/FAILED"
  exit "$STATUS"
fi
