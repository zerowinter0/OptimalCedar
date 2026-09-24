#!/usr/bin/env bash
# Continuation driver: waits for the running campaign to finish, then repairs
# the three workloads whose profiles failed (coco, llava_pretrain,
# wikitext103) and runs their cells, then re-assembles everything.
#
# Why a separate driver: the main campaign is already occupying the machine and
# the remote Ray cluster; profiling or executing concurrently would corrupt the
# throughput measurements of the running cells.
#
# Usage (inside the container, normally under tmux):
#   bash scripts/pico_final_continuation_20260925.sh
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

LOG=outputs/pico_final_w_only_20260924/campaign.log
echo "waiting for the main campaign to finish ($(date -Is))"
while true; do
  if grep -q "=== DONE" "$LOG" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "pico_final_all_20260924" > /dev/null; then
    # Driver gone without DONE: still continue, the per-cell resume logic will
    # pick up whatever is missing.
    break
  fi
  sleep 60
done
echo "main campaign finished; starting repairs ($(date -Is))"

# Rebuild the shipped snapshot so the fixes (non-spatial payload guard, llava
# and wikitext dataset wiring) reach both the driver and the Ray workers.
for dir in outputs/affine_repr_profile_20260924/modules \
           outputs/pico_final_w_only_20260924/modules; do
  rm -rf "$dir"
  mkdir -p "$dir"
  cp -r cedar "$dir"/cedar
  mkdir -p "$dir/evaluation"
  find evaluation -maxdepth 1 -type f -name '*.py' -exec cp {} "$dir/evaluation/" \;
  cp -r evaluation/pipelines "$dir/evaluation/pipelines"
  find "$dir" -name '__pycache__' -type d -prune -exec rm -rf {} +
done
# Drop the failed profiles so the resume logic regenerates them.
for workload in coco llava_pretrain wikitext103; do
  rm -f "outputs/affine_repr_profile_20260924/$workload/shared.yaml"
done

echo "=== profiles ($(date -Is)) ==="
OUT=outputs/affine_repr_profile_20260924 \
  bash scripts/pico_final_profiles_20260924.sh coco llava_pretrain wikitext103

echo "=== campaign for the repaired workloads ($(date -Is)) ==="
OUT=outputs/pico_final_w_only_20260924 ROUNDS=3 \
  bash scripts/pico_final_w_only_campaign.sh coco llava_pretrain wikitext103

echo "=== assembly ($(date -Is)) ==="
python -u tmp_analysis/assemble_final_delivery.py || true
echo "=== CONTINUATION DONE ($(date -Is)) ==="
