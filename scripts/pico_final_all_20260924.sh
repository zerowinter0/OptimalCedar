#!/usr/bin/env bash
# End-to-end driver for the final W-only PICO delivery.
#
#   profiles -> smoke gate -> main campaign -> W scaling -> assembly
#
# Every stage is resumable: an existing profile/result cell is skipped, and the
# assembly step can be re-run at any time (it rebuilds the CSV/JSON set from the
# raw results that exist).
#
# Usage (inside the container, normally under tmux):
#   bash scripts/pico_final_all_20260924.sh
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

PROFILE_DIR=${PROFILE_DIR:-outputs/affine_repr_profile_20260924}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
WORKLOADS=${WORKLOADS:-"simclrv2 simclrv2_cache commonvoice coco llava_pretrain wikitext103"}
ROUNDS=${ROUNDS:-3}

echo "=== stage 0: smoke gate ($(date -Is)) ==="
OUT=outputs/pico_final_smoke_20260924 ROUNDS=1 SIMCLRV2_EPOCHS=1 \
  SIMCLRV2_SAMPLES=200 PLAN_TIMEOUT=900 \
  bash scripts/pico_final_w_only_campaign.sh simclrv2 2>&1 | tail -20
if [ ! -s outputs/pico_final_smoke_20260924/simclrv2/results/main_mean.json ]; then
  echo "SMOKE GATE FAILED; stopping before the long campaign"
  exit 1
fi
echo "=== smoke gate passed ==="

echo "=== stage 1: profiles ($(date -Is)) ==="
OUT="$PROFILE_DIR" bash scripts/pico_final_profiles_20260924.sh $WORKLOADS

echo "=== stage 2: main campaign ($(date -Is)) ==="
PROFILE_DIR="$PROFILE_DIR" OUT="$OUT" ROUNDS="$ROUNDS" \
  bash scripts/pico_final_w_only_campaign.sh $WORKLOADS

echo "=== stage 3: W scaling on SimCLRv2 ($(date -Is)) ==="
plan=$(ls -t "$OUT"/simclrv2/plans/*.yaml 2>/dev/null | head -1 || true)
if [ -n "${plan:-}" ]; then
  OUT="$OUT" bash scripts/pico_final_w_scaling_20260924.sh simclrv2 "$plan" || true
fi

echo "=== stage 4: assembly ($(date -Is)) ==="
python -u tmp_analysis/assemble_final_delivery.py || true
python -u tmp_analysis/build_affine_repr_deliverable.py || true

echo "=== DONE ($(date -Is)) ==="
