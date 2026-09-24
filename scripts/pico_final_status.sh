#!/usr/bin/env bash
# Progress and ETA for the final W-only campaign.
#
# Usage: bash scripts/pico_final_status.sh   (host or container)
set -uo pipefail
ROOT=${ROOT:-/home/xieruiyang/OptimalCedar}
OUT="$ROOT/outputs/pico_final_w_only_20260924"

echo "== cells finished (results/*.json) =="
find "$OUT" -path "*/results/*.json" ! -name "*.failed.json" 2>/dev/null | sort |
  sed "s#$OUT/##"
echo "== failed cells =="
find "$OUT" -name "*.failed.json" 2>/dev/null | sed "s#$OUT/##"
echo "== running cell =="
tail -3 "$OUT/campaign.log" 2>/dev/null
echo "== per-optimizer progress in the running log =="
log=$(ls -t "$OUT"/*/logs/*.log 2>/dev/null | head -1)
if [ -n "${log:-}" ]; then
  echo "log: ${log}"
  grep -o "Starting [a-z_]* repeat [0-9]/[0-9]" "$log" | tail -4
  tail -1 "$log" | cut -c1-120
fi
echo "== profiles =="
for profile in "$ROOT"/outputs/affine_repr_profile_20260924/*/shared.yaml; do
  [ -f "$profile" ] && echo "  ok  $(basename "$(dirname "$profile")")"
done
