#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

ch6_log "Chapter 6 result directory: ${CH6_OUT_DIR}"
ch6_log "Workloads: ${CH6_WORKLOADS}"

bash "${SCRIPT_DIR}/run_profile_workloads.sh"
bash "${SCRIPT_DIR}/run_runtime.sh"
bash "${SCRIPT_DIR}/run_optimizer_overhead.sh"
bash "${SCRIPT_DIR}/run_plan_quality.sh"
bash "${SCRIPT_DIR}/run_ablation.sh"
bash "${SCRIPT_DIR}/run_cache_epoch.sh"

if [[ "${CH6_RUN_PROFILE_SENSITIVITY:-0}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_profile_sensitivity.sh"
fi

python "${SCRIPT_DIR}/summarize_results.py" \
  --results_dir "${CH6_OUT_DIR}" \
  --output_dir "${CH6_OUT_DIR}/summary"

ch6_log "Done. Summary: ${CH6_OUT_DIR}/summary/chapter6_summary.md"
