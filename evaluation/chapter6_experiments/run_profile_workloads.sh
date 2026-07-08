#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

for workload in $(ch6_workloads); do
  profile_path="$(ch6_workload_profile "${workload}")"
  log_path="${CH6_OUT_DIR}/logs/profile_${workload}.log"
  if [[ -f "${profile_path}" && "${CH6_FORCE_PROFILE:-0}" != "1" ]]; then
    ch6_log "Profile exists for ${workload}, skipping: ${profile_path}"
    continue
  fi
  ch6_log "Profiling ${workload} -> ${profile_path}"
  ch6_profile_workload "${workload}" "${profile_path}" "${CH6_PROFILE_SAMPLES}" "${log_path}"
done
