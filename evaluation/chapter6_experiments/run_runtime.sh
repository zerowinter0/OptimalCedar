#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

runtime_dir="${CH6_OUT_DIR}/raw/runtime"
mkdir -p "${runtime_dir}"

for workload in $(ch6_workloads); do
  output_json="${runtime_dir}/${workload}_runtime.json"
  log_path="${CH6_OUT_DIR}/logs/runtime_${workload}.log"
  ch6_run_compare_optimizer_perf "${workload}" "${output_json}" "${log_path}"
done
