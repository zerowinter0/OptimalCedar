#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

plan_dir="${CH6_OUT_DIR}/raw/plan_quality"
mkdir -p "${plan_dir}"

for workload in $(ch6_workloads); do
  output_json="${plan_dir}/${workload}_plan_cost.json"
  log_path="${CH6_OUT_DIR}/logs/plan_quality_${workload}.log"
  ch6_run_compare_optimizer_perf "${workload}" "${output_json}" "${log_path}" \
    --calculate_plan_cost
done
