#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

: "${CH6_OVERHEAD_MIN_OPS:=0}"
: "${CH6_OVERHEAD_MAX_OPS:=10}"
: "${CH6_REORDER_TIMEOUT_SEC:=300}"
: "${CH6_OVERHEAD_OPTIMIZERS:=cedar dp_cedar_optimizer dp_two_stage_optimizer dp_optimizer}"

overhead_dir="${CH6_OUT_DIR}/raw/optimizer_overhead"
mkdir -p "${overhead_dir}"

for optimizer in ${CH6_OVERHEAD_OPTIMIZERS}; do
  csv_path="${overhead_dir}/${optimizer}_synthetic_reorder.csv"
  figure_path="${overhead_dir}/${optimizer}_synthetic_reorder.svg"
  log_path="${CH6_OUT_DIR}/logs/overhead_synthetic_${optimizer}.log"
  ch6_run_and_log "${log_path}" \
    python evaluation/benchmark_cedar_reorder_time.py \
      --optimizer "${optimizer}" \
      --min-ops "${CH6_OVERHEAD_MIN_OPS}" \
      --max-ops "${CH6_OVERHEAD_MAX_OPS}" \
      --repeats "${CH6_REPEATS}" \
      --timeout-sec "${CH6_REORDER_TIMEOUT_SEC}" \
      --csv "${csv_path}" \
      --figure "${figure_path}"
done

for workload in $(ch6_workloads); do
  output_json="${overhead_dir}/${workload}_plan_only.json"
  log_path="${CH6_OUT_DIR}/logs/overhead_real_${workload}.log"
  ch6_run_compare_optimizer_perf "${workload}" "${output_json}" "${log_path}" --plan_only
done
