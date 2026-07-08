#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

sensitivity_dir="${CH6_OUT_DIR}/raw/profile_sensitivity"
mkdir -p "${sensitivity_dir}"

for workload in $(ch6_workloads); do
  for sample_count in ${CH6_PROFILE_SAMPLE_SWEEP:-20 50 100 200}; do
    profile_path="${CH6_OUT_DIR}/profiles/${workload}_profile_n${sample_count}.yml"
    profile_log="${CH6_OUT_DIR}/logs/profile_sensitivity_${workload}_n${sample_count}_profile.log"
    quality_log="${CH6_OUT_DIR}/logs/profile_sensitivity_${workload}_n${sample_count}_quality.log"
    output_json="${sensitivity_dir}/${workload}_n${sample_count}_plan_cost.json"

    ch6_log "Profile sensitivity: ${workload}, n=${sample_count}"
    ch6_profile_workload "${workload}" "${profile_path}" "${sample_count}" "${profile_log}"

    old_profile_override=""
    case "${workload}" in
      bloom_oscar)
        old_profile_override="${BLOOM_PROFILE_PATH:-}"
        BLOOM_PROFILE_PATH="${profile_path}"
        ;;
      llava_pretrain)
        old_profile_override="${LLAVA_PROFILE_PATH:-}"
        LLAVA_PROFILE_PATH="${profile_path}"
        ;;
      wikitext103)
        old_profile_override="${WIKITEXT_PROFILE_PATH:-}"
        WIKITEXT_PROFILE_PATH="${profile_path}"
        ;;
      simclrv2)
        old_profile_override="${SIMCLR_PROFILE_PATH:-}"
        SIMCLR_PROFILE_PATH="${profile_path}"
        ;;
      commonvoice)
        old_profile_override="${COMMONVOICE_PROFILE_PATH:-}"
        COMMONVOICE_PROFILE_PATH="${profile_path}"
        ;;
      coco)
        old_profile_override="${COCO_PROFILE_PATH:-}"
        COCO_PROFILE_PATH="${profile_path}"
        ;;
    esac

    ch6_run_compare_optimizer_perf "${workload}" "${output_json}" "${quality_log}" \
      --calculate_plan_cost

    case "${workload}" in
      bloom_oscar)
        BLOOM_PROFILE_PATH="${old_profile_override}"
        ;;
      llava_pretrain)
        LLAVA_PROFILE_PATH="${old_profile_override}"
        ;;
      wikitext103)
        WIKITEXT_PROFILE_PATH="${old_profile_override}"
        ;;
      simclrv2)
        SIMCLR_PROFILE_PATH="${old_profile_override}"
        ;;
      commonvoice)
        COMMONVOICE_PROFILE_PATH="${old_profile_override}"
        ;;
      coco)
        COCO_PROFILE_PATH="${old_profile_override}"
        ;;
    esac
  done
done
