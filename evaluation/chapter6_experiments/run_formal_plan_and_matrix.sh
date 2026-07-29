#!/usr/bin/env bash
# Generate fresh W=8 plans and run the complete workload matrix offline.
#
# Example (inside optimalcedar-torch201-dev):
#   nohup bash evaluation/chapter6_experiments/run_w8_plan_and_matrix.sh \
#     > evaluation/chapter6_experiments/w8_matrix.nohup 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
PROFILE_DIR="${PROFILE_DIR:-${BASE_DIR}/formal_results/profiles}"
MATRIX_OUTPUT_ROOT="${MATRIX_OUTPUT_ROOT:-${BASE_DIR}}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
CPU_BUDGET="${CPU_BUDGET:-64}"
LOCAL_WORKERS="${LOCAL_WORKERS:-8}"
REPEATS="${REPEATS:-1}"
OPTIMIZER_PLAN_TIMEOUT_SEC="${OPTIMIZER_PLAN_TIMEOUT_SEC:-300}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
OPTIMIZER_SET="${OPTIMIZER_SET:-all}"
PLAN_ONLY="${PLAN_ONLY:-0}"
SELECTED_WORKLOADS="all"

usage() {
  cat <<'EOF'
Usage: run_w8_plan_and_matrix.sh [--workloads workload[,workload...]]

Default: run all workloads. Selected names: coco, commonvoice,
commonvoice_cache, llava_pretrain, redpajama_c4, simclrv2, simclrv2_cache,
wikitext103, wikitext103_cache, stackexchange.

OPTIMIZER_SET=required runs only dj_optimizer, dp_cedar_optimizer, and
dp_optimizer. OPTIMIZER_SET=paper runs the five optimizers in the current
paper figure. OPTIMIZER_SET=complete runs their union with dp_two_stage_optimizer.
The default "all" preserves the project-standard five-optimizer matrix.
OPTIMIZER_SET=dp_only regenerates only dp_optimizer.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --workloads)
      [[ "$#" -ge 2 ]] || { echo "--workloads requires a value" >&2; exit 2; }
      SELECTED_WORKLOADS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

workload_selected() {
  [[ "${SELECTED_WORKLOADS}" == "all" ]] && return 0
  case ",${SELECTED_WORKLOADS}," in
    *,"$1",*) return 0 ;;
    *) return 1 ;;
  esac
}

case "${OPTIMIZER_SET}" in
  all)
    OPTIMIZERS=(
      optimizer
      dj_optimizer
      dp_cedar_optimizer
      dp_optimizer
      dp_two_stage_optimizer
    )
    ;;
  required)
    OPTIMIZERS=(dj_optimizer dp_cedar_optimizer dp_optimizer)
    ;;
  paper)
    OPTIMIZERS=(
      optimizer
      dj_optimizer
      dp_cedar_optimizer
      dp_optimizer
      pecan_optimizer
    )
    ;;
  complete)
    OPTIMIZERS=(
      optimizer
      dj_optimizer
      dp_cedar_optimizer
      dp_optimizer
      dp_two_stage_optimizer
      pecan_optimizer
    )
    ;;
  dp_only)
    OPTIMIZERS=(dp_optimizer)
    ;;
  *)
    echo "OPTIMIZER_SET must be all, required, paper, complete, or dp_only." >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
# Project convention: all code runs in the container's virtual environment.
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
ulimit -n 65536

if [[ "${LOCAL_WORKERS}" != "8" ]]; then
  echo "This script is the fixed-W=8 ablation; LOCAL_WORKERS must be 8." >&2
  exit 2
fi
if [[ "${REPEATS}" != "1" && "${REPEATS}" != "3" ]]; then
  echo "REPEATS must be 1 (legacy reduced run) or 3 (paper run)." >&2
  exit 2
fi
if [[ "${RESUME_EXISTING}" != "0" && "${RESUME_EXISTING}" != "1" ]]; then
  echo "RESUME_EXISTING must be 0 or 1." >&2
  exit 2
fi
if [[ "${PLAN_ONLY}" != "0" && "${PLAN_ONLY}" != "1" ]]; then
  echo "PLAN_ONLY must be 0 or 1." >&2
  exit 2
fi

ray_healthy() {
  timeout 20s python - "${RAY_ADDRESS}" <<'PY' >/dev/null 2>&1
import ray
import sys

ray.init(address=sys.argv[1], logging_level="ERROR")
resources = ray.cluster_resources()
ray.shutdown()
assert resources.get("CPU", 0) >= 64, resources
PY
}

if ! ray_healthy; then
  ray stop --force >/dev/null 2>&1 || true
  ray start --head --node-ip-address=127.0.0.1 --port=6379 \
    --num-cpus="${CPU_BUDGET}" --disable-usage-stats >/dev/null
  ray_healthy || {
    echo "Failed to start a healthy local Ray cluster at ${RAY_ADDRESS}." >&2
    exit 1
  }
fi

TMP_PLAN="/tmp/cedar_optimized_plan.yml"
TMP_PLAN_BACKUP=""
if [[ -f "${TMP_PLAN}" ]]; then
  TMP_PLAN_BACKUP="$(mktemp /tmp/cedar_optimized_plan.backup.XXXXXX.yml)"
  cp -p "${TMP_PLAN}" "${TMP_PLAN_BACKUP}"
fi
cleanup_tmp_plan() {
  if [[ -n "${TMP_PLAN_BACKUP}" ]]; then
    mv -f "${TMP_PLAN_BACKUP}" "${TMP_PLAN}"
  else
    rm -f "${TMP_PLAN}"
  fi
}
trap cleanup_tmp_plan EXIT

reset_workload_dir() {
  local workload="$1"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"

  # These are generated artifacts only. Profiles and source files are kept.
  rm -rf "${root}/plans" "${root}/results" "${root}/warmup_results" \
    "${root}/logs" "${root}/cache"
  rm -f "${root}/nohup.log" "${root}/metadata.txt"
  mkdir -p "${root}/plans" "${root}/results" "${root}/warmup_results" \
    "${root}/logs"
}

write_metadata() {
  local workload="$1" profile="$2" samples="$3" cache_mode="$4" kwargs="$5"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  {
    printf 'ablation=fixed_local_workers_w8\n'
    printf 'measurement_protocol=fixed_w8_%s_round_robin_repeats\n' "${REPEATS}"
    printf 'profile_source=%s\n' "${profile#${REPO_ROOT}/}"
    printf 'cpu_budget=%s\n' "${CPU_BUDGET}"
    printf 'local_workers=%s\n' "${LOCAL_WORKERS}"
    printf 'repeats=%s\n' "${REPEATS}"
    printf 'optimizer_plan_timeout_sec=%s\n' "${OPTIMIZER_PLAN_TIMEOUT_SEC}"
    printf 'cache=%s\n' "${cache_mode}"
    printf 'optimizers=%s\n' "${OPTIMIZERS[*]}"
    printf 'dataset_kwargs=%s\n' "${kwargs}"
    if [[ "${workload}" == "llava_pretrain" || "${workload}" == "redpajama_c4" || "${workload}" == "stackexchange" ]]; then
      printf 'data_juicer_reference_commit=%s\n' \
        "$(git -C data-juicer -c safe.directory="${REPO_ROOT}/data-juicer" rev-parse HEAD)"
    fi
    if [[ "${workload}" == "llava_pretrain" ]]; then
      printf 'scale_note=caption_only_20000_source_for_5000_outputs\n'
    fi
    if [[ "${workload}" == "redpajama_c4" ]]; then
      printf 'scale_note=bounded_raw_c4_1975058466B_829916_source_for_20000_outputs\n'
    fi
    if [[ "${workload}" == "stackexchange" ]]; then
      printf 'scale_note=bounded_unique_redpajama_stackexchange_35000_source_for_10000_outputs\n'
      printf 'omitted_operator=document_simhash_deduplicator\n'
    fi
    if [[ "${workload}" == "wikitext103" || "${workload}" == "wikitext103_cache" ]]; then
      printf 'scale_note=reduced_protocol_100000_line_prefix\n'
    fi
  } > "${root}/metadata.txt"
}

generate_plan() {
  local workload="$1" dataset="$2" profile="$3" samples="$4"
  local cache_mode="$5" kwargs="$6" optimizer="$7"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  local log="${root}/logs/plan__${optimizer}.log"
  local result="${root}/warmup_results/plan_only__${optimizer}.json"
  local -a args=(
    python evaluation/compare_optimizer_perf.py
    --dataset_file "${dataset}"
    --profiled_stats "${profile}"
    --num_total_samples "${samples}"
    --num_epochs 1
    --num_repeats 1
    --warmup_runs 0
    --full_data_run
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --enable_local_parallelism
    --match_profile_resources
    --cpu_budget "${CPU_BUDGET}"
    --fixed_local_workers_ablation "${LOCAL_WORKERS}"
    --disable_cedar_runtime_timeout
    --plan_only
    --optimizers "${optimizer}"
    --results_path "${result}"
  )
  [[ "${cache_mode}" == "off" ]] && args+=(--disable_caching)
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")

  echo "[$(date -Is)] PLAN ${workload}/${optimizer}" | tee -a "${root}/nohup.log"
  printf 'command:' > "${log}"
  printf ' %q' "${args[@]}" >> "${log}"
  printf '\n' >> "${log}"
  # Do not copy a stale plan if a previous optimizer timed out.
  rm -f "${TMP_PLAN}"
  set +e
  timeout --signal=TERM --kill-after=30s "${OPTIMIZER_PLAN_TIMEOUT_SEC}" \
    "${args[@]}" >> "${log}" 2>&1
  local status=$?
  set -e
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    rm -f "${TMP_PLAN}"
    printf '{\n  "optimizer": "%s",\n  "status": "unavailable",\n' \
      "${optimizer}" > "${root}/plans/${optimizer}.unavailable.json"
    printf '  "reason": "plan_generation_timeout",\n  "timeout_sec": %s\n}\n' \
      "${OPTIMIZER_PLAN_TIMEOUT_SEC}" \
      >> "${root}/plans/${optimizer}.unavailable.json"
    echo "[$(date -Is)] UNAVAILABLE ${workload}/${optimizer}: plan generation exceeded ${OPTIMIZER_PLAN_TIMEOUT_SEC}s" \
      | tee -a "${root}/nohup.log"
    return 0
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Plan generation failed for ${workload}/${optimizer}; see ${log}" >&2
    return "${status}"
  fi
  [[ -f "${TMP_PLAN}" ]] || {
    echo "Plan generation produced no ${TMP_PLAN}" >&2
    return 1
  }
  cp -p "${TMP_PLAN}" "${root}/plans/${optimizer}.yaml"
}

warm_cache() {
  local workload="$1" dataset="$2" samples="$3" kwargs="$4" optimizer="$5"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  local warmup_request=$((samples + 1))
  local -a args=(
    python evaluation/eval_cedar.py
    --dataset_file "${dataset}"
    --master_feature_config "${root}/plans/${optimizer}.yaml"
    # Request one item past the bounded source so the cache writer observes
    # natural exhaustion and atomically commits its manifest.
    --num_total_samples "${warmup_request}"
    --num_epochs 1
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --results_path "${root}/warmup_results/warmup__${optimizer}.json"

  )
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")

  export CEDAR_CACHE_ROOT="${root}/cache"
  export CEDAR_CACHE_NAMESPACE="${workload}__${optimizer}"
  unset CEDAR_CACHE_SHARD
  echo "[$(date -Is)] WARMUP ${workload}/${optimizer}" | tee -a "${root}/nohup.log"
  "${args[@]}" > "${root}/logs/warmup__${optimizer}.log" 2>&1

  # A bounded Cedar iteration can exit successfully without exhausting the
  # source.  Such a run leaves no complete cache manifest and must never be
  # accepted as a formal cache warmup.
  python - "${root}/cache/${workload}__${optimizer}" "${samples}" <<'PY'
import json
import pathlib
import sys

cache_namespace = pathlib.Path(sys.argv[1])
expected_items = int(sys.argv[2])
manifests = sorted(cache_namespace.glob("**/.manifest.json"))
if not manifests:
    raise SystemExit(f"cache warmup produced no manifests: {cache_namespace}")

total_items = 0
for path in manifests:
    with path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not True:
        raise SystemExit(f"incomplete cache manifest: {path}")
    total_items += int(manifest.get("num_items", -1))

if total_items != expected_items:
    raise SystemExit(
        f"cache item mismatch for {cache_namespace}: "
        f"expected {expected_items}, found {total_items}"
    )
PY
}

run_workload() {
  local workload="$1" dataset="$2" profile="$3" samples="$4"
  local cache_mode="$5" kwargs="$6"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  local optimizer round offset i tag
  local -a available_optimizers=()

  if ! workload_selected "${workload}"; then
    echo "[$(date -Is)] SKIP ${workload}: not selected by --workloads"
    return 0
  fi

  [[ -f "${profile}" ]] || {
    echo "Missing profile for ${workload}: ${profile}" >&2
    return 1
  }
  if [[ "${RESUME_EXISTING}" == "0" ]]; then
    reset_workload_dir "${workload}"
  else
    mkdir -p "${root}/plans" "${root}/results" \
      "${root}/warmup_results" "${root}/logs"
  fi
  write_metadata "${workload}" "${profile}" "${samples}" "${cache_mode}" "${kwargs}"

  for optimizer in "${OPTIMIZERS[@]}"; do
    if [[ "${RESUME_EXISTING}" == "1" ]] && \
       { [[ -f "${root}/plans/${optimizer}.yaml" ]] || \
         [[ -f "${root}/plans/${optimizer}.unavailable.json" ]]; }; then
      echo "[$(date -Is)] REUSE ${workload}/${optimizer} plan status" \
        | tee -a "${root}/nohup.log"
      continue
    fi
    generate_plan "${workload}" "${dataset}" "${profile}" "${samples}" \
      "${cache_mode}" "${kwargs}" "${optimizer}"
  done

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    echo "[$(date -Is)] PLAN-ONLY COMPLETE ${workload}" \
      | tee -a "${root}/nohup.log"
    return 0
  fi

  if [[ "${cache_mode}" == "on" ]]; then
    for optimizer in "${OPTIMIZERS[@]}"; do
      [[ -f "${root}/plans/${optimizer}.yaml" ]] || continue
      if [[ "${RESUME_EXISTING}" == "1" && \
            -f "${root}/results/round1__${optimizer}.json" ]]; then
        echo "[$(date -Is)] REUSE ${workload}/${optimizer} cache warmup" \
          | tee -a "${root}/nohup.log"
        continue
      fi
      warm_cache "${workload}" "${dataset}" "${samples}" "${kwargs}" "${optimizer}"
    done
  else
    unset CEDAR_CACHE_ROOT CEDAR_CACHE_NAMESPACE CEDAR_CACHE_SHARD
  fi

  for ((round = 1; round <= REPEATS; round++)); do
    offset=$(((round - 1) % ${#OPTIMIZERS[@]}))
    for ((i = 0; i < ${#OPTIMIZERS[@]}; i++)); do
      optimizer="${OPTIMIZERS[$(((offset + i) % ${#OPTIMIZERS[@]}))]}"
      [[ -f "${root}/plans/${optimizer}.yaml" ]] || continue
      tag="round${round}__${optimizer}"
      if [[ "${RESUME_EXISTING}" == "1" && \
            -f "${root}/results/${tag}.json" ]]; then
        echo "[$(date -Is)] REUSE ${workload}/${tag} result" \
          | tee -a "${root}/nohup.log"
        continue
      fi
      local -a args=(
        python evaluation/eval_cedar.py
        --dataset_file "${dataset}"
        --master_feature_config "${root}/plans/${optimizer}.yaml"
        --num_total_samples "${samples}"
        --num_epochs 1
        --use_ray
        --ray_ip "${RAY_ADDRESS}"
        --results_path "${root}/results/${tag}.json"
      )
      [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
      if [[ "${cache_mode}" == "on" ]]; then
        export CEDAR_CACHE_ROOT="${root}/cache"
        export CEDAR_CACHE_NAMESPACE="${workload}__${optimizer}"
        unset CEDAR_CACHE_SHARD
      fi
      echo "[$(date -Is)] RUN ${workload}/${tag}" | tee -a "${root}/nohup.log"
      "${args[@]}" > "${root}/logs/${tag}.log" 2>&1
      echo "[$(date -Is)] DONE ${workload}/${tag}" | tee -a "${root}/nohup.log"
    done
  done
  echo "[$(date -Is)] COMPLETE ${workload}" | tee -a "${root}/nohup.log"
  if [[ "${MATRIX_OUTPUT_ROOT}" == "${BASE_DIR}" ]]; then
    python evaluation/chapter6_experiments/analyze_w8_acceptance.py \
      --json-output "${BASE_DIR}/formal_results/w8_acceptance_latest.json" \
      --markdown-output "${BASE_DIR}/formal_results/w8_acceptance_latest.md" \
      | tee -a "${root}/nohup.log"
  fi
}

run_workload coco \
  evaluation/pipelines/coco/cedar_dataset.py \
  "${PROFILE_DIR}/coco.yaml" 5000 off "split=val2017"
run_workload commonvoice \
  evaluation/pipelines/commonvoice/cedar_dataset.py \
  "${PROFILE_DIR}/commonvoice.yaml" 10000 off "max_samples=10000"
run_workload commonvoice_cache \
  evaluation/pipelines/commonvoice/cedar_cache_dataset.py \
  "${PROFILE_DIR}/commonvoice_cache.yaml" 10000 on "max_samples=10000"
run_workload llava_pretrain \
  evaluation/pipelines/llava_pretrain/cedar_dataset.py \
  "${PROFILE_DIR}/llava_pretrain.yaml" 5000 off \
  "dataset_path=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl,image_root=evaluation/datasets/llava_pretrain"
run_workload redpajama_c4 \
  evaluation/pipelines/redpajama_c4/cedar_dataset.py \
  "${PROFILE_DIR}/redpajama_c4.yaml" 20000 off \
  "dataset_path=datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl"
run_workload stackexchange \
  evaluation/pipelines/stackexchange/cedar_dataset.py \
  "${PROFILE_DIR}/stackexchange.yaml" 10000 off \
  "dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl"
run_workload simclrv2 \
  evaluation/pipelines/simclrv2/cedar_dataset.py \
  "${PROFILE_DIR}/simclrv2.yaml" 9469 off ""
run_workload simclrv2_cache \
  evaluation/pipelines/simclrv2/cedar_cache_dataset.py \
  "${PROFILE_DIR}/simclrv2_cache.yaml" 9469 on ""
run_workload wikitext103 \
  evaluation/pipelines/wikitext103/cedar_dataset.py \
  "${PROFILE_DIR}/wikitext103.yaml" 100000 off "max_samples=100000"
run_workload wikitext103_cache \
  evaluation/pipelines/wikitext103/cedar_cache_dataset.py \
  "${PROFILE_DIR}/wikitext103_cache.yaml" 100000 on "max_samples=100000"

echo "[$(date -Is)] W=8 matrix complete"
