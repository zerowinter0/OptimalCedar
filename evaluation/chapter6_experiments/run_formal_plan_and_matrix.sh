#!/usr/bin/env bash
# Generate fresh W=8 plans and run the complete workload matrix offline.
#
# Example (inside optimalcedar-torch201-dev):
#   nohup bash evaluation/chapter6_experiments/run_w8_plan_and_matrix.sh \
#     > evaluation/chapter6_experiments/w8_matrix.nohup 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
PROFILE_DIR="${PROFILE_DIR:-${BASE_DIR}/formal_results/paper_artifacts/optimizer/profiles}"
MATRIX_OUTPUT_ROOT="${MATRIX_OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/optimizer_matrix}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
CPU_BUDGET="${CPU_BUDGET:-64}"
LOCAL_WORKERS="${LOCAL_WORKERS:-8}"
REPEATS="${REPEATS:-1}"
TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-3600}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
OPTIMIZER_SET="${OPTIMIZER_SET:-all}"
PLAN_ONLY="${PLAN_ONLY:-0}"
COCO_SAMPLES="${COCO_SAMPLES:-5000}"
COCO_DATASET_KWARGS="${COCO_DATASET_KWARGS:-split=val2017}"
ALPACA_COT_SAMPLES="${ALPACA_COT_SAMPLES:-20000}"
COMMONVOICE_SAMPLES="${COMMONVOICE_SAMPLES:-240000}"
COMMONVOICE_DATASET_PATHS="${COMMONVOICE_DATASET_PATHS:-datasets/commonvoice/cv15_en_train_5shards;evaluation/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips}"
REDPAJAMA_ARXIV_SAMPLES="${REDPAJAMA_ARXIV_SAMPLES:-20000}"
GENERAL_VIDEO_REFINE_SAMPLES="${GENERAL_VIDEO_REFINE_SAMPLES:-5000}"
VIDEO_SELF_EVOLUTION_SAMPLES="${VIDEO_SELF_EVOLUTION_SAMPLES:-5000}"
PILE_EUROPARL_SAMPLES="${PILE_EUROPARL_SAMPLES:-10000}"
PILE_HACKERNEWS_SAMPLES="${PILE_HACKERNEWS_SAMPLES:-40000}"
PILE_PUBMED_SAMPLES="${PILE_PUBMED_SAMPLES:-85000}"
PILE_USPTO_SAMPLES="${PILE_USPTO_SAMPLES:-50000}"
REDPAJAMA_CODE_SAMPLES="${REDPAJAMA_CODE_SAMPLES:-60000}"
REDPAJAMA_CODE_DATASET_PATH="${REDPAJAMA_CODE_DATASET_PATH:-datasets/redpajama_code/redpajama-github-raw-100000.jsonl}"
STACKEXCHANGE_SAMPLES="${STACKEXCHANGE_SAMPLES:-160000}"
STACKEXCHANGE_DATASET_PATH="${STACKEXCHANGE_DATASET_PATH:-datasets/stackexchange/redpajama-stackexchange-400000.jsonl}"
SELECTED_WORKLOADS="all"

usage() {
  cat <<'EOF'
Usage: run_w8_plan_and_matrix.sh [--workloads workload[,workload...]]

Default: run all workloads. Selected names: alpaca_cot, redpajama_arxiv, coco, commonvoice,
commonvoice_cache, llava_pretrain, redpajama_c4, simclrv2, simclrv2_cache,
wikitext103, wikitext103_cache, stackexchange, general_video_refine,
video_self_evolution, pile_europarl, pile_hackernews,
pile_pubmed_abstracts, pile_uspto_backgrounds, redpajama_code.

OPTIMIZER_SET=required runs only dj_optimizer, dp_cedar_optimizer, and
dp_optimizer. OPTIMIZER_SET=paper runs the five optimizers in the current
paper figure. OPTIMIZER_SET=complete runs their union with dp_two_stage_optimizer.
The default "all" is the project-standard six-optimizer matrix.
OPTIMIZER_SET=dp_only regenerates only dp_optimizer.
OPTIMIZER_SET=dp_two_stage_only runs only dp_two_stage_optimizer.
OPTIMIZER_SET=dj_two_stage_only runs only dj_two_stage_optimizer.
OPTIMIZER_SET=simple_dp_only runs the joint DP with Cedar's original profile
and cost model.
OPTIMIZER_SET=pico_simple_gate compares PICO and Simple-DP round-robin.
OPTIMIZER_SET=legacy_and_two_stage runs original Cedar and the DP two-stage
ablation for supplementing older formal matrices.
OPTIMIZER_SET=operator_scaling_study runs the revised DP optimizer plus
pecan_two_stage_optimizer and dj_two_stage_optimizer.
OPTIMIZER_SET=formal_seven runs Cedar, DJ, Pecan, both policy two-stage
baselines, Simple DP, and the revised DP optimizer.
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
      pecan_optimizer
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
  dp_two_stage_only)
    OPTIMIZERS=(dp_two_stage_optimizer)
    ;;
  dj_two_stage_only)
    OPTIMIZERS=(dj_two_stage_optimizer)
    ;;
  simple_dp_only)
    OPTIMIZERS=(simple_dp_optimizer)
    ;;
  pico_simple_gate)
    OPTIMIZERS=(simple_dp_optimizer dp_optimizer)
    ;;
  legacy_and_two_stage)
    OPTIMIZERS=(optimizer dp_two_stage_optimizer)
    ;;
  new_two_stage_only)
    OPTIMIZERS=(pecan_two_stage_optimizer dj_two_stage_optimizer)
    ;;
  operator_scaling_study)
    OPTIMIZERS=(dp_optimizer pecan_two_stage_optimizer dj_two_stage_optimizer)
    ;;
  formal_seven)
    OPTIMIZERS=(
      optimizer
      dj_optimizer
      pecan_optimizer
      dj_two_stage_optimizer
      pecan_two_stage_optimizer
      simple_dp_optimizer
      dp_optimizer
    )
    ;;
  *)
    echo "Unsupported OPTIMIZER_SET=${OPTIMIZER_SET}." >&2
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
if [[ ! "${TASK_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TASK_TIMEOUT_SEC must be a positive integer: ${TASK_TIMEOUT_SEC}" >&2
  exit 2
fi
for sample_setting in COCO_SAMPLES ALPACA_COT_SAMPLES COMMONVOICE_SAMPLES REDPAJAMA_ARXIV_SAMPLES GENERAL_VIDEO_REFINE_SAMPLES VIDEO_SELF_EVOLUTION_SAMPLES PILE_EUROPARL_SAMPLES PILE_HACKERNEWS_SAMPLES PILE_PUBMED_SAMPLES PILE_USPTO_SAMPLES REDPAJAMA_CODE_SAMPLES STACKEXCHANGE_SAMPLES; do
  sample_value="${!sample_setting}"
  if [[ ! "${sample_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${sample_setting} must be a positive integer: ${sample_value}" >&2
    exit 2
  fi
done

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
    printf 'dp_runtime_cpu_reserve_per_worker=%s\n' \
      "${CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER:-1}"
    printf 'local_workers=%s\n' "${LOCAL_WORKERS}"
    printf 'repeats=%s\n' "${REPEATS}"
    printf 'task_timeout_sec=%s\n' "${TASK_TIMEOUT_SEC}"
    printf 'task_boundary=optimization_plus_first_execution\n'
    printf 'timeout_policy=skip_remaining_repeats_and_continue_matrix\n'
    printf 'cache=%s\n' "${cache_mode}"
    printf 'samples=%s\n' "${samples}"
    printf 'optimizers=%s\n' "${OPTIMIZERS[*]}"
    if [[ " ${OPTIMIZERS[*]} " == *" dp_optimizer "* || " ${OPTIMIZERS[*]} " == *" pecan_two_stage_optimizer "* || " ${OPTIMIZERS[*]} " == *" dj_two_stage_optimizer "* ]]; then
      printf 'dp_objective=multi_bottleneck\n'
      printf 'dp_pareto_global_epsilon=%s\n' "${CEDAR_DP_PARETO_EPSILON:-0.10}"
      printf 'dp_gpu_objective=single_shared_gpu_serial_service_demand\n'
      printf 'ray_gpu_accounting=fractional_sum_exactly_one_gpu\n'
    fi
    printf 'dataset_kwargs=%s\n' "${kwargs}"
    if [[ "${workload}" == "alpaca_cot" || "${workload}" == "redpajama_arxiv" || "${workload}" == "video_self_evolution" || "${workload}" == "llava_pretrain" || "${workload}" == "redpajama_c4" || "${workload}" == "stackexchange" ]]; then
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
      printf 'scale_note=bounded_unique_redpajama_stackexchange_400000_source_for_%s_outputs\n' \
        "${STACKEXCHANGE_SAMPLES}"
      printf 'omitted_operator=document_simhash_deduplicator\n'
    fi
    if [[ "${workload}" == "alpaca_cot" ]]; then
      printf 'data_source=QingyiSi/Alpaca-CoT@18add89e3b884703ec869a5c6e2bcf1412ee7edc/Chain-of-Thought/CoT_data.json\n'
      printf 'operator_count=8\n'
      printf 'scenario=instruction_reasoning\n'
      printf 'omitted_operators=document_deduplicator,document_simhash_deduplicator\n'
    fi
    if [[ "${workload}" == "redpajama_arxiv" ]]; then
      printf 'data_source=togethercomputer/RedPajama-Data-1T@398f92572e94f4793e41c22ab7ea2a788d9e7de4/arxiv_source_order_3gib_prefix\n'
      printf 'operator_count=18\n'
      printf 'scenario=scientific_latex_document_cleaning\n'
      printf 'omitted_operator=document_simhash_deduplicator\n'
    fi
    if [[ "${workload}" == "general_video_refine" ]]; then
      printf 'data_source=nisav/MSR-VTT@a9c822473969ee469e224da2187fda193c62e960\n'
      printf 'data_juicer_recipe=datajuicer/data-juicer-hub@47fc34588b5d4258c13747cea37c2b63cf4e11b0/refined_recipes/video/general-video-refine-example.yaml\n'
      printf 'operator_count=7\n'
      printf 'scenario=general_video_text_quality_refinement\n'
      printf 'gpu=single_RTX_A6000_shared_by_all_optimizers\n'
      printf 'sample_construction=caption_round_then_numeric_video_id\n'
      printf 'scale_note=reduced_protocol_5000_outputs_to_fit_unified_one_hour_budget\n'
    fi
    if [[ "${workload}" == "video_self_evolution" ]]; then
      printf 'data_source=nisav/MSR-VTT@a9c822473969ee469e224da2187fda193c62e960\n'
      printf 'data_juicer_hub_recipe=refined_recipes/video/data-juicer-sandbox-self-evolution.yaml\n'
      printf 'data_juicer_hub_commit=47fc34588b5d4258c13747cea37c2b63cf4e11b0\n'
      printf 'cedar_operator_count=8\n'
      printf 'official_filter_count=5\n'
      printf 'scenario=video_text_self_evolution_filtering\n'
      printf 'gpu=single_RTX_A6000_shared_by_all_optimizers\n'
      printf 'sample_construction=caption_round_then_numeric_video_id\n'
    fi
    if [[ "${workload}" == "wikitext103" || "${workload}" == "wikitext103_cache" ]]; then
      printf 'scale_note=reduced_protocol_100000_line_prefix\n'
    fi
  } > "${root}/metadata.txt"
}

generate_plan() {
  local workload="$1" dataset="$2" profile="$3" samples="$4"
  local cache_mode="$5" kwargs="$6" optimizer="$7" timeout_sec="$8"
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
    --optimizer_time_limit_sec "${timeout_sec}"
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
  timeout --signal=TERM --kill-after=30s "${timeout_sec}" \
    "${args[@]}" >> "${log}" 2>&1
  local status=$?
  set -e
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    rm -f "${TMP_PLAN}"
    printf '{\n  "optimizer": "%s",\n  "status": "unavailable",\n' \
      "${optimizer}" > "${root}/plans/${optimizer}.unavailable.json"
    printf '  "reason": "plan_generation_timeout",\n  "timeout_sec": %s\n}\n' \
      "${timeout_sec}" \
      >> "${root}/plans/${optimizer}.unavailable.json"
    echo "[$(date -Is)] UNAVAILABLE ${workload}/${optimizer}: plan generation exhausted ${timeout_sec}s task budget" \
      | tee -a "${root}/nohup.log"
    return 0
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "Plan generation failed for ${workload}/${optimizer}; see ${log}" >&2
    return "${status}"
  fi
  [[ -f "${TMP_PLAN}" ]] || {
    # compare_optimizer_perf may enforce its own optimizer limit and return
    # successfully with a structured unavailable result but no plan. Record
    # that optimizer as unavailable and continue the matrix.
    if [[ -s "${result}" ]] && python - "${result}" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
records = payload if isinstance(payload, list) else [payload]
if not any(
    isinstance(record, dict)
    and (
        record.get("workload_skipped") is True
        or
        record.get("status") in {"unavailable", "skipped"}
        or record.get("skip_reason")
        or any(
            isinstance(run, dict) and run.get("skip_reason")
            for run in record.get("runs", [])
        )
    )
    for record in records
):
    raise SystemExit(1)
PY
    then
      python - "${optimizer}" "${result}" \
        "${root}/plans/${optimizer}.unavailable.json" <<'PY'
import json
import sys

optimizer, result_path, output_path = sys.argv[1:]
with open(result_path) as handle:
    payload = json.load(handle)
with open(output_path, "w") as handle:
    json.dump(
        {
            "optimizer": optimizer,
            "status": "unavailable",
            "reason": "optimizer_reported_unavailable",
            "plan_only_result": payload,
        },
        handle,
        indent=2,
    )
    handle.write("\n")
PY
      echo "[$(date -Is)] UNAVAILABLE ${workload}/${optimizer}: optimizer returned no plan" \
        | tee -a "${root}/nohup.log"
      return 0
    fi
    echo "Plan generation produced no ${TMP_PLAN}" >&2
    return 1
  }
  cp -p "${TMP_PLAN}" "${root}/plans/${optimizer}.yaml"
}

run_guarded_result() {
  local timeout_sec="$1" result="$2" log="$3"
  shift 3
  local pid status=0 start_time="${SECONDS}"

  rm -f "${result}"
  setsid "$@" > "${log}" 2>&1 &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    if [[ -s "${result}" ]]; then
      # eval_cedar writes its result only after the measured epoch. Allow
      # normal teardown briefly, then reclaim the complete process group so
      # orphaned Ray/SMP workers cannot stall the remaining matrix.
      sleep 10
      kill -TERM -- "-${pid}" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    if ((SECONDS - start_time >= timeout_sec)); then
      kill -TERM -- "-${pid}" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 124
    fi
    sleep 5
  done
  wait "${pid}" || status=$?
  [[ "${status}" -eq 0 && -s "${result}" ]]
}

validate_result() {
  local result="$1" expected_samples="$2"
  python - "${result}" "${expected_samples}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
with open(path) as handle:
    result = json.load(handle)

if result.get("num_epochs") != 1:
    raise SystemExit(f"unexpected num_epochs in {path}: {result.get('num_epochs')}")
epoch_samples = result.get("epoch_num_samples")
if epoch_samples != [expected]:
    raise SystemExit(
        f"sample count mismatch in {path}: expected [{expected}], "
        f"found {epoch_samples}"
    )
epoch_times = result.get("epoch_run_times")
if (
    not isinstance(epoch_times, list)
    or len(epoch_times) != 1
    or not math.isfinite(float(epoch_times[0]))
    or float(epoch_times[0]) <= 0
):
    raise SystemExit(f"invalid epoch_run_times in {path}: {epoch_times}")
PY
}

plan_has_cache() {
  grep -q "ObjectDiskCachePipe" "$1"
}

warm_cache() {
  local workload="$1" dataset="$2" samples="$3" kwargs="$4" optimizer="$5"
  local timeout_sec="$6"
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
  if ! run_guarded_result "${timeout_sec}" \
      "${root}/warmup_results/warmup__${optimizer}.json" \
      "${root}/logs/warmup__${optimizer}.log" "${args[@]}"; then
    echo "Cache warmup failed or exhausted ${timeout_sec}s remaining task budget for ${workload}/${optimizer}" >&2
    return 1
  fi
  validate_result "${root}/warmup_results/warmup__${optimizer}.json" "${samples}"

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

run_execution_round() {
  local workload="$1" dataset="$2" samples="$3" cache_mode="$4"
  local kwargs="$5" optimizer="$6" round="$7" timeout_sec="$8"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  local tag="round${round}__${optimizer}"
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
  echo "[$(date -Is)] RUN ${workload}/${tag} budget=${timeout_sec}s" \
    | tee -a "${root}/nohup.log"
  run_guarded_result "${timeout_sec}" \
    "${root}/results/${tag}.json" \
    "${root}/logs/${tag}.log" "${args[@]}"
}

record_task_timeout() {
  local workload="$1" optimizer="$2" round="$3" reason="$4"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  local tag="round${round}__${optimizer}"
  python - "${optimizer}" "${round}" "${reason}" "${TASK_TIMEOUT_SEC}" \
    "${root}/results/${tag}.timeout.json" <<'PY'
import json
import sys

optimizer, repeat, reason, timeout_sec, output = sys.argv[1:]
with open(output, "w") as handle:
    json.dump(
        {
            "optimizer": optimizer,
            "repeat": int(repeat),
            "status": "timeout",
            "reason": reason,
            "task_timeout_sec": int(timeout_sec),
            "task_boundary": (
                "optimization_plus_first_execution"
                if int(repeat) == 1
                else "execution"
            ),
        },
        handle,
        indent=2,
    )
    handle.write("\n")
PY
}

run_workload() {
  local workload="$1" dataset="$2" profile="$3" samples="$4"
  local cache_mode="$5" kwargs="$6"
  local root="${MATRIX_OUTPUT_ROOT}/${workload}"
  local optimizer round offset i tag
  local -a available_optimizers=()
  local -A execution_timed_out=()

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

  # Round 1 is one unified task: optimize first, then execute with whatever
  # remains of the same one-hour deadline. A timeout suppresses later rounds.
  for optimizer in "${OPTIMIZERS[@]}"; do
    local task_start="${SECONDS}" remaining status
    if [[ "${RESUME_EXISTING}" == "1" && \
          -f "${root}/results/round1__${optimizer}.timeout.json" ]]; then
      execution_timed_out["${optimizer}"]=1
      echo "[$(date -Is)] REUSE ${workload}/round1__${optimizer} task timeout" \
        | tee -a "${root}/nohup.log"
      continue
    fi
    if [[ "${RESUME_EXISTING}" == "1" && \
          -f "${root}/plans/${optimizer}.unavailable.json" ]]; then
      if grep -q 'plan_generation_timeout' \
          "${root}/plans/${optimizer}.unavailable.json"; then
        execution_timed_out["${optimizer}"]=1
      fi
      echo "[$(date -Is)] REUSE ${workload}/${optimizer} unavailable plan" \
        | tee -a "${root}/nohup.log"
      continue
    fi
    if [[ "${RESUME_EXISTING}" == "1" && \
          -f "${root}/results/round1__${optimizer}.json" ]] &&
       validate_result \
         "${root}/results/round1__${optimizer}.json" "${samples}"; then
      echo "[$(date -Is)] REUSE ${workload}/round1__${optimizer} result" \
        | tee -a "${root}/nohup.log"
      continue
    fi
    if [[ "${RESUME_EXISTING}" != "1" || \
          ! -f "${root}/plans/${optimizer}.yaml" ]]; then
      generate_plan "${workload}" "${dataset}" "${profile}" "${samples}" \
        "${cache_mode}" "${kwargs}" "${optimizer}" "${TASK_TIMEOUT_SEC}"
    fi
    if [[ ! -f "${root}/plans/${optimizer}.yaml" ]]; then
      if [[ -f "${root}/plans/${optimizer}.unavailable.json" ]] && \
         grep -q 'plan_generation_timeout' \
           "${root}/plans/${optimizer}.unavailable.json"; then
        execution_timed_out["${optimizer}"]=1
        record_task_timeout \
          "${workload}" "${optimizer}" 1 "unified_task_timeout_during_optimization"
      fi
      continue
    fi
    if [[ "${PLAN_ONLY}" == "1" ]]; then
      continue
    fi

    remaining=$((TASK_TIMEOUT_SEC - (SECONDS - task_start)))
    if ((remaining <= 0)); then
      execution_timed_out["${optimizer}"]=1
      record_task_timeout \
        "${workload}" "${optimizer}" 1 "unified_task_timeout_after_optimization"
      continue
    fi
    if [[ "${cache_mode}" == "on" ]] && \
       plan_has_cache "${root}/plans/${optimizer}.yaml"; then
      set +e
      warm_cache "${workload}" "${dataset}" "${samples}" "${kwargs}" \
        "${optimizer}" "${remaining}"
      status=$?
      set -e
      if [[ "${status}" -ne 0 ]]; then
        execution_timed_out["${optimizer}"]=1
        record_task_timeout \
          "${workload}" "${optimizer}" 1 "unified_task_timeout_during_cache_warmup"
        continue
      fi
      remaining=$((TASK_TIMEOUT_SEC - (SECONDS - task_start)))
    else
      unset CEDAR_CACHE_ROOT CEDAR_CACHE_NAMESPACE CEDAR_CACHE_SHARD
    fi
    if ((remaining <= 0)); then
      execution_timed_out["${optimizer}"]=1
      record_task_timeout \
        "${workload}" "${optimizer}" 1 "unified_task_timeout_before_execution"
      continue
    fi
    set +e
    run_execution_round "${workload}" "${dataset}" "${samples}" \
      "${cache_mode}" "${kwargs}" "${optimizer}" 1 "${remaining}"
    status=$?
    set -e
    if [[ "${status}" -eq 124 ]]; then
      execution_timed_out["${optimizer}"]=1
      record_task_timeout \
        "${workload}" "${optimizer}" 1 "unified_task_timeout_during_execution"
      echo "[$(date -Is)] TIMEOUT ${workload}/round1__${optimizer}: unified task exceeded ${TASK_TIMEOUT_SEC}s" \
        | tee -a "${root}/nohup.log"
      continue
    elif [[ "${status}" -ne 0 ]]; then
      echo "Execution failed for ${workload}/round1__${optimizer}" >&2
      return "${status}"
    fi
    validate_result \
      "${root}/results/round1__${optimizer}.json" "${samples}"
    echo "[$(date -Is)] DONE ${workload}/round1__${optimizer}" \
      | tee -a "${root}/nohup.log"
  done

  if [[ "${PLAN_ONLY}" == "1" ]]; then
    echo "[$(date -Is)] PLAN-ONLY COMPLETE ${workload}" \
      | tee -a "${root}/nohup.log"
    return 0
  fi

  for ((round = 2; round <= REPEATS; round++)); do
    offset=$(((round - 1) % ${#OPTIMIZERS[@]}))
    for ((i = 0; i < ${#OPTIMIZERS[@]}; i++)); do
      optimizer="${OPTIMIZERS[$(((offset + i) % ${#OPTIMIZERS[@]}))]}"
      [[ -f "${root}/plans/${optimizer}.yaml" ]] || continue
      tag="round${round}__${optimizer}"
      if [[ -f "${root}/results/${tag}.timeout.json" ]]; then
        execution_timed_out["${optimizer}"]=1
        echo "[$(date -Is)] REUSE ${workload}/${tag} execution timeout" \
          | tee -a "${root}/nohup.log"
        continue
      fi
      if [[ -n "${execution_timed_out[${optimizer}]:-}" ]]; then
        printf '{\n  "optimizer": "%s",\n  "repeat": %s,\n' \
          "${optimizer}" "${round}" \
          > "${root}/results/${tag}.skipped.json"
        printf '  "status": "skipped_after_timeout",\n' \
          >> "${root}/results/${tag}.skipped.json"
        printf '  "skip_reason": "earlier_task_timeout",\n' \
          >> "${root}/results/${tag}.skipped.json"
        printf '  "task_timeout_sec": %s\n}\n' \
          "${TASK_TIMEOUT_SEC}" \
          >> "${root}/results/${tag}.skipped.json"
        echo "[$(date -Is)] SKIP ${workload}/${tag}: earlier repeat timed out" \
          | tee -a "${root}/nohup.log"
        continue
      fi
      if [[ "${RESUME_EXISTING}" == "1" && \
            -f "${root}/results/${tag}.json" ]] &&
         validate_result "${root}/results/${tag}.json" "${samples}"; then
        echo "[$(date -Is)] REUSE ${workload}/${tag} result" \
          | tee -a "${root}/nohup.log"
        continue
      fi
      set +e
      run_execution_round "${workload}" "${dataset}" "${samples}" \
        "${cache_mode}" "${kwargs}" "${optimizer}" "${round}" \
        "${TASK_TIMEOUT_SEC}"
      local execution_status=$?
      set -e
      if [[ "${execution_status}" -eq 124 ]]; then
        record_task_timeout \
          "${workload}" "${optimizer}" "${round}" "execution_task_timeout"
        execution_timed_out["${optimizer}"]=1
        echo "[$(date -Is)] TIMEOUT ${workload}/${tag}: exceeded ${TASK_TIMEOUT_SEC}s" \
          | tee -a "${root}/nohup.log"
        continue
      elif [[ "${execution_status}" -ne 0 ]]; then
        echo "Execution failed for ${workload}/${tag}; see ${root}/logs/${tag}.log" >&2
        return "${execution_status}"
      fi
      validate_result "${root}/results/${tag}.json" "${samples}"
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
  "${PROFILE_DIR}/coco.yaml" "${COCO_SAMPLES}" off "${COCO_DATASET_KWARGS}"
run_workload commonvoice \
  evaluation/pipelines/commonvoice/cedar_dataset.py \
  "${PROFILE_DIR}/commonvoice.yaml" "${COMMONVOICE_SAMPLES}" off \
  "dataset_paths=${COMMONVOICE_DATASET_PATHS},max_samples=${COMMONVOICE_SAMPLES}"
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
  "${PROFILE_DIR}/stackexchange.yaml" "${STACKEXCHANGE_SAMPLES}" off \
  "dataset_path=${STACKEXCHANGE_DATASET_PATH}"
run_workload alpaca_cot \
  evaluation/pipelines/alpaca_cot/cedar_dataset.py \
  "${PROFILE_DIR}/alpaca_cot.yaml" "${ALPACA_COT_SAMPLES}" off \
  "dataset_path=datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl"
run_workload redpajama_arxiv \
  evaluation/pipelines/redpajama_arxiv/cedar_dataset.py \
  "${PROFILE_DIR}/redpajama_arxiv.yaml" "${REDPAJAMA_ARXIV_SAMPLES}" off \
  "dataset_path=datasets/redpajama_arxiv/redpajama-arxiv-raw-3gib.jsonl"
run_workload general_video_refine \
  evaluation/pipelines/general_video_refine/cedar_dataset.py \
  "${PROFILE_DIR}/general_video_refine.yaml" \
  "${GENERAL_VIDEO_REFINE_SAMPLES}" off \
  "dataset_path=datasets/general_video_refine/msrvtt-video-text-200000.jsonl,video_root=datasets/general_video_refine/videos"
run_workload video_self_evolution \
  evaluation/pipelines/video_self_evolution/cedar_dataset.py \
  "${PROFILE_DIR}/video_self_evolution.yaml" \
  "${VIDEO_SELF_EVOLUTION_SAMPLES}" off \
  "dataset_path=datasets/general_video_refine/msrvtt-video-text-200000.jsonl,video_root=datasets/general_video_refine/videos"
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
run_workload pile_europarl \
  evaluation/pipelines/pile_europarl/cedar_dataset.py \
  "${PROFILE_DIR}/pile_europarl.yaml" "${PILE_EUROPARL_SAMPLES}" off \
  "dataset_path=datasets/pile_europarl/pile-europarl-raw.jsonl"
run_workload pile_hackernews \
  evaluation/pipelines/pile_hackernews/cedar_dataset.py \
  "${PROFILE_DIR}/pile_hackernews.yaml" "${PILE_HACKERNEWS_SAMPLES}" off \
  "dataset_path=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
run_workload pile_pubmed_abstracts \
  evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py \
  "${PROFILE_DIR}/pile_pubmed_abstracts.yaml" "${PILE_PUBMED_SAMPLES}" off \
  "dataset_path=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl"
run_workload pile_uspto_backgrounds \
  evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py \
  "${PROFILE_DIR}/pile_uspto_backgrounds.yaml" "${PILE_USPTO_SAMPLES}" off \
  "dataset_path=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl"
run_workload redpajama_code \
  evaluation/pipelines/redpajama_code/cedar_dataset.py \
  "${PROFILE_DIR}/redpajama_code.yaml" "${REDPAJAMA_CODE_SAMPLES}" off \
  "dataset_path=${REDPAJAMA_CODE_DATASET_PATH}"

echo "[$(date -Is)] W=8 matrix complete"
