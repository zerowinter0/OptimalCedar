#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${CH6_RESULT_ROOT:-${REPO_ROOT}/evaluation/chapter6_experiments/formal_results}"
PROFILE_ROOT="${CH6_PROFILE_ROOT:-${RESULT_ROOT}/profiles}"
RUN_ID="${CH6_PROFILE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RESULT_ROOT}/profile_runs/${RUN_ID}"
LOG_ROOT="${RUN_ROOT}/logs"
STAGING_ROOT="${RUN_ROOT}/profiles"
ARCHIVE_ROOT="${RUN_ROOT}/replaced_profiles"
RAY_ADDRESS="127.0.0.1:6379"
CPU_BUDGET=64
PROFILE_TIMEOUT_SEC=3600
SELECTED_WORKLOADS="all"
INCREMENTAL_BACKEND_COMPUTE="${INCREMENTAL_BACKEND_COMPUTE:-0}"
export INCREMENTAL_BACKEND_COMPUTE

usage() {
  cat <<'EOF'
Usage: run_formal_profiles.sh [--workloads workload[,workload...]]

Default: profile all formal workloads.  The filter is useful when a dataset
changes and only its profile must be regenerated under the same protocol.
Data-Juicer candidates: pile_europarl, redpajama_code, pile_hackernews,
pile_pubmed_abstracts, pile_freelaw, pile_uspto_backgrounds.
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

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_FILTER_SELECTIVITY=1
unset CEDAR_LOCAL_WORKERS
ulimit -n 65536

mkdir -p "${PROFILE_ROOT}" "${LOG_ROOT}" "${STAGING_ROOT}" "${ARCHIVE_ROOT}"
exec > >(tee -a "${RUN_ROOT}/profile_matrix.log") 2>&1

validate_profile() {
  local profile="$1"
  python - "${profile}" <<'PY'
import sys
import os
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as stream:
    profile = yaml.safe_load(stream)

expected = {
    "schema_version": 1,
    "profile_scope": "single_local_worker",
    "profile_local_workers": 1,
    "actors_per_stage": 1,
    "ray_actors_per_stage": 1,
    "smp_procs_per_stage": 1,
}
if not isinstance(profile, dict):
    raise RuntimeError(f"Profile is not a mapping: {path}")
actual = profile.get("resource_config")
if actual != expected:
    raise RuntimeError(
        f"Invalid resource_config in {path}: expected={expected}, actual={actual}"
    )
for key in ("baseline", "disk_info", "offloads"):
    if key not in profile:
        raise RuntimeError(f"Profile {path} is missing required section: {key}")
offloads = profile["offloads"]
if not isinstance(offloads, dict) or "RAY" not in offloads or "SMP" not in offloads:
    raise RuntimeError(f"Profile {path} does not contain both RAY and SMP results")
if os.environ.get("INCREMENTAL_BACKEND_COMPUTE") == "1":
    updated = 0
    for variant_name, pipe_profiles in offloads.items():
        for pipe_id, pipe_profile in pipe_profiles.items():
            timing = pipe_profile.get("backend_compute")
            if not isinstance(timing, dict):
                raise RuntimeError(
                    f"Missing backend_compute for {variant_name} pipe {pipe_id}"
                )
            if int(timing.get("count", 0)) < 1:
                raise RuntimeError(
                    f"Empty backend_compute for {variant_name} pipe {pipe_id}"
                )
            updated += 1
    if updated < 1:
        raise RuntimeError(f"No backend timings found in {path}")
if os.environ.get("CEDAR_PROFILE_FILTER_SELECTIVITY") == "1":
    baseline = profile["baseline"]
    selectivity_keys = (
        "input_counts",
        "output_counts",
        "selectivities",
        "selectivity_observation_sources",
    )
    missing_selectivity_keys = [
        key for key in selectivity_keys if key not in baseline
    ]
    # Legacy profiles for pipelines without a FilterPipe contain none of
    # these mappings. Incremental backend timing intentionally preserves that
    # profile instead of inventing unrelated selectivity observations.
    if missing_selectivity_keys and not (
        os.environ.get("INCREMENTAL_BACKEND_COMPUTE") == "1"
        and len(missing_selectivity_keys) == len(selectivity_keys)
    ):
        raise RuntimeError(
            f"Selectivity-aware profile {path} is missing "
            f"{missing_selectivity_keys}"
        )
    if missing_selectivity_keys:
        print(f"Validated complete profile: {path}: {actual}")
        raise SystemExit(0)
    filter_ids = set(baseline["input_counts"])
    if filter_ids != set(baseline["output_counts"]):
        raise RuntimeError(f"Filter count key mismatch in {path}")
    if filter_ids != set(baseline["selectivities"]):
        raise RuntimeError(f"Filter selectivity key mismatch in {path}")
    if filter_ids != set(baseline["selectivity_observation_sources"]):
        raise RuntimeError(
            f"Filter selectivity source key mismatch in {path}"
        )
    for pipe_id in filter_ids:
        inputs = baseline["input_counts"][pipe_id]
        outputs = baseline["output_counts"][pipe_id]
        selectivity = baseline["selectivities"][pipe_id]
        if (
            not isinstance(inputs, int)
            or not isinstance(outputs, int)
            or inputs < 0
            or outputs < 0
            or outputs > inputs
        ):
            raise RuntimeError(
                f"Invalid filter counts for pipe {pipe_id} in {path}: "
                f"inputs={inputs}, outputs={outputs}"
            )
        expected = outputs / inputs if inputs else 1.0
        if abs(float(selectivity) - expected) > 1e-12:
            raise RuntimeError(
                f"Invalid filter selectivity for pipe {pipe_id} in {path}: "
                f"expected={expected}, actual={selectivity}"
            )
        source = baseline["selectivity_observation_sources"][pipe_id]
        if not isinstance(source, str) or not source:
            raise RuntimeError(
                f"Invalid filter selectivity source for pipe {pipe_id} "
                f"in {path}: {source!r}"
            )
print(f"Validated complete profile: {path}: {actual}")
PY
}

archive_existing_profile() {
  local name="$1"
  local target="${PROFILE_ROOT}/${name}.yaml"
  if [[ -f "${target}" && ! -f "${ARCHIVE_ROOT}/${name}.yaml" ]]; then
    cp -p "${target}" "${ARCHIVE_ROOT}/${name}.yaml"
  fi
}

run_profile() {
  local name="$1"
  local dataset="$2"
  local kwargs="${3:-}"
  local staged_profile="${STAGING_ROOT}/${name}.yaml"
  local target_profile="${PROFILE_ROOT}/${name}.yaml"
  local log="${LOG_ROOT}/${name}.log"
  local -a args=(
    taskset -c 0 python evaluation/eval_cedar.py
    --dataset_file "${dataset}"
    --profiled_stats "${staged_profile}"
    --run_profiling
    --disable_optimizer
    --disable_controller
    --disable_prefetch
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
  )
  if [[ "${INCREMENTAL_BACKEND_COMPUTE}" == "1" ]] && \
     [[ -f "${target_profile}" ]] && \
     validate_profile "${target_profile}" >> "${log}" 2>&1; then
    echo "[$(date -Is)] REUSE ${name}: backend timings already complete"
    return 0
  fi
  if [[ -n "${kwargs}" ]]; then
    args+=(--dataset_kwargs "${kwargs}")
  fi
  if [[ "${INCREMENTAL_BACKEND_COMPUTE}" == "1" ]]; then
    [[ -f "${target_profile}" ]] || {
      echo "Missing base profile for incremental update: ${target_profile}" >&2
      return 1
    }
    args=(env "CEDAR_INCREMENTAL_PROFILE_FROM=${target_profile}" "${args[@]}")
  fi

  echo "[$(date -Is)] Profiling ${name} -> ${target_profile}"
  printf 'command:' > "${log}"
  printf ' %q' "${args[@]}" >> "${log}"
  printf '\n' >> "${log}"
  timeout --signal=TERM --kill-after=30s "${PROFILE_TIMEOUT_SEC}" \
    "${args[@]}" >> "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    echo "[$(date -Is)] PROFILE_TIMEOUT ${name}: exceeded ${PROFILE_TIMEOUT_SEC}s; see ${log}"
    return 1
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "[$(date -Is)] FAILED ${name}: profiler command exit status ${status}; see ${log}"
    return 1
  fi
  if ! validate_profile "${staged_profile}" | tee -a "${log}"; then
    echo "[$(date -Is)] FAILED ${name}: validation failed; see ${log}"
    return 1
  fi
  archive_existing_profile "${name}"
  mv -f "${staged_profile}" "${target_profile}"
  echo "[$(date -Is)] Completed ${name} -> ${target_profile}"
}

run_selected_profile() {
  local name="$1"
  shift
  if ! workload_selected "${name}"; then
    echo "[$(date -Is)] SKIP ${name}: not selected by --workloads"
    return 0
  fi
  run_profile "${name}" "$@"
}

echo "[$(date -Is)] Starting formal profile run ${RUN_ID}"
echo "[$(date -Is)] Incremental backend compute: ${INCREMENTAL_BACKEND_COMPUTE}"
echo "[$(date -Is)] Profile width: local_workers=1 ray_actors_per_stage=1 smp_procs_per_stage=1"

echo "[$(date -Is)] Per-workload profile timeout: ${PROFILE_TIMEOUT_SEC}s"
ray stop --force || true
if ! ray start --head --node-ip-address=127.0.0.1 --port=6379 \
  --num-cpus="${CPU_BUDGET}" --disable-usage-stats; then
  echo "[$(date -Is)] FAILED to start Ray"
  exit 1
fi

failures=()
run_selected_profile coco evaluation/pipelines/coco/cedar_dataset.py || failures+=(coco)
run_selected_profile commonvoice evaluation/pipelines/commonvoice/cedar_dataset.py || failures+=(commonvoice)
run_selected_profile commonvoice_cache evaluation/pipelines/commonvoice/cedar_cache_dataset.py || failures+=(commonvoice_cache)
run_selected_profile llava_pretrain evaluation/pipelines/llava_pretrain/cedar_dataset.py \
  "dataset_path=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl,image_root=evaluation/datasets/llava_pretrain" || failures+=(llava_pretrain)
run_selected_profile redpajama_c4 evaluation/pipelines/redpajama_c4/cedar_dataset.py \
  "dataset_path=datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl" || failures+=(redpajama_c4)
run_selected_profile stackexchange evaluation/pipelines/stackexchange/cedar_dataset.py \
  "dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl" || failures+=(stackexchange)
run_selected_profile pile_europarl evaluation/pipelines/pile_europarl/cedar_dataset.py \
  "dataset_path=datasets/pile_europarl/pile-europarl-raw.jsonl" || failures+=(pile_europarl)
run_selected_profile redpajama_code evaluation/pipelines/redpajama_code/cedar_dataset.py \
  "dataset_path=datasets/redpajama_code/redpajama-github-raw-50000.jsonl" || failures+=(redpajama_code)
run_selected_profile pile_hackernews evaluation/pipelines/pile_hackernews/cedar_dataset.py \
  "dataset_path=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl" || failures+=(pile_hackernews)
run_selected_profile pile_pubmed_abstracts evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py \
  "dataset_path=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl" || failures+=(pile_pubmed_abstracts)
run_selected_profile pile_freelaw evaluation/pipelines/pile_freelaw/cedar_dataset.py \
  "dataset_path=datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl" || failures+=(pile_freelaw)
run_selected_profile pile_uspto_backgrounds evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py \
  "dataset_path=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl" || failures+=(pile_uspto_backgrounds)
run_selected_profile simclrv2 evaluation/pipelines/simclrv2/cedar_dataset.py || failures+=(simclrv2)
run_selected_profile simclrv2_cache evaluation/pipelines/simclrv2/cedar_cache_dataset.py || failures+=(simclrv2_cache)
run_selected_profile wikitext103 evaluation/pipelines/wikitext103/cedar_dataset.py || failures+=(wikitext103)
run_selected_profile wikitext103_cache evaluation/pipelines/wikitext103/cedar_cache_dataset.py || failures+=(wikitext103_cache)

ray stop --force || true
if (( ${#failures[@]} )); then
  printf '[%s] Profile run completed with failures:' "$(date -Is)"
  printf ' %s' "${failures[@]}"
  printf '\n'
  exit 1
fi

echo "[$(date -Is)] All formal profiles completed successfully"
