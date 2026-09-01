#!/usr/bin/env bash
# Operator-scaling and policy-two-stage paper experiment.
# Run inside optimalcedar-torch201-dev with the project virtualenv available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY_ROOT="${STUDY_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/operator_scaling_two_stage_study}"
PROFILE_ROOT="${STUDY_ROOT}/profiles"
MATRIX_ROOT="${STUDY_ROOT}/matrix_bottleneck_objective"
LOG_ROOT="${STUDY_ROOT}/logs"
PHASE="${PHASE:-all}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
PROFILE_TIMEOUT_SEC="${PROFILE_TIMEOUT_SEC:-3600}"

case "${PHASE}" in
  profile|matrix|all) ;;
  *) echo "PHASE must be profile, matrix, or all" >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
ulimit -n 65536
mkdir -p "${PROFILE_ROOT}" "${MATRIX_ROOT}" "${LOG_ROOT}"

cat > "${STUDY_ROOT}/PROTOCOL.md" <<'EOF'
# Operator-scaling and policy-two-stage study

- Workloads: Alpaca-CoT, general-video-refine, RP-Code, Pile HackerNews,
  Pile USPTO Backgrounds, Pile EuroParl, Pile PubMed Abstracts, StackExchange.
- Added complex workloads: PubMed Abstracts and StackExchange, selected from
  already completed formal workloads with strong DP gains.
- Profiles: frozen formal numeric profiles plus an incremental 10-second
  in-process legal-input size/latency observation. Existing baseline,
  offload, boundary, and disk measurements are preserved.
- Optimizers rerun: DP, Pecan-two-stage, DJ-two-stage. Existing Cedar, DJ,
  DP-Cedar, DP-two-stage, and Pecan results remain immutable comparison data.
- Execution: W=8, CPU budget 64, three round-robin repeats, cache disabled,
  3600-second plan and per-repeat execution limits. A first-repeat timeout
  skips later repeats.
- DP objective: the maximum accumulated service demand among local, Ray, SMP,
  and GPU resource families. Pipelines with at most eight operators
  use an exact Pareto frontier; larger pipelines use deterministic
  multiplicative trimming with a 10% whole-search error bound
  (`CEDAR_DP_PARETO_EPSILON=0.10`).
- General-video-refine uses 6,000 outputs uniformly for every optimizer. The
  stopped 7,500-output pilot reached only 6,468--6,908 outputs in one hour;
  6,000 preserves a substantial video workload with reproducible headroom.
EOF

validate_incremental_profile() {
  local source="$1" target="$2"
  python - "${source}" "${target}" <<'PY'
import copy
import sys
import yaml

source_path, target_path = sys.argv[1:]
with open(source_path) as stream:
    source = yaml.safe_load(stream)
with open(target_path) as stream:
    target = yaml.safe_load(stream)
for payload in (source, target):
    payload.pop("operator_compute_scaling", None)
    payload.pop("incremental_compute_scaling", None)
if source != target:
    raise SystemExit("Incremental profile changed frozen numeric measurements")
with open(target_path) as stream:
    target = yaml.safe_load(stream)
metadata = target.get("operator_compute_scaling")
if not isinstance(metadata, dict) or not metadata:
    raise SystemExit("Missing operator_compute_scaling metadata")
valid_modes = {"explicit", "inferred", "default"}
for pipe_id, entry in metadata.items():
    if entry.get("scaling") not in {"per_data", "per_byte", "per_record"}:
        raise SystemExit(f"Invalid scaling for pipe {pipe_id}: {entry}")
    if entry.get("mode") not in valid_modes:
        raise SystemExit(f"Invalid mode for pipe {pipe_id}: {entry}")
PY
}

profile_one() {
  local name="$1" dataset="$2" source_profile="$3" kwargs="$4"
  local target="${PROFILE_ROOT}/${name}.yaml"
  local log="${LOG_ROOT}/profile__${name}.log"
  if [[ -f "${target}" ]] && validate_incremental_profile \
      "${source_profile}" "${target}"; then
    echo "[$(date -Is)] REUSE profile ${name}"
    return 0
  fi
  local -a args=(
    taskset -c 0 python evaluation/eval_cedar.py
    --dataset_file "${dataset}"
    --profiled_stats "${target}"
    --run_profiling
    --disable_optimizer
    --disable_controller
    --disable_prefetch
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
  )
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
  echo "[$(date -Is)] PROFILE ${name}"
  printf 'command:' > "${log}"
  printf ' %q' env \
    "CEDAR_INCREMENTAL_PROFILE_FROM=${source_profile}" \
    CEDAR_INCREMENTAL_COMPUTE_SCALING=1 \
    CEDAR_PROFILE_INFER_COMPUTE_SCALING=1 \
    "${args[@]}" >> "${log}"
  printf '\n' >> "${log}"
  env "CEDAR_INCREMENTAL_PROFILE_FROM=${source_profile}" \
    CEDAR_INCREMENTAL_COMPUTE_SCALING=1 \
    CEDAR_PROFILE_INFER_COMPUTE_SCALING=1 \
    timeout --signal=TERM --kill-after=30s "${PROFILE_TIMEOUT_SEC}" \
    "${args[@]}" >> "${log}" 2>&1
  validate_incremental_profile "${source_profile}" "${target}"
}

materialize_annotations() {
  local name="$1" dataset="$2" kwargs="$3"
  local -a args=(
    python evaluation/chapter6_experiments/materialize_operator_scaling_annotations.py
    --profile "${PROFILE_ROOT}/${name}.yaml"
    --dataset-file "${dataset}"
    --require-all-operators-annotated
  )
  [[ -n "${kwargs}" ]] && args+=(--dataset-kwargs "${kwargs}")
  "${args[@]}"
}

run_profiles() {
  local optimizer_profiles="evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles"
  local diverse_profiles="evaluation/chapter6_experiments/formal_results/paper_artifacts/datajuicer_diverse/profiles"
  profile_one alpaca_cot evaluation/pipelines/alpaca_cot/cedar_dataset.py \
    "${diverse_profiles}/alpaca_cot.yaml" \
    "dataset_path=datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl"
  profile_one general_video_refine evaluation/pipelines/general_video_refine/cedar_dataset.py \
    "${diverse_profiles}/general_video_refine.yaml" \
    "dataset_path=datasets/general_video_refine/msrvtt-video-text-200000.jsonl,video_root=datasets/general_video_refine/videos"
  profile_one redpajama_code evaluation/pipelines/redpajama_code/cedar_dataset.py \
    "${diverse_profiles}/redpajama_code.yaml" \
    "dataset_path=datasets/redpajama_code/redpajama-github-raw-50000.jsonl"
  profile_one pile_hackernews evaluation/pipelines/pile_hackernews/cedar_dataset.py \
    "${optimizer_profiles}/pile_hackernews.yaml" \
    "dataset_path=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
  profile_one pile_uspto_backgrounds evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py \
    "${optimizer_profiles}/pile_uspto_backgrounds.yaml" \
    "dataset_path=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl"
  profile_one pile_europarl evaluation/pipelines/pile_europarl/cedar_dataset.py \
    "${optimizer_profiles}/pile_europarl.yaml" \
    "dataset_path=datasets/pile_europarl/pile-europarl-raw.jsonl"
  profile_one pile_pubmed_abstracts evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py \
    "${optimizer_profiles}/pile_pubmed_abstracts.yaml" \
    "dataset_path=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl"
  profile_one stackexchange evaluation/pipelines/stackexchange/cedar_dataset.py \
    "${optimizer_profiles}/stackexchange.yaml" \
    "dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl"

  materialize_annotations alpaca_cot \
    evaluation/pipelines/alpaca_cot/cedar_dataset.py \
    "dataset_path=datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl"
  materialize_annotations general_video_refine \
    evaluation/pipelines/general_video_refine/cedar_dataset.py \
    "dataset_path=datasets/general_video_refine/msrvtt-video-text-200000.jsonl,video_root=datasets/general_video_refine/videos"
  materialize_annotations redpajama_code \
    evaluation/pipelines/redpajama_code/cedar_dataset.py \
    "dataset_path=datasets/redpajama_code/redpajama-github-raw-50000.jsonl"
  materialize_annotations pile_hackernews \
    evaluation/pipelines/pile_hackernews/cedar_dataset.py \
    "dataset_path=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
  materialize_annotations pile_uspto_backgrounds \
    evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py \
    "dataset_path=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl"
  materialize_annotations pile_europarl \
    evaluation/pipelines/pile_europarl/cedar_dataset.py \
    "dataset_path=datasets/pile_europarl/pile-europarl-raw.jsonl"
  materialize_annotations pile_pubmed_abstracts \
    evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py \
    "dataset_path=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl"
  materialize_annotations stackexchange \
    evaluation/pipelines/stackexchange/cedar_dataset.py \
    "dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl"

  python - "${PROFILE_ROOT}" "${STUDY_ROOT}/scaling_inference_summary.json" <<'PY'
import json
import pathlib
import sys
import yaml

profile_root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
summary = {}
for path in sorted(profile_root.glob("*.yaml")):
    profile = yaml.safe_load(path.read_text())
    counts = {"explicit": 0, "inferred": 0, "default": 0}
    details = []
    for pipe_id, entry in profile["operator_compute_scaling"].items():
        counts[entry["mode"]] += 1
        details.append({"pipe_id": pipe_id, **entry})
    summary[path.stem] = {"counts": counts, "operators": details}
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
  touch "${STUDY_ROOT}/PROFILE_COMPLETE"
}

run_matrix() {
  [[ -f "${STUDY_ROOT}/PROFILE_COMPLETE" ]] || {
    echo "Scaling profiles are incomplete" >&2
    exit 1
  }
  PROFILE_DIR="${PROFILE_ROOT}" \
  MATRIX_OUTPUT_ROOT="${MATRIX_ROOT}" \
  OPTIMIZER_SET=operator_scaling_study \
  CEDAR_DP_PARETO_EPSILON=0.10 \
  REPEATS=3 \
  RESUME_EXISTING="${RESUME_EXISTING:-0}" \
  ALPACA_COT_SAMPLES=65000 \
  GENERAL_VIDEO_REFINE_SAMPLES=6000 \
  PILE_EUROPARL_SAMPLES=2500 \
  PILE_HACKERNEWS_SAMPLES=20000 \
  PILE_PUBMED_SAMPLES=20000 \
  PILE_USPTO_SAMPLES=20000 \
  REDPAJAMA_CODE_SAMPLES=20000 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads alpaca_cot,general_video_refine,redpajama_code,pile_hackernews,pile_uspto_backgrounds,pile_europarl,pile_pubmed_abstracts,stackexchange
  touch "${STUDY_ROOT}/MATRIX_COMPLETE"
}

[[ "${PHASE}" == "profile" || "${PHASE}" == "all" ]] && run_profiles
[[ "${PHASE}" == "matrix" || "${PHASE}" == "all" ]] && run_matrix

echo "[$(date -Is)] PHASE ${PHASE} complete"
