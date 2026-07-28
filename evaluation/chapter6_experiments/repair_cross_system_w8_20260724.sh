#!/usr/bin/env bash
set -euo pipefail

REPO="/home/xieruiyang/OptimalCedar"
RUN_ID="cross_system_w8_formal_20260724_164724"
RUN_ROOT="${REPO}/evaluation/chapter6_experiments/formal_results/cross_system_w8_runs/${RUN_ID}"
BACKUP="${RUN_ROOT}/pre_bugfix_results_20260724"
ACTIVE_SERVICE="${ACTIVE_SERVICE:-optimalcedar-cross-system-w8-20260724_164724-resume2.service}"

echo "[$(date -Is)] Waiting for ${ACTIVE_SERVICE} to finish"
while systemctl --user is-active --quiet "${ACTIVE_SERVICE}"; do
  sleep 30
done

mkdir -p "${BACKUP}/status" "${BACKUP}/systems" "${BACKUP}/logs"

archive_cell() {
  local workload="$1"
  local entity="$2"
  local status="${RUN_ROOT}/status/${workload}/${entity}.tsv"
  if [[ -f "${status}" ]]; then
    mkdir -p "${BACKUP}/status/${workload}"
    mv "${status}" "${BACKUP}/status/${workload}/${entity}.tsv"
  fi

  local artifact
  for artifact in "${RUN_ROOT}/systems/${workload}/${entity}"*; do
    if [[ -e "${artifact}" ]]; then
      mkdir -p "${BACKUP}/systems/${workload}"
      mv "${artifact}" "${BACKUP}/systems/${workload}/"
    fi
  done

  for artifact in \
      "${RUN_ROOT}/logs/${workload}__${entity}.log" \
      "${RUN_ROOT}/logs/${workload}__${entity}_profile.log" \
      "${RUN_ROOT}/logs/${workload}__${entity}_optimize.log"; do
    if [[ -e "${artifact}" ]]; then
      mv "${artifact}" "${BACKUP}/logs/"
    fi
  done
}

# COCO native baselines previously dropped 48 images without annotations.
for entity in pytorch tensorflow ray; do
  archive_cell coco "${entity}"
done

# FastFlow previously launched both worker groups in-process, so the custom
# runtime saw no remote workers.
archive_cell coco fastflow

# Ray Data requires every map output to be a row dictionary. The old adapter
# returned a scalar string only after completing each expensive block, causing
# full-block failures and retries.
archive_cell redpajama_c4 ray

# Plumber's old report interpreted the first tensor dimension as batch size.
for workload in \
    coco commonvoice commonvoice_cache \
    simclrv2 simclrv2_cache wikitext103 wikitext103_cache; do
  archive_cell "${workload}" plumber
done

# These pipelines contain Python callbacks and cannot execute remotely.
# Removing any old fallback result lets the fixed launcher mark them unsupported.
for workload in \
    commonvoice commonvoice_cache wikitext103 wikitext103_cache; do
  archive_cell "${workload}" fastflow
done

# These five cells hit the asynchronous cache-manifest commit race.
for entity in \
    cedar_optimizer cedar_dp_optimizer cedar_dp_cedar_optimizer \
    cedar_dj_optimizer cedar_pecan_optimizer; do
  archive_cell commonvoice_cache "${entity}"
done

echo "[$(date -Is)] Starting repaired resume"
cd "${REPO}"
exec env RUN_ID="${RUN_ID}" \
  bash evaluation/chapter6_experiments/run_cross_system_w8_matrix.sh \
    --run-id "${RUN_ID}" \
    --workloads \
      coco,commonvoice,commonvoice_cache,redpajama_c4,simclrv2,simclrv2_cache,wikitext103,wikitext103_cache \
    --resume
