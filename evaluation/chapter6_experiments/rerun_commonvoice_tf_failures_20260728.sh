#!/usr/bin/env bash
# Archive and rerun only the CommonVoice cells invalidated by the old
# tf.data **/*.mp3 input glob. Successful cells in the existing formal run
# remain recorded and are skipped by --resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="coco_cv_enlarged_w8_formal_20260727"
RUN_ROOT="${HOST_REPO}/evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_runs/${RUN_ID}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="${RUN_ROOT}/repair_backups/commonvoice_tf_glob_${STAMP}"
PID_FILE="${HOST_REPO}/evaluation/chapter6_experiments/formal_results/commonvoice_tf_failures_rerun_20260728.pid"

printf '%s\n' "$$" > "${PID_FILE}"
echo "[$(date -Is)] START CommonVoice TensorFlow-family repair rerun pid=$$"

[[ -d "${RUN_ROOT}" ]] || {
  echo "Missing formal run: ${RUN_ROOT}" >&2
  exit 1
}

mkdir -p "${BACKUP_ROOT}"

archive_if_present() {
  local source="$1"
  [[ -e "${source}" ]] || return 0
  local relative="${source#${RUN_ROOT}/}"
  mkdir -p "${BACKUP_ROOT}/$(dirname "${relative}")"
  mv -- "${source}" "${BACKUP_ROOT}/${relative}"
}

for workload in commonvoice commonvoice_cache; do
  for system in tensorflow plumber fastflow; do
    for round in 1 2 3; do
      tag="attempt1__round${round}"
      status="${RUN_ROOT}/status/${workload}/${system}/${tag}.tsv"
      state="$(cut -f1 "${status}" 2>/dev/null || true)"
      [[ "${state}" == failed ]] || continue

      archive_if_present "${status}"
      archive_if_present "${RUN_ROOT}/logs/${workload}__${system}__${tag}.log"
      archive_if_present "${RUN_ROOT}/systems/${workload}/${system}/${tag}.json"
      archive_if_present "${RUN_ROOT}/systems/${workload}/${system}/${tag}.pb"
      printf 'queued\t%s\t%s\t%s\n' "${workload}" "${system}" "${tag}" \
        >> "${BACKUP_ROOT}/rerun_cells.tsv"
    done
  done
done

queued="$(wc -l < "${BACKUP_ROOT}/rerun_cells.tsv" 2>/dev/null || true)"
if [[ "${queued:-0}" -eq 0 ]]; then
  echo "No failed CommonVoice TensorFlow-family cells found."
  exit 0
fi

echo "[$(date -Is)] Archived and queued ${queued} cells under ${BACKUP_ROOT}"
cd "${HOST_REPO}"
exec bash evaluation/chapter6_experiments/run_scaled_reuse_plan_matrix.sh \
  --workloads commonvoice,commonvoice_cache \
  --run-id "${RUN_ID}" --resume
