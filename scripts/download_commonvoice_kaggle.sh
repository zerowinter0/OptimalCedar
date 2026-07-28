#!/usr/bin/env bash
# Download the exact Common Voice release used by Cedar's CommonVoice
# workload from Kaggle, then normalize Kaggle's nested archive layout to the
# path expected by evaluation/pipelines/commonvoice/cedar*_dataset.py.
set -euo pipefail

repo_root=/workspace/OptimalCedar
dataset_ref=saadawadh/cv-corpus-150-new
release=cv-corpus-15.0-delta-2023-09-08
dataset_root="${repo_root}/evaluation/datasets/commonvoice"
clips_dir="${dataset_root}/${release}/en/clips"
staging_dir="${dataset_root}/.kaggle_${release}_staging"

cd "${repo_root}"
source env/bin/activate
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/root/.config/kaggle}"

if [[ ! -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]]; then
  echo "Missing Kaggle credentials: ${KAGGLE_CONFIG_DIR}/kaggle.json" >&2
  exit 2
fi

if [[ -d "${clips_dir}" ]] && find "${clips_dir}" -maxdepth 1 -type f -name '*.mp3' -print -quit | grep -q .; then
  echo "Common Voice is already ready: ${clips_dir}"
  exit 0
fi

mkdir -p "${staging_dir}" "${clips_dir}"
echo "[$(date -Is)] Downloading ${dataset_ref} into ${staging_dir}"
kaggle datasets download "${dataset_ref}" --path "${staging_dir}" --unzip

# Kaggle preserves the publisher's deeply nested paths.  Cedar consumes only
# raw MP3 clips, whose Common Voice filenames are unique, so flatten those
# clips into the expected release/language directory.
echo "[$(date -Is)] Normalizing Common Voice MP3 layout"
while IFS= read -r -d '' source_file; do
  destination="${clips_dir}/$(basename "${source_file}")"
  if [[ ! -e "${destination}" ]]; then
    mv "${source_file}" "${destination}"
  fi
done < <(find "${staging_dir}" -type f -name '*.mp3' -print0)

clip_count="$(find "${clips_dir}" -maxdepth 1 -type f -name '*.mp3' | wc -l)"
if [[ "${clip_count}" -eq 0 ]]; then
  echo "No MP3 clips were found after Kaggle extraction." >&2
  exit 3
fi

echo "[$(date -Is)] Common Voice ready: ${clip_count} clips in ${clips_dir}"
