#!/usr/bin/env bash
# Download the datasets required by the project experiment matrix.  This is
# intended to run inside the OptimalCedar Docker container after activating
# its repository environment.
set -euo pipefail

repo_root=/workspace/OptimalCedar
datasets_dir="${repo_root}/datasets"
redpajama_dir="${repo_root}/datasets/redpajama_c4"
redpajama_url=https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-c4-refine-result.jsonl

cd "${repo_root}"
source env/bin/activate

echo "[$(date -Is)] Dataset download started"

# These four workload inputs are already present.  They are checked here so
# the launch log documents the exact data state used for the later experiment.
test -f "${datasets_dir}/bloom_oscar/c4_en_50000_for_bloom_oscar.jsonl"
test -f "${datasets_dir}/llava_pretrain/blip_laion_cc_sbu_558k.jsonl"
test -d "${datasets_dir}/imagenette2/imagenette2/train"
test -f "${datasets_dir}/wikitext103/wikitext-103/wiki.train.tokens"
echo "[$(date -Is)] Existing BLOOM, LLaVA, SimCLRv2, and WikiText103 inputs verified"

# Cedar's COCO download recipe uses val2017 images and the corresponding
# train/validation annotations.  curl --continue-at makes interruption
# recovery safe in the project container, which intentionally lacks wget.
coco_dir="${datasets_dir}/coco"
coco_download_dir="${coco_dir}/downloads"
mkdir -p "${coco_dir}" "${coco_download_dir}"
if [[ ! -f "${coco_dir}/annotations/instances_val2017.json" ]]; then
  echo "[$(date -Is)] Downloading COCO val2017 and annotations"
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "${coco_download_dir}/val2017.zip" \
    http://images.cocodataset.org/zips/val2017.zip
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "${coco_download_dir}/annotations_trainval2017.zip" \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  python - "${coco_download_dir}" "${coco_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

download_dir, coco_dir = map(Path, sys.argv[1:])
for name in ("val2017.zip", "annotations_trainval2017.zip"):
    archive = download_dir / name
    print(f"Extracting {archive} to {coco_dir}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(coco_dir)
PY
fi
test -f "${coco_dir}/annotations/instances_val2017.json"
test -d "${coco_dir}/val2017"
echo "[$(date -Is)] COCO ready"

# The RedPajama-C4 workload is defined against Data-Juicer's published
# refined C4 corpus.  Store it at the exact path hard-coded by the workload.
mkdir -p "${redpajama_dir}"
if [[ ! -s "${redpajama_dir}/redpajama-c4-refined.jsonl" ]]; then
  echo "[$(date -Is)] Downloading Data-Juicer refined RedPajama-C4"
  curl --fail --location --retry 5 --retry-all-errors --range 0-21474836479 \
    --output "${redpajama_dir}/redpajama-c4-refined.jsonl" "${redpajama_url}"
fi
test -s "${redpajama_dir}/redpajama-c4-refined.jsonl"
echo "[$(date -Is)] RedPajama-C4 ready"

# Cedar's CommonVoice helper downloads the required cv-corpus-15.0 English
# archive from the project bucket and expands it into the workload's expected
# path.  It intentionally relies on standard Google application credentials.
commonvoice_clips="${datasets_dir}/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips"
if [[ ! -d "${commonvoice_clips}" ]]; then
  echo "[$(date -Is)] Downloading Cedar CommonVoice archive"
  python evaluation/pipelines/commonvoice/download.py
fi
test -d "${commonvoice_clips}"
echo "[$(date -Is)] CommonVoice ready"

echo "[$(date -Is)] All required datasets verified"
