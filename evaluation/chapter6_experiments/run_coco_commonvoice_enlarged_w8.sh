#!/usr/bin/env bash
# Prepare larger, non-repeated COCO/CommonVoice inputs and then run the strict
# W=8 optimizer/system matrix. Intended to be launched with nohup.

set -euo pipefail

HOST_REPO="/home/xieruiyang/OptimalCedar"
CEDAR_CONTAINER="${CEDAR_CONTAINER:-optimalcedar-torch201-dev}"
SCRIPT_DIR="${HOST_REPO}/evaluation/chapter6_experiments"
COCO_READY="${HOST_REPO}/datasets/coco/train2017.download.json"
CV_DATA_REL="datasets/commonvoice/cv15_en_train_5shards"
CV_ARCHIVE_REL="datasets/commonvoice/cv15_en_train_5shards_archives"
CV_READY="${HOST_REPO}/${CV_ARCHIVE_REL}/ready.json"
RUN_ID="${RUN_ID:-coco_cv_enlarged_w8_formal_$(date -u +%Y%m%dT%H%M%SZ)}"

echo "[$(date -Is)] PREPARE run_id=${RUN_ID}"

# A previously queued COCO download may already be active. Do not start a
# competing transfer; resume it only if it exited before producing metadata.
while [[ ! -s "${COCO_READY}" ]] && \
      pgrep -f 'curl .*train2017.zip.*images.cocodataset.org' >/dev/null; do
  size="$(stat -c %s "${HOST_REPO}/datasets/coco/train2017.zip" 2>/dev/null || echo 0)"
  echo "[$(date -Is)] WAIT COCO train2017 archive_bytes=${size}"
  sleep 60
done
if [[ ! -s "${COCO_READY}" ]]; then
  echo "[$(date -Is)] RESUME COCO train2017 download"
  bash "${SCRIPT_DIR}/download_coco_train2017_after_formal.sh"
fi

docker exec -i "${CEDAR_CONTAINER}" bash -lc '
  set -euo pipefail
  cd /workspace/OptimalCedar
  source env/bin/activate

  data_root="datasets/commonvoice/cv15_en_train_5shards"
  archive_root="datasets/commonvoice/cv15_en_train_5shards_archives"
  mkdir -p "${data_root}" "${archive_root}"

  sizes=(1807292928 1787189248 1747609600 1698287616 1702616064)
  hashes=(
    4e2db220551afee0c8ad3ad0d5f5d422c3a86463c1e21782d8679237bfcffd8e
    40fc0659198c30cdd21c189b2b66f255e42036cedbd4ad39215a9bb0ba554090
    d8188a0d3f8f73a89eddfeca2eb07671a7b857f022f947e09d9d696ec66c4571
    54ae2e3309edd9b2039acc725b04f6862ed9d0cb178e20839f901f2381d54b68
    2585d71fdb728f7521d4b3707d33b6baf1d102fa39c6fa61865c71014c5ed898
  )
  base="https://huggingface.co/datasets/fsicoli/common_voice_15_0/resolve/main/audio/en/train"

  download_shard() {
    local index="$1"
    local archive="${archive_root}/en_train_${index}.tar"
    local marker="${archive_root}/en_train_${index}.extracted"
    local url="${base}/en_train_${index}.tar?download=true"
    local log="${archive_root}/en_train_${index}.download.log"
    [[ -f "${marker}" ]] && return 0
    echo "[$(date -Is)] DOWNLOAD CommonVoice shard=${index} connections=12"
    while true; do
      if env -u all_proxy -u ALL_PROXY /usr/bin/aria2c \
        --continue=true --max-connection-per-server=12 --split=12 \
        --min-split-size=1M --piece-length=1M --file-allocation=none \
        --auto-file-renaming=false --allow-overwrite=true \
        --max-tries=5 --retry-wait=3 --timeout=30 \
        --summary-interval=30 --console-log-level=notice \
        --dir="${archive_root}" --out="$(basename "${archive}")" \
        "${url}" >> "${log}" 2>&1; then
        break
      fi
      [[ -f "${archive}.aria2" ]] || {
        echo "CommonVoice shard ${index} failed without resumable state" >&2
        return 1
      }
      echo "[$(date -Is)] RETRY CommonVoice shard=${index}" | tee -a "${log}"
      sleep 5
    done
  }

  # Five shards x twelve HTTP Range connections = at most 60 concurrent
  # transfers, staying below the requested cap of 64.
  pids=()
  for index in 0 1 2 3 4; do
    download_shard "${index}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || exit 1

  for index in 0 1 2 3 4; do
    archive="${archive_root}/en_train_${index}.tar"
    marker="${archive_root}/en_train_${index}.extracted"
    [[ -f "${marker}" ]] && continue
    actual="$(stat -c %s "${archive}")"
    [[ "${actual}" == "${sizes[${index}]}" ]] || {
      echo "CommonVoice shard ${index} size mismatch: ${actual}" >&2
      exit 1
    }
    echo "${hashes[${index}]}  ${archive}" | sha256sum --check -
    tar -tf "${archive}" >/dev/null
    # Each tar has one en_train_N/ prefix. Strip it so Cedar Ray-DS and all
    # external systems see the same flat directory of MP3 files.
    tar -xf "${archive}" -C "${data_root}" --strip-components=1
    touch "${marker}"
  done

  count="$(find "${data_root}" -maxdepth 1 -type f -name "*.mp3" | wc -l)"
  (( count >= 160000 )) || {
    echo "CommonVoice enlarged input has only ${count} clips" >&2
    exit 1
  }
  python - "${count}" <<"PY"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path("datasets/commonvoice/cv15_en_train_5shards_archives/ready.json").write_text(
    json.dumps(
        {
            "dataset": "Common Voice 15.0 English train, shards 0-4",
            "source": "fsicoli/common_voice_15_0 Hugging Face mirror",
            "clip_count": int(sys.argv[1]),
            "benchmark_samples": 160000,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
  echo "[$(date -Is)] CommonVoice enlarged input ready clips=${count}"
'

[[ -s "${CV_READY}" ]]
echo "[$(date -Is)] START strict-W8 enlarged matrix run_id=${RUN_ID}"
cd "${HOST_REPO}"
FORCE_REGENERATE_PLANS=1 \
  bash evaluation/chapter6_experiments/run_scaled_reuse_plan_matrix.sh \
    --workloads coco,commonvoice,commonvoice_cache --run-id "${RUN_ID}"
