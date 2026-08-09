#!/usr/bin/env bash
# Queue the official COCO train2017 image archive after the formal matrix so
# network/disk activity cannot contaminate throughput measurements.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="${HOST_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CEDAR_CONTAINER="${CEDAR_CONTAINER:-optimalcedar-torch201-dev}"
FORMAL_PID_FILE="${HOST_REPO}/evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_formal.pid"
URL="https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/main/train2017.zip?download=true"
EXPECTED_BYTES=19336861798
EXPECTED_IMAGES=118287
EXPECTED_SHA256=69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929

echo "[$(date -Is)] COCO train2017 download queued"
if [[ -s "${FORMAL_PID_FILE}" ]]; then
  formal_pid="$(cat "${FORMAL_PID_FILE}")"
  while ps -p "${formal_pid}" -o args= 2>/dev/null \
      | grep -q 'run_scaled_reuse_plan_matrix.sh'; do
    echo "[$(date -Is)] Waiting for formal matrix PID ${formal_pid}"
    sleep 60
  done
fi

echo "[$(date -Is)] Formal matrix inactive; starting COCO download"
docker exec -i \
  -e COCO_URL="${URL}" \
  -e COCO_EXPECTED_BYTES="${EXPECTED_BYTES}" \
  -e COCO_EXPECTED_IMAGES="${EXPECTED_IMAGES}" \
  -e COCO_EXPECTED_SHA256="${EXPECTED_SHA256}" \
  "${CEDAR_CONTAINER}" bash -lc '
    set -euo pipefail
    cd /workspace/OptimalCedar
    source env/bin/activate
    archive="datasets/coco/train2017.zip"
    mirror_archive="datasets/coco/train2017.hf-mirror.zip"
    official_partial="datasets/coco/train2017.official-partial.zip"
    mkdir -p datasets/coco

    # A curl-created sequential partial cannot be repartitioned efficiently by
    # aria2. Preserve it for recovery and start a fresh parallel control file.
    if [[ -f "${archive}" && ! -f "${mirror_archive}" && \
          "$(stat -c %s "${archive}")" != "${COCO_EXPECTED_BYTES}" ]]; then
      mv "${archive}" "${official_partial}"
    fi

    # The official COCO endpoint delivered only tens of KiB/s on this VPN.
    # This byte-identical Hugging Face mirror supports parallel HTTP ranges.
    # aria2 rejects the socks5h all_proxy syntax, so retain the working HTTP(S)
    # proxy variables while removing only all_proxy for this command.
    # Hugging Face Xet range URLs are signed. Individual CDN connections can
    # eventually return 403 while a long transfer is still active. Restarting
    # aria2 refreshes those signatures and resumes exactly from its control
    # file, without discarding completed pieces.
    while true; do
      if env -u all_proxy -u ALL_PROXY /usr/bin/aria2c \
        --continue=true --max-connection-per-server=16 --split=16 \
        --min-split-size=1M --piece-length=1M --file-allocation=none \
        --auto-file-renaming=false --allow-overwrite=true \
        --max-tries=5 --retry-wait=3 --timeout=30 \
        --summary-interval=30 --console-log-level=notice \
        --dir="$(dirname "${mirror_archive}")" --out="$(basename "${mirror_archive}")" \
        "${COCO_URL}"; then
        break
      fi
      [[ -f "${mirror_archive}.aria2" ]] || {
        echo "aria2 failed without a resumable control file" >&2
        exit 1
      }
      echo "[$(date -Is)] RETRY COCO mirror with refreshed signed URLs"
      sleep 5
    done

    actual_bytes="$(stat -c %s "${mirror_archive}")"
    if [[ "${actual_bytes}" != "${COCO_EXPECTED_BYTES}" ]]; then
      echo "Archive size mismatch: expected=${COCO_EXPECTED_BYTES} actual=${actual_bytes}" >&2
      exit 1
    fi
    unzip -tq "${mirror_archive}"
    echo "${COCO_EXPECTED_SHA256}  ${mirror_archive}" | sha256sum --check -
    mv "${mirror_archive}" "${archive}"
    unzip -oq "${archive}" -d datasets/coco

    actual_images="$(find datasets/coco/train2017 -maxdepth 1 \
      -type f -name "*.jpg" | wc -l)"
    if [[ "${actual_images}" != "${COCO_EXPECTED_IMAGES}" ]]; then
      echo "Image count mismatch: expected=${COCO_EXPECTED_IMAGES} actual=${actual_images}" >&2
      exit 1
    fi

    sha256="${COCO_EXPECTED_SHA256}"
    python - "${COCO_URL}" "${actual_bytes}" "${actual_images}" \
      "${sha256}" <<"PY"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "dataset": "COCO train2017 images",
    "url": sys.argv[1],
    "archive_bytes": int(sys.argv[2]),
    "image_count": int(sys.argv[3]),
    "archive_sha256": sys.argv[4],
    "zip_integrity_checked": True,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
Path("datasets/coco/train2017.download.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    echo "[$(date -Is)] COCO train2017 ready: images=${actual_images} sha256=${sha256}"
  '

echo "[$(date -Is)] COCO train2017 download and extraction complete"
