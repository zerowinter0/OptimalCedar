#!/usr/bin/env bash
# Segmented, resumable download of ILSVRC2012 (ImageNet-1k) for the simclrv2
# workloads.
#
# The official download page asks for an institution account, but the three
# archive URLs themselves answer without authentication (verified 2026-09-20,
# HTTP 200 through the lab proxy), which is the method described in the CSDN
# walkthrough. Single-connection throughput here is ~5.4 MB/s and four
# parallel segments reach ~8 MB/s, so the 147.9 GB train archive is a
# multi-hour download; every segment resumes, so the script can be restarted.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${DEST:-$REPO_ROOT/datasets/imagenet/ILSVRC2012}
SEGMENTS=${SEGMENTS:-4}
BASE=https://image-net.org/data/ILSVRC/2012
TRAIN=ILSVRC2012_img_train.tar
TRAIN_BYTES=147897477120
VAL=ILSVRC2012_img_val.tar
VAL_BYTES=6744924160
DEVKIT=ILSVRC2012_devkit_t12.tar.gz
DEVKIT_BYTES=2568145

mkdir -p "$DEST"
cd "$DEST" || exit 1
log() { echo "[$(date '+%F %T')] $*"; }

log "destination $DEST (segments=$SEGMENTS, train=$TRAIN_BYTES bytes)"

# Small archives first: val is 6.74 GB, the devkit 2.5 MB. A partially
# downloaded file is resumed until it reaches the announced size; an earlier
# campaign left a 35 MB stub behind exactly that way.
for spec in "$VAL:$VAL_BYTES" "$DEVKIT:$DEVKIT_BYTES"; do
    name=${spec%%:*}
    want=${spec##*:}
    while :; do
        have=$(stat -c%s "$name" 2>/dev/null || echo 0)
        if [ "$have" -eq "$want" ]; then
            log "$name complete ($have bytes)"
            break
        fi
        log "$name: have=$have want=$want"
        curl -fsS -L --retry 20 --retry-delay 10 -C - -o "$name" "$BASE/$name" \
            || sleep 15
    done
done

part_size=$(( (TRAIN_BYTES + SEGMENTS - 1) / SEGMENTS ))
pids=()
for i in $(seq 0 $((SEGMENTS - 1))); do
    first=$((i * part_size))
    last=$((first + part_size - 1))
    if [ "$last" -ge "$TRAIN_BYTES" ]; then
        last=$((TRAIN_BYTES - 1))
    fi
    want=$((last - first + 1))
    (
        while :; do
            have=$(stat -c%s "part_$i" 2>/dev/null || echo 0)
            if [ "$have" -ge "$want" ]; then
                log "segment $i complete ($have bytes)"
                break
            fi
            log "segment $i: have=$have want=$want"
            curl -fsS -L --retry 5 --retry-delay 10 --max-time 3600 \
                -r "$((first + have))-$last" "$BASE/$TRAIN" >> "part_$i" \
                || sleep 15
        done
    ) &
    pids+=("$!")
done
wait "${pids[@]}"

total=$(du -cb part_* 2>/dev/null | tail -1 | cut -f1)
if [ "${total:-0}" -ne "$TRAIN_BYTES" ]; then
    log "ERROR: assembled $total bytes, expected $TRAIN_BYTES"
    exit 1
fi
if [ ! -s "$TRAIN" ]; then
    log "concatenating segments into $TRAIN"
    cat $(seq -f 'part_%g' 0 $((SEGMENTS - 1))) > "$TRAIN"
fi
log "done: $(ls -la "$TRAIN" "$TRAIN").tar 2>/dev/null"
ls -la "$DEST"
