#!/usr/bin/env bash
# Extract the ILSVRC2012 archives fetched by download_imagenet1k.sh.
#
# The train archive holds 1000 per-class tar files; they are expanded into
# train/<wnid>/ so the simclrv2 pipeline can list images recursively. The
# validation images stay in one flat directory, which is all this workload
# needs (it never looks at the class folders).
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${DEST:-$REPO_ROOT/datasets/imagenet/ILSVRC2012}
PARALLEL=${PARALLEL:-8}
cd "$DEST" || exit 1
log() { echo "[$(date '+%F %T')] $*"; }

val_bytes=$(stat -c%s ILSVRC2012_img_val.tar 2>/dev/null || echo 0)
if [ "$val_bytes" -eq 6744924160 ]; then
    if [ ! -d val ] || [ "$(find val -maxdepth 1 -type f | wc -l)" -lt 1000 ]; then
        log "extracting validation images"
        mkdir -p val
        tar xf ILSVRC2012_img_val.tar -C val
    fi
    log "val images: $(find val -maxdepth 1 -type f | wc -l)"
else
    log "validation archive absent or incomplete ($val_bytes bytes); skipping"
fi

if [ ! -s ILSVRC2012_img_train.tar ]; then
    log "train archive absent; the class subset downloader produces train/ directly"
elif [ ! -d class_tars ] || [ "$(find class_tars -maxdepth 1 -name '*.tar' | wc -l)" -lt 1000 ]; then
    log "extracting the outer train archive into class_tars/"
    mkdir -p class_tars
    tar xf ILSVRC2012_img_train.tar -C class_tars
fi
log "class tars: $(find class_tars -maxdepth 1 -name '*.tar' | wc -l)"

if [ ! -d train ] || [ "$(find train -type f -name '*.JPEG' | wc -l)" -lt 1000000 ]; then
    log "extracting $(find class_tars -maxdepth 1 -name '*.tar' | wc -l) class archives with $PARALLEL workers"
    mkdir -p train
    find class_tars -maxdepth 1 -name '*.tar' -print0 | \
        xargs -0 -P "$PARALLEL" -I{} bash -c '
            name=$(basename "{}" .tar)
            mkdir -p "'"$DEST"'/train/$name"
            tar xf "{}" -C "'"$DEST"'/train/$name"
        '
fi
log "train images: $(find train -type f | wc -l)"
log "done; duplicate class archives can be removed to reclaim ~150 GB"
