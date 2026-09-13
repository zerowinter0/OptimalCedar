#!/bin/bash
# Wait for the dino profile, stop the unbounded comparison, then run the
# remaining comparisons with the bounded Cedar reorder timeout.
set -u
cd /workspace/OptimalCedar
source env/bin/activate
OUT=/workspace/OptimalCedar/outputs/plumber_bench_20260912
DRIVER=261433
WATCHER=262708

echo "[finish] waiting for dino profile"
for _ in $(seq 1 720); do
  [ -f "$OUT/dino_profile.yaml" ] && break
  kill -0 "$DRIVER" 2>/dev/null || break
  sleep 30
done
echo "[finish] dino profile present: $([ -f "$OUT/dino_profile.yaml" ] && echo yes || echo no)"

kill -9 "$DRIVER" 2>/dev/null || true
kill -9 "$WATCHER" 2>/dev/null || true
sleep 5
pkill -9 -f "workload=dino" 2>/dev/null || true
pkill -9 -f "run_plumber_bench.py" 2>/dev/null || true
sleep 5

run_workload () {
  local name=$1
  local dataset_file=$2
  local kwargs=$3
  echo "[finish] comparing $name"
  python -u tmp_analysis/compare_one.py \
    --output "$OUT" --workload "$name" --dataset-file "$dataset_file" \
    --dataset-kwargs "$kwargs" --samples 1000 --repeats 3 \
    --cedar-timeout 600 --optimizer-limit 1800 \
    && echo "[finish] $name ok" || echo "[finish] $name failed"
}

run_workload swav evaluation/pipelines/target_pipeline/cedar_dataset.py \
  "workload=swav,dataset_path=/workspace/OptimalCedar/datasets/target_pipeline_bench/swav.jsonl"
run_workload dino evaluation/pipelines/target_pipeline/cedar_dataset.py \
  "workload=dino,dataset_path=/workspace/OptimalCedar/datasets/target_pipeline_bench/dino.jsonl"
run_workload clip evaluation/pipelines/target_pipeline/cedar_dataset.py \
  "workload=clip,dataset_path=/workspace/OptimalCedar/datasets/target_pipeline_bench/clip.jsonl,tokenizer_path=/workspace/OptimalCedar/evaluation/datasets/target_pipeline/clip_tokenizer"

python -u tmp_analysis/summarize_plumber_bench.py "$OUT" > "$OUT/SUMMARY.txt" 2>&1
echo "[finish] DONE" >> "$OUT/SUMMARY.txt"
echo "[finish] all steps complete"
