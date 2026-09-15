#!/bin/bash
# Per-workload harness parameters for the small-data measurement matrix.
#
#   source tmp_analysis/workload_env.sh <workload>
#
# Exports DATASET_FILE, DATASET_KWARGS, PROFILE and DATA for
# tmp_analysis/run_plan_busy.sh, plus PLAN_PREFIX (the optimizer prefix used
# by the recorded comparison runs of that workload).
set -u

WORKLOAD=$1
PILE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SMALL=/tmp/small

case "$WORKLOAD" in
  alpaca_cot)
    DATASET_FILE=evaluation/pipelines/alpaca_cot/cedar_dataset.py
    DATA=$SMALL/alpaca_2k.jsonl
    ;;
  pile_hackernews)
    DATASET_FILE=evaluation/pipelines/pile_hackernews/cedar_dataset.py
    DATA=$SMALL/pile_hackernews_2k.jsonl
    ;;
  pile_pubmed_abstracts)
    DATASET_FILE=evaluation/pipelines/target_pipeline/hub/pile_pubmed_abstracts/cedar_dataset.py
    DATA=$SMALL/pile_pubmed_2k.jsonl
    ;;
  pile_uspto_backgrounds)
    DATASET_FILE=evaluation/pipelines/target_pipeline/hub/pile_uspto_backgrounds/cedar_dataset.py
    DATA=$SMALL/pile_uspto_2k.jsonl
    ;;
  bloom_oscar)
    DATASET_FILE=evaluation/pipelines/bloom_oscar/cedar_dataset.py
    DATA=$SMALL/bloom_oscar_2k.jsonl
    ;;
  blip)
    DATASET_FILE=evaluation/pipelines/target_pipeline/blip/cedar_dataset.py
    DATA=datasets/target_pipeline_bench/blip.jsonl
    ;;
  clip)
    DATASET_FILE=evaluation/pipelines/target_pipeline/clip/cedar_dataset.py
    DATA=datasets/target_pipeline_bench/clip.jsonl
    ;;
  dino)
    DATASET_FILE=evaluation/pipelines/target_pipeline/dino/cedar_dataset.py
    DATA=datasets/target_pipeline_bench/dino.jsonl
    ;;
  simclr)
    DATASET_FILE=evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py
    DATA=
    ;;
  *)
    echo "unknown workload: $WORKLOAD" >&2
    return 1
    ;;
esac

# The Data-Juicer Hub entry points expose ``get_target_dataset``; every other
# workload module exposes ``get_dataset``.
case "$WORKLOAD" in
  pile_pubmed_abstracts|pile_uspto_backgrounds)
    DATASET_FUNC=get_target_dataset
    ;;
  *)
    DATASET_FUNC=get_dataset
    ;;
esac

PROFILE=outputs/pico_drained_20260914/profiles/${WORKLOAD}_profile.yaml
if [ ! -f "$PILE/$PROFILE" ]; then
  echo "missing profile: $PROFILE" >&2
  return 1
fi

case "$WORKLOAD" in
  blip|clip)
    DATASET_KWARGS="workload=$WORKLOAD,dataset_path=$DATA"
    ;;
  dino)
    DATASET_KWARGS="workload=dino,dataset_path=$DATA,views=2"
    ;;
  simclr)
    DATASET_KWARGS="workload=simclr"
    ;;
  *)
    DATASET_KWARGS="dataset_path=$DATA"
    ;;
esac

export DATASET_FILE DATASET_KWARGS DATASET_FUNC PROFILE DATA
