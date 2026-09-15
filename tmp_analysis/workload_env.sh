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
  pile_europarl)
    DATASET_FILE=evaluation/pipelines/pile_europarl/cedar_dataset.py
    DATA=$SMALL/pile_europarl_2k.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/pile_europarl/pile-europarl-raw.jsonl
    ;;
  pile_freelaw)
    DATASET_FILE=evaluation/pipelines/pile_freelaw/cedar_dataset.py
    DATA=$SMALL/pile_freelaw_2k.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl
    ;;
  redpajama_arxiv)
    DATASET_FILE=evaluation/pipelines/redpajama_arxiv/cedar_dataset.py
    DATA=$SMALL/redpajama_arxiv_2k.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/redpajama_arxiv/redpajama-arxiv-raw-3gib.jsonl
    ;;
  redpajama_code)
    DATASET_FILE=evaluation/pipelines/redpajama_code/cedar_dataset.py
    DATA=$SMALL/redpajama_code_2k.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/redpajama_code/redpajama-github-raw-100000.jsonl
    ;;
  redpajama_c4)
    DATASET_FILE=evaluation/pipelines/redpajama_c4/cedar_dataset.py
    DATA=$SMALL/redpajama_c4_2k.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl
    ;;
  stackexchange)
    DATASET_FILE=evaluation/pipelines/stackexchange/cedar_dataset.py
    DATA=$SMALL/stackexchange_2k.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/stackexchange/redpajama-stackexchange-10000.jsonl
    ;;
  swav)
    DATASET_FILE=evaluation/pipelines/target_pipeline/swav/cedar_dataset.py
    DATA=datasets/target_pipeline_bench/swav.jsonl
    ;;
  swav_single)
    # Single-view SwAV: the multi-crop recipe replicates the whole augmentation
    # chain once per view (59 operators with the default views=8).  One view
    # keeps the pipeline semantics and the dataset, and stays inside the joint
    # DP's subset limit; the multi-view shape is a training-time choice, not a
    # different data pipeline.
    DATASET_FILE=evaluation/pipelines/target_pipeline/swav/cedar_dataset.py
    DATA=datasets/target_pipeline_bench/swav.jsonl
    ;;
  dino_single)
    DATASET_FILE=evaluation/pipelines/target_pipeline/dino/cedar_dataset.py
    DATA=datasets/target_pipeline_bench/dino.jsonl
    ;;
  general_video_refine)
    DATASET_FILE=evaluation/pipelines/general_video_refine/cedar_dataset.py
    DATA=$SMALL/general_video_refine_500.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/general_video_refine/msrvtt-video-text-200000.jsonl
    ;;
  video_self_evolution)
    DATASET_FILE=evaluation/pipelines/video_self_evolution/cedar_dataset.py
    DATA=$SMALL/general_video_refine_500.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/general_video_refine/msrvtt-video-text-200000.jsonl
    ;;
  commonvoice)
    DATASET_FILE=evaluation/pipelines/commonvoice/cedar_dataset.py
    DATA=datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en
    ;;
  coco)
    DATASET_FILE=evaluation/pipelines/coco/cedar_dataset.py
    DATA=datasets/coco
    ;;
  llava_pretrain)
    # Data-Juicer Hub image/llava-pretrain-refine.yaml: nine cheap text
    # filters, three image filters, then the CLIP/BLIP image-text filters on
    # the accelerator.  The recipe is the selective-filter counterpart of the
    # augmentation workloads above.
    DATASET_FILE=evaluation/pipelines/llava_pretrain/cedar_dataset.py
    SUBSET=${SUBSET:-2k}
    DATA=$SMALL/llava_${SUBSET}.jsonl
    SOURCE=/workspace/OptimalCedar/evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl
    ;;
  multimodal_running_example)
    # Data-Juicer's multimodal running example over COCO image/caption pairs
    # (CLIP + BLIP + perplexity + sharpness + aesthetics predicates).  The
    # manifest is built by the workload's own fixture builder from the
    # original COCO annotations.
    DATASET_FILE=evaluation/pipelines/multimodal_running_example/cedar_dataset.py
    DATA=/tmp/small/multimodal_fixture.jsonl/calibration.jsonl
    ;;
  wikitext103)
    DATASET_FILE=evaluation/pipelines/wikitext103/cedar_dataset.py
    DATA=datasets/wikitext103/wikitext-103/wiki.train.tokens
    ;;
  simclrv2)
    DATASET_FILE=evaluation/pipelines/simclrv2/cedar_dataset.py
    DATA=datasets/imagenette2
    ;;
  simclrv2_views4)
    # Same recipe, four views instead of the default eight: the augmentation
    # chain is replicated per view, so this is the multi-view shape the
    # chain-partition DP is meant to handle.
    DATASET_FILE=evaluation/pipelines/simclrv2/cedar_dataset.py
    DATA=datasets/imagenette2
    ;;
  simclrv2_cache)
    # The same recipe with Cedar's object-disk cache enabled: a second stage
    # boundary (cache write/read) that the joint DP can price.
    DATASET_FILE=evaluation/pipelines/simclrv2/cedar_cache_dataset.py
    DATA=datasets/imagenette2
    ;;
  pile_hackernews)
    DATASET_FILE=evaluation/pipelines/pile_hackernews/cedar_dataset.py
    SUBSET=${SUBSET:-2k}
    DATA=$SMALL/pile_hackernews_${SUBSET}.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl
    ;;
  pile_pubmed_abstracts)
    DATASET_FILE=evaluation/pipelines/target_pipeline/hub/pile_pubmed_abstracts/cedar_dataset.py
    SUBSET=${SUBSET:-2k}
    DATA=$SMALL/pile_pubmed_${SUBSET}.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl
    ;;
  pile_uspto_backgrounds)
    DATASET_FILE=evaluation/pipelines/target_pipeline/hub/pile_uspto_backgrounds/cedar_dataset.py
    SUBSET=${SUBSET:-2k}
    DATA=$SMALL/pile_uspto_${SUBSET}.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl
    ;;
  bloom_oscar)
    DATASET_FILE=evaluation/pipelines/bloom_oscar/cedar_dataset.py
    SUBSET=${SUBSET:-2k}
    DATA=$SMALL/bloom_oscar_${SUBSET}.jsonl
    SOURCE=/workspace/OptimalCedar/datasets/bloom_oscar/c4_en_50000_for_bloom_oscar.jsonl
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
if [ ! -f "$PILE/$PROFILE" ] && [ -f "$PILE/outputs/plumber_bench_20260912/${WORKLOAD}_profile.yaml" ]; then
  PROFILE=outputs/plumber_bench_20260912/${WORKLOAD}_profile.yaml
fi
if [ ! -f "$PILE/$PROFILE" ]; then
  echo "[env] profiling needed for $WORKLOAD (expected $PROFILE)" >&2
fi

case "$WORKLOAD" in
  blip|clip)
    DATASET_KWARGS="workload=$WORKLOAD,dataset_path=$DATA"
    ;;
  general_video_refine)
    DATASET_KWARGS="dataset_path=$DATA,video_root=/workspace/OptimalCedar/datasets/general_video_refine/videos"
    ;;
  video_self_evolution)
    DATASET_KWARGS="dataset_path=$DATA,video_root=/workspace/OptimalCedar/datasets/general_video_refine/videos"
    ;;
  commonvoice)
    DATASET_KWARGS="dataset_path=$DATA,max_samples=300"
    ;;
  coco)
    DATASET_KWARGS="split=val2017"
    ;;
  llava_pretrain)
    DATASET_KWARGS="dataset_path=$DATA,image_root=/workspace/OptimalCedar/evaluation/datasets/llava_pretrain"
    ;;
  multimodal_running_example)
    DATASET_KWARGS="dataset_path=$DATA,threshold_path=/workspace/OptimalCedar/.multimodal_smoke_threshold.json,image_root=/workspace/OptimalCedar/datasets/coco/train2017"
    ;;
  wikitext103)
    DATASET_KWARGS="max_samples=800"
    ;;
  simclrv2)
    DATASET_KWARGS=""
    ;;
  simclrv2_views4)
    DATASET_KWARGS="views=4"
    ;;
  simclrv2_cache)
    DATASET_KWARGS=""
    ;;
  swav)
    DATASET_KWARGS="workload=swav,dataset_path=$DATA"
    ;;
  swav_single)
    DATASET_KWARGS="workload=swav,dataset_path=$DATA,views=1"
    ;;
  dino_single)
    DATASET_KWARGS="workload=dino,dataset_path=$DATA,views=1"
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
