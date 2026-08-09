# General video refinement workload protocol

This protocol was frozen before profiling, plan generation, or execution of
the workload. It extends the varied-operator-count study with a video--text
pipeline and is independent of the earlier text-only candidate selection rule.

## Recipe and implementation

- Official recipe repository: `datajuicer/data-juicer-hub`.
- Recipe revision: `47fc34588b5d4258c13747cea37c2b63cf4e11b0`.
- Recipe: `refined_recipes/video/general-video-refine-example.yaml`.
- Data-Juicer operator revision: `bb3d88aac183cc22b6f816262a812a9e5d5abb57`.
- All seven per-sample filters and their published thresholds are retained in
  recipe order before optimization. No operator or threshold may be removed or
  changed after observing a profile or execution result.

The seven filters are language-ID confidence, text perplexity, video-frame
aesthetics, video-frame/text CLIP similarity, optical-flow motion score, video
NSFW score, and video watermark probability. JSON parsing, video-path
resolution, and final projection are fixed Cedar mappers and are not reordered.

## Frozen data source and sample construction

- Dataset: complete MSR-VTT 10K video archive and official annotations.
- Immutable mirror: `nisav/MSR-VTT@a9c822473969ee469e224da2187fda193c62e960`.
- Source artifacts: `MSRVTT_Videos.zip` and `raw_data/MSRVTT_data.json`.
- The archive contains 10,000 distinct clips and the annotation contains
  200,000 distinct video-caption pairs. No synthetic record or duplicated
  caption is introduced.
- The Cedar JSONL enumerates caption rounds: the first caption for every video,
  then the second caption for every video, and so on. This prevents a bounded
  run from repeatedly processing only a small source-order prefix of videos.
- Formal execution requests 10,000 retained outputs, matching the natural
  number of distinct videos. If filtering requires more source records, Cedar
  continues into later genuine caption rounds.

The downloaded archive is below the existing 3-GiB input policy. Preparation
records SHA-256 digests, source revision, record counts, and extracted-video
coverage. Test MP4 files from Data-Juicer are used only for correctness tests,
never as formal measurements.

## Resources and measurement

- Local execution on the paper server with one RTX A6000, `W=8`, and
  `CPU_BUDGET=64`.
- Non-cache workload; caching is disabled for every optimizer.
- The same saved profile is used by every optimizer. The profile uses one
  local worker and one actor/process for each Ray/SMP stage, with a nominal
  ten-second measurement window.
- Optimizers: Cedar, Data-Juicer ordering, DP-Cedar, PICO, DP-two-stage, and
  Pecan. Plans are generated before execution.
- Plan timeout: 3,600 seconds per optimizer.
- Execution timeout: 10,800 seconds per optimizer repetition. If the first
  repetition times out, later repetitions for that optimizer are recorded as
  skipped rather than rerun.
- Successful optimizers run three repetitions in round-robin order. Setup,
  model download, profile generation, and plan generation are excluded from
  execution time.

All CUDA model weights are prefetched before profiling. CUDA models are loaded
lazily inside Cedar workers so multiprocessing never forks a live CUDA context.
Every optimizer uses the same visible GPU and model cache. Cedar invokes the
pinned per-sample accelerator operators under `torch.inference_mode()`, matching
Data-Juicer's outer inference execution context and preventing autograd
workspaces from consuming the GPU shared by the eight workers. This changes no
model, threshold, score, or filter decision.

## Acceptance checks

Before formal profiling, the migration must pass:

1. dataset revision, digest, 10,000-video, and 200,000-pair validation;
2. one-record comparisons of Cedar adapter statistics/decisions with the
   pinned Data-Juicer operator implementation;
3. final-output equality between the recipe-order Cedar feature and a direct
   pinned-operator reference on the test fixture;
4. serialization and Cedar in-process smoke tests;
5. profile schema/resource validation before any optimizer consumes it.

Failures and timeouts remain in the result directory and are not replaced by
smaller inputs, relaxed thresholds, or projected runtimes.
