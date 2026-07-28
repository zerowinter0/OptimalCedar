# Enlarged-data reused-plan protocol

This protocol enlarges workloads whose previous formal run used only a prefix
of an already materialized dataset. It does not repeat profiling. A persisted
physical plan is reused only when every recorded `n_local_workers` value is
exactly 8. Otherwise it is regenerated from the same saved profile, with a
one-hour optimizer limit. This check prevents older W=21/W=32 artifacts from
silently entering the fixed-W=8 matrix.

## Scale

| workload | previous outputs | enlarged outputs |
|---|---:|---:|
| coco | 5,000 val2017 images | 50,000 train2017 images |
| commonvoice / commonvoice_cache | 10,000 delta clips | 160,000 train clips |
| llava_pretrain | 5,000 | 20,000 |
| redpajama_c4 | 20,000 | 100,000 |
| stackexchange | 10,000 | 20,000 |
| wikitext103 / wikitext103_cache | 100,000 | 500,000 |

COCO train2017 is downloaded from the official COCO image archive. CommonVoice
uses the first five CV15 English train shards from the Hugging Face mirror and
verifies each shard against its LFS SHA-256. Neither enlarged workload repeats
records. SimCLRv2 remains at all 9,469 locally available training files. The
six predeclared extension workloads remain at 20,000 accepted outputs.

## Fixed resources and timing

- W=8 and CPU budget 64.
- Three round-robin repetitions.
- Optimization timeout: 3,600 seconds.
- Execution timeout: 3,600 seconds per cell.
- Profile, plan optimization, framework setup, and cache warmup are recorded
  separately from steady-state execution.
- Data-Juicer as a system remains excluded by request; `dj_optimizer` remains
  an in-Cedar optimizer baseline.
- The unoptimized Cedar plan (`no_optimizer`) is excluded because its execution
  time is orders of magnitude longer and it is not part of the optimizer matrix.
- Cache is warmed once per optimizer/attempt. Its marker is stored outside the
  container-owned cache tree so all three measured rounds reuse the same cache.

## Automatic scale fallback

All five Cedar optimizers are attempted at the enlarged output count. If more
than two distinct optimizers time out, the output target is halved and the
workload is restarted. Every rejected attempt remains on disk. This repeats
until at most two optimizers time out or the previous formal output count is
reached. Unavailable plans do not count as execution timeouts.

## External systems

PyTorch, tf.data, and Ray Data implementations of EuroParl, RedPajama Code,
HackerNews, PubMed Abstracts, FreeLaw, and USPTO Backgrounds use the exact
frozen per-record operator parameters shared with Cedar. A test prevents the
native metadata and Cedar registry from diverging.

Plumber and FastFlow are marked unsupported for these six pipelines. Their
graph optimizers cannot serialize or reconstruct the required Python-backed
FastText, SentencePiece, KenLM, and quality-filter callbacks; reporting the
local `tf.py_function` path as an optimized Plumber/FastFlow execution would
not be a valid system comparison.

## Entry point

Run from the host; every workload command is executed in its actual Docker
environment:

```bash
nohup bash evaluation/chapter6_experiments/run_scaled_reuse_plan_matrix.sh \
  > evaluation/chapter6_experiments/scaled_reuse_plan_formal.nohup 2>&1 &
```

The final JSON and Markdown reports are written to
`evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_latest.*`.

For the enlarged COCO/CommonVoice correction run, use the preparation wrapper.
It resumes data downloads, validates and extracts them, and then starts only
`coco,commonvoice,commonvoice_cache` with forced strict-W=8 plan generation:

```bash
nohup env RUN_ID=coco_cv_enlarged_w8_formal_YYYYMMDD \
  bash evaluation/chapter6_experiments/run_coco_commonvoice_enlarged_w8.sh \
  > evaluation/chapter6_experiments/coco_cv_enlarged_w8_formal_YYYYMMDD.nohup \
  2>&1 &
```
