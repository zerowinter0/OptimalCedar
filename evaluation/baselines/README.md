# Unified external-system baselines

This directory is the authoritative entry point for comparisons with PyTorch,
tf.data, Ray Data, Data-Juicer, Plumber, and FastFlow. The sixteen workloads
are:

`coco`, `commonvoice`, `commonvoice_cache`, `llava_pretrain`,
`redpajama_c4`, `simclrv2`, `simclrv2_cache`, `stackexchange`,
`wikitext103`, `wikitext103_cache`, `pile_europarl`, `redpajama_code`,
`pile_hackernews`, `pile_pubmed_abstracts`, `pile_freelaw`, and
`pile_uspto_backgrounds`.

## Inspect and validate

Run inside `optimalcedar-torch201-dev` after `source env/bin/activate`:

```bash
python -m evaluation.baselines.run --matrix
python -m evaluation.baselines.run --validate
```

The support matrix is executable metadata. Every supported entry names a real
module, recipe, shell runner, or FastFlow app. Unsupported entries include a
system-level reason.

## Native framework runner

PyTorch, TensorFlow, and Ray use one result schema:

```bash
python -m evaluation.baselines.run \
  --system tensorflow \
  --workload simclrv2 \
  --workers 8 \
  --num-samples 9469 \
  --results-path evaluation/baselines/results/tensorflow/simclrv2.json
```

`--dataset-path` overrides every native workload's default input. For
Data-Juicer-derived workloads, extra values such as `image_root` can be supplied
with `--dataset-kwargs '{"image_root": "..."}'`.

For `_cache` workloads, tf.data uses its file-backed final-output cache and Ray
uses `Dataset.materialize()`, with an unmeasured warmup recorded separately.
PyTorch has no native dataset cache and therefore runs its native pipeline
without an invented optimizer layer; the result metadata records this.

The FM tf.data implementations use `tf.py_function` because their exact
Data-Juicer/Hugging Face operators are Python functions. This is a valid native
tf.data execution path, but those internals are opaque to Grappler, Plumber,
and FastFlow. The backend is recorded as `tf_py_function`.

## Pecan AutoOrder

Pecan is implemented as Cedar optimizer selector `8` and as
`pecan_optimizer` in `evaluation/compare_optimizer_perf.py`. It implements
AutoOrder Algorithm 2 using Cedar's profiled data-size ratios; `Pipe.fix()`
marks Pecan fixed-position/rank-changing boundaries, and Cedar
`depends_on()` constraints are preserved. It is intentionally not added to the
existing five-optimizer default experiment order: include it explicitly in a
new paper comparison so old experiment protocols remain reproducible.

## Separate environments

Data-Juicer, Plumber, and FastFlow cannot share Cedar's pinned Python
environment:

- Data-Juicer uses `docker-compose.baselines.yml` and the nested
  `data-juicer/` checkout. Run `bootstrap_datajuicer.sh` to clone or verify the
  exact locked revision; it refuses to alter a mismatched or dirty checkout.
- Plumber requires its patched TensorFlow plus `plumber_analysis`; see
  `evaluation/plumber/README.md`.
- FastFlow requires its FastFlow TensorFlow build/package; see
  `evaluation/fastflow/README.md`.

`run_external.sh` provides a common launch surface. Its pinned defaults are
`optimalcedar-plumber` and `optimalcedar-fastflow`; either can be overridden
with `PLUMBER_CONTAINER` or `FASTFLOW_CONTAINER`. Data-Juicer's container is
defined locally:

```bash
bash evaluation/baselines/bootstrap_datajuicer.sh
docker compose -f docker-compose.baselines.yml build datajuicer
bash evaluation/baselines/run_external.sh \
  datajuicer redpajama_c4 \
  /workspace/OptimalCedar/datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl
```

The external launcher writes a common metadata envelope to the requested
`.json` path and keeps the system-native result/data artifact beside it. This
prevents Data-Juicer's transformed JSONL output or FastFlow's epoch timing JSON
from being confused with cross-system run metadata.

## Fairness rules

- Use identical source files, sample caps, CPU/GPU budgets, and output semantics.
- Record both the outer OptimalCedar commit and nested Data-Juicer commit plus
  dirty state.
- Do not include setup, model download, profiling, or cache warmup in measured
  steady-state time; report them separately.
- Run repeats in round-robin system order and retain raw per-repeat results.
- Do not claim Plumber/FastFlow support for opaque Python FM pipelines.
- Do not include Data-Juicer's global SimHash deduplicator in the main
  per-sample throughput comparison; it changes the logical workload.
- The current task added implementations only. No performance numbers in this
  directory should be interpreted as paper results until the formal matrix is
  run.
