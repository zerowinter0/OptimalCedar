# Plumber baselines

The original `coco/` and `simclr/` directories are retained from Cedar's
artifact, but they contain author-machine paths and are not the formal runner.
`run_workload.sh` is the shared entry used by the OptimalCedar support matrix.
It:

1. builds the same `tf.data` workload as the TensorFlow baseline;
2. profiles it into a Plumber `stats.pb` performance model;
3. applies `DataPipelineOptimizer.apply_optimizations()`;
4. instantiates and measures the rewritten graph; and
5. writes a JSON result.

Plumber requires its patched TensorFlow build; stock TensorFlow cannot emit or
load `PlumberPerformanceModel`. Pin both upstream repositories:

- `mkuchnik/PlumberApp@6123f5bce36eec7dc75b6b9298054b493d930bdc`
- `mkuchnik/PlumberTensorflow@08bf144ec13b0c27f2a02aaba975546506ee0f6a`

Supported base workloads are `coco`, `commonvoice`, `simclrv2`, and
`wikitext103`. Their `_cache` variants use the same logical pipeline and allow
Plumber to make its own cache recommendation. LLaVA, RedPajama-C4, and
StackExchange are not claimed: their TensorFlow adapters use opaque
`tf.py_function` callbacks, so Plumber cannot model or rewrite their internal
operators.

Example, inside the pinned Plumber environment:

```bash
PYTHONPATH=/workspace/OptimalCedar \
  bash evaluation/plumber/run_workload.sh simclrv2
```
