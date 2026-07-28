# FastFlow baselines

The files under `examples/` originate from Cedar's comparison artifact. The
formal workload implementations are under `workloads/` and use an explicit
`APP_CLASS`, repository-relative imports, configurable data paths, sample caps,
and JSON epoch timing.

Supported base workloads are COCO, CommonVoice, SimCLRv2, and WikiText-103.
The `_cache` variants execute the same deterministic workload, but FastFlow has
no cache-placement policy; this limitation is recorded in the support matrix.
LLaVA, RedPajama-C4, and StackExchange are unsupported because their TensorFlow
adapters contain `tf.py_function`/Hugging Face callbacks that tf.data service
cannot serialize for FastFlow's remote partitioning.

COCO uses a dedicated padded `from_tensor_slices` source. This is logically
equivalent to the TensorFlow generator source but serializable by tf.data
service, matching the approach in Cedar's original FastFlow artifact.

Example in a FastFlow environment:

```bash
python evaluation/fastflow/examples/eval_app_runner.py \
  evaluation/fastflow/workloads/simclrv2_app.py \
  /workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train \
  ff evaluation/fastflow/examples/config.yaml \
  --epochs 1 --batch 1 \
  --results_path evaluation/fastflow/results/simclrv2.json
```

FastFlow's public repository referenced by the VLDB'23 paper is no longer
anonymously cloneable at the original URL as of this implementation. Preserve
the FastFlow source/wheel commit used to build the experiment image and record
its hash in every result; do not silently substitute stock TensorFlow.
