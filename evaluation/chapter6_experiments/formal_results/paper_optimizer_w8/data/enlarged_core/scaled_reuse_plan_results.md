# Enlarged-data reused-plan W=8 matrix

Execution time excludes profile, plan optimization, cache warmup, and setup. Values are mean±sample SD over three round-robin repetitions.

| workload | samples | Cedar | DJ | DP-Cedar | DP | Pecan | raw plan | PyTorch | TensorFlow | Ray | Plumber | FastFlow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coco | 50000 | 561.125±22.317 | 906.954±2.472 | 553.111±6.894 | 562.044±8.130 | 566.687±13.358 | N/A | 1639.461±14.122 | 447.546±4.622 | 1331.646±2.547 | 399.599±4.869 | N/A |
| commonvoice | 160000 | 719.136±3.157 | 731.702±2.502 | 719.451±1.702 | 607.055±4.546 | 720.364±1.233 | N/A | 1537.115±4.843 | N/A | 795.988±10.742 | N/A | N/A |
| commonvoice_cache | 160000 | 119.952±50.708 | 88.872±1.243 | 89.108±0.807 | 89.667±2.703 | 89.621±0.470 | N/A | 1561.820±14.734 | N/A | 280.074±30.887 | N/A | N/A |

## Optimizer time

Reused plans retain their prior formal optimization time; only previously missing plans are timed in this run.

| workload | Cedar | DJ | DP-Cedar | DP | Pecan |
|---|---:|---:|---:|---:|---:|
| coco | 9.061 | 8.683 | 8.802 | 8.468 | 8.656 |
| commonvoice | 8.424 | 8.477 | 8.479 | 8.381 | 8.273 |
| commonvoice_cache | 8.627 | 8.434 | 8.362 | 8.373 | 8.368 |
