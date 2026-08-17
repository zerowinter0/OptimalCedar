# Operator input-size scaling artifact

This directory is the paper artifact produced by the existing 19-operator
`StackExchangeFeature` Data-Juicer pipeline. It contains the publication figure,
all plotted measurements, and sufficient metadata to audit every point.

## Main result

For the fitted model `latency = c * input_bytes^elasticity`:

- 16 explicit per-data operators have median elasticity **0.948** with range
  **0.759–1.066**.
- 3 explicit per-record operators have median elasticity **0.143** with range
  **0.129–0.206**.
- The two ranges do not overlap; the profiler's `0.35` elasticity boundary
  separates all 19 explicit labels on this pipeline.

Each operator has twelve real legal-input points and every point is the median
of seven single-CPU wall-clock repeats. See `PROTOCOL.md` and `metadata.json`
for the exact method and provenance.

## Files

- `operator_rate_vs_input_bytes.pdf` and `.png`: publication figure.
- `raw_results.csv`: scalar table used for plotting.
- `raw_results.json`: complete repeats and timing data.
- `metadata.json`: dataset hash, operator order, input coverage, fitted
  elasticities, and fit quality.

Reproduce from the repository root inside the project container with:

```bash
OUTPUT_ROOT=outputs/chapter6_experiments/operator_input_size_rate_reproduction \
  bash evaluation/chapter6_experiments/run_operator_input_size_rate.sh
```
