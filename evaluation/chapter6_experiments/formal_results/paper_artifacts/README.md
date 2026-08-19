# Canonical paper experiment artifacts

This is the only retained result archive for Chapter 6. It contains every
measurement consumed by the paper figures, the exact profiles and physical
plans needed for audit, machine-readable figure tables, and publication-ready
PDF/SVG/PNG outputs. Historical runs, caches, logs, failed repairs, and
superseded figures are intentionally excluded.

All execution experiments use local execution, `W=8`, and `CPU_BUDGET=64`.
Successful cells contain three round-robin repetitions. Cache workloads are
warmed independently, and warmup is excluded from reported execution time.

## Paper figures and sources

| Paper figure | Canonical figure | Source data |
|---|---|---|
| Optimizer execution | `figures/optimizer_execution.pdf` | `optimizer/data/formal_seven/` |
| Cross-system execution | `figures/cross_system_execution_time.pdf` | `cross_system/results/`, `cross_system/status/`, optimizer TSV |
| Optimizer overhead | `figures/optimizer_overhead.pdf` | `optimizer/data/formal_seven/` |
| DP scalability and exactness | `figures/dp_search_evaluation.pdf` | `search/data/` |
| Cost-model accuracy | `figures/cost_model_accuracy.pdf` | `cost_model/analysis.json`, `optimizer/data/` |

The `figures/` directory is the publication interface: its concise filenames
are copied verbatim into the English and Chinese paper trees. Component
directories retain their own source tables and editable figure formats.

## Reproduction

Run inside `optimalcedar-torch201-dev` after `source env/bin/activate`:

```bash
# Optimizer execution and overhead (seven optimizers, eight workloads)
python evaluation/chapter6_experiments/plot_formal_seven_optimizer_matrix.py \
  --matrix-root evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/formal_seven/matrix \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/figures

# Cross-system absolute execution time
python evaluation/chapter6_experiments/plot_paper_cross_system_absolute.py \
  --run-root evaluation/chapter6_experiments/formal_results/paper_artifacts/cross_system \
  --optimizer-tsv evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/figures/latest_optimizer_data.tsv \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_artifacts/cross_system/figures

# Search validation and cost-model accuracy
python evaluation/chapter6_experiments/plot_search_validation.py
python evaluation/chapter6_experiments/plot_formal_cost_model_accuracy.py
```

`MANIFEST.tsv` records the size and SHA-256 digest of every retained artifact.
Run `python evaluation/chapter6_experiments/verify_paper_artifacts.py --write`
after intentionally regenerating figures or changing canonical data.

## Deliberate exclusions

- FreeLaw: no valid profile completed within three hours.
- Data-Juicer as a native-system baseline: removed from the comparison by the
  experiment protocol; Data-Juicer remains only as an optimizer baseline.
- RP-Code, HackerNews, PubMed, and USPTO in the cost-model figure: their
  aggregate runtimes remain in the optimizer archive, but their exact YAML
  plans were not retained, so pairing regenerated plans with old runtimes
  would be invalid.
- CommonVoice-cache and WikiText-cache are visible as N/A in the cost-model
  figure because of tied predictions and insufficient shared scoring coverage.
