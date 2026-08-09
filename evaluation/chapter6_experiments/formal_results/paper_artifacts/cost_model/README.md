# Formal cost-model accuracy comparison

This directory contains a read-only re-analysis of the canonical W=8 optimizer
archive. No pipeline was re-executed. Cedar's legacy cost and the exact DP
objective score the same optimizer-generated physical plans using the frozen
profile for each workload.

## Dataset construction

- Source archive: `../optimizer`.
- Identical physical plans are deduplicated by canonical SHA-256 within each
  workload.
- When multiple optimizers produced the same plan, all corresponding
  three-round measurements are pooled before computing its mean runtime.
- In total, the audit covers 11 workloads, 42 unique plans, and 156 recorded
  executions.
- The main accuracy comparison uses nine informative workloads, 36 unique
  plans, and 126 executions.
- CommonVoice-cache is reported as N/A because both models assign a constant
  score to all four candidates. WikiText-103-cache is N/A because only one of
  its two plans can be replayed by the DP objective. These exclusions are shown
  in the figure and are not silently removed.
- RP-Code, HackerNews, PubMed, and USPTO are not included because their
  canonical archive retains aggregate runtimes but not the corresponding YAML
  physical plans. Regenerating plans with the current code and pairing them
  with old runtimes would not be a valid cost-accuracy measurement.

Per-workload Spearman correlation evaluates plan ordering. Selection regret is
`runtime(minimum-cost plan) / runtime(fastest candidate) - 1`. Aggregate values
are macro averages over the same nine informative workloads, so large
workloads do not receive extra weight.

## Artifacts

- Main figure: `figures/formal_cost_model_accuracy_comparison.pdf`
- Machine-readable analysis: `analysis.json`
- Figure table and aggregate metrics: `data/`
- Scoring code: `../../../analyze_formal_cost_model_accuracy.py`
- Plotting code: `../../../plot_formal_cost_model_accuracy.py`

## Suggested caption

**Cost-model accuracy on formal optimizer-generated plans.** We rescore the
same deduplicated physical plans with Cedar's legacy cost and the DP objective,
and compare the predicted order against runtime averaged over the archived
three-round executions. The DP objective raises macro-average Spearman
correlation from 0.24 to 0.46, reduces mean selected-plan regret from 36.0% to
1.3%, and increases Top-1 selection accuracy from 3/9 to 7/9 workloads. Gray
columns are retained as N/A: CommonVoice-cache has tied predictions for all
candidates, while WikiText-103-cache has insufficient shared scoring coverage.
