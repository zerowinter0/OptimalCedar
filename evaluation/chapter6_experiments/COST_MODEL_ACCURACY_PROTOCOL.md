# Cost-model accuracy protocol

The optimizer cost model is validated offline against already completed formal
W=8 executions. The evaluator canonicalizes and deduplicates physical plans
within each workload, pools the three round-robin runtime measurements for each
unique plan, and replays every plan through Cedar's original cost and the DP
objective. It does not execute a data pipeline.

Run inside the project container and activated environment:

```bash
python evaluation/chapter6_experiments/analyze_formal_cost_model_accuracy.py \
  --reference-analysis OLD_ANALYSIS.json \
  --output NEW_ANALYSIS.json
```

## Primary metric: pairwise Q-error

For two plans `i` and `j` from the same workload:

```text
predicted_ratio = predicted_cost_i / predicted_cost_j
observed_ratio  = mean_runtime_i / mean_runtime_j

pairwise_qerror = max(
    predicted_ratio / observed_ratio,
    observed_ratio / predicted_ratio
)
```

One is exact; two means that the predicted relative performance differs from
measurement by a factor of two. Pairwise ratios cancel any workload-wide
multiplicative scale between the service-demand objective and wall-clock time,
so the metric tests the part of the model that actually ranks candidate plans.
The report includes geometric mean, median, P90, maximum, and pairwise ordering
accuracy. Spearman correlation, selected-plan regret, and top-1 accuracy remain
secondary decision metrics.

When comparing model revisions, use `shared_coverage_summary`: it restricts all
models to exactly the same workload-plan pairs. This matters after correctness
fixes reject historical plans that violated newly explicit semantic
dependencies. Comparing each model's independent coverage would otherwise be
misleading.

This is a regression and validation metric, not a substitute for final
end-to-end measurement of a newly selected plan. It can reject a broadly worse
cost-model change without rerunning pipelines; accepted changes still require
the paper's normal execution protocol before their runtime is reported.
