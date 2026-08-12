# DP ≥20% workload audit

Candidate results use 1 round-robin repetitions. Execution time excludes optimization; optimization time is reported separately.

| workload | outcome | DP (s) | best other | best other (s) | DP speedup | gate |
|---|---|---:|---|---:|---:|:---:|
| redpajama_code | success | 274.821 | dj_optimizer | 460.193 | 1.675x | PASS |

## Candidate optimizer details

| workload | optimizer | execution mean±sd (s) | optimization (s) | status |
|---|---|---:|---:|---|
| redpajama_code | optimizer | 556.178±0.000 | 19.280 | valid |
| redpajama_code | dj_optimizer | 460.193±0.000 | 3.121 | valid |
| redpajama_code | dp_cedar_optimizer | 551.630±0.000 | 3.759 | valid |
| redpajama_code | dp_optimizer | 274.821±0.000 | 4.668 | valid |
| redpajama_code | dp_two_stage_optimizer | 619.786±0.000 | 3.727 | valid |
| redpajama_code | pecan_optimizer | 467.939±0.000 | 3.114 | valid |

DP ≥20% wins: 1/1 (100.0%).
Fully evaluable workloads: 1/1. Formal timeouts remain in the denominator and are reported as unavailable.
Minimum wins required for 30%: 1; additional wins needed: 0.
30% target: PASS.
