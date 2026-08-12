# DP ≥20% workload audit

Candidate results use 3 round-robin repetitions. Execution time excludes optimization; optimization time is reported separately.

| workload | outcome | DP (s) | best other | best other (s) | DP speedup | gate |
|---|---|---:|---|---:|---:|:---:|
| redpajama_code | success | 274.699 | dj_optimizer | 460.249 | 1.675x | PASS |

## Candidate optimizer details

| workload | optimizer | execution mean±sd (s) | optimization (s) | status |
|---|---|---:|---:|---|
| redpajama_code | optimizer | 552.280±1.211 | 19.108 | valid |
| redpajama_code | dj_optimizer | 460.249±1.089 | 3.107 | valid |
| redpajama_code | dp_cedar_optimizer | 553.268±1.619 | 3.742 | valid |
| redpajama_code | dp_optimizer | 274.699±0.770 | 4.700 | valid |
| redpajama_code | dp_two_stage_optimizer | 622.776±1.917 | 3.774 | valid |
| redpajama_code | pecan_optimizer | 466.864±0.825 | 3.089 | valid |

DP ≥20% wins: 1/1 (100.0%).
Fully evaluable workloads: 1/1. Formal timeouts remain in the denominator and are reported as unavailable.
Minimum wins required for 30%: 1; additional wins needed: 0.
30% target: PASS.
