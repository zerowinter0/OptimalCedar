# Multimodal Running Example and Motivation Experiments

## Objective

Build one six-operator image--text preprocessing workload that can serve as
the running example in Sections 2--4. Use its optimizer-generated plans and
measured behavior to replace Figure 2 and Figure 5. The workload must expose
interactions among reordering, offloading, fusion, and parallelism without
injecting artificial delays or hand-editing optimizer statistics.

This task does not change PICO's core cost model. The new measurements may
motivate a later, separately reviewed extension for modality-specific work
units.

## Workload

### Data

Use COCO 2017 validation image--caption pairs, ordered by image ID and caption
ID. The first 500 records form a threshold-calibration split, the next 500 form
a pilot split, the following 3,000 form the formal plan-comparison split, and
the final 1,000 form the operator-scaling split. The fixture builder records
the source annotation checksum, ordered image IDs, caption IDs, image paths,
dimensions, and output checksum. Both optimizers and every formal repetition
consume the same materialized comparison split in the same order.

### Operators

The logical pipeline contains exactly six operators:

| ID | Operator | Consumed data | Resource | Kind |
| --- | --- | --- | --- | --- |
| `N` | text normalization | caption | CPU | map |
| `P` | perplexity filter | caption | CPU/SMP | filter |
| `Q` | image-sharpness filter | image pixels | CPU/SMP | filter |
| `A` | aesthetic filter | image | CUDA-Ray | filter |
| `C` | CLIP similarity filter | image and caption | CUDA-Ray | filter |
| `B` | BLIP matching filter | image and caption | CUDA-Ray | filter |

`N`, `P`, `C`, and `B` reuse the behavior of the existing LLaVA/Data-Juicer
workload. `A` uses Data-Juicer's default cached aesthetic model. `Q` computes
the variance of a Laplacian response over the decoded luminance image and
rejects blurred images. This deterministic predicate makes the CPU image work
explicitly pixel-dependent.

Candidate filter thresholds are derived only from the 500-record calibration
split. Quantiles are computed independently from each operator's raw score.
The initial configuration makes `P`, `Q`, `A`, `C`, and `B` retain 35%, 80%,
60%, 80%, and 80%, respectively. A bounded pilot search may vary the `P`
retention over {20%, 35%, 50%}, `Q` over {70%, 80%, 90%}, and `A` over
{40%, 60%, 80%}; `C` and `B` remain fixed at 80%. This gives 27 declared
configurations. The search changes thresholds, not operator implementations.

Each candidate is profiled and planned on the calibration data, then executed
twice in alternating optimizer order on the separate 500-record pilot split.
A candidate qualifies when both plans are valid and output-equivalent, the
joint plan uses at least two resource backends, its pilot median runtime is
lower than the staged plan's, and Cedar's native score ranks the staged plan
ahead of it. Among qualifying candidates, select the one with the largest
pilot speedup; break ties by the largest output cardinality and then by
lexicographic threshold tuple. This rule is fixed before pilot results are
observed.

The selected thresholds are frozen before the 3,000-record formal comparison.
No threshold is changed after inspecting formal plans or runtimes. All 27
pilot outcomes, including unsuccessful configurations, are archived. Neither
the formal comparison split nor the operator-scaling split participates in
parameter selection.

The semantic constraints are

```
N < P
Q < A
P < C < B
A < C < B
```

The text-only and image-only chains may be interleaved. Every legal order
therefore preserves the accepted record set.

### Expected physical contrast

The motivating contrast is:

```
staged: [Q]CPU -> [A]GPU -> [N,P]SMP -> [C,B]GPU
joint:  [N,P,Q]SMP -> [A,C,B]GPU
```

These layouts are hypotheses, not hard-coded plans. Figure 2 will display the
actual plans returned by `dp_two_stage_optimizer` and `dp_optimizer`. If the
measured plans differ from the hypothesis, the artifact records them unchanged.

## Experiment A: Joint versus Staged Planning

### Protocol

- Disable cache and enable reorder, fusion, offload, prefetch, and parallelism.
- Profile once and share that profile between both optimizers.
- Profile every Ray/SMP stage with one actor/process for ten seconds.
- Fix `W=8`, `CPU_BUDGET=64`, and one RTX A6000 GPU.
- Use the same fixture size and fixed thresholds for both optimizers.
- Materialize both generated plans before timed execution.
- Warm model weights and runtime actors outside the measured region.
- Execute three repetitions in round-robin order.
- Report individual repetitions, median execution time, and dispersion.
- Apply the existing 60-minute planning limit without reducing the workload.

All plans, profiles, metadata, logs, warmup records, raw timings, and summary
JSON files live under a new experiment-specific output directory. The launcher
uses `nohup` and writes one top-level log and PID file.

### Cedar cost-model audit

Replay both materialized plans through Cedar's native `Optimizer.calculate_cost`
using the same frozen profile. Record Cedar's score and PICO's objective score
alongside measured runtime. A Cedar ranking error is established only if
Cedar assigns the lower score to the plan with the higher median runtime.

The declared pilot search is the only permitted parameter selection. No score,
profile statistic, plan, formal runtime, resource budget, model revision, or
operator implementation may be modified to force a reversal. If no candidate
qualifies, archive the complete pilot grid before proposing a revised grid;
the formal comparison does not begin until one configuration passes the pilot
criteria.

### Figure 2

Produce a full-width vector figure with three aligned panels:

1. the six logical operators, their modality, and semantic constraints;
2. the actual staged and joint physical plans, using stage boxes colored by
   backend and fused blocks drawn as one stage;
3. measured execution time and Cedar score in separate small plots with their
   own units.

The caption distinguishes generated plans, measured runtime, and model scores.
It states the number of records and repetitions and avoids presenting an
unobserved expected layout as a result. The surrounding text identifies this
as a controlled motivating counterexample selected by the archived pilot
protocol, rather than evidence that Cedar misranks every multimodal workload.

## Experiment B: Operator Input Sensitivity

### Controlled inputs

Use the first 128 records of the held-out scaling split; these do not overlap
the plan-comparison fixture. Keep all 128 records at every point. Perform
operator/model warmup before measurement and measure seven repetitions per
point in an interleaved order.

- For `N` and `P`, use caption lengths of 16, 32, 64, 128, 256, and
  512 tokenizer tokens while holding the image fixed.
- For `Q` and `A`, use square decoded images with sides 128, 256, 512, and
  1,024 pixels while holding the caption fixed.
- For `C` and `B`, use the Cartesian product of 16, 32, 64, and 128 tokens
  with the four image-side values above. Model-side truncation and resizing
  remain enabled because they are part of the real operators.

Caption scaling repeats complete natural-language clauses rather than random
characters. Image scaling resamples the same source images, preventing image
content from changing across resolutions. Raw values, median, and interquartile
range are archived in CSV and JSON.

### Figure 5

Generate Figure 5 from archived measurements, never from embedded constants:

- two line panels for `N` and `P` versus caption tokens;
- two line panels for `Q` and `A` versus image megapixels;
- two heatmaps for `C` and `B` over the token--pixel grid.

All panels report per-record execution time. Line panels show medians with IQR
bands. Heatmaps share a normalized color scale only when their ranges make that
comparison meaningful; otherwise each includes an explicit scale. The plotting
script emits PDF and PNG from the same data file.

## Implementation Boundaries

New workload code belongs under
`evaluation/pipelines/multimodal_running_example/` and follows the current
`cedar_dataset.py`/operator-module structure. Launch, analysis, and plotting
code belongs under `evaluation/motivation_multimodal/`; immutable results belong
under `outputs/motivation_multimodal/`. Tests cover fixture
determinism, operator modality isolation, semantic equivalence of legal orders,
plan parsing, cost-score replay, and plot-data provenance.

The task does not alter Chapter 6 results, existing workloads, or PICO's cost
model. Paper TeX and Figure 2/Figure 5 assets are updated only after the raw
experiments pass their validity checks.

## Validation and Failure Handling

Before launching the formal run:

1. Resolve the three cached model snapshots offline at their frozen revisions:
   CLIP `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, BLIP
   `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`, and the aesthetic model
   `684098de3856fa4678bf800efc05635de5b6cde5`.
2. Run operator unit tests inside `optimalcedar-torch201-dev` after activating
   `/workspace/OptimalCedar/env`.
3. Run a small end-to-end smoke test that exercises both CPU/SMP and CUDA-Ray
   stages. A smoke test may use fewer records only to detect failures; its
   results are never plotted.
4. Confirm that staged and joint outputs contain identical record IDs.
5. Validate the generated plan resource assignments and global CPU/GPU budgets.

An experiment failure is recorded with its command, environment metadata, and
traceback. The launcher does not silently retry with smaller inputs, fewer
operators, or altered resource budgets. Threshold changes occur only through
the declared pilot grid and are recorded before the formal run.

## Acceptance Criteria

- The workload executes all six real operators on COCO image--caption records.
- The selected configuration comes from the archived 27-point pilot grid.
- Every displayed plan is optimizer-generated and semantically equivalent.
- The staged/joint comparison follows the frozen-profile, three-round protocol.
- Cedar scores are calculated by its native model from the same profile.
- Figure 2 is generated from archived plans and measurements.
- Figure 5 is generated from seven-run controlled operator measurements.
- Plotting scripts reproduce both PDF figures without network access.
- Any claimed Cedar ranking error and joint speedup are directly supported by
  the archived results.
