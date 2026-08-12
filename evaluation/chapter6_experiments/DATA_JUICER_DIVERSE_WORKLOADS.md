# Data-Juicer diverse-workload study

This document records the workload search used to broaden the Chapter 6
evaluation beyond structurally similar 15--18-operator text-refinement
pipelines. It is a screening ledger, not a rule for silently discarding
unfavourable measurements.

## Source snapshot

- Repository: `https://github.com/datajuicer/data-juicer-hub.git`
- Local checkout: `data-juicer-hub/`
- Revision: `47fc34588b5d4258c13747cea37c2b63cf4e11b0`
- Revision date: 2026-02-11

The checkout was the upstream `main` revision observed when this study started
on 2026-08-12. Operator semantics use the separately frozen local
`data-juicer` checkout. Cross-record deduplicators are outside Cedar's
per-record reorder/fusion search and are excluded consistently with
`DATA_JUICER_CANDIDATE_PROTOCOL.md`.

## Recipe classification

Operator counts refer to entries in each recipe's `process` list and include
cross-record deduplication when present.

| Family | Hub recipes | Operator-count range | Distinguishing work |
|---|---:|---:|---|
| General pre-training text refinement | 18 | 15--18 | normalization, statistical filters, language ID, perplexity, SimHash |
| Source-code refinement | 3 | 1--15 | copyright/link cleanup and code-specific line filters |
| Instruction/post-tuning refinement | 2 | 6--7 | short instruction-response filtering and deduplication |
| Reproduced corpus construction | 5 | 1--14 | BLOOM multilingual and RedPajama domain processing |
| Image-text refinement/synthesis | 2 | 4--13 | metadata, CLIP/BLIP, diffusion and MLLM generation |
| Video refinement | 3 | 2--7 | frame sampling, vision scoring, motion and duration analysis |

Counting every Pile and RedPajama domain as a separate system workload would
exaggerate diversity: most share the same fixed normalization prefix and the
same reorderable filter set. Text domains are retained only when they add a
different application scenario or materially different selectivity/cost
behaviour.

## Candidate pool and Cedar status

The pool spans task, modality, and pipeline size. Existing formal results are
reused only when they use W=8, CPU budget 64, a shared profile, and three
successful repetitions.

| Candidate | Family/scenario | Cedar operators | Initial status |
|---|---|---:|---|
| `alpaca_cot` | instruction/post-tuning | 8 | implemented; fresh screening |
| `redpajama_arxiv` | scientific documents | 18 | implemented; fresh screening |
| `redpajama_code` | source code | 18 | implemented; formal result available |
| `pile_europarl` | parliamentary text | 19 | implemented; formal result available |
| `pile_hackernews` | social/news discussion | 18 | implemented; formal result available |
| `pile_pubmed_abstracts` | biomedical abstracts | 19 | implemented; formal result available |
| `pile_uspto_backgrounds` | patent/legal-technical text | 19 | implemented; formal result available |
| `llava_pretrain` | image-text refinement | 18 | implemented; formal result available |
| `general_video_refine` | video-text filtering | 10 | implemented; corrected DP result available |
| `video_self_evolution` | video self-evolution filtering | 8 | implemented as a same-family replacement candidate; queued screening |
| `bloom_oscar` | multilingual web text | 17 | implemented, but the official OSCAR source is unavailable locally; fallback only |

## Screening and final-selection rule

1. Reuse protocol-compatible formal measurements without rerunning them.
2. Profile and screen `alpaca_cot` and `redpajama_arxiv`, which add scenarios
   absent from the existing formal candidate batch.
3. Use one full-data execution per optimizer for screening. The input prefix,
   profile, W=8 resource signature, cache setting, and optimizer flags are
   identical. The comparison includes original Cedar, Data-Juicer-style,
   DP-Cedar, joint DP, two-stage DP, and Pecan. Screening reduces only
   repetitions, not semantics or resources.
4. Select six to eight workloads that maximize family coverage. At most 40%
   of the final set may fail to make DP at least 20% faster than the fastest
   non-DP optimizer. All screened outcomes remain in this ledger.
5. A selected new workload receives three round-robin repetitions. Its sample
   count must be large enough for stable timing and small enough that every
   successful optimizer execution completes within 3,600 seconds. Source
   exhaustion and timeouts are reported rather than repaired by duplication or
   recipe-threshold changes.
6. A cost-model change is admissible only when it represents an explicit,
   workload-independent execution property and passes existing-plan regression
   checks unless a workload opts into new metadata.

## Existing formal evidence

Speedup is fastest non-DP execution time divided by DP execution time.

| Workload | DP speedup | At least 20% faster? |
|---|---:|---:|
| `pile_europarl` | 1.496x | yes |
| `pile_hackernews` | 2.762x | yes |
| `pile_pubmed_abstracts` | 1.507x | yes |
| `pile_uspto_backgrounds` | 2.450x | yes |
| `redpajama_code` | 0.906x | no |
| `llava_pretrain` | 0.883x | no |
| `general_video_refine` | 0.980x | no; corrected DP is within 2.1% of Data-Juicer |

The video comparison uses the corrected three-run DP mean (2926.480 s) and
the existing three-run Data-Juicer mean (2866.715 s). A fresh complete
round-robin matrix is required before using that cross-run comparison as a
paper claim.

## Fresh screening ledger

Both profiles use one local worker, one Ray actor or SMP process per stage,
and ten seconds per measured stage.

| Workload | Samples | Profile | Screening outcome | Decision |
|---|---:|---|---|---|
| `alpaca_cot` | 20,000 | complete | DP 29.074 s; fastest competitor 28.573 s (0.983x) | below 1.20x; retain as a negative screening result |
| `redpajama_arxiv` | 20,000 | complete | DJ timed out after 3,600 s at 3,367 outputs | scale infeasible; no optimizer-speedup claim |
| `redpajama_code` | 20,000 | reused validated formal profile | fresh additive-objective screening running | pending |
| `redpajama_arxiv` | 2,500 | reused fresh profile | queued after Code | pending |

The 20,000-output ArXiv attempt was stopped after the completed Data-Juicer
timeout established that the scale cannot satisfy the one-hour rule. The
partially observed DP-Cedar run (1,673 outputs in approximately 1,933 seconds)
is not used as performance evidence. The replacement uses the same immutable
source prefix and recipe, changes only the requested retained-output count,
and covers roughly 140 MiB of long scientific text at the source file's mean
record size. This is large enough to be a substantive execution while leaving
headroom under the threshold according to the completed rate measurement.

The final set and aggregate win fraction will be recorded after the queued
screening runs finish. In addition, original Cedar and DP two-stage plans
behind reused Pile results are being re-audited with the current 3,600-second
plan threshold; historical 300-second unavailability records alone are not
treated as sufficient under the current protocol. Missing original-Cedar and
two-stage evidence for the reused LLaVA matrix is supplemented under the same
threshold before that workload can be considered valid.

The final report is generated by
`analyze_datajuicer_diverse_workloads.py`. It refuses to run until the
60-minute Cedar audit and every required three-repeat result are complete. The
selection rule keeps only as many of the four frozen, scenario-distinct Pile
results as are required alongside the instruction/video representatives to
satisfy the six-workload and 40% gates. Pile candidates are ranked by measured
DP speedup, while all four remain in the ledger. A newly formalized Code or
ArXiv workload is added only if it clears 1.20x; LLaVA is admitted only when
its image-text coverage does not push non-wins above 40%.
All candidates, including screened non-wins, remain in the machine-readable
ledger even when they are not in the final six-to-eight-workload subset.

The Hub's `data-juicer-sandbox-self-evolution.yaml` is also screened as the
predeclared same-family replacement for the neutral general-video result. Its
five filters (NSFW, frame-text similarity, motion, aesthetics, duration) and
all official arguments are preserved; Cedar adds only parse/root/projection
pipes. It reuses the frozen MSR-VTT source, profiles independently, screens
2,000 retained outputs, and receives a 5,000-output three-repeat matrix only
if the screen reaches 1.20x. Profile failure or source infeasibility is kept as
a negative feasibility outcome rather than repaired by changing thresholds.
The frozen source contains 200,000 caption records over 10,000 distinct local
videos (approximately 2.1 GiB of video data), so the 5,000-output formal target
is not a tiny fixture despite being smaller than text-pipeline record counts.
