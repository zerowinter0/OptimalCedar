# Data-Juicer diverse-workload result

All selected results use W=8, CPU budget 64, cache disabled, one shared profile per workload, and three measured repetitions. A win means DP is at least 1.20x faster than the fastest available non-DP optimizer.

Hub operator counts include every entry in the official recipe, including global deduplicators. Cedar operator counts exclude the source and omitted cross-record deduplicators, but include fixed parse, path-resolution, synchronization, and projection adapters actually executed by Cedar.

| workload | scenario | modality | Hub ops | Cedar ops | samples | DP (s) | best other (s) | speedup | selected |
|---|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `pile_europarl` | parliamentary proceedings | text | 17 | 19 | 2500 | 1227.819 | 1836.513 | 1.496x | yes |
| `pile_hackernews` | online discussion | text | 16 | 18 | 20000 | 467.732 | 1292.086 | 2.762x | yes |
| `pile_pubmed_abstracts` | biomedical abstracts | text | 17 | 19 | 20000 | 201.517 | 298.980 | 1.484x | no |
| `pile_uspto_backgrounds` | patent background | text | 17 | 19 | 20000 | 338.417 | 829.252 | 2.450x | yes |
| `alpaca_cot` | instruction/reasoning tuning | text | 7 | 8 | 65000 | 124.849 | 120.372 | 0.964x | yes |
| `llava_pretrain` | image-caption refinement | image-text | 13 | 16 | 5000 | 84.878 | 80.114 | 0.944x | no |
| `general_video_refine` | video-text quality refinement | video-text | 7 | 10 | 7500 | 2153.033 | 1979.108 | 0.919x | yes |
| `video_self_evolution` | video self-evolution filtering | video-text | 5 | 8 | N/A | N/A | N/A | N/A | no (formal result incomplete) |
| `redpajama_code` | source-code refinement | code | 15 | 17 | 20000 | 274.699 | 460.249 | 1.675x | yes |
| `redpajama_arxiv` | scientific long documents | text | 16 | 18 | N/A | N/A | N/A | N/A | no (formal result incomplete) |

Selected: 6 workloads; DP ≥1.20x wins: 4; non-wins: 2 (33.3%).
Coverage: 6 scenarios, 3 modalities, Hub operator counts 7--17, and Cedar operator counts 8--19.

Every non-selected screened outcome remains in the JSON ledger; selection does not erase negative evidence.
