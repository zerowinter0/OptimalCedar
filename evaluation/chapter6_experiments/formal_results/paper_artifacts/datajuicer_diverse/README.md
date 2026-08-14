# Formal Data-Juicer diverse-workload artifacts

This directory is generated only after every selected workload has complete W=8/CPU-64, three-repeat, 3,600-second evidence for all six optimizers and the reused Pile workloads pass the current 3,600-second original-Cedar and DP two-stage plan audit.

Selected workloads: `pile_hackernews`, `pile_uspto_backgrounds`, `pile_europarl`, `alpaca_cot`, `general_video_refine`, `redpajama_code`.

`final_selection.json` is authoritative. `screening/` retains non-selected outcomes; `profiles/` and `evidence/` contain the exact inputs to the aggregate; `source_metadata/` records frozen dataset provenance, size, and content hashes; `MANIFEST.tsv` provides size and SHA-256 for every archived file.
