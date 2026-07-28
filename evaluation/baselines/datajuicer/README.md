# OptimalCedar Data-Juicer baselines

These configs run the native Data-Juicer implementation of the three
Data-Juicer-derived OptimalCedar workloads:

- `llava_pretrain.yaml`
- `redpajama_c4.yaml`
- `stackexchange.yaml`

The source recipes are from `datajuicer/data-juicer-hub` commit
`47fc34588b5d4258c13747cea37c2b63cf4e11b0`. The Data-Juicer implementation
itself is an ignored nested checkout pinned to
`bb3d88aac183cc22b6f816262a812a9e5d5abb57` by
`versions.lock.json` and `bootstrap_datajuicer.sh`; record both revisions in
every experiment result.

The C4 and StackExchange upstream recipes include a final
`document_simhash_deduplicator`. OptimalCedar's compared Cedar workloads omit
that global, cross-sample operator because it is outside Cedar's per-sample
optimizer search space. These configs omit it as well so the systems execute
the same logical workload. A separate experiment may include global
deduplication, but it must not be mixed into the throughput comparison.

`op_fusion: true` and `fusion_strategy: probe` enable Data-Juicer's native
probe-based filter ordering/fusion. Cache and tracing are disabled. The two
CPU-only workloads use the formal 64-CPU budget. LLaVA uses one process because
its CLIP and BLIP operators are GPU-backed, matching the Cedar workload's
single global GPU-operator concurrency.

Paths and parallelism can be overridden without editing the configs:

```bash
dj-process \
  --config /optimalcedar-configs/redpajama_c4.yaml \
  --dataset_path /absolute/input.jsonl \
  --export_path /absolute/output.jsonl \
  --np 64
```

Do not benchmark from a working tree whose `git status --short` is omitted
from the result metadata.
