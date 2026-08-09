# Formal cross-system results (W=8)

This directory combines all five Cedar optimizer results with native PyTorch
DataLoader, tf.data, Ray Data, Plumber, and FastFlow. The primary paper figure
selects twelve workloads: CommonVoice, SimCLR-v2, and WikiText-103 with both
non-cache and explicitly labeled cache variants, plus StackExchange, HackerNews,
PubMed Abstracts, USPTO Backgrounds, EuroParl, and RP-C4. Workloads are displayed in
ascending logical Cedar operator count. Every supported
cell has three round-robin repeats, an 8-worker setting, a 64-CPU budget, and
the exact output count in
`../optimizer/figures/latest_optimizer_data.tsv`. The primary figure,
`figures/optimizer_and_system_execution_time.pdf`, reports absolute execution
time in seconds. Every workload has an independent linear y-axis; the figure
uses neither a speedup axis nor a logarithmic axis. Each title reports the
logical Cedar operator count, excluding the source.

Unsupported cells and one-hour feasibility timeouts remain explicit in status/.
Superseded attempts are preserved by failure class under invalidated_attempts/;
the runner ignores that archive when resuming or plotting.
