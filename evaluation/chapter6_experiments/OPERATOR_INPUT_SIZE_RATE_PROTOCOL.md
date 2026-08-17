# Operator input-size versus execution-rate experiment

This experiment uses the existing `StackExchangeFeature`, which implements
Data-Juicer's `redpajama-pile-stackexchange-refine` per-record recipe. No
operator or order is redefined in the benchmark.

1. Up to 5,000 real source records traverse the original 19-operator pipeline.
2. Immediately before every operator, the benchmark snapshots the legal value
   produced by all original prefix operators. Rejected records do not reach
   later operators.
3. Inputs are stratified into half-octave Cedar logical-size buckets. At most
   twelve representative real inputs are selected per operator.
4. Each point reports the median of seven wall-clock repeats. Only the operator
   callable is timed; source I/O, prefix execution, and snapshot restoration are
   excluded. Interquartile ranges are retained in raw data and plotted.
5. The process is pinned to one CPU because one profiled Cedar Ray/SMP worker is
   allocated one CPU. Library thread counts are fixed to one.

The y-axis is records per second, not bytes per second. Thus a per-data
operator is expected to slow as record bytes increase, whereas a per-record
operator should be comparatively size-insensitive. The figure separates the
16 explicit per-data operators from the 3 explicit per-record operators.
