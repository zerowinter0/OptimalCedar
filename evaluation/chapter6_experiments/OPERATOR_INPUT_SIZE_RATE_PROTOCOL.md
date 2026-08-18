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

The x-axis is input bytes per record and the y-axis is records per second;
both use base-2 logarithmic scales. All 19 operator curves are shown together
without semantic class labels or a per-operator legend. The figure is intended
to expose the empirical split--some rates decrease with input size while
others remain approximately stable--before the two scaling classes are
introduced in the paper text.

The companion relative-rate figure divides every operator's rates and
interquartile bounds by that operator's median rate at its smallest observed
legal input. Consequently, each curve begins at 1 while later points may
exceed 1. Its y-axis is also logarithmic with base 2, so the principal ticks
are `1`, `2^-1`, `2^-2`, and so on; zero itself cannot appear on a logarithmic
axis.
