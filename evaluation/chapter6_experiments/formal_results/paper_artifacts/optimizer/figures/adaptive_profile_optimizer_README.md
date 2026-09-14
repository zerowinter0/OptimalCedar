# Adaptive-profile optimizer comparison

This figure substitutes only the latest adaptive-profile PICO measurements;
all other optimizer cells remain immutable formal comparison data.

- Seven text/image baseline workloads: `outputs/chapter6_experiments/formal_seven_optimizer_matrix`
- GenerateVideo 5,000-sample baselines: `outputs/chapter6_experiments/formal_seven_optimizer_matrix_general_video_5000`
- CommonVoice and SimCLR-cache baselines: `outputs/chapter6_experiments/gpu_aware_formal_matrix`
- Adaptive-profile PICO replacement: `outputs/chapter6_experiments/adaptive_all_ten_dp_validation`

Every successful cell uses the same workload-specific sample count, W=8,
and three measured executions. StackExchange PICO is shown as a unified-task
timeout because planning plus the first execution exceeded one hour; no
execution time is imputed.

The PDF/PNG/SVG files are generated from the exported JSON/TSV data by
`evaluation/chapter6_experiments/plot_adaptive_profile_optimizer_matrix.py`.
The Draw.io file wraps the generated SVG for editable paper layout.
