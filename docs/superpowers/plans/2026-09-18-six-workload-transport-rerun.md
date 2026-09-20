# Six-workload transport rerun implementation plan

Goal: run simple_dp_boundary, simple_dp_workers_boundary and
simple_dp_workers_width_boundary on all six workloads with valid new profiles.
Architecture: extend layered profiling with opt-in SMPActor/Queue calibration
using immutable snapshots of real legal boundary objects; retain the current
objective and resource accounting. Use the existing sequential matrix runner.
Constraints: Docker and active venv; no prior output changes; local/Ray CPU
budgets 64/64; Ray actors exclusively remote; automatic W; one round; 7200-second
cell limit. Non-cache workloads disable cache; cache workload prewarms and
excludes warmup from official timing. Sizes: SimCLR 9469, CV15000, COCO5000,
LLaVA1000, StackExchange2000. Keep three repeats in calibration curves.

Steps:
1. Add a test using real ndarray and dictionary snapshots and an actual
   SMPActor queue pair; ensure measured counts, bytes and throughput are positive.
2. Implement cedar/client/smp_transport_profiler.py: independent driver/actor
   pairs, bounded queue window, startup/warmup excluded, synchronized repeated
   one-second measurements, raw observations retained, strict cleanup.
3. Attach the curve during opt-in layered dataset profiling; preserve
   nonmonotonic points and validate queue configuration with existing lookup.
4. Create an isolated dated queue runner selecting only the three requested
   methods. Regenerate shared layered profiles and supplemental curves for
   every workload. Write metadata, hashes, plans, cache warmup and results.
5. Run targeted tests, compile changed files, check diff, start with nohup and
   verify the first profiling process is live. Do not wait for long experiments.
