LOCAL_PARALLELISM_SCALING_FACTOR = 0.8
OFFLOAD_THRESHOLD_FRAC = 0.05
FUSED_PIPE_NAME = "FusedPipe"

RAY_SUBMIT_BATCH_SIZE = 30
RAY_AVAILABLE_PARALLELISM = 32
# RAY_SUBMIT_BATCH_SCALING_FACTOR = 2000
RAY_SUBMIT_BATCH_SCALING_FACTOR = 2000000

# Threshold at which we forbid local workers due to serialization bottlenecks
LOCAL_PARALLELISM_THRESHOLD = 100000000

SMP_AVAILABLE_PARALLELISM = 8


# Threshold for samples/s at which local parallelism is forbidden
LOCAL_PARALLELISM_SAMPLES_PER_SEC_THRESHOLD = 100

# A single SMP item larger than this already saturates the local serialization
# budget at the minimum sample rate for which Cedar enables its throughput
# guard (100 MB/s / 100 samples/s).  Multi-operator SMP stages must account for
# both their input and output boundaries; checking only the input lets an
# expanding stage create an unmodelled multi-megabyte return transfer.
SMP_MAX_SERIALIZED_SAMPLE_SIZE = (
    LOCAL_PARALLELISM_THRESHOLD
    / LOCAL_PARALLELISM_SAMPLES_PER_SEC_THRESHOLD
)

# The same-machine Ray boundary microbenchmark in
# evaluation/chapter6_experiments/formal_results/raw/
# ray_boundary_microbenchmark.json measured about 10.4 GB/s for the
# incremental input+output boundary of a second Ray stage.  Use a slightly
# conservative round value in the optimizer's separable transport model.
RAY_STAGE_BOUNDARY_THROUGHPUT = 10_000_000_000

# Amdahl inversion becomes singular when an observed end-to-end speedup
# reaches the maximum attributable to one operator.  Retain the strong
# improvement signal without ever turning that operator into a zero-cost
# candidate.
MAX_UNIDENTIFIABLE_OPERATOR_SPEEDUP = 64.0

# Small isolated-stage slowdowns are dominated by queue/dispatch noise that
# disappears inside a fused stage. Larger regressions are retained as a real
# backend-execution penalty.
FUSED_BACKEND_SLOWDOWN_TOLERANCE = 1.25

# Conservative reduction in per-operator dispatch/call overhead when several
# Python operators execute inside one parallel stage. This is deliberately
# much smaller than Cedar's legacy IO-ratio discount of the full compute cost.
FUSED_OPERATOR_DISPATCH_DISCOUNT = 0.01
