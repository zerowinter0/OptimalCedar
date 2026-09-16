#!/bin/bash
# Strict "every planner gets Cedar's W" control experiment.
#
# Protocol
#   * W is fixed to Cedar's own choice on these five workloads: W=32
#     (64 local CPUs; Cedar's rule asks for one core per worker plus one
#     reserved runtime core, i.e. 64 // 2 = 32).
#   * ``CEDAR_WORKER_SEARCH_SET=32`` + ``CEDAR_DP_WORKER_LADDER=0`` pin the
#     planner-side candidate set to exactly W=32, so no planner can answer the
#     protocol with a cheaper worker count.
#   * ``FIXED_W=32`` forces the *executed* worker count of every plan.
#   * ``CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER=0`` removes the extra reserved
#     core, so at W=32 each worker still owns one core for an SMP slot
#     (32 workers + 32 SMP processes = 64 cores) and two Ray actors per worker
#     fit the remote pool.  Without it the residual per-worker budget at W=32
#     is zero, per-operator SMP becomes *unavailable*, and the only way an
#     optimizer could still reach a parallel stage would be to pick a smaller
#     W -- i.e. the worker count would silently turn into the confound this
#     control is supposed to remove.
#   * SMP, Ray placement and Ray widths stay available to every planner: the
#     four systems differ only in plan structure and in the prices they attach
#     to it, never in the worker count.
cd /home/xieruiyang/OptimalCedar

run() {
  local w=$1 n=$2
  echo "########## $w samples=$n strict-same-W=32 $(date +%H:%M:%S)"
  FIXED_W=32 CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER=0 \
    CEDAR_WORKER_SEARCH_SET=32 CEDAR_DP_WORKER_LADDER=0 \
    ACTOR_READY_TIMEOUT=900 WORKER_READY_TIMEOUT=1800 \
    SPEC=600 PICO_PLAN_BUDGET=600 OUT=/tmp/screen_samew_fixedw \
    SELECTIVITY_SEC=45 RESULTS_DIR=outputs/screen_samew_fixedw \
    OPTIMIZERS="optimizer plumber_optimizer dp_optimizer simple_dp_optimizer" \
    timeout 7200 bash tmp_analysis/screen_workload.sh "$w" "$n"
}

run commonvoice 300
run wikitext103 800
run simclrv2 9472
run simclrv2_cache 2000
run coco 20000
echo STRICT_DONE
