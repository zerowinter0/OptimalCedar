#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

readonly WORKFLOW_ROOT="/workspace/OptimalCedar/outputs/motivation_multimodal/simclrv2_multimodal_gpu_profile_fixed_20260909"
readonly WORKFLOW_GATE_ROOT="${WORKFLOW_ROOT}/gate"
readonly WORKFLOW_RUNTIME_ROOT="${WORKFLOW_ROOT}/runtime"

mkdir -p "${WORKFLOW_ROOT}" "${WORKFLOW_GATE_ROOT}" "${WORKFLOW_RUNTIME_ROOT}"
rm -f "${WORKFLOW_ROOT}/COMPLETE" "${WORKFLOW_ROOT}/FAILED"
trap 'touch "${WORKFLOW_ROOT}/FAILED"' ERR

echo "[workflow] profiling and plan gate"
OUTPUT_ROOT="${WORKFLOW_GATE_ROOT}" \
    bash evaluation/pipelines/simclrv2_multimodal/run_plan_gate.sh \
    > "${WORKFLOW_GATE_ROOT}/workflow.log" 2>&1

echo "[workflow] gate passed; running three-repeat round robin"
GATE_ROOT="${WORKFLOW_GATE_ROOT}" OUTPUT_ROOT="${WORKFLOW_RUNTIME_ROOT}" \
    bash evaluation/pipelines/simclrv2_multimodal/run_runtime.sh \
    > "${WORKFLOW_RUNTIME_ROOT}/workflow.log" 2>&1

touch "${WORKFLOW_ROOT}/COMPLETE"
rm -f "${WORKFLOW_ROOT}/FAILED"
echo "[workflow] complete"
