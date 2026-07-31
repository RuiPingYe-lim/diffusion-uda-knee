#!/usr/bin/env bash
# Minimal matrix for K diversity and task-gradient attribution.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-7}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-busi_to_breast_trsc}"

run_arm() {
  local arm="$1"
  local seed="$2"
  local references="$3"
  local task_scale="$4"

  echo "[TRSC joint] arm=${arm} seed=${seed} K=${references} task_scale=${task_scale}"
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${arm}_s${seed}" \
  SEED="${seed}" \
  NUM_REFERENCES="${references}" \
  LAMBDA_TASK="${task_scale}" \
    bash "${SCRIPT_ROOT}/run_unsb_trsc_joint_breast.sh"
}

for seed in ${SEEDS}; do
  # K=1 versus K=3 isolates reference diversity under equal per-case weight.
  run_arm "k1_joint" "${seed}" 1 1.0

  # This arm still trains TRSC with its native losses, but blocks classifier CE
  # at the candidate image so the task-gradient contribution is isolated.
  run_arm "k3_no_task_gradient" "${seed}" 3 0.0

  # Requested canonical path: three unfiltered U1 views and full task gradient.
  run_arm "k3_joint" "${seed}" 3 1.0
done
