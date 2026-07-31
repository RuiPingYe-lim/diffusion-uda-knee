#!/usr/bin/env bash
# Minimal DA-BRF attribution matrix on the validated K=3 U1 joint baseline.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-7}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-busi_to_breast_trsc_dabrf}"

run_arm() {
  local arm="$1"
  local seed="$2"
  local repair_mode="$3"
  local fixed_scale="$4"
  local radius_ratio="$5"
  local diagnostic_weight="$6"
  local progress_weight="$7"
  local radius_weight="$8"

  echo "[DA-BRF] arm=${arm} seed=${seed} mode=${repair_mode}"
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${arm}_s${seed}" \
  SEED="${seed}" \
  NUM_REFERENCES=3 \
  LAMBDA_TASK=1.0 \
  DABRF_MODE="${repair_mode}" \
  DABRF_FIXED_SCALE="${fixed_scale}" \
  DABRF_MAX_RADIUS_RATIO="${radius_ratio}" \
  LAMBDA_DABRF_DIAG="${diagnostic_weight}" \
  LAMBDA_DABRF_PROGRESS="${progress_weight}" \
  LAMBDA_DABRF_RADIUS="${radius_weight}" \
    bash "${SCRIPT_ROOT}/run_unsb_trsc_dabrf_breast.sh"
}

for seed in ${SEEDS}; do
  # Exact K3 joint control through the same model and checkpoint path.
  run_arm "identity" "${seed}" identity 1.0 1.0 0.0 0.0 0.0

  # Non-learned controls test whether uniform attenuation explains any gain.
  run_arm "fixed_scale08" "${seed}" fixed_scale 0.8 1.0 0.0 0.0 0.0
  run_arm "norm_clip08" "${seed}" norm_clip 1.0 0.8 0.0 0.0 0.0

  # This arm learns the residual gate from source CE only.
  run_arm "learned_task_only" "${seed}" learned 1.0 1.0 0.0 0.0 0.0

  # Canonical DA-BRF: calibrated diagnosis, target-style progress, and radius.
  run_arm "learned_full" "${seed}" learned 1.0 1.0 1.0 1.0 1.0
done
