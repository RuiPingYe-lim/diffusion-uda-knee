#!/usr/bin/env bash
# Minimal DA-BRF attribution matrix on the validated K=3 U1 joint baseline.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-7}"
ARMS="${ARMS:-identity fixed_scale08 norm_clip08 learned_task_only learned_full}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-busi_to_breast_trsc_dabrf_detfix_v2}"

run_arm() {
  local arm="$1"
  local seed="$2"
  local repair_mode="$3"
  local fixed_scale="$4"
  local radius_ratio="$5"
  local diagnostic_weight="$6"
  local progress_weight="$7"
  local radius_weight="$8"

  if [[ "${repair_mode}" == "identity" ]]; then
    # A control is valid only if it is the historical K3 implementation, not
    # a new model that happens to return the same pixels. This path therefore
    # instantiates no repair net, teacher, queue, optimizer, or extra CUDA work.
    echo "[DA-BRF] arm=${arm} seed=${seed} model=trsc_joint_sb (exact K3 control)"
    EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${arm}_s${seed}" \
    MODEL_NAME="trsc_joint_sb" \
    SEED="${seed}" \
    NUM_REFERENCES=3 \
    LAMBDA_TASK=1.0 \
      bash "${SCRIPT_ROOT}/run_unsb_trsc_joint_breast.sh"
    return
  fi

  echo "[DA-BRF] arm=${arm} seed=${seed} model=trsc_dabrf_joint_sb mode=${repair_mode}"
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${arm}_s${seed}" \
  MODEL_NAME="trsc_dabrf_joint_sb" \
  SEED="${seed}" \
  NUM_REFERENCES=3 \
  LAMBDA_TASK=1.0 \
  DABRF_MODE="${repair_mode}" \
  DABRF_FIXED_SCALE="${fixed_scale}" \
  DABRF_MAX_RADIUS_RATIO="${radius_ratio}" \
  LAMBDA_DABRF_DIAG="${diagnostic_weight}" \
  LAMBDA_DABRF_PROGRESS="${progress_weight}" \
  LAMBDA_DABRF_RADIUS="${radius_weight}" \
    bash "${SCRIPT_ROOT}/run_unsb_trsc_joint_breast.sh"
}

for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    case "${arm}" in
      identity)
        # Exact K3 joint control through the original model and checkpoint path.
        run_arm "identity" "${seed}" identity 1.0 1.0 0.0 0.0 0.0
        ;;
      fixed_scale08)
        # Test whether uniform attenuation explains any gain.
        run_arm "fixed_scale08" "${seed}" fixed_scale 0.8 1.0 0.0 0.0 0.0
        ;;
      norm_clip08)
        # Test whether a per-case residual bound explains any gain.
        run_arm "norm_clip08" "${seed}" norm_clip 1.0 0.8 0.0 0.0 0.0
        ;;
      learned_task_only)
        # Learn the residual gate from source CE only.
        run_arm "learned_task_only" "${seed}" learned 1.0 1.0 0.0 0.0 0.0
        ;;
      learned_full)
        # Calibrated diagnosis, target-style progress, and residual radius.
        run_arm "learned_full" "${seed}" learned 1.0 1.0 1.0 1.0 1.0
        ;;
      *)
        echo "Unknown DA-BRF arm: ${arm}" >&2
        exit 2
        ;;
    esac
  done
done
