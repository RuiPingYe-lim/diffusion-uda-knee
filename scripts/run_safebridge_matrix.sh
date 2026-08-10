#!/usr/bin/env bash
# Stepwise attribution matrix for DSSR and TS-DNF.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-7}"
ARMS="${ARMS:-raw_only dssr_only stable_residual target_projected full}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-busi_to_breast_safebridge_v1}"

run_arm() {
  local arm="$1"
  local seed="$2"
  local fusion_mode="$3"
  echo "[SafeBridge] arm=${arm} seed=${seed} fusion=${fusion_mode}"
  EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${arm}_s${seed}" \
  SEED="${seed}" \
  FUSION_MODE="${fusion_mode}" \
    bash "${SCRIPT_ROOT}/run_unsb_safebridge_breast.sh"
}

for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    case "${arm}" in
      raw_only)
        run_arm "raw_only" "${seed}" "raw_only"
        ;;
      dssr_only)
        run_arm "dssr_only" "${seed}" "selected_feature"
        ;;
      stable_residual)
        run_arm "stable_residual" "${seed}" "stable_residual"
        ;;
      target_projected)
        run_arm "target_projected" "${seed}" "target_projected"
        ;;
      full)
        run_arm "full" "${seed}" "full"
        ;;
      *)
        echo "Unknown SafeBridge arm: ${arm}" >&2
        exit 2
        ;;
    esac
  done
done
