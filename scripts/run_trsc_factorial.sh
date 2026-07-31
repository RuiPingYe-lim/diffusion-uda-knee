#!/usr/bin/env bash
# Train the minimal target-reference conditioning factorial.
#
# Arms:
#   core            target-reference conditioning only
#   cidp            core + output-level CIDP
#   legacy_controls cidp + the rejected projection/GRL leakage controls
#
# Run one screening seed first:
#   SEEDS="7" bash scripts/run_trsc_factorial.sh
# Confirm only after the source-validation and downstream protocol is frozen:
#   SEEDS="7 16 42" bash scripts/run_trsc_factorial.sh
set -euo pipefail

: "${UNSB_ROOT:?Set UNSB_ROOT to the upstream cyclomon/UNSB checkout}"
TRSC_DATA_ROOT="${TRSC_DATA_ROOT:-${DOSC_DATA_ROOT:-}}"
TRSC_TEACHER="${TRSC_TEACHER:-${DOSC_TEACHER:-}}"
: "${TRSC_DATA_ROOT:?Set TRSC_DATA_ROOT to the target-reference UNSB dataset}"
: "${TRSC_TEACHER:?Set TRSC_TEACHER to the render-robust TorchScript teacher}"

SEEDS="${SEEDS:-7}"
ARMS="${ARMS:-core cidp legacy_controls}"
RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_unsb_trsc_breast.sh"

run_arm() {
  local arm="$1"
  local seed="$2"
  local projection=0
  local lambda_diag=0.0
  local lambda_safe=0.0

  case "${arm}" in
    core)
      ;;
    cidp)
      lambda_safe=0.50
      ;;
    legacy_controls)
      projection=1
      lambda_diag=0.10
      lambda_safe=0.50
      ;;
    *)
      echo "Unknown arm: ${arm}" >&2
      exit 2
      ;;
  esac

  local experiment_name="busi_to_breast_trsc_${arm}_s${seed}"
  local generator="${CHECKPOINTS_DIR:-${UNSB_ROOT}/checkpoints}/${experiment_name}/latest_net_G.pth"
  local style="${CHECKPOINTS_DIR:-${UNSB_ROOT}/checkpoints}/${experiment_name}/latest_net_S.pth"
  local result="${RESULTS_DIR:-${UNSB_ROOT}/results}/${experiment_name}/test_latest/images"
  if [[ -f "${generator}" && -f "${style}" && -d "${result}" ]]; then
    echo "skip ${experiment_name}: checkpoint and translations already exist"
    return
  fi

  echo "### ${experiment_name}: projection=${projection} diag=${lambda_diag} safe=${lambda_safe}"
  EXPERIMENT_NAME="${experiment_name}" \
  SEED="${seed}" \
  ENABLE_PROJECTION="${projection}" \
  LAMBDA_DIAG="${lambda_diag}" \
  LAMBDA_SAFE="${lambda_safe}" \
  TRSC_DATA_ROOT="${TRSC_DATA_ROOT}" \
  TRSC_TEACHER="${TRSC_TEACHER}" \
  bash "${RUNNER}"
}

for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    run_arm "${arm}" "${seed}"
  done
done

echo "TRSC_FACTORIAL_DONE"
