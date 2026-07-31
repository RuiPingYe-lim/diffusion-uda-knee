#!/usr/bin/env bash
# Train TRSC K-view joint adaptation with DA-BRF U1 residual repair.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="trsc_dabrf_joint_sb" \
  bash "${SCRIPT_ROOT}/run_unsb_trsc_joint_breast.sh"
