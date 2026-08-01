#!/usr/bin/env bash
# Train matched two-view classifiers for the TRSC translator factorial.
#
# This stage never loads target labels. It writes per-epoch target probabilities
# that are unlocked by eval_trsc_downstream.py only after the full run matrix is
# complete.
set -euo pipefail

: "${TARGET_CSV:?Set TARGET_CSV to the target inference manifest}"
: "${SRC_TEST_CSV:?Set SRC_TEST_CSV to the independent BUSI test manifest}"

MANIFEST="${MANIFEST:-}"
MANIFEST_TEMPLATE="${MANIFEST_TEMPLATE:-}"
if [[ -n "${MANIFEST}" && -n "${MANIFEST_TEMPLATE}" ]]; then
  echo "Set only one of MANIFEST or MANIFEST_TEMPLATE" >&2
  exit 2
fi
if [[ -z "${MANIFEST}" && -z "${MANIFEST_TEMPLATE}" ]]; then
  echo "Set MANIFEST, or MANIFEST_TEMPLATE containing {seed}" >&2
  exit 2
fi
if [[ -n "${MANIFEST_TEMPLATE}" && "${MANIFEST_TEMPLATE}" != *"{seed}"* ]]; then
  echo "MANIFEST_TEMPLATE must contain the literal token {seed}" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-7 16 42}"
DEFAULT_CONDITIONS="raw core_U1 cidp_U1 legacy_controls_U1"
DEFAULT_CONDITIONS+=" core_U5 cidp_U5 legacy_controls_U5"
CONDITIONS="${CONDITIONS:-${DEFAULT_CONDITIONS}}"
EPOCHS="${EPOCHS:-50}"
RUNS_ROOT="${RUNS_ROOT:-/root/autodl-tmp/breast/trsc_downstream/runs}"
SEALED_ROOT="${SEALED_ROOT:-/root/autodl-tmp/breast/trsc_downstream/sealed}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${RUNS_ROOT}" "${SEALED_ROOT}"

for seed in ${SEEDS}; do
  seed_manifest="${MANIFEST}"
  if [[ -n "${MANIFEST_TEMPLATE}" ]]; then
    seed_manifest="${MANIFEST_TEMPLATE//\{seed\}/${seed}}"
  fi
  if [[ ! -f "${seed_manifest}" ]]; then
    echo "manifest not found for seed ${seed}: ${seed_manifest}" >&2
    exit 2
  fi
  for condition in ${CONDITIONS}; do
    run_dir="${RUNS_ROOT}/${condition}_s${seed}"
    if [[ -f "${run_dir}/config.json" && -f "${run_dir}/history.csv" ]]; then
      echo "skip ${condition} seed ${seed}: run is complete"
      continue
    fi
    echo "### downstream ${condition} seed ${seed}"
    "${PYTHON}" "${SCRIPT_ROOT}/train_da_route.py" \
      --cond_col "${condition}" \
      --run_name "${condition}" \
      --manifest "${seed_manifest}" \
      --target_csv "${TARGET_CSV}" \
      --src_test_csv "${SRC_TEST_CSV}" \
      --out_dir "${RUNS_ROOT}" \
      --sealed_dir "${SEALED_ROOT}" \
      --epochs "${EPOCHS}" \
      --seed "${seed}" \
      > "${RUNS_ROOT}/${condition}_s${seed}.log" 2>&1
    tail -n 2 "${RUNS_ROOT}/${condition}_s${seed}.log"
  done
done

echo "TRSC_DOWNSTREAM_FACTORIAL_DONE"
