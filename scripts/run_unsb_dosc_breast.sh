#!/usr/bin/env bash
set -euo pipefail

: "${UNSB_ROOT:?Set UNSB_ROOT to the upstream cyclomon/UNSB checkout}"
TRSC_DATA_ROOT="${TRSC_DATA_ROOT:-${DOSC_DATA_ROOT:-}}"
TRSC_TEACHER="${TRSC_TEACHER:-${DOSC_TEACHER:-}}"
: "${TRSC_DATA_ROOT:?Set TRSC_DATA_ROOT to the target-reference UNSB dataset}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-busi_to_breast_trsc}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${UNSB_ROOT}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${UNSB_ROOT}/results}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_EPOCHS="${N_EPOCHS:-15}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-15}"
NUM_TEST="${NUM_TEST:-99999}"
STYLE_DIM="${STYLE_DIM:-128}"
CIDP_QUEUE_SIZE="${CIDP_QUEUE_SIZE:-128}"
CIDP_MIN_PER_CLASS="${CIDP_MIN_PER_CLASS:-8}"
CIDP_RANK_WEIGHT="${CIDP_RANK_WEIGHT:-1.0}"
CIDP_RANK_TOLERANCE="${CIDP_RANK_TOLERANCE:-0.10}"
ENABLE_PROJECTION="${ENABLE_PROJECTION:-0}"
LAMBDA_DIAG="${LAMBDA_DIAG:-0.0}"
LAMBDA_DOMAIN="${LAMBDA_DOMAIN:-0.10}"
LAMBDA_INSTANCE="${LAMBDA_INSTANCE:-0.10}"
LAMBDA_RECON="${LAMBDA_RECON:-1.00}"
LAMBDA_SAFE="${LAMBDA_SAFE:-0.0}"

if [[ -z "${TRSC_TEACHER}" && ! "${LAMBDA_SAFE}" =~ ^0+([.]0+)?$ ]]; then
  echo "TRSC_TEACHER is required when LAMBDA_SAFE is non-zero" >&2
  exit 2
fi
if [[ "${ENABLE_PROJECTION}" != "0" && "${ENABLE_PROJECTION}" != "1" ]]; then
  echo "ENABLE_PROJECTION must be 0 or 1" >&2
  exit 2
fi

projection_args=()
if [[ "${ENABLE_PROJECTION}" == "1" ]]; then
  projection_args+=(--dosc_enable_projection)
fi

export PYTHONPATH="${UNSB_ROOT}:${PYTHONPATH:-}"

python "${UNSB_ROOT}/train.py" \
  --dataroot "${TRSC_DATA_ROOT}" \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --model trsc_sb \
  --dataset_mode trsc_unaligned \
  --direction AtoB \
  --dosc_source_manifest "${TRSC_DATA_ROOT}/trainA_manifest.csv" \
  --dosc_teacher_path "${TRSC_TEACHER}" \
  --dosc_style_dim "${STYLE_DIM}" \
  --dosc_noise_ratio 0.10 \
  --lambda_DOSC_diag "${LAMBDA_DIAG}" \
  --lambda_DOSC_domain "${LAMBDA_DOMAIN}" \
  --lambda_DOSC_instance "${LAMBDA_INSTANCE}" \
  --lambda_DOSC_recon "${LAMBDA_RECON}" \
  --lambda_DOSC_safe "${LAMBDA_SAFE}" \
  --dosc_safe_mode cidp \
  --dosc_safe_tolerance 0.10 \
  --dosc_safe_warmup_steps 1000 \
  --dosc_cidp_queue_size "${CIDP_QUEUE_SIZE}" \
  --dosc_cidp_min_per_class "${CIDP_MIN_PER_CLASS}" \
  --dosc_cidp_rank_weight "${CIDP_RANK_WEIGHT}" \
  --dosc_cidp_rank_tolerance "${CIDP_RANK_TOLERANCE}" \
  --mode sb \
  --lambda_SB 1.0 \
  --lambda_NCE 1.0 \
  --batch_size "${BATCH_SIZE}" \
  --n_epochs "${N_EPOCHS}" \
  --n_epochs_decay "${N_EPOCHS_DECAY}" \
  --seed "${SEED}" \
  "${projection_args[@]}" \
  --gpu_ids "${GPU_IDS}"

python "${UNSB_ROOT}/test.py" \
  --dataroot "${TRSC_DATA_ROOT}" \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --model trsc_sb \
  --dataset_mode trsc_unaligned \
  --direction AtoB \
  --dosc_style_dim "${STYLE_DIM}" \
  --dosc_noise_ratio 0.0 \
  --mode sb \
  --phase test \
  --epoch latest \
  --eval \
  --serial_batches \
  --num_test "${NUM_TEST}" \
  "${projection_args[@]}" \
  --gpu_ids "${GPU_IDS}"
