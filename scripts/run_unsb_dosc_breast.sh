#!/usr/bin/env bash
set -euo pipefail

: "${UNSB_ROOT:?Set UNSB_ROOT to the upstream cyclomon/UNSB checkout}"
: "${DOSC_DATA_ROOT:?Set DOSC_DATA_ROOT to the dataset built by build_breast_dosc_unsb_dataset.py}"
: "${DOSC_TEACHER:?Set DOSC_TEACHER to the exported TorchScript source classifier}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-busi_to_breast_dosc}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${UNSB_ROOT}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${UNSB_ROOT}/results}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_EPOCHS="${N_EPOCHS:-15}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-15}"
NUM_TEST="${NUM_TEST:-99999}"
STYLE_DIM="${STYLE_DIM:-128}"

export PYTHONPATH="${UNSB_ROOT}:${PYTHONPATH:-}"

python "${UNSB_ROOT}/train.py" \
  --dataroot "${DOSC_DATA_ROOT}" \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --model dosc_sb \
  --dataset_mode dosc_unaligned \
  --direction AtoB \
  --dosc_source_manifest "${DOSC_DATA_ROOT}/trainA_manifest.csv" \
  --dosc_teacher_path "${DOSC_TEACHER}" \
  --dosc_style_dim "${STYLE_DIM}" \
  --dosc_noise_ratio 0.10 \
  --lambda_DOSC_diag 0.10 \
  --lambda_DOSC_domain 0.10 \
  --lambda_DOSC_instance 0.10 \
  --lambda_DOSC_recon 1.00 \
  --lambda_DOSC_safe 0.50 \
  --dosc_safe_tolerance 0.10 \
  --dosc_safe_warmup_steps 1000 \
  --mode sb \
  --lambda_SB 1.0 \
  --lambda_NCE 1.0 \
  --batch_size "${BATCH_SIZE}" \
  --n_epochs "${N_EPOCHS}" \
  --n_epochs_decay "${N_EPOCHS_DECAY}" \
  --gpu_ids "${GPU_IDS}"

python "${UNSB_ROOT}/test.py" \
  --dataroot "${DOSC_DATA_ROOT}" \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --model dosc_sb \
  --dataset_mode dosc_unaligned \
  --direction AtoB \
  --dosc_style_dim "${STYLE_DIM}" \
  --dosc_noise_ratio 0.0 \
  --mode sb \
  --phase test \
  --epoch latest \
  --eval \
  --serial_batches \
  --num_test "${NUM_TEST}" \
  --gpu_ids "${GPU_IDS}"
