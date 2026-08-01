#!/usr/bin/env bash
# Do not pass --mode here. The model's own options are registered only after the base
# parser has read --model, so at that point argparse still prefix-matches --mode to
# --model and silently rewrites "--model trsc_joint_sb" to "sb". The whole TRSC option
# group then never registers and train.py aborts with "unrecognized arguments:
# --trsc_num_references ...". --mode already defaults to sb.
# Train the requested raw + K unfiltered U1 end-to-end TRSC baseline.
set -euo pipefail

: "${UNSB_ROOT:?Set UNSB_ROOT to the upstream cyclomon/UNSB checkout}"
: "${TRSC_DATA_ROOT:?Set TRSC_DATA_ROOT to the BUSI-to-BrEaST TRSC dataset}"
: "${TRSC_INIT_DIR:?Set TRSC_INIT_DIR to the pretrained TRSC G/F/D/E/S checkpoint directory}"
: "${SOURCE_CLASSIFIER_CKPT:?Set SOURCE_CLASSIFIER_CKPT to the source-only custom_resnet50_space checkpoint}"

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-busi_to_breast_trsc_joint_k3}"
MODEL_NAME="${MODEL_NAME:-trsc_joint_sb}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${UNSB_ROOT}/checkpoints}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_REFERENCES="${NUM_REFERENCES:-3}"
LAMBDA_TASK="${LAMBDA_TASK:-1.0}"
VIEW_WEIGHTING="${VIEW_WEIGHTING:-equal_groups}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-4}"
GENERATOR_LR="${GENERATOR_LR:-2e-4}"
TRSC_INIT_EPOCH="${TRSC_INIT_EPOCH:-latest}"
N_EPOCHS="${N_EPOCHS:-15}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-15}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
TARGET_INFERENCE_CSV="${TARGET_INFERENCE_CSV:-}"
TARGET_ROOT_DIR="${TARGET_ROOT_DIR:-}"
TARGET_PATH_COL="${TARGET_PATH_COL:-image_path}"
TARGET_CASE_COL="${TARGET_CASE_COL:-case_id}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}/target_predictions.csv}"

DABRF_TEACHER="${DABRF_TEACHER:-}"
DABRF_MODE="${DABRF_MODE:-learned}"
DABRF_HIDDEN_CHANNELS="${DABRF_HIDDEN_CHANNELS:-32}"
DABRF_FIXED_SCALE="${DABRF_FIXED_SCALE:-0.8}"
DABRF_MAX_RADIUS_RATIO="${DABRF_MAX_RADIUS_RATIO:-1.0}"
DABRF_GATE_FLOOR="${DABRF_GATE_FLOOR:-0.0}"
DABRF_GATE_INIT="${DABRF_GATE_INIT:-0.95}"
DABRF_LR="${DABRF_LR:-1e-4}"
DABRF_PROGRESS_RETENTION="${DABRF_PROGRESS_RETENTION:-0.8}"
LAMBDA_DABRF_DIAG="${LAMBDA_DABRF_DIAG:-1.0}"
LAMBDA_DABRF_PROGRESS="${LAMBDA_DABRF_PROGRESS:-1.0}"
LAMBDA_DABRF_RADIUS="${LAMBDA_DABRF_RADIUS:-1.0}"

if ! [[ "${NUM_REFERENCES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_REFERENCES must be a positive integer" >&2
  exit 2
fi
if [[ "${VIEW_WEIGHTING}" != "equal_groups" && "${VIEW_WEIGHTING}" != "equal_views" ]]; then
  echo "VIEW_WEIGHTING must be equal_groups or equal_views" >&2
  exit 2
fi
if [[ "${MODEL_NAME}" != "trsc_joint_sb" && "${MODEL_NAME}" != "trsc_dabrf_joint_sb" ]]; then
  echo "MODEL_NAME must be trsc_joint_sb or trsc_dabrf_joint_sb" >&2
  exit 2
fi

model_args=()
if [[ "${MODEL_NAME}" == "trsc_dabrf_joint_sb" ]]; then
  model_args=(
    --dabrf_teacher_path "${DABRF_TEACHER}"
    --dabrf_mode "${DABRF_MODE}"
    --dabrf_hidden_channels "${DABRF_HIDDEN_CHANNELS}"
    --dabrf_fixed_scale "${DABRF_FIXED_SCALE}"
    --dabrf_max_radius_ratio "${DABRF_MAX_RADIUS_RATIO}"
    --dabrf_gate_floor "${DABRF_GATE_FLOOR}"
    --dabrf_gate_init "${DABRF_GATE_INIT}"
    --dabrf_lr "${DABRF_LR}"
    --dabrf_progress_retention "${DABRF_PROGRESS_RETENTION}"
    --lambda_DABRF_diag "${LAMBDA_DABRF_DIAG}"
    --lambda_DABRF_progress "${LAMBDA_DABRF_PROGRESS}"
    --lambda_DABRF_radius "${LAMBDA_DABRF_RADIUS}"
  )
fi

export PYTHONPATH="${UNSB_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${SEED}"

python "${UNSB_ROOT}/train.py" \
  --dataroot "${TRSC_DATA_ROOT}" \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --model "${MODEL_NAME}" \
  --dataset_mode trsc_unaligned \
  --direction AtoB \
  --dosc_source_manifest "${TRSC_DATA_ROOT}/trainA_manifest.csv" \
  --trsc_num_references "${NUM_REFERENCES}" \
  --lambda_TRSC_task "${LAMBDA_TASK}" \
  --trsc_view_weighting "${VIEW_WEIGHTING}" \
  --trsc_source_classifier_path "${SOURCE_CLASSIFIER_CKPT}" \
  --trsc_translator_init_dir "${TRSC_INIT_DIR}" \
  --trsc_translator_init_epoch "${TRSC_INIT_EPOCH}" \
  --trsc_classifier_lr "${CLASSIFIER_LR}" \
  --trsc_deterministic true \
  "${model_args[@]}" \
  --dosc_noise_ratio 0.0 \
  --lambda_DOSC_diag 0.0 \
  --lambda_DOSC_domain 0.10 \
  --lambda_DOSC_instance 0.10 \
  --lambda_DOSC_recon 1.00 \
  --lambda_DOSC_safe 0.0 \
  --lambda_GAN 1.0 \
  --lambda_SB 1.0 \
  --lambda_NCE 1.0 \
  --lr "${GENERATOR_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --n_epochs "${N_EPOCHS}" \
  --n_epochs_decay "${N_EPOCHS_DECAY}" \
  --save_epoch_freq "${SAVE_EPOCH_FREQ}" \
  --seed "${SEED}" \
  --display_id -1 \
  --no_html \
  --gpu_ids "${GPU_IDS}"

if [[ -n "${TARGET_INFERENCE_CSV}" ]]; then
  root_args=()
  if [[ -n "${TARGET_ROOT_DIR}" ]]; then
    root_args=(--root_dir "${TARGET_ROOT_DIR}")
  fi
  python "${SCRIPT_ROOT}/eval_trsc_joint_classifier.py" \
    --checkpoint "${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}/latest_net_C.pth" \
    --input_csv "${TARGET_INFERENCE_CSV}" \
    "${root_args[@]}" \
    --path_col "${TARGET_PATH_COL}" \
    --case_col "${TARGET_CASE_COL}" \
    --out "${PREDICTIONS_OUT}"
fi
