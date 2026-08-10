#!/usr/bin/env bash
# Train SafeBridge-UDA on labeled BUSI and unlabeled BrEaST target-train images.
set -euo pipefail

: "${UNSB_ROOT:?Set UNSB_ROOT to the upstream cyclomon/UNSB checkout}"
: "${TRSC_DATA_ROOT:?Set TRSC_DATA_ROOT to the BUSI-to-BrEaST TRSC dataset}"
: "${TRSC_INIT_DIR:?Set TRSC_INIT_DIR to pretrained TRSC G/F/D/E/S checkpoints}"
: "${SOURCE_CLASSIFIER_CKPT:?Set SOURCE_CLASSIFIER_CKPT to the source-only classifier}"
: "${SAFEBRIDGE_TEACHER:?Set SAFEBRIDGE_TEACHER to the frozen render-robust anchor}"

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-busi_to_breast_safebridge_full}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${UNSB_ROOT}/checkpoints}"
GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_REFERENCES="${NUM_REFERENCES:-1}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-5}"
CANDIDATE_STATES="${CANDIDATE_STATES:-1,3,5}"
FUSION_MODE="${FUSION_MODE:-full}"
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-4}"
DOMAIN_LR="${DOMAIN_LR:-1e-4}"
GENERATOR_LR="${GENERATOR_LR:-2e-4}"
TRSC_INIT_EPOCH="${TRSC_INIT_EPOCH:-latest}"
N_EPOCHS="${N_EPOCHS:-15}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-15}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"

DIAG_QUEUE_SIZE="${DIAG_QUEUE_SIZE:-128}"
DIAG_MIN_PER_CLASS="${DIAG_MIN_PER_CLASS:-8}"
DIAG_MARGIN_TOLERANCE="${DIAG_MARGIN_TOLERANCE:-0.10}"
DIAG_RANK_TOLERANCE="${DIAG_RANK_TOLERANCE:-0.10}"
DIAG_MAX_RANK_VIOLATION="${DIAG_MAX_RANK_VIOLATION:-0.10}"
MINIMUM_STABILITY="${MINIMUM_STABILITY:-0.60}"
STATE_MINIMUM_DOMAIN_GAIN="${STATE_MINIMUM_DOMAIN_GAIN:-0.0}"

TARGET_RANK="${TARGET_RANK:-8}"
TARGET_QUEUE_SIZE="${TARGET_QUEUE_SIZE:-128}"
TARGET_MIN_SAMPLES="${TARGET_MIN_SAMPLES:-16}"
TARGET_QUANTILE="${TARGET_QUANTILE:-0.25}"
DOMAIN_MIN_ACCURACY="${DOMAIN_MIN_ACCURACY:-0.60}"

NOISE_FLOOR="${NOISE_FLOOR:-0.01}"
SPATIAL_THRESHOLD="${SPATIAL_THRESHOLD:-1.0}"
CHANNEL_THRESHOLD="${CHANNEL_THRESHOLD:-1.0}"
BACKTRACKING_ALPHAS="${BACKTRACKING_ALPHAS:-1.0,0.5,0.25,0.125}"
BACKTRACKING_DOMAIN_GAIN="${BACKTRACKING_DOMAIN_GAIN:-0.0}"
BACKTRACKING_GAIN_RETENTION="${BACKTRACKING_GAIN_RETENTION:-0.50}"
BACKTRACKING_MARGIN_TOLERANCE="${BACKTRACKING_MARGIN_TOLERANCE:-0.10}"

TARGET_INFERENCE_CSV="${TARGET_INFERENCE_CSV:-}"
TARGET_ROOT_DIR="${TARGET_ROOT_DIR:-}"
TARGET_PATH_COL="${TARGET_PATH_COL:-image_path}"
TARGET_CASE_COL="${TARGET_CASE_COL:-case_id}"
SOURCE_VAL_CSV="${SOURCE_VAL_CSV:-}"
SOURCE_VAL_ROOT_DIR="${SOURCE_VAL_ROOT_DIR:-}"
SOURCE_VAL_PATH_COL="${SOURCE_VAL_PATH_COL:-image_path}"
SOURCE_VAL_LABEL_COL="${SOURCE_VAL_LABEL_COL:-label}"
SOURCE_VAL_CASE_COL="${SOURCE_VAL_CASE_COL:-case_id}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}/target_predictions.csv}"
SAFEBRIDGE_AUDIT="${SAFEBRIDGE_AUDIT:-${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}/safebridge_audit.jsonl}"
ALLOW_AUDIT_APPEND="${ALLOW_AUDIT_APPEND:-0}"

if ! [[ "${NUM_REFERENCES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_REFERENCES must be a positive integer" >&2
  exit 2
fi
if [[ ! "${FUSION_MODE}" =~ ^(raw_only|selected_feature|stable_residual|target_projected|full)$ ]]; then
  echo "Unsupported FUSION_MODE: ${FUSION_MODE}" >&2
  exit 2
fi
if [[ -e "${SAFEBRIDGE_AUDIT}" && "${ALLOW_AUDIT_APPEND}" != "1" ]]; then
  echo "Audit log already exists: ${SAFEBRIDGE_AUDIT}" >&2
  echo "Use a new EXPERIMENT_NAME, or set ALLOW_AUDIT_APPEND=1 only for an intentional resume." >&2
  exit 2
fi

export PYTHONPATH="${UNSB_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${SEED}"

python "${UNSB_ROOT}/train.py" \
  --dataroot "${TRSC_DATA_ROOT}" \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --model safebridge_joint_sb \
  --dataset_mode trsc_unaligned \
  --direction AtoB \
  --dosc_source_manifest "${TRSC_DATA_ROOT}/trainA_manifest.csv" \
  --trsc_num_references "${NUM_REFERENCES}" \
  --lambda_TRSC_task 0.0 \
  --trsc_view_weighting equal_groups \
  --trsc_source_classifier_path "${SOURCE_CLASSIFIER_CKPT}" \
  --trsc_translator_init_dir "${TRSC_INIT_DIR}" \
  --trsc_translator_init_epoch "${TRSC_INIT_EPOCH}" \
  --trsc_classifier_lr "${CLASSIFIER_LR}" \
  --trsc_deterministic true \
  --safebridge_candidate_states "${CANDIDATE_STATES}" \
  --safebridge_fusion_mode "${FUSION_MODE}" \
  --safebridge_teacher_path "${SAFEBRIDGE_TEACHER}" \
  --safebridge_diag_queue_size "${DIAG_QUEUE_SIZE}" \
  --safebridge_diag_min_per_class "${DIAG_MIN_PER_CLASS}" \
  --safebridge_diag_margin_tolerance "${DIAG_MARGIN_TOLERANCE}" \
  --safebridge_diag_rank_tolerance "${DIAG_RANK_TOLERANCE}" \
  --safebridge_diag_max_rank_violation "${DIAG_MAX_RANK_VIOLATION}" \
  --safebridge_minimum_stability "${MINIMUM_STABILITY}" \
  --safebridge_state_minimum_domain_gain "${STATE_MINIMUM_DOMAIN_GAIN}" \
  --safebridge_domain_lr "${DOMAIN_LR}" \
  --safebridge_target_rank "${TARGET_RANK}" \
  --safebridge_target_queue_size "${TARGET_QUEUE_SIZE}" \
  --safebridge_target_min_samples "${TARGET_MIN_SAMPLES}" \
  --safebridge_target_quantile "${TARGET_QUANTILE}" \
  --safebridge_domain_min_accuracy "${DOMAIN_MIN_ACCURACY}" \
  --safebridge_noise_floor "${NOISE_FLOOR}" \
  --safebridge_spatial_threshold "${SPATIAL_THRESHOLD}" \
  --safebridge_channel_threshold "${CHANNEL_THRESHOLD}" \
  --safebridge_backtracking_alphas "${BACKTRACKING_ALPHAS}" \
  --safebridge_backtracking_domain_gain "${BACKTRACKING_DOMAIN_GAIN}" \
  --safebridge_backtracking_gain_retention "${BACKTRACKING_GAIN_RETENTION}" \
  --safebridge_backtracking_margin_tolerance "${BACKTRACKING_MARGIN_TOLERANCE}" \
  --safebridge_freeze_classifier_bn true \
  --safebridge_audit_jsonl "${SAFEBRIDGE_AUDIT}" \
  --num_timesteps "${NUM_TIMESTEPS}" \
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

CLASSIFIER_FOR_INFERENCE="${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}/latest_net_C.pth"
if [[ -n "${SOURCE_VAL_CSV}" ]]; then
  source_root_args=()
  if [[ -n "${SOURCE_VAL_ROOT_DIR}" ]]; then
    source_root_args=(--root_dir "${SOURCE_VAL_ROOT_DIR}")
  fi
  CLASSIFIER_FOR_INFERENCE="${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}/best_source_val_net_C.pth"
  python "${SCRIPT_ROOT}/select_safebridge_source_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINTS_DIR}/${EXPERIMENT_NAME}" \
    --source_val_csv "${SOURCE_VAL_CSV}" \
    "${source_root_args[@]}" \
    --path_col "${SOURCE_VAL_PATH_COL}" \
    --label_col "${SOURCE_VAL_LABEL_COL}" \
    --case_col "${SOURCE_VAL_CASE_COL}" \
    --out_checkpoint "${CLASSIFIER_FOR_INFERENCE}"
fi

if [[ -n "${TARGET_INFERENCE_CSV}" ]]; then
  root_args=()
  if [[ -n "${TARGET_ROOT_DIR}" ]]; then
    root_args=(--root_dir "${TARGET_ROOT_DIR}")
  fi
  python "${SCRIPT_ROOT}/eval_trsc_joint_classifier.py" \
    --checkpoint "${CLASSIFIER_FOR_INFERENCE}" \
    --input_csv "${TARGET_INFERENCE_CSV}" \
    "${root_args[@]}" \
    --path_col "${TARGET_PATH_COL}" \
    --case_col "${TARGET_CASE_COL}" \
    --out "${PREDICTIONS_OUT}"
fi
