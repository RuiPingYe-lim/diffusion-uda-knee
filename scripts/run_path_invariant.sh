#!/usr/bin/env bash
# C2 gate: source-only vs endpoint vs continuous path vs Brownian path.
#
# Usage:
#   bash scripts/run_path_invariant.sh \
#     <source_train.csv> <source_val.csv> <target_train_unlabeled.csv> \
#     <target_test.csv> <output_root> [source_test.csv]
#
# Optional environment overrides:
#   MODES="none endpoint linear brownian" SEEDS="42" EPOCHS=40 BS=16 WORKERS=8
#   BACKBONE=custom_resnet50_space PRETRAINED=imagenet
#
# The script never selects a checkpoint with target-test labels.  Each run uses
# source validation AUC, then evaluates the selected checkpoint once on target.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <src_train.csv> <src_val.csv> <tgt_train.csv> <tgt_test.csv> <out_root> [src_test.csv]" >&2
  exit 2
fi

SRC_TRAIN=$1
SRC_VAL=$2
TGT_TRAIN=$3
TGT_TEST=$4
OUT_ROOT=$5
SRC_TEST=${6:-}

MODES=${MODES:-"none endpoint linear brownian"}
SEEDS=${SEEDS:-"42"}
EPOCHS=${EPOCHS:-40}
BS=${BS:-16}
WORKERS=${WORKERS:-8}
BACKBONE=${BACKBONE:-custom_resnet50_space}
PRETRAINED=${PRETRAINED:-imagenet}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PY=$ROOT/src/bbdm_strict/train_path_invariant_classifier.py
mkdir -p "$OUT_ROOT"

for seed in $SEEDS; do
  for mode in $MODES; do
    run_dir=$OUT_ROOT/${mode}_seed${seed}
    args=(
      --src_train_csv "$SRC_TRAIN"
      --src_val_csv "$SRC_VAL"
      --tgt_test_csv "$TGT_TEST"
      --output_dir "$run_dir"
      --path_mode "$mode"
      --seed "$seed"
      --epochs "$EPOCHS"
      --batch_size "$BS"
      --num_workers "$WORKERS"
      --backbone "$BACKBONE"
      --pretrained "$PRETRAINED"
      --metric_for_best case_auc
      --amp
    )
    if [[ "$mode" != "none" ]]; then
      args+=(
        --tgt_train_csv "$TGT_TRAIN"
        --style_bank_cache "$OUT_ROOT/target_style_bank_seed${seed}.json"
      )
    fi
    if [[ -n "$SRC_TEST" ]]; then
      args+=(--src_test_csv "$SRC_TEST")
    fi

    echo "===== C2 mode=$mode seed=$seed ====="
    python "$TRAIN_PY" "${args[@]}" 2>&1 | tee "$OUT_ROOT/${mode}_seed${seed}.log"
  done
done

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*_seed*/final_metrics.json")):
    data = json.loads(path.read_text())
    target = data.get("tgt_test") or {}
    source = data.get("src_val") or {}
    rows.append(
        (
            path.parent.name,
            source.get("case_auc", float("nan")),
            target.get("case_auc", float("nan")),
        )
    )
print("\n===== C2 summary (selection=source val only) =====")
print("run\tsrc_val_case_auc\ttgt_test_case_auc")
for name, source_auc, target_auc in rows:
    print(f"{name}\t{source_auc:.4f}\t{target_auc:.4f}")
PY
