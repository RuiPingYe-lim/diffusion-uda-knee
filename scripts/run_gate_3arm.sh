#!/usr/bin/env bash
# Matched three-arm gate (review v2): direct / repeat_before / true_fakes, 5 seeds.
# Minimal config first: stat_prior OFF, supcon 0. Uses fusion_gate.py (NOT the old
# fusion_classifier.py). Saves per-case predictions for every (control, seed).
set -euo pipefail
PY=/root/miniconda3/bin/python
C=/root/autodl-tmp/breast/cache
E=/root/autodl-tmp/breast/exp/gate3arm
SRC=/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt
GEN=/root/autodl-tmp/UNSB/checkpoints/b2u_SB/latest_net_G.pth
EVAL=$C/fusion_eval_breast_diag.csv
FAKES=fake_1,fake_2,fake_3,fake_4,fake_5
REPEAT=before_png,before_png,before_png,before_png,before_png   # repeat_before: K non-fake cols (copies built in code)
SEEDS="${SEEDS:-42 43 44 45 46}"
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
mkdir -p "$E"

run() {  # control other_cols_train other_cols_eval seed  (train_csv/val only for non-direct)
  ctrl=$1; octr=$2; oev=$3; seed=$4; d=$E/${ctrl}_s${seed}; mkdir -p "$d"
  if [ "$ctrl" = "direct" ]; then
    $PY fusion_gate.py --mode train --control direct --source_ckpt "$SRC" --generator_ckpt "$GEN" \
      --train_csv "$C/fusion_train_busi.csv" --out_dir "$d" --seed "$seed"
  else
    $PY fusion_gate.py --mode train --control "$ctrl" --source_ckpt "$SRC" --generator_ckpt "$GEN" \
      --train_csv "$C/fusion_train_busi.csv" --val_csv "$C/fusion_val_busi.csv" \
      --other_cols "$octr" --out_dir "$d" --epochs 30 --batch_size 16 --lr 1e-4 --seed "$seed"
  fi
  rm -f "$d/percase.csv"   # never let a stale eval survive a failed re-run
  $PY fusion_gate.py --mode eval --control "$ctrl" --weights "$d/best.pt" --test_csv "$EVAL" \
    --other_cols "$oev" --out_csv "$d/percase.csv"
}

for seed in $SEEDS; do
  run direct        "$FAKES"  "$FAKES"  "$seed"
  run repeat_before "$REPEAT" "$REPEAT" "$seed"
  run true_fakes    "$FAKES"  "$FAKES"  "$seed"
done
echo "ALL_DONE"
