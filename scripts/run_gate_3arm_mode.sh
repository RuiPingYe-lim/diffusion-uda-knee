#!/usr/bin/env bash
# Matched three-arm gate, parameterised by FUSION MODE.
#
# WHY: the original 3-arm result (true_fakes - repeat_before = -0.0122 paired,
# 3/3 seeds negative) was produced entirely in mode orig_kv. A lesion test then
# showed only ~2% of the adapter's residual variance is driven by `others` at
# all -- which follows from the architecture: in orig_kv the attention VALUES are
# `before`'s own tokens and the output is norm(anchor + fused) with anchor also
# from `before`, so a translation view can only re-weight the original and can
# never contribute content. That makes the orig_kv null uninformative about the
# translations themselves.
#
# orig_query (translations = Key/Value) gives the translation a real value path.
# This script re-runs the SAME matched protocol there, so the two modes are
# directly comparable arm-for-arm and seed-for-seed.
#
# `direct` contains no trainable parameters and is mode-independent, so it is
# NOT re-run: reuse gate3arm/direct_s42/percase.csv (case-AUC 0.7909).
set -euo pipefail
PY=/root/miniconda3/bin/python
C=/root/autodl-tmp/breast/cache
SRC=/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt
GEN=/root/autodl-tmp/UNSB/checkpoints/b2u_SB/latest_net_G.pth
EVAL=$C/fusion_eval_breast_diag.csv
FAKES=fake_1,fake_2,fake_3,fake_4,fake_5
REPEAT=before_png,before_png,before_png,before_png,before_png
MODE="${MODE:-orig_query}"
SEEDS="${SEEDS:-42 43 44}"
E=/root/autodl-tmp/breast/exp/gate3arm_${MODE}
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
mkdir -p "$E"

run() {  # control other_cols seed
  ctrl=$1; cols=$2; seed=$3; d=$E/${ctrl}_s${seed}; mkdir -p "$d"
  $PY fusion_gate.py --mode train --control "$ctrl" --fusion_mode "$MODE" \
    --source_ckpt "$SRC" --generator_ckpt "$GEN" \
    --train_csv "$C/fusion_train_busi.csv" --val_csv "$C/fusion_val_busi.csv" \
    --other_cols "$cols" --out_dir "$d" --epochs 30 --batch_size 16 --lr 1e-4 --seed "$seed" \
    > "$d/train.log" 2>&1
  rm -f "$d/percase.csv"   # never let a stale eval survive a failed re-run
  $PY fusion_gate.py --mode eval --weights "$d/best.pt" --test_csv "$EVAL" \
    --other_cols "$cols" --out_csv "$d/percase.csv"
}

for seed in $SEEDS; do
  run repeat_before "$REPEAT" "$seed"
  run true_fakes    "$FAKES"  "$seed"
  # lesion check in this mode: is the `others` pathway actually live here?
  for sh in 12345 777; do
    $PY fusion_gate.py --mode eval --weights "$E/true_fakes_s${seed}/best.pt" --test_csv "$EVAL" \
      --other_cols "$FAKES" --shuffle_others --shuffle_seed "$sh" \
      --out_csv "$E/true_fakes_s${seed}/percase_shuf${sh}.csv"
  done
done
echo "GATE_${MODE}_DONE"
