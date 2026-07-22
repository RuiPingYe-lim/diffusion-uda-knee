#!/usr/bin/env bash
# DIAGNOSTIC 1 -- does the adapter actually READ the other views?
#
# 1a LESION (free, no training): re-evaluate the EXISTING true_fakes checkpoints
#    with the before<->others correspondence broken (fixed derangement). If AUC
#    and the per-case residual are unchanged, the residual is a function of
#    `before` alone and the whole `others` pathway is dead -> architecture bug,
#    NOT evidence about the information content of the translations.
# 1b MATCHED ARM (trains): train+eval with the pairing broken throughout, so the
#    arm is directly comparable to true_fakes / repeat_before.
set -euo pipefail
PY=/root/miniconda3/bin/python
C=/root/autodl-tmp/breast/cache
E=/root/autodl-tmp/breast/exp/gate3arm
D=/root/autodl-tmp/breast/exp/diag
SRC=/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt
GEN=/root/autodl-tmp/UNSB/checkpoints/b2u_SB/latest_net_G.pth
EVAL=$C/fusion_eval_breast_diag.csv
FAKES=fake_1,fake_2,fake_3,fake_4,fake_5
SHUF_SEED=12345
SEEDS="${SEEDS:-42 43 44}"
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
mkdir -p "$D"

echo "########## 1a: lesion eval of existing true_fakes checkpoints ##########"
for s in $SEEDS; do
  d=$E/true_fakes_s${s}
  [ -f "$d/best.pt" ] || { echo "SKIP seed $s (no best.pt)"; continue; }
  rm -f "$d/percase_shuffled.csv"
  $PY fusion_gate.py --mode eval --weights "$d/best.pt" --test_csv "$EVAL" \
    --other_cols "$FAKES" --shuffle_others --shuffle_seed "$SHUF_SEED" \
    --out_csv "$d/percase_shuffled.csv"
done
echo "LESION_DONE"

echo "########## 1b: matched shuffled_fakes arm (train+eval shuffled) ##########"
for s in $SEEDS; do
  o=$D/shuffled_fakes_s${s}; mkdir -p "$o"
  $PY fusion_gate.py --mode train --control true_fakes --shuffle_others --shuffle_seed "$SHUF_SEED" \
    --source_ckpt "$SRC" --generator_ckpt "$GEN" \
    --train_csv "$C/fusion_train_busi.csv" --val_csv "$C/fusion_val_busi.csv" \
    --other_cols "$FAKES" --out_dir "$o" --epochs 30 --batch_size 16 --lr 1e-4 --seed "$s"
  rm -f "$o/percase.csv"
  $PY fusion_gate.py --mode eval --weights "$o/best.pt" --test_csv "$EVAL" \
    --other_cols "$FAKES" --shuffle_others --shuffle_seed "$SHUF_SEED" \
    --out_csv "$o/percase.csv"
done
echo "SHUFFLE_ARM_DONE"
