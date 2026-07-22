#!/usr/bin/env bash
# DIAGNOSTIC 2 -- supervised ORACLE on the target diagnostic set (5-fold CV).
#
# The adapter is allowed to see TARGET LABELS. This is deliberately NOT a UDA
# result and must never be reported as one; it is an upper reference for how
# much the fusion pathway could possibly extract.
#   oracle_true_fakes - oracle_repeat_before = information in the translation views
#   oracle_repeat_before - direct            = what target labels alone buy
#
# RUN IN BOTH FUSION MODES (review must-fix #1). In orig_kv the attention VALUES
# all come from `before` (fusion_classifier.py::_rep: kv = before_tok, and the
# output is norm(anchor + fused) with anchor = before_tok.mean), so `others` can
# only re-weight the original's own tokens and can never inject content -- a null
# there says nothing about the translations. orig_query gives the translation a
# real value path. Reporting a boundary condition requires the orig_query result.
#
# THREE SEEDS (review must-fix #2): the effect being ruled out (MCID 0.01) is the
# same order as the seed noise observed in the main protocol.
#
# Each fold also evaluates on its own TRAIN split (percase_trainfit.csv) so that
# "cannot fit" (capacity) is distinguishable from "fits but does not generalise".
#
# The LOCKED 51-case BrEaST test set is NOT used anywhere here.
set -euo pipefail
PY=/root/miniconda3/bin/python
C=/root/autodl-tmp/breast/cache
F=$C/oracle_folds
O=/root/autodl-tmp/breast/exp/diag_oracle
SRC=/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt
GEN=/root/autodl-tmp/UNSB/checkpoints/b2u_SB/latest_net_G.pth
FAKES=fake_1,fake_2,fake_3,fake_4,fake_5
REPEAT=before_png,before_png,before_png,before_png,before_png
NFOLDS="${NFOLDS:-5}"
EPOCHS="${EPOCHS:-60}"     # more budget than the main run: a generous oracle strengthens a null
SEEDS="${SEEDS:-42 43 44}"
MODES="${MODES:-orig_kv orig_query}"
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
mkdir -p "$O"
[ -f "$F/f0_train.csv" ] || { echo "FATAL: fold CSVs missing; run build_oracle_folds.py first"; exit 1; }

for M in $MODES; do
 for s in $SEEDS; do
  for k in $(seq 0 $((NFOLDS-1))); do
    for arm in repeat_before true_fakes; do
      if [ "$arm" = "repeat_before" ]; then cols=$REPEAT; else cols=$FAKES; fi
      d=$O/${M}_${arm}_f${k}_s${s}; mkdir -p "$d"
      $PY fusion_gate.py --mode train --control "$arm" --fusion_mode "$M" \
        --source_ckpt "$SRC" --generator_ckpt "$GEN" \
        --train_csv "$F/f${k}_train.csv" --val_csv "$F/f${k}_val.csv" \
        --other_cols "$cols" --out_dir "$d" --epochs "$EPOCHS" --batch_size 16 --lr 1e-4 --seed "$s" \
        > "$d/train.log" 2>&1
      rm -f "$d/percase.csv" "$d/percase_trainfit.csv"
      $PY fusion_gate.py --mode eval --weights "$d/best.pt" --test_csv "$F/f${k}_test.csv" \
        --other_cols "$cols" --out_csv "$d/percase.csv"
      $PY fusion_gate.py --mode eval --weights "$d/best.pt" --test_csv "$F/f${k}_train.csv" \
        --other_cols "$cols" --out_csv "$d/percase_trainfit.csv" > /dev/null
      # 60 folds x ~98 MB would fill the disk. The per-case CSVs already record
      # checkpoint/source/generator/manifest sha256 and the full config is echoed
      # in train.log, so the RESULT stays traceable; only the weights of these
      # diagnostic folds are not retained. Headline runs (gate3arm*) keep theirs.
      [ "${KEEP_CKPT:-0}" = "1" ] || rm -f "$d/best.pt"
    done
  done
  # coverage assert (review must-fix #3): the pooled out-of-fold set must be the
  # SAME 201 cases the `direct` reference was computed on, or the AUC difference
  # is a population difference. MultiImageDataset silently drops non-{0,1} labels.
  for arm in repeat_before true_fakes; do
    $PY - "$O" "$M" "$arm" "$s" "$NFOLDS" <<'PYEOF'
import sys, pandas as pd
O, M, arm, s, nf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
d = pd.concat([pd.read_csv(f"{O}/{M}_{arm}_f{k}_s{s}/percase.csv") for k in range(nf)])
n = d.case_id.nunique()
assert n == 201 and len(d) == 201, f"out-of-fold coverage {n} unique / {len(d)} rows != 201 ({M} {arm} s{s})"
print(f"[coverage OK] {M} {arm} seed{s}: 201 unique cases, no duplicates, no dropped rows")
PYEOF
  done
  echo "ORACLE_${M}_s${s}_DONE"
 done
done
echo "ORACLE_ALL_DONE"
