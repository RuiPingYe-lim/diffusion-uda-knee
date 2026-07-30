#!/usr/bin/env bash
# STAGE 1 -- 3-seed variance re-estimation over the 7 matched arms.
#
# PURPOSE IS VARIANCE. The calibration's seed count (21 under last-5 averaging) was
# measured on the plain augmentation arms; adding SupCon and attention pooling can
# change the variance, so the seed count must be re-derived here before committing
# to the full run. The MEANS of these 3 seeds are NOT a result.
#
# Two ablation toggles run alongside, so the contribution of each new component is
# separable rather than confounded:
#   --lambda_sup 0   = attention pooling only, no contrastive  (isolates SupCon)
#   --pool gap       = contrastive only, average pooling       (isolates attention)
#
# Nothing here reads the locked 51-case BrEaST test set.
set -euo pipefail
PY=/root/miniconda3/bin/python
B=/root/autodl-tmp/breast
OUT=$B/stage1/runs
SEALED=$B/stage1/sealed
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-30}"
cd "$B"
mkdir -p "$OUT" "$SEALED"

run() {  # arm  extra-args...
  local arm=$1; shift
  local tag
  tag=$($PY - "$arm" "$@" <<'EOF'
import sys
arm=sys.argv[1]; rest=sys.argv[2:]
pool="attn"; sup="1"
for i,a in enumerate(rest):
    if a=="--pool": pool=rest[i+1]
    if a=="--lambda_sup": sup=rest[i+1]
print(f"{arm}_{pool}_sup{float(sup):g}_s{__import__('os').environ['SEED']}")
EOF
)
  if [ -f "$OUT/$tag/config.json" ]; then echo "skip $tag (done)"; return; fi
  echo "### $tag ###"
  $PY train_stage1.py --arm "$arm" --seed "$SEED" --epochs "$EPOCHS" \
    --out_dir "$OUT" --sealed_dir "$SEALED" "$@" > "$OUT/$tag.log" 2>&1
  tail -1 "$OUT/$tag.log"
}

for SEED in $SEEDS; do
  export SEED
  # main configuration: attention pooling + SupCon, all 7 arms
  for arm in A0 P1 F1 U1 P5 F5 U5; do run "$arm"; done
  # ablations, only on the arms the primary contrast needs
  for arm in A0 P1 U1; do run "$arm" --lambda_sup 0; done   # no contrastive
  for arm in A0 P1 U1; do run "$arm" --pool gap;      done   # no attention pooling
  echo "SEED_${SEED}_DONE"
done
echo "STAGE1_DONE"
