#!/usr/bin/env bash
# STEP 4 -- 3-seed variance calibration over the 13 matched arms.
#
# 13 arms = 7 base conditions (A0, P1, F1, U1, P5, F5, U5) plus the `+C`
# consistency variant of the six P/F/U arms. A0 has no +C variant.
#
# The consistency loss is applied to P and F as well as U -- otherwise a gain in
# U+C could not be separated from the regulariser's own effect.
#
# PURPOSE: estimate SD of the seed-paired differences, to size the real trial.
# The MEANS of these 3 seeds are NOT a result and must not be reported as one.
#
# Nothing here reads the locked 51-case BrEaST test. Target-set probabilities go
# to a write-only sealed directory and are not opened until every run finishes.
set -euo pipefail
PY=/root/miniconda3/bin/python
B=/root/autodl-tmp/breast
OUT=$B/da_route/runs
SEALED=$B/da_route/sealed
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-50}"
cd "$B"
mkdir -p "$OUT" "$SEALED"

run() {  # arm [--consistency]
  local arm=$1; shift
  local tag="${arm}${1:+C}"
  local d="$OUT/${tag}_s${SEED}"
  if [ -f "$d/config.json" ]; then echo "skip $tag seed $SEED (done)"; return; fi
  echo "### $tag  seed $SEED ###"
  $PY train_da_route.py --arm "$arm" "$@" --seed "$SEED" --epochs "$EPOCHS" \
    --out_dir "$OUT" --sealed_dir "$SEALED" > "$OUT/${tag}_s${SEED}.log" 2>&1
  tail -1 "$OUT/${tag}_s${SEED}.log"
}

for SEED in $SEEDS; do
  export SEED
  # no-consistency block first, so partial results are already interpretable
  for arm in A0 P1 F1 U1 P5 F5 U5; do run "$arm"; done
  # consistency block (A0 has no +C variant)
  for arm in P1 F1 U1 P5 F5 U5; do run "$arm" --consistency; done
  echo "SEED_${SEED}_DONE"
done
echo "CALIBRATION_DONE"
