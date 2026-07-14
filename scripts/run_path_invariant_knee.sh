#!/usr/bin/env bash
# AutoDL-ready wrapper for the mean-projection knee benchmark.
# Default: run the forward MRNet -> KneeMRI C2 gate with one seed.
#
# Examples:
#   bash scripts/run_path_invariant_knee.sh
#   DIRECTIONS="m2k k2m" SEEDS="42 43 44" bash scripts/run_path_invariant_knee.sh
#   MODES="none endpoint linear" EPOCHS=5 bash scripts/run_path_invariant_knee.sh
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MEANPROJ_ROOT=${MEANPROJ_ROOT:-/root/autodl-tmp/meanproj_stage}
OUTPUT_ROOT=${OUTPUT_ROOT:-$MEANPROJ_ROOT/c2_path_invariant}
DIRECTIONS=${DIRECTIONS:-"m2k"}

for direction in $DIRECTIONS; do
  case "$direction" in
    m2k)
      src_train=$MEANPROJ_ROOT/mrnet_train.csv
      src_val=$MEANPROJ_ROOT/mrnet_valid.csv
      src_test=$MEANPROJ_ROOT/mrnet_test.csv
      tgt_train=$MEANPROJ_ROOT/kneemri_train.csv
      tgt_test=$MEANPROJ_ROOT/kneemri_test.csv
      ;;
    k2m)
      src_train=$MEANPROJ_ROOT/kneemri_train.csv
      src_val=$MEANPROJ_ROOT/kneemri_valid.csv
      src_test=$MEANPROJ_ROOT/kneemri_test.csv
      tgt_train=$MEANPROJ_ROOT/mrnet_train.csv
      tgt_test=$MEANPROJ_ROOT/mrnet_test.csv
      ;;
    *)
      echo "Unknown direction '$direction'; expected m2k or k2m" >&2
      exit 2
      ;;
  esac

  echo "########## C2 $direction ##########"
  bash "$ROOT/scripts/run_path_invariant.sh" \
    "$src_train" "$src_val" "$tgt_train" "$tgt_test" \
    "$OUTPUT_ROOT/$direction" "$src_test"
done
