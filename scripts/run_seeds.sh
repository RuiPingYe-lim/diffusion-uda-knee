#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
CODE=/root/autodl-tmp/knee/code2/idea2_diffusion_baseline
AS=/root/autodl-tmp/knee_allslices
FD=$AS/fuse_sup_pseudo
CLF=$AS/exp/cls_allslices/best_checkpoint.pt
LOG=$AS/seeds_result.log
cd $CODE/bbdm_strict
exec > >(tee -a $LOG) 2>&1
echo "############ FUSION MULTI-SEED start $(date) ############"

for S in 1 2 3; do
  echo "========== SEED $S : train diffusion-fusion =========="
  python fusion_classifier.py --mode train --train_csv $FD/tr.csv --val_csv $FD/va.csv \
    --before_col before_png --other_cols translated_png,sample_0,sample_1 --label_col label \
    --out_dir $FD/run_s$S --epochs 12 --batch_size 16 --backbone resnet50 --seed $S || { echo "SEED_${S}_FAIL"; continue; }
  echo "@@@ SEED $S RESULT @@@"
  python eval_late_fusion.py --src_clf $CLF --fusion_ckpt $FD/run_s$S/best.pt \
    --pairs_csv $FD/tgt/fusion_pairs.csv
done
echo "SEEDS_DONE $(date)"
