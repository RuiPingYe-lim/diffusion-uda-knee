#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
CODE=/root/autodl-tmp/knee/code2/idea2_diffusion_baseline
AS=/root/autodl-tmp/knee_allslices
CLF=$AS/exp/cls_kneemri
LOG=$AS/reverse_result.log
cd $CODE
exec > >(tee -a $LOG) 2>&1
echo "############ REVERSE (source=KneeMRI, target=MRNet) start $(date) ############"

echo "===== [1] train KneeMRI classifier (source=KneeMRI) ====="
python train_source_classifier.py \
  --src_train_csv $AS/kneemri_train/allslices.csv \
  --src_val_csv $AS/kneemri_test/allslices.csv \
  --output_dir $CLF \
  --backbone custom_resnet50_space --pretrained imagenet --image_size 224 \
  --epochs 20 --batch_size 32 --num_workers 8 --amp --metric_for_best auc \
  || echo "(train script exited nonzero; final-eval torch.load may crash — best_checkpoint saved before that)"

test -f $CLF/best_checkpoint.pt || { echo NO_CLF_FAIL; exit 1; }
echo "source ceiling (KneeMRI val) best:"; grep -o "best_auc=[0-9.]*\|best_metric_value.*" $CLF/train_log.csv 2>/dev/null | tail -1 || true

echo "===== [2] direct transfer vs moment-match (target=MRNet valid) ====="
echo "@@@ REVERSE_STYLE @@@"
cd $CODE/bbdm_strict
python eval_style_match.py \
  --clf_ckpt $CLF/best_checkpoint.pt \
  --source_csv $AS/kneemri_train/allslices.csv \
  --target_csv $AS/mrnet_valid/allslices.csv

echo "REVERSE_DONE $(date)"
