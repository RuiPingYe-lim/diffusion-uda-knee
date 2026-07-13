#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
CODE=/root/autodl-tmp/knee/code2/idea2_diffusion_baseline
AS=/root/autodl-tmp/knee_allslices
CFG=$CODE/bbdm_strict/configs/bbdm_knee_allslices_supcon_pseudo.json
CLF=$AS/exp/cls_allslices/best_checkpoint.pt
BCK=$AS/exp/bbdm_runs/bbdm_allslices_supcon_pseudo/checkpoints/latest.pt
FD=$AS/fuse_sup_pseudo
LOG=$AS/pseudo_sup_result.log
cd $CODE/bbdm_strict
exec > >(tee -a $LOG) 2>&1
echo "############ PSEUDO + SUPCON RUN start $(date) ############"

echo "===== [1] train BBDM (pseudo pairing + SupCon lambda=0.1) ====="
python train_strict_bbdm.py --config $CFG || { echo TRAIN_FAIL; exit 1; }

echo "===== [2] +1+2 translate eval (volume) [PSEUDO] ====="
echo "@@@ RESULT_12 @@@"
python eval_volume.py --clf_ckpt $CLF --slice_csv $AS/kneemri_test/allslices.csv \
  --mode translate --bbdm_config $CFG --bbdm_ckpt $BCK --num_inference_steps 50 --image_size 128

echo "===== [3] +1+2+3 sampling-steps sweep [PSEUDO] ====="
echo "@@@ RESULT_123 @@@"
for S in 5 10 25 50; do
  echo "--- steps=$S ---"
  python eval_volume.py --clf_ckpt $CLF --slice_csv $AS/kneemri_test/allslices.csv \
    --mode translate --bbdm_config $CFG --bbdm_ckpt $BCK --num_inference_steps $S --image_size 128 | grep volume_AUC
done

echo "===== [4] gen fusion pairs (src=mrnet_sub6k, tgt=kneemri_test) ====="
python gen_fusion_pairs.py --config $CFG --checkpoint $BCK \
  --input_csv $AS/mrnet_sub6k.csv --out_dir $FD/src --num_samples 2 --reverse_eta 0.35 || { echo GEN_SRC_FAIL; exit 1; }
python gen_fusion_pairs.py --config $CFG --checkpoint $BCK \
  --input_csv $AS/kneemri_test/allslices.csv --out_dir $FD/tgt --num_samples 2 --reverse_eta 0.35 || { echo GEN_TGT_FAIL; exit 1; }

echo "===== [5] split src -> tr/va by case ====="
python - <<'PY'
import pandas as pd, numpy as np
d=pd.read_csv("/root/autodl-tmp/knee_allslices/fuse_sup_pseudo/src/fusion_pairs.csv")
rng=np.random.RandomState(42)
if "case_id" in d.columns:
    cs=d["case_id"].unique(); va_c=set(rng.choice(cs,size=max(1,int(len(cs)*0.15)),replace=False))
    va=d[d["case_id"].isin(va_c)]; tr=d[~d["case_id"].isin(va_c)]
else:
    idx=rng.permutation(len(d)); k=int(len(d)*0.15); va=d.iloc[idx[:k]]; tr=d.iloc[idx[k:]]
tr.to_csv("/root/autodl-tmp/knee_allslices/fuse_sup_pseudo/tr.csv",index=False)
va.to_csv("/root/autodl-tmp/knee_allslices/fuse_sup_pseudo/va.csv",index=False)
print("tr",len(tr),"va",len(va))
PY

echo "===== [6] train fusion classifier ====="
python fusion_classifier.py --mode train \
  --train_csv $FD/tr.csv --val_csv $FD/va.csv \
  --before_col before_png --other_cols translated_png,sample_0,sample_1 --label_col label \
  --out_dir $FD/run --epochs 12 --batch_size 16 --backbone resnet50 || { echo FUSION_TRAIN_FAIL; exit 1; }

echo "===== [7] +1+2+3+4 fusion eval (volume) [PSEUDO] ====="
echo "@@@ RESULT_1234 @@@"
python fusion_classifier.py --mode eval --weights $FD/run/best.pt \
  --test_csv $FD/tgt/fusion_pairs.csv --before_col before_png \
  --other_cols translated_png,sample_0,sample_1 --label_col label

echo "PSEUDO_SUP_ALL_DONE $(date)"
