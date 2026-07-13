#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
CODE=/root/autodl-tmp/knee/code2/idea2_diffusion_baseline
AS=/root/autodl-tmp/knee_allslices
FD=$AS/fuse_sup_pseudo
LOG=$AS/fuse_moment_result.log
cd $CODE/bbdm_strict
exec > >(tee -a $LOG) 2>&1
echo "############ FUSE + MOMENT start $(date) ############"

echo "===== [1] add moment_png column to src/tgt pairs ====="
python gen_moment_col.py --pairs_csv $FD/src/fusion_pairs.csv --source_csv $AS/mrnet_train/allslices.csv --out_dir $FD/src/moment || { echo MOM_SRC_FAIL; exit 1; }
python gen_moment_col.py --pairs_csv $FD/tgt/fusion_pairs.csv --source_csv $AS/mrnet_train/allslices.csv --out_dir $FD/tgt/moment || { echo MOM_TGT_FAIL; exit 1; }

echo "===== [2] re-split src -> tr/va by case (with moment col) ====="
python - <<'PY'
import pandas as pd, numpy as np
d=pd.read_csv("/root/autodl-tmp/knee_allslices/fuse_sup_pseudo/src/fusion_pairs.csv")
rng=np.random.RandomState(42)
cs=d["case_id"].unique(); va_c=set(rng.choice(cs,size=max(1,int(len(cs)*0.15)),replace=False))
va=d[d["case_id"].isin(va_c)]; tr=d[~d["case_id"].isin(va_c)]
tr.to_csv("/root/autodl-tmp/knee_allslices/fuse_sup_pseudo/tr_m.csv",index=False)
va.to_csv("/root/autodl-tmp/knee_allslices/fuse_sup_pseudo/va_m.csv",index=False)
print("tr",len(tr),"va",len(va))
PY

# V1: 矩匹配图当 query, 其它=翻译+采样
echo "===== [V1] query=moment, others=translated+samples ====="
python fusion_classifier.py --mode train --train_csv $FD/tr_m.csv --val_csv $FD/va_m.csv \
  --before_col moment_png --other_cols translated_png,sample_0,sample_1 --label_col label \
  --out_dir $FD/run_v1 --epochs 12 --batch_size 16 --backbone resnet50 || { echo V1_FAIL; exit 1; }
echo "@@@ RESULT_V1 @@@"
python fusion_classifier.py --mode eval --weights $FD/run_v1/best.pt \
  --test_csv $FD/tgt/fusion_pairs.csv --before_col moment_png \
  --other_cols translated_png,sample_0,sample_1 --label_col label

# V2: 原图当 query, 额外并入矩匹配图
echo "===== [V2] query=before, others=moment+translated+samples ====="
python fusion_classifier.py --mode train --train_csv $FD/tr_m.csv --val_csv $FD/va_m.csv \
  --before_col before_png --other_cols moment_png,translated_png,sample_0,sample_1 --label_col label \
  --out_dir $FD/run_v2 --epochs 12 --batch_size 16 --backbone resnet50 || { echo V2_FAIL; exit 1; }
echo "@@@ RESULT_V2 @@@"
python fusion_classifier.py --mode eval --weights $FD/run_v2/best.pt \
  --test_csv $FD/tgt/fusion_pairs.csv --before_col before_png \
  --other_cols moment_png,translated_png,sample_0,sample_1 --label_col label

echo "FUSE_MOMENT_DONE $(date)"
