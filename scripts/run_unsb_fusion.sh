#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
M=/root/autodl-tmp/meanproj_stage
UK=/root/autodl-tmp/UNSB/results/k2m_SB/test_latest/images        # KneeMRI 翻译(目标)
UM=/root/autodl-tmp/UNSB/results_mrnet/k2m_SB/test_latest/images  # MRNet 翻译(源)
OUT=/root/autodl-tmp/unsb_fusion
mkdir -p $OUT
LOG=/root/autodl-tmp/unsb_fusion.log
exec > >(tee -a $LOG) 2>&1
echo "###### UNSB + 交叉注意力融合 start ######"

python - <<'PY'
import os, pandas as pd
M="/root/autodl-tmp/meanproj_stage"
UK="/root/autodl-tmp/UNSB/results/k2m_SB/test_latest/images"
UM="/root/autodl-tmp/UNSB/results_mrnet/k2m_SB/test_latest/images"
OUT="/root/autodl-tmp/unsb_fusion"
def build(label_csv, before_dir, unsb_dir, out_csv):
    df=pd.read_csv(label_csv); rows=[]
    for _,r in df.iterrows():
        c=str(r["case_id"]); bp=f"{before_dir}/{c}.png"
        f1,f3,f5=f"{unsb_dir}/fake_1/{c}.png",f"{unsb_dir}/fake_3/{c}.png",f"{unsb_dir}/fake_5/{c}.png"
        if not all(os.path.isfile(p) for p in [bp,f1,f3,f5]): continue
        rows.append({"before_png":bp,"f1":f1,"f3":f3,"f5":f5,"label":int(r["label"]),"case_id":c})
    pd.DataFrame(rows).to_csv(out_csv,index=False); print(out_csv, len(rows))
build(f"{M}/mrnet_train.csv", f"{M}/mrnet_train", UM, f"{OUT}/src_train.csv")
build(f"{M}/mrnet_valid.csv", f"{M}/mrnet_valid", UM, f"{OUT}/src_val.csv")
build(f"{M}/kneemri_test.csv",f"{M}/kneemri_test",UK, f"{OUT}/tgt_test.csv")
PY

echo "===== 训练交叉注意力融合(before=原图, others=UNSB fake_1/3/5) ====="
python fusion_classifier.py --mode train \
  --train_csv $OUT/src_train.csv --val_csv $OUT/src_val.csv \
  --before_col before_png --other_cols f1,f3,f5 --label_col label \
  --out_dir $OUT/run --epochs 15 --batch_size 16 --backbone resnet50 --seed 1 || { echo FUSION_FAIL; exit 1; }

echo "@@@ UNSB_FUSION_RESULT @@@"
python fusion_classifier.py --mode eval --weights $OUT/run/best.pt \
  --test_csv $OUT/tgt_test.csv --before_col before_png --other_cols f1,f3,f5 --label_col label
echo "UNSB_FUSION_DONE"
