#!/usr/bin/env bash
# Multi-seed 2x2 fusion ablation on breast + knee, to average out the
# source-val model-selection noise. Reports per-cell mean +/- std over seeds.
PY=/root/miniconda3/bin/python
C=/root/autodl-tmp/breast/cache
M=/root/autodl-tmp/meanproj_stage
E=/root/autodl-tmp/breast/exp/multiseed
mkdir -p $E
RES=$E/results.csv
echo "dataset,cell,seed,auc" > $RES
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
COLS=fake_1,fake_2,fake_3,fake_4,fake_5

run_one() {
  ds=$1; tr=$2; va=$3; ev=$4; cell=$5; extra=$6; seed=$7
  od=$E/${ds}_${cell}_s${seed}; mkdir -p $od
  $PY fusion_classifier.py --mode train --train_csv $tr --val_csv $va \
    --before_col before_png --other_cols $COLS --label_col label \
    --fusion_mode orig_kv --out_dir $od --epochs 30 --batch_size 16 --lr 1e-4 --seed $seed $extra >/dev/null 2>&1
  auc=$($PY fusion_classifier.py --mode eval --weights $od/best.pt --test_csv $ev \
    --before_col before_png --other_cols $COLS --label_col label 2>/dev/null \
    | grep -oE "slice_AUC=[0-9.]+" | head -1 | cut -d= -f2)
  echo "$ds,$cell,$seed,$auc" >> $RES
  echo "done $ds/$cell/seed$seed -> auc=$auc"
}

for dsspec in \
  "breast|$C/fusion_train_busi.csv|$C/fusion_val_busi.csv|$C/fusion_eval_breast_diag.csv" \
  "knee|$M/knee_fusion_train_mrnet.csv|$M/knee_fusion_val_mrnet.csv|$M/knee_fusion_eval_kneemri.csv" ; do
  IFS='|' read -r ds tr va ev <<< "$dsspec"
  for seed in 42 43 44 ; do
    for cellspec in "base|" "supcon|--supcon_weight 0.1" "stat|--stat_prior" "stat_supcon|--stat_prior --supcon_weight 0.1" ; do
      cell="${cellspec%%|*}"; extra="${cellspec#*|}"
      run_one "$ds" "$tr" "$va" "$ev" "$cell" "$extra" "$seed"
    done
  done
  echo "===${ds} DONE==="
done

echo "===AGGREGATE==="
$PY - <<PYEOF
import pandas as pd
df=pd.read_csv("$RES")
g=df.groupby(["dataset","cell"])["auc"].agg(["mean","std","count"]).reset_index()
for _,r in g.iterrows():
    print(f"{r['dataset']:6s} {r['cell']:12s} mean={r['mean']:.4f} std={r['std']:.4f} n={int(r['count'])}")
PYEOF
echo ALL_DONE
