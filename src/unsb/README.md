# UNSB integration (newer translation baseline)

We compared BBDM against **UNSB — Unpaired Neural Schrödinger Bridge** (a newer, sharper unpaired
image-to-image translation method).

> **UNSB itself is an external repository — NOT included here.**
> Clone it separately: https://github.com/cyclomon/UNSB (ICLR 2024, built on CUT).
> This folder only contains **our integration and evaluation** on top of UNSB's outputs.

## What we did

- **Direction**: translate target → source style (e.g. KneeMRI → MRNet), matching BBDM, so results
  are comparable.
- **Representation**: the mean-projection images produced by `../bbdm_strict/precompute_meanproj.py`
  are placed into UNSB's CUT-style dataset folders (`trainA/ trainB/ testA/ testB/`).
- UNSB outputs several NFE steps per image (`fake_1 … fake_5`); more steps = stronger style.

## Files here

| file | purpose |
|---|---|
| `eval_unsb_translation.py` | classify UNSB translation-only outputs (fake_1/3/5) with the source classifier; compare to direct transfer |
| `build_unsb_fusion_csv.py` | build fusion-classifier CSVs (before + fake_1/3/5) from UNSB outputs |

The cross-attention fusion classifier itself is `../bbdm_strict/fusion_classifier.py`.
End-to-end driver scripts are in `../../scripts/run_unsb_fusion.sh` and `run_unsb_final.sh`.

## Workflow

```bash
# 0. clone UNSB separately and train it on the mean-projection data
#    (trainA = target domain, trainB = source domain)
git clone https://github.com/cyclomon/UNSB
python UNSB/train.py --dataroot <cut_dataset> --name k2m_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0
python UNSB/test.py  --dataroot <cut_dataset> --name k2m_SB --mode sb --eval --phase test --num_test 999

# 1. translation-only AUC vs direct transfer
python eval_unsb_translation.py --clf_ckpt <src_clf.pt> --label_csv <tgt_test.csv> \
    --before_dir <tgt meanproj dir> --unsb_dir <UNSB results .../images>

# 2. cross-attention fusion (original + UNSB views)
python build_unsb_fusion_csv.py --label_csv <src_train.csv> --before_dir <src meanproj> --unsb_dir <UNSB(src) results> --out_csv src_train.csv
python build_unsb_fusion_csv.py --label_csv <tgt_test.csv>  --before_dir <tgt meanproj> --unsb_dir <UNSB(tgt) results> --out_csv tgt_test.csv
python ../bbdm_strict/fusion_classifier.py --mode train --train_csv src_train.csv --val_csv src_val.csv \
    --before_col before_png --other_cols f1,f3,f5 --label_col label --out_dir run
python ../bbdm_strict/fusion_classifier.py --mode eval  --weights run/best.pt \
    --test_csv tgt_test.csv --before_col before_png --other_cols f1,f3,f5 --label_col label
```

## Finding

UNSB produces **visibly sharper, structure-preserving** translations than BBDM, **but the downstream
classification AUC did not improve** — translation-only stayed below direct transfer, and the fusion
gain came from the original image, not the translation. Image quality and discriminative usefulness
are decoupled for this task.
