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
| `dosc_modules.py` | target-reference style encoder, CIDP, and legacy projection/GRL ablations |
| `dosc_sb_model.py` | upstream UNSB model overlay using target exemplar conditions instead of random style noise |
| `dosc_unaligned_dataset.py` | strict source-labeled / target-unlabeled dataset overlay |
| `trsc_sb_model.py` | canonical model alias for new runs (`--model trsc_sb`) |
| `trsc_joint_sb_model.py` | warm-started end-to-end classifier + TRSC training on raw and \(K\) unfiltered U1 views |
| `trsc_joint_modules.py` | exact source-classifier architecture, task-gradient routing, and K-invariant multi-view CE |
| `trsc_dabrf_joint_sb_model.py` | U1-only diagnosis-aware residual repair on the K-view joint baseline |
| `dabrf_modules.py` | spatial residual gate, hard radius projection, and fixed target-style progress metric |
| `trsc_unaligned_dataset.py` | strict source labels plus \(K\) unique unlabeled target references |
| `style_swap_metrics.py` | validated common-noise reference-swap and fixed-reference noise-control statistics |
| `TRSC.md` | design, causal evidence boundary, training, and downstream attribution protocol |
| `DABRF.md` | DA-BRF formulation, gradient boundary, attribution matrix, and stop rule |
| `DOSC.md` | compatibility note for the retired method name |

The cross-attention fusion classifier itself is `../bbdm_strict/fusion_classifier.py`.
End-to-end driver scripts are in `../../scripts/run_unsb_fusion.sh` and `run_unsb_final.sh`.

## Target-reference source → target augmentation

The TRSC path reverses the old target→source inference direction: it translates labeled source
images toward target style so the translated images can augment classifier training. It extracts a
style code from an unlabeled target reference and injects it through UNSB's existing style-modulated
residual blocks. Projection and GRL did not reduce held-out diagnostic leakage and are disabled by
default. A common-noise reference-swap audit found no significant reference-driven change in output
diagnosis, and the completed three-arm downstream audit found no stable CIDP gain, so all three
controls are ablations rather than defaults. The current experiment uses \(K=3\) unique target
references, keeps every U1 candidate, warm-starts both translator and classifier, and sends the
classification CE through the candidates into the translator. See [TRSC.md](TRSC.md) for the exact
evidence boundary, losses, and commands.

DA-BRF is a new, unvalidated U1-only experiment layered on that K-view baseline. It learns to
attenuate the existing source-to-U1 residual locally, while a detached constraint branch combines
calibration-decoupled source diagnosis, target-reference style progress, and a hard residual-radius
bound. It does not restore U3/U5, and it does not claim a gain before the identity/simple-scaling
controls are run. See [DABRF.md](DABRF.md).

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
