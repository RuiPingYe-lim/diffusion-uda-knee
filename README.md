# Diffusion-based Unsupervised Domain Adaptation for Knee MRI

Unsupervised domain adaptation (UDA) between two knee-MRI datasets (**MRNet** and **KneeMRI**) via
generative image translation. A source classifier is trained on the labeled source domain; target
images are translated toward the source style and classified. The repo also contains the analysis
and diagnostic experiments that motivated the design.

> Companion to the first paper (feature-space / NMF-prototype UDA). This repo covers the second line
> of work: **diffusion / generative image translation** for the same benchmark.

## Method

1. **KL-VAE autoencoder** (`src/vqgan_pretrain/`) — compresses images into a continuous latent space
   (skip-free, so all information passes through the bottleneck, as required for latent diffusion).
2. **Brownian Bridge Diffusion Model, BBDM** (`src/bbdm_strict/`) — a two-endpoint-anchored bridge
   that translates target latents toward source latents:
   `x_t = (1-m_t)·x_A + m_t·x_B + σ_t·ε`, bridge target `bb_t = x_t - x_A`.
3. **Analysis & targeted transforms** (`src/bbdm_strict/style_gap.py`, `eval_style_match.py`, ...) —
   quantify where the domain gap lives (intensity/frequency) and test simple, content-preserving
   alternatives (moment matching, histogram matching).
4. **Cross-attention fusion classifier** (`src/bbdm_strict/fusion_classifier.py`) — classify using the
   original image as query, cross-attending to translated / sampled views.

## Repository layout

```
src/
├── train_source_classifier.py          # train the source-domain classifier
├── eval_existing_classifier_on_csv.py   # model builder + eval helpers
├── pseudo_label_target.py               # pseudo-label target for unsupervised pairing
├── vqgan_pretrain/                       # KL-VAE autoencoder
│   ├── models_vqgan.py                  # KLVAE / VQGAN models
│   ├── train_vqgan.py                   # AE training
│   └── losses_vqgan.py, datasets_vqgan.py, utils_vqgan.py, reconstruct_vqgan.py
└── bbdm_strict/                          # Brownian-bridge diffusion + analysis + fusion
    ├── models_strict_bbdm.py            # bridge U-Net
    ├── bridge_scheduler.py              # Brownian-bridge scheduler
    ├── ae_frontend.py                   # VAE frontend (encode/decode latents)
    ├── datasets_strict_bbdm.py          # (pseudo-)paired dataset
    ├── contrastive.py                   # supervised contrastive loss (advisor suggestion 2)
    ├── train_strict_bbdm.py             # BBDM training
    ├── sample_strict_bbdm.py            # reverse sampling (translation)
    ├── sample_guided.py                 # classifier-guided sampling
    ├── eval_volume.py                   # volume-level AUC eval (direct / translate)
    ├── fusion_classifier.py             # cross-attention fusion classifier (suggestions 4/5)
    ├── gen_fusion_pairs.py              # generate before/translated/sample views for fusion
    ├── gen_moment_col.py                # add moment-matched column to fusion pairs
    ├── style_gap.py                     # intensity + FFT domain-gap analysis
    ├── eval_style_match.py, eval_style_match2.py   # moment / histogram matching
    ├── eval_vae_recon.py                # isolate VAE vs bridge information loss
    ├── eval_partial_translate.py        # partial-translation (t0) sweep
    ├── eval_freqmix.py                  # frequency-decoupled translation prototype
    ├── eval_late_fusion.py              # decision-level ensemble
    ├── precompute_meanproj.py           # mean-projection preprocessing (paper-1 aligned pipeline)
    ├── train_eval_meanproj.py           # source-only train/eval on mean-projection representation
    └── configs/                          # BBDM / VAE configs (JSON)
└── unsb/                                 # integration with the external UNSB translator (see below)
    ├── eval_unsb_translation.py         # classify UNSB translation-only outputs
    ├── build_unsb_fusion_csv.py         # build fusion CSVs from UNSB outputs
    └── README.md
scripts/                                  # one-shot driver scripts (edit paths before use)
```

## Setup

```bash
pip install -r requirements.txt
```
Tested with PyTorch 2.x. Set `PYTHONPATH` to the repo `src/` (some modules import the package):
```bash
export PYTHONPATH=/path/to/repo/src:$PYTHONPATH
```

## Data

Not included. Each dataset is a set of grayscale slices/volumes with a CSV of `image_path,label,case_id`.
Config JSONs and scripts contain **absolute paths that must be edited** to your data locations.

## Typical workflow

```bash
# 1. train KL-VAE
python src/vqgan_pretrain/train_vqgan.py   --config <klvae_config.json>
# 2. train BBDM (uses the frozen VAE)
python src/bbdm_strict/train_strict_bbdm.py --config src/bbdm_strict/configs/bbdm_knee_allslices.json
# 3. translate target + volume-level AUC
python src/bbdm_strict/eval_volume.py --mode translate --clf_ckpt <cls.pt> \
    --slice_csv <target_test.csv> --bbdm_config <cfg.json> --bbdm_ckpt <bbdm_latest.pt>
# 4. domain-gap analysis + moment matching
python src/bbdm_strict/style_gap.py       --source_csv <src.csv> --target_csv <tgt.csv>
python src/bbdm_strict/eval_style_match.py --clf_ckpt <cls.pt> --source_csv <src.csv> --target_csv <tgt.csv>
# 5. cross-attention fusion
python src/bbdm_strict/gen_fusion_pairs.py  --config <cfg.json> --checkpoint <bbdm.pt> --input_csv <csv> --out_dir <dir>
python src/bbdm_strict/fusion_classifier.py --mode train  ...
python src/bbdm_strict/fusion_classifier.py --mode eval   ...
```

## Notes

- **UNSB comparison** (`src/unsb/`): a newer, sharper unpaired translator — Unpaired Neural Schrödinger
  Bridge — was compared against BBDM. **UNSB itself is an external repo**
  ([cyclomon/UNSB](https://github.com/cyclomon/UNSB)) and is **not vendored here**; `src/unsb/` contains
  only our integration (translation-only eval, fusion-CSV builder) and the drivers are
  `scripts/run_unsb_*.sh`. Finding: UNSB images look much better than BBDM, but downstream AUC did not
  improve — image quality and discriminative usefulness are decoupled. See `src/unsb/README.md`.
- **Pipeline alignment**: `precompute_meanproj.py` / `train_eval_meanproj.py` reproduce the first paper's
  mean-projection representation so both papers share one benchmark.
- Configs/scripts contain **absolute paths** — edit them for your environment.
