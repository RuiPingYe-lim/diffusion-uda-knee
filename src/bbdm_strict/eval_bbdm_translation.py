#!/usr/bin/env python
"""Evaluate a trained BBDM translation with the FROZEN source classifier.

Reads a BBDM sampling run's sample_pairs.csv (columns include target_path,
target_label, before_png, ae_recon_png, translated_png) and scores three arms
with the frozen BUSI source-only classifier, all inside the SAME 128-latent
pipeline so they are mutually comparable:

  before      : the target image at the BBDM resolution (128) -> classifier
  ae_recon    : target through the KL-VAE only (no bridge) -> isolates the VAE cost
  translated  : target through KL-VAE + Brownian bridge -> the full BBDM translation

Three paired case-level bootstrap CIs on shared indices:
  translated - before     (does the full BBDM translation help vs raw-128?)
  translated - ae_recon    (net contribution of the bridge over the VAE alone)
  ae_recon   - before      (VAE bottleneck cost)

Decision (translated - before, mirrors the b' rule):
  Delta >= 0.01 -> BBDM translation gives a gain worth pursuing
  Delta <= 0    -> no gain
  0<Delta<0.01  -> very weak

Note: the saved PNGs encode a [-1,1] tensor as [0,255]; ToTensor->Normalize(0.5)
recovers [-1,1], matching the classifier's eval transform. The 'before' arm is
the 128-pipeline baseline, NOT the full-res b' direct (0.7909).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

_THIS = Path(__file__).resolve()
for _p in (_THIS.parent, _THIS.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eval_existing_classifier_on_csv import build_model  # noqa: E402

_TT = T.ToTensor()
_MIN_EFFECT = 0.01


def load_clf_input(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    g = T.Resize((image_size, image_size), antialias=True)(_TT(img)).clamp(0, 1)  # [1,S,S]
    return (g.repeat(3, 1, 1) * 2.0 - 1.0)  # [-1,1], matches Normalize(0.5,0.5)


@torch.no_grad()
def score_column(model, paths: List[str], image_size, device, batch=64) -> np.ndarray:
    probs: List[float] = []
    buf: List[torch.Tensor] = []

    def flush():
        if buf:
            x = torch.stack(buf).to(device)
            probs.extend(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy().tolist())
            buf.clear()

    for p in paths:
        buf.append(load_clf_input(p, image_size))
        if len(buf) >= batch:
            flush()
    flush()
    return np.asarray(probs)


def three_contrast_bootstrap(y, p_before, p_recon, p_trans, n_boot, seed) -> Dict:
    rng = np.random.RandomState(seed); n = len(y)
    d_tb, d_tr, d_rb = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n); yl = y[idx]
        if yl.min() == yl.max():
            continue
        a_b = roc_auc_score(yl, p_before[idx])
        a_r = roc_auc_score(yl, p_recon[idx])
        a_t = roc_auc_score(yl, p_trans[idx])
        d_tb.append(a_t - a_b); d_tr.append(a_t - a_r); d_rb.append(a_r - a_b)

    def ci(a):
        a = np.asarray(a)
        return {"lo": round(float(np.percentile(a, 2.5)), 4), "hi": round(float(np.percentile(a, 97.5)), 4),
                "boot_mean": round(float(a.mean()), 4)}
    return {"translated_minus_before": ci(d_tb), "translated_minus_ae_recon": ci(d_tr),
            "ae_recon_minus_before": ci(d_rb)}


def decide(delta: float) -> str:
    if delta >= _MIN_EFFECT:
        return f"translated-before Delta={delta:+.4f} >= 0.01 -> BBDM translation gain worth pursuing"
    if delta <= 0:
        return f"translated-before Delta={delta:+.4f} <= 0 -> no gain from BBDM translation"
    return f"translated-before Delta={delta:+.4f} in (0,0.01) -> very weak"


def main() -> None:
    ap = argparse.ArgumentParser("evaluate trained BBDM translation with frozen classifier")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--sample_pairs_csv", type=Path, required=True)
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" not in args.device or torch.cuda.is_available()) else "cpu")
    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    ck = torch.load(args.classifier, map_location=device, weights_only=False)
    state = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    r = model.load_state_dict(state, strict=False)
    bad = [k for k in list(r.missing_keys) + list(r.unexpected_keys) if "rsa" not in k.lower()]
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:6]}")
    model.eval()

    df = pd.read_csv(args.sample_pairs_csv)
    for c in ["target_path", "target_label", "before_png", "ae_recon_png", "translated_png"]:
        if c not in df.columns:
            raise ValueError(f"missing column {c} in {args.sample_pairs_csv}")
    # case key = target_path (verified 1 img/case); dedupe defensively
    df = df.drop_duplicates(subset=["target_path"]).reset_index(drop=True)
    y = df["target_label"].to_numpy(int)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())

    p_before = score_column(model, df["before_png"].tolist(), args.image_size, device)
    p_recon = score_column(model, df["ae_recon_png"].tolist(), args.image_size, device)
    p_trans = score_column(model, df["translated_png"].tolist(), args.image_size, device)

    auc_before = float(roc_auc_score(y, p_before))
    auc_recon = float(roc_auc_score(y, p_recon))
    auc_trans = float(roc_auc_score(y, p_trans))
    delta_tb = auc_trans - auc_before
    ci = three_contrast_bootstrap(y, p_before, p_recon, p_trans, args.n_boot, args.seed)

    out = {
        "n_cases": int(len(df)), "n_pos": n_pos, "n_neg": n_neg,
        "auc_before_128": round(auc_before, 4),
        "auc_ae_recon": round(auc_recon, 4),
        "auc_translated": round(auc_trans, 4),
        "CI_translated_minus_before": ci["translated_minus_before"],
        "CI_translated_minus_ae_recon": ci["translated_minus_ae_recon"],
        "CI_ae_recon_minus_before": ci["ae_recon_minus_before"],
        "decision": decide(delta_tb),
    }
    print("\n=========== BBDM translation eval (frozen classifier) ===========")
    print(f"  n={out['n_cases']} (pos {n_pos}/neg {n_neg})  [128-latent pipeline]")
    print(f"  AUC  before(128)={auc_before:.4f}  ae_recon={auc_recon:.4f}  translated={auc_trans:.4f}")
    for name, key in [("translated - before", "translated_minus_before"),
                      ("translated - ae_recon", "translated_minus_ae_recon"),
                      ("ae_recon - before", "ae_recon_minus_before")]:
        c = ci[key]
        print(f"  {name:22s} boot_mean={c['boot_mean']:+.4f}  CI[{c['lo']:+.4f}, {c['hi']:+.4f}]")
    print(f"  DECISION: {out['decision']}")
    print("=================================================================\n")

    if args.out_csv:
        pd.DataFrame({"target_path": df["target_path"], "label": y,
                      "prob_before": p_before, "prob_ae_recon": p_recon, "prob_translated": p_trans}
                     ).to_csv(args.out_csv, index=False)
        print(f"[write] {args.out_csv}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
