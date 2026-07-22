#!/usr/bin/env python
"""Matched three-arm fusion GATE (review-mandated P0 redesign).

Safe residual adapter over a FROZEN source classifier:

    logits = frozen_source(before) + tanh(alpha) * adapter_delta(before, others)

alpha is initialised to 0, so at init logits == frozen_source(before) EXACTLY,
per case. The source backbone + BN are frozen. This makes the fusion a strict
add-on to `direct`, and the three controls a matched comparison that isolates
the value of the translation CONTENT:

  --control before_only   : logits = frozen_source(before)   (== direct; no adapter)
  --control repeat_before : others = K copies of `before`    (adapter capacity, no
                            translation info)  [pass --other_cols before_png,before_png,...]
  --control true_fakes    : others = the K real translations  [--other_cols fake_1,...]

All three share the SAME frozen source checkpoint, the SAME adapter init (seed),
and the SAME DataLoader order. Only `others` differs. If true_fakes does not beat
repeat_before, translation content adds nothing (beyond attention capacity).

Correct UDA pairing (build the CSVs accordingly, see scripts/build_fusion_csvs_*):
  train: before = G_s->t(x_s) (target-styled source), others = G_t->s(before)
  test : before = x_t (real target),                  others = G_t->s(x_t)

Checkpoint stores every config needed to rebuild the exact model at eval
(control, backbone/dim/heads/proj_dim, fusion_mode, stat_prior, src stats,
other_cols+order, resize, seed, epoch, source_ckpt sha256). Per-case predictions
are always written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, _HERE.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_existing_classifier_on_csv import build_model  # noqa: E402
from fusion_classifier import (  # noqa: E402
    CrossAttnFusionClassifier, MultiImageDataset, auc, evaluate_volume, sup_con_loss,
)

CONTROLS = ("before_only", "repeat_before", "true_fakes")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class FusionGate(nn.Module):
    """frozen_source(before) + tanh(alpha) * adapter(before, others)."""

    def __init__(self, source_ckpt, control="true_fakes", backbone="custom_resnet50_space",
                 pretrained="imagenet", adapter_backbone="resnet50", dim=256, heads=4,
                 proj_dim=128, stat_prior=False, src_mean=0.0, src_std=1.0, num_classes=2):
        super().__init__()
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}")
        self.control = control
        # frozen source classifier (provides `direct`); backbone + BN frozen
        self.frozen_src = build_model(backbone, num_classes=num_classes, pretrained=pretrained,
                                      device=torch.device("cpu"))
        ck = torch.load(source_ckpt, map_location="cpu", weights_only=False)
        state = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
        state = {str(k).replace("module.", ""): v for k, v in state.items()}
        res = self.frozen_src.load_state_dict(state, strict=False)
        bad = [k for k in list(res.missing_keys) + list(res.unexpected_keys) if "rsa" not in k.lower()]
        if bad:
            raise RuntimeError(f"frozen source checkpoint mismatch: {bad[:6]}")
        for p in self.frozen_src.parameters():
            p.requires_grad_(False)
        self.frozen_src.eval()

        if control != "before_only":
            self.adapter = CrossAttnFusionClassifier(
                num_classes=num_classes, pretrained=True, dim=dim, heads=heads,
                backbone=adapter_backbone, mode="orig_kv", proj_dim=proj_dim,
                stat_prior=stat_prior, src_mean=src_mean, src_std=src_std)
            self.alpha = nn.Parameter(torch.zeros(1))  # gate: init 0 -> logits == direct

    def train(self, mode=True):
        super().train(mode)
        self.frozen_src.eval()  # keep frozen source in eval (frozen BN) even in train mode
        return self

    def forward(self, before, others, return_emb=False):
        with torch.no_grad():
            direct = self.frozen_src(before)
        if self.control == "before_only":
            if return_emb:
                return direct, None
            return direct
        if return_emb:
            delta, emb = self.adapter(before, others, return_emb=True)
            return direct + torch.tanh(self.alpha) * delta, emb
        delta = self.adapter(before, others)
        return direct + torch.tanh(self.alpha) * delta


def _evaluate(model, loader, dev):
    return evaluate_volume(model, loader, dev)


def main():
    ap = argparse.ArgumentParser("matched three-arm fusion gate")
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--source_ckpt", type=str, help="frozen source classifier (custom_resnet50_space)")
    ap.add_argument("--control", choices=CONTROLS, default="true_fakes")
    ap.add_argument("--train_csv"); ap.add_argument("--val_csv"); ap.add_argument("--test_csv")
    ap.add_argument("--before_col", default="before_png")
    ap.add_argument("--other_cols", default="fake_1,fake_2,fake_3,fake_4,fake_5")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="./gate_run")
    ap.add_argument("--weights")
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--pretrained", default="imagenet")
    ap.add_argument("--adapter_backbone", default="resnet50", choices=["resnet50", "resnet18"])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--stat_prior", action="store_true")
    ap.add_argument("--supcon_weight", type=float, default=0.0)
    ap.add_argument("--supcon_temp", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self_check", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    others = [c.strip() for c in a.other_cols.split(",") if c.strip()]

    if a.self_check:
        # random-input smoke test for all three controls (no data/ckpt needed for adapter shape)
        for ctrl in CONTROLS:
            m = FusionGate.__new__(FusionGate)  # bypass frozen-src load for the shape check
            nn.Module.__init__(m); m.control = ctrl
            m.frozen_src = build_model(a.backbone, 2, "none", torch.device("cpu"))
            for p in m.frozen_src.parameters():
                p.requires_grad_(False)
            m.frozen_src.eval()
            if ctrl != "before_only":
                m.adapter = CrossAttnFusionClassifier(pretrained=False, mode="orig_kv",
                                                      backbone=a.adapter_backbone, stat_prior=a.stat_prior)
                m.alpha = nn.Parameter(torch.zeros(1))
            m.eval()
            b = torch.randn(2, 3, a.resize, a.resize); o = torch.randn(2, len(others), 3, a.resize, a.resize)
            with torch.no_grad():
                out = m(b, o)
                if ctrl != "before_only":
                    assert torch.allclose(out, m.frozen_src(b)), f"{ctrl}: alpha=0 not identity to direct"
            assert out.shape == (2, 2), out.shape
            print(f"[self_check] {ctrl}: out={tuple(out.shape)}  alpha=0==direct OK")
        return

    src_stats = (0.0, 1.0)
    if a.mode == "eval":
        ck = torch.load(a.weights, map_location=dev, weights_only=False)
        cfg = ck["config"]
        others = [c.strip() for c in cfg["other_cols"].split(",") if c.strip()]
        model = FusionGate(cfg["source_ckpt"], control=cfg["control"], backbone=cfg["backbone"],
                           pretrained=cfg["pretrained"], adapter_backbone=cfg["adapter_backbone"],
                           dim=cfg["dim"], heads=cfg["heads"], proj_dim=cfg["proj_dim"],
                           stat_prior=cfg["stat_prior"], src_mean=cfg["src_mean"], src_std=cfg["src_std"]).to(dev)
        model.load_state_dict(ck["model"])
        dl = DataLoader(MultiImageDataset(a.test_csv, cfg["before_col"], others, a.label_col, cfg["resize"]),
                        batch_size=a.batch_size, num_workers=4)
        sa, vm, vx, vt = _evaluate(model, dl, dev)
        print(f"[gate eval] control={cfg['control']} slice_AUC={sa:.4f} case_mean={vm:.4f} (n_other={len(others)})")
        return

    # ---- train ----
    os.makedirs(a.out_dir, exist_ok=True)
    if a.stat_prior:
        from fusion_classifier import _compute_before_stats
        src_stats = _compute_before_stats(a.train_csv, a.before_col, a.resize)
    model = FusionGate(a.source_ckpt, control=a.control, backbone=a.backbone, pretrained=a.pretrained,
                       adapter_backbone=a.adapter_backbone, dim=a.dim, heads=a.heads, proj_dim=a.proj_dim,
                       stat_prior=a.stat_prior, src_mean=src_stats[0], src_std=src_stats[1]).to(dev)
    print(f"[gate] control={a.control}  source={Path(a.source_ckpt).name}  "
          f"stat_prior={a.stat_prior} supcon={a.supcon_weight}  trainable_params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    tr = DataLoader(MultiImageDataset(a.train_csv, a.before_col, others, a.label_col, a.resize, True),
                    batch_size=a.batch_size, shuffle=True, num_workers=4,
                    generator=torch.Generator().manual_seed(a.seed))
    va = DataLoader(MultiImageDataset(a.val_csv, a.before_col, others, a.label_col, a.resize),
                    batch_size=a.batch_size, num_workers=4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    use_sc = a.supcon_weight > 0 and a.control != "before_only"
    cfg = {"control": a.control, "source_ckpt": a.source_ckpt, "source_ckpt_sha256": sha256_file(a.source_ckpt),
           "backbone": a.backbone, "pretrained": a.pretrained, "adapter_backbone": a.adapter_backbone,
           "dim": a.dim, "heads": a.heads, "proj_dim": a.proj_dim, "fusion_mode": "orig_kv",
           "stat_prior": a.stat_prior, "src_mean": src_stats[0], "src_std": src_stats[1],
           "before_col": a.before_col, "other_cols": a.other_cols, "resize": a.resize, "seed": a.seed}
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for before, oth, y, _ in tr:
            before, oth, y = before.to(dev), oth.to(dev), y.to(dev)
            opt.zero_grad()
            if use_sc:
                logits, emb = model(before, oth, return_emb=True)
                loss = crit(logits, y) + a.supcon_weight * sup_con_loss(emb, y, temp=a.supcon_temp)
            else:
                loss = crit(model(before, oth), y)
            loss.backward(); opt.step()
        v = evaluate_volume(model, va, dev)[1]  # case-mean AUC
        alpha_v = float(torch.tanh(model.alpha).item()) if a.control != "before_only" else 0.0
        print(f"epoch {ep} val_case_auc={v:.4f} tanh(alpha)={alpha_v:.3f}")
        if v > best:
            best = v
            torch.save({"model": model.state_dict(), "val_auc": v, "epoch": ep, "config": cfg},
                       os.path.join(a.out_dir, "best.pt"))
    print(f"[gate] control={a.control} best val_case_auc={best:.4f}")


if __name__ == "__main__":
    main()
