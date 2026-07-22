#!/usr/bin/env python
"""Matched three-arm fusion GATE (review v2: zero-init residual head, no alpha).

Binary logit-MARGIN residual over a FROZEN source classifier:

    direct_margin   = frozen_source(before)[:,1] - frozen_source(before)[:,0]
    residual_margin = residual_head( adapter.encode(before, others) )   # Linear(dim,1), ZERO-init
    final_margin    = direct_margin + residual_margin
    loss = BCEWithLogits(final_margin, label) + residual_weight * residual_margin^2.mean()

Why not `direct + tanh(alpha)*delta` (the old design): with alpha=0 the adapter's
CE gradient is tanh(0)*(...) = 0, so the adapter never moves on the first batch --
a cold-start deadlock (review). A ZERO-INIT residual head instead gives (a) init
output == direct EXACTLY (per case), AND (b) a NON-zero gradient from batch 1
(d loss / d W_head = d loss / d margin * features, features != 0).

Controls (matched: same frozen source, same adapter init/seed, same DataLoader):
  --control direct         : frozen_source only; NO adapter, NO training (reference line)
  --control repeat_before  : others := K copies of `before`, CONSTRUCTED IN CODE
                             (other_cols content ignored; fake cols rejected)
  --control true_fakes     : others := the K real translation views

Primary paper comparison is true_fakes - repeat_before (both have the adapter);
`direct` is only a reference line, not a matched arm.

Correct UDA pairing (built by scripts/build_fusion_csvs_*):
  train: before = G_s->t(x_s) (target-styled source), others = G_t->s(before)
  test : before = x_t (real target),                  others = G_t->s(x_t)

Eval ALWAYS writes per-case predictions (case_id,label,direct/residual/final
margin, prob, control, seed, and checkpoint/source/manifest sha256).
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
from fusion_classifier import CrossAttnFusionClassifier, MultiImageDataset, auc, sup_con_loss  # noqa: E402

CONTROLS = ("direct", "repeat_before", "true_fakes")


def sha256_file(p):
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class FusionGate(nn.Module):
    """final_margin = direct_margin + residual_head(adapter.encode(before, others))."""

    def __init__(self, source_ckpt, control="true_fakes", backbone="custom_resnet50_space",
                 pretrained="imagenet", adapter_backbone="resnet50", dim=256, heads=4,
                 proj_dim=128, stat_prior=False, src_mean=0.0, src_std=1.0, num_classes=2):
        super().__init__()
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}")
        self.control = control
        self.dim = dim
        # frozen source classifier (provides direct margin); backbone + BN frozen
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

        if control != "direct":
            self.adapter = CrossAttnFusionClassifier(
                num_classes=num_classes, pretrained=True, dim=dim, heads=heads,
                backbone=adapter_backbone, mode="orig_kv", proj_dim=proj_dim,
                stat_prior=stat_prior, src_mean=src_mean, src_std=src_std)
            self.residual_head = nn.Linear(dim, 1)         # scalar margin correction
            nn.init.zeros_(self.residual_head.weight)      # ZERO-init -> init == direct, but grad != 0
            nn.init.zeros_(self.residual_head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.frozen_src.eval()  # keep frozen source in eval (frozen BN) even in train mode
        return self

    def forward(self, before, others, return_parts=False):
        with torch.no_grad():
            d = self.frozen_src(before)                    # [B,2]
        direct_margin = d[:, 1] - d[:, 0]                  # [B]
        if self.control == "direct":
            zero = direct_margin.new_zeros(direct_margin.shape)
            return (direct_margin, direct_margin, zero, None) if return_parts else direct_margin
        if self.control == "repeat_before":
            k = others.shape[1]
            others = before.unsqueeze(1).expand(-1, k, -1, -1, -1)  # ENFORCE K copies of before
        feat = self.adapter._rep(before, others)           # [B, dim]
        residual_margin = self.residual_head(feat).squeeze(1)      # [B]
        final_margin = direct_margin + residual_margin
        if return_parts:
            return final_margin, direct_margin, residual_margin, feat
        return final_margin


@torch.no_grad()
def evaluate_percase(model, loader, dev):
    """Return (case_auc, per_case_df). prob = sigmoid(final_margin), aggregated by case_id."""
    model.eval()
    rows = []
    for before, others, y, case in loader:
        fm, dm, rm, _ = model(before.to(dev), others.to(dev), return_parts=True)
        fm = fm.cpu().numpy(); dm = dm.cpu().numpy(); rm = rm.cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-fm))
        for i in range(len(y)):
            rows.append({"case_id": str(case[i]), "label": int(y[i]),
                         "direct_margin": float(dm[i]), "residual_margin": float(rm[i]),
                         "final_margin": float(fm[i]), "prob_positive": float(prob[i])})
    df = pd.DataFrame(rows)
    cg = df.groupby("case_id", sort=False).agg(label=("label", "first"),
                                               prob_positive=("prob_positive", "mean")).reset_index()
    a = auc(cg["label"].values, cg["prob_positive"].values)
    return a, df, cg


def main():
    ap = argparse.ArgumentParser("matched three-arm fusion gate (v2)")
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--source_ckpt", type=str)
    ap.add_argument("--generator_ckpt", type=str, default=None, help="G_t->s ckpt, for provenance hash")
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
    ap.add_argument("--residual_weight", type=float, default=0.0, help="lambda on residual_margin^2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=Path, default=None, help="per-case predictions (defaults to out_dir/eval)")
    ap.add_argument("--self_check", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    others = [c.strip() for c in a.other_cols.split(",") if c.strip()]

    # code-level guard: repeat_before must NOT be fed fake columns; true_fakes needs >=2 distinct
    if a.control == "repeat_before" and any("fake" in c.lower() for c in others):
        raise ValueError("--control repeat_before must NOT pass fake_* in --other_cols (copies are built in code)")
    if a.control == "true_fakes" and len(set(others)) < 2:
        raise ValueError("--control true_fakes needs >=2 distinct fake columns")

    if a.self_check:
        for ctrl in CONTROLS:
            m = FusionGate.__new__(FusionGate); nn.Module.__init__(m); m.control = ctrl; m.dim = 256
            m.frozen_src = build_model(a.backbone, 2, "none", torch.device("cpu"))
            for p in m.frozen_src.parameters():
                p.requires_grad_(False)
            m.frozen_src.eval()
            if ctrl != "direct":
                m.adapter = CrossAttnFusionClassifier(pretrained=False, mode="orig_kv", backbone=a.adapter_backbone)
                m.residual_head = nn.Linear(256, 1); nn.init.zeros_(m.residual_head.weight); nn.init.zeros_(m.residual_head.bias)
            b = torch.randn(2, 3, a.resize, a.resize); o = torch.randn(2, 5, 3, a.resize, a.resize)
            fm = m(b, o)
            dm = (m.frozen_src(b)[:, 1] - m.frozen_src(b)[:, 0])
            assert torch.allclose(fm, dm, atol=1e-5), f"{ctrl}: init margin != direct"
            if ctrl != "direct":  # residual head must have gradient from batch 1
                loss = F.binary_cross_entropy_with_logits(m(b, o), torch.tensor([0., 1.]))
                loss.backward()
                g = m.residual_head.weight.grad.abs().sum().item()
                assert g > 0, f"{ctrl}: residual head got zero gradient (cold-start not fixed)"
                print(f"[self_check] {ctrl}: init==direct OK, residual_head grad={g:.3e} (nonzero)")
            else:
                print(f"[self_check] direct: init==direct OK (no adapter)")
        return

    # ---------- eval ----------
    if a.mode == "eval":
        ck = torch.load(a.weights, map_location=dev, weights_only=False)
        cfg = ck["config"]
        ocols = [c.strip() for c in cfg["other_cols"].split(",") if c.strip()]
        model = FusionGate(cfg["source_ckpt"], control=cfg["control"], backbone=cfg["backbone"],
                           pretrained=cfg["pretrained"], adapter_backbone=cfg["adapter_backbone"],
                           dim=cfg["dim"], heads=cfg["heads"], proj_dim=cfg["proj_dim"],
                           stat_prior=cfg["stat_prior"], src_mean=cfg["src_mean"], src_std=cfg["src_std"]).to(dev)
        model.load_state_dict(ck["model"])
        dl = DataLoader(MultiImageDataset(a.test_csv, cfg["before_col"], ocols, a.label_col, cfg["resize"]),
                        batch_size=a.batch_size, num_workers=4)
        acase, df, _ = evaluate_percase(model, dl, dev)
        df["control"] = cfg["control"]; df["seed"] = cfg["seed"]
        df["checkpoint_sha256"] = sha256_file(a.weights)
        df["source_checkpoint_sha256"] = cfg.get("source_ckpt_sha256")
        df["generator_checkpoint_sha256"] = cfg.get("generator_ckpt_sha256")
        df["manifest_sha256"] = sha256_file(a.test_csv)
        outp = a.out_csv or (Path(a.out_dir) / f"percase_{cfg['control']}_seed{cfg['seed']}.csv")
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"[gate eval] control={cfg['control']} case_AUC={acase:.4f} n_case={df['case_id'].nunique()} -> {outp}")
        return

    # ---------- train ----------
    os.makedirs(a.out_dir, exist_ok=True)
    # `direct` never trains: just evaluate the frozen source (reference line)
    if a.control == "direct":
        model = FusionGate(a.source_ckpt, control="direct", backbone=a.backbone, pretrained=a.pretrained).to(dev)
        cfg = {"control": "direct", "source_ckpt": a.source_ckpt, "source_ckpt_sha256": sha256_file(a.source_ckpt),
               "generator_ckpt_sha256": sha256_file(a.generator_ckpt), "backbone": a.backbone,
               "pretrained": a.pretrained, "adapter_backbone": a.adapter_backbone, "dim": a.dim, "heads": a.heads,
               "proj_dim": a.proj_dim, "stat_prior": False, "src_mean": 0.0, "src_std": 1.0,
               "before_col": a.before_col, "other_cols": a.other_cols, "resize": a.resize, "seed": a.seed}
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": 0, "val_auc": float("nan")},
                   os.path.join(a.out_dir, "best.pt"))
        print("[gate] control=direct: no training; frozen source saved as best.pt (evaluate with --mode eval)")
        return

    src_stats = (0.0, 1.0)
    if a.stat_prior:
        from fusion_classifier import _compute_before_stats
        src_stats = _compute_before_stats(a.train_csv, a.before_col, a.resize)
    model = FusionGate(a.source_ckpt, control=a.control, backbone=a.backbone, pretrained=a.pretrained,
                       adapter_backbone=a.adapter_backbone, dim=a.dim, heads=a.heads, proj_dim=a.proj_dim,
                       stat_prior=a.stat_prior, src_mean=src_stats[0], src_std=src_stats[1]).to(dev)
    print(f"[gate] control={a.control} source={Path(a.source_ckpt).name} "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    tr = DataLoader(MultiImageDataset(a.train_csv, a.before_col, others, a.label_col, a.resize, True),
                    batch_size=a.batch_size, shuffle=True, num_workers=4,
                    generator=torch.Generator().manual_seed(a.seed))
    va = DataLoader(MultiImageDataset(a.val_csv, a.before_col, others, a.label_col, a.resize),
                    batch_size=a.batch_size, num_workers=4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    use_sc = a.supcon_weight > 0
    cfg = {"control": a.control, "source_ckpt": a.source_ckpt, "source_ckpt_sha256": sha256_file(a.source_ckpt),
           "generator_ckpt_sha256": sha256_file(a.generator_ckpt), "backbone": a.backbone, "pretrained": a.pretrained,
           "adapter_backbone": a.adapter_backbone, "dim": a.dim, "heads": a.heads, "proj_dim": a.proj_dim,
           "fusion_mode": "orig_kv", "stat_prior": a.stat_prior, "src_mean": src_stats[0], "src_std": src_stats[1],
           "before_col": a.before_col, "other_cols": a.other_cols, "resize": a.resize, "seed": a.seed}
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for before, oth, y, _ in tr:
            before, oth, y = before.to(dev), oth.to(dev), y.to(dev)
            opt.zero_grad()
            fm, dm, rm, feat = model(before, oth, return_parts=True)
            loss = F.binary_cross_entropy_with_logits(fm, y.float()) + a.residual_weight * rm.pow(2).mean()
            if use_sc:
                emb = F.normalize(model.adapter.proj_head(feat), dim=1)
                loss = loss + a.supcon_weight * sup_con_loss(emb, y, temp=a.supcon_temp)
            loss.backward(); opt.step()
        vauc, _, _ = evaluate_percase(model, va, dev)
        print(f"epoch {ep} val_case_auc={vauc:.4f}")
        if vauc > best:
            best = vauc
            torch.save({"model": model.state_dict(), "val_auc": vauc, "epoch": ep, "config": cfg},
                       os.path.join(a.out_dir, "best.pt"))
    print(f"[gate] control={a.control} best val_case_auc={best:.4f}")


if __name__ == "__main__":
    main()
