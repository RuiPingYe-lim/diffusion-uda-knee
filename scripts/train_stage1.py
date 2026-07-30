#!/usr/bin/env python
"""STAGE 1 -- frozen generator + translation-pair contrastive learning.

The classifier is retrained (not frozen). Each item contributes TWO views that
share one geometric draw: view1 = raw source image, view2 = the arm's condition
image. The arms differ ONLY in what view2 is:

    A0        raw again                  (controls sample count and gradient steps)
    P1 / P5   photometric,   style-strength-matched to U1 / U5
    F1 / F5   FDA amplitude, style-strength-matched to U1 / U5
    U1 / U5   UNSB G_s->t at bridge step 1 / 5

    loss = CE(view1) + lambda_a * CE(view2) + lambda_sup * SupCon(both views)

SupCon (Khosla et al. 2020) treats SAME CLASS as positive, so a sample's own
translation is a guaranteed positive and same-class samples across both styles are
pulled together. The naive instance-only alternative would push two different
benign lesions apart and fight the classifier -- that trap is why SupCon, not
SimCLR, is the right framework here.

ATTENTION POOLING replaces global average pooling: the model learns a weight per
spatial location and pools accordingly, so it focuses on discriminative regions by
itself. The scoring conv is ZERO-INITIALISED, so at initialisation the softmax is
uniform and the pooling is EXACTLY global average pooling -- a clean ablation
baseline, and any difference is something the model chose to learn. No lesion mask
is required, so this stays usable on X-ray / MRI where masks do not exist.

SELECTION: fixed epoch budget, predictions averaged over the last 5 epochs. The
calibration measured source-validation selection to cost 0.036 -- 3.6x the MCID --
while last-5 averaging costs 0.022 and cuts the seed-paired SD from 0.0275 to
0.0157 (64 seeds needed -> 21). No target label is involved.

Target-set probabilities are written per epoch to a SEALED directory and are never
read during training or selection. The LOCKED 51-case BrEaST test set is not
referenced anywhere in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

SIZE = 256
ARMS = {"A0": "raw", "P1": "P1", "F1": "F1", "U1": "U1", "P5": "P5", "F5": "F5", "U5": "U5"}


def sha256_file(p):
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- data
class TwoViewDataset(Dataset):
    """view1 = raw, view2 = the arm's condition image, sharing ONE geometric draw.

    The geometric draw must be shared: otherwise the contrastive term would also have
    to undo a flip/rotation and would stop measuring appearance invariance.
    """

    def __init__(self, df, cond_col, resize=224, train=False, aug_seed=0):
        self.df = df.reset_index(drop=True)
        self.cond, self.train, self.resize = cond_col, train, resize
        self.rng_aug = random.Random(aug_seed)
        self.post = T.Compose([T.ToTensor(),
                               T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                               T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.df)

    def _load(self, p, flip, ang):
        im = Image.open(p).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if ang:
            im = im.rotate(ang, resample=Image.BILINEAR)
        return self.post(im.resize((self.resize, self.resize), Image.BILINEAR))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        flip = self.train and (self.rng_aug.random() < 0.5)
        ang = self.rng_aug.uniform(-10, 10) if self.train else 0.0
        return (self._load(r["raw"], flip, ang), self._load(r[self.cond], flip, ang),
                int(r["label"]), str(r["case_id"]))


class EvalDataset(Dataset):
    def __init__(self, paths, labels, cases, resize=224):
        self.p, self.y, self.c = list(paths), list(labels), list(cases)
        self.resize = resize
        self.post = T.Compose([T.ToTensor(),
                               T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                               T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = Image.open(self.p[i]).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        return self.post(im.resize((self.resize, self.resize), Image.BILINEAR)), int(self.y[i]), str(self.c[i])


# ---------------------------------------------------------------- model
class AttnPoolNet(nn.Module):
    """ResNet50 -> (attention | average) pooling -> {classifier head, projection head}.

    The SAME pooled vector feeds both heads, so the contrastive term shapes exactly
    the representation the classifier consumes.
    """

    def __init__(self, pool="attn", proj_dim=128, num_classes=2, pretrained=True):
        super().__init__()
        w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        b = models.resnet50(weights=w)
        self.backbone = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool,
                                      b.layer1, b.layer2, b.layer3, b.layer4)
        d = 2048
        self.pool = pool
        if pool == "attn":
            self.attn = nn.Conv2d(d, 1, 1)
            nn.init.zeros_(self.attn.weight)   # uniform softmax at init == global average pooling
            nn.init.zeros_(self.attn.bias)
        self.fc = nn.Linear(d, num_classes)
        self.proj = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(inplace=True), nn.Linear(d // 4, proj_dim))

    def features(self, x):
        f = self.backbone(x)                                   # [B, 2048, h, w]
        if self.pool == "attn":
            a = self.attn(f).flatten(2)                        # [B, 1, hw]
            a = F.softmax(a, dim=2).unsqueeze(1)               # [B, 1, 1, hw]
            return (f.flatten(2).unsqueeze(1) * a).sum(-1).squeeze(1)
        return f.mean(dim=(2, 3))

    def forward(self, x):
        z = self.features(x)
        return self.fc(z), F.normalize(self.proj(z), dim=1)


def sup_con_loss(emb, labels, temp=0.07):
    """SupCon over a multiviewed batch. Positives = same class (any view), self excluded.

    Anchors with no same-class partner contribute 0 (not NaN), which happens in small
    batches when a class appears once.
    """
    n = emb.shape[0]
    sim = emb @ emb.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()        # numerical stability
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    exp = torch.exp(sim).masked_fill(eye, 0)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    cnt = pos.sum(1)
    valid = cnt > 0
    if not valid.any():
        return emb.sum() * 0.0
    mean_lp = (log_prob * pos).sum(1)[valid] / cnt[valid]
    return -mean_lp.mean()


# ---------------------------------------------------------------- eval
def supcon_floor(labels, temp):
    """Analytic floor of SupCon for THIS batch composition.

    With multiple positives the loss cannot reach 0: even for a perfect embedding
    (same class coincident, classes antipodal) an anchor with P positives pays
    about log(P). Reporting the raw value without this floor makes a converged
    SupCon look 'stuck'. Returns the mean floor over anchors.
    """
    lab = labels.detach().cpu().numpy()
    vals = []
    for c in np.unique(lab):
        p = int((lab == c).sum()) - 1          # positives excluding self
        if p > 0:
            vals += [float(np.log(p))] * (p + 1)
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def representation_metrics(model, df, cond, dev, resize=224, n=96):
    """Does the contrastive term actually do its job, at the REPRESENTATION level?

    Two label-free/label-light diagnostics on held-out source data:
      style_cos     mean cosine between the features of an image and its translation.
                    This is exactly what the contrastive term is supposed to raise;
                    if it does not move, the mechanism failed rather than the idea.
      class_sep     between-class distance / within-class distance on the raw view.
                    Rises when classes cluster more tightly.
    A high style_cos with no AUC gain means style invariance was achieved and simply
    is not the bottleneck -- a mechanism answer, not a tuning answer.
    """
    model.eval()
    sub = df.head(n)
    ds = TwoViewDataset(sub, cond, resize=resize, train=False, aug_seed=0)
    dl = DataLoader(ds, batch_size=32, num_workers=4)
    A, B, Y = [], [], []
    for v1, v2, y, _ in dl:
        A.append(model.features(v1.to(dev)).cpu())
        B.append(model.features(v2.to(dev)).cpu())
        Y.append(y)
    A, B, Y = torch.cat(A), torch.cat(B), torch.cat(Y).numpy()
    An, Bn = F.normalize(A, dim=1), F.normalize(B, dim=1)
    style_cos = float((An * Bn).sum(1).mean())
    d = torch.cdist(An, An).numpy()
    same = (Y[:, None] == Y[None, :]) & ~np.eye(len(Y), dtype=bool)
    diff = Y[:, None] != Y[None, :]
    sep = float(d[diff].mean() / max(d[same].mean(), 1e-9)) if same.any() and diff.any() else float("nan")
    return style_cos, sep


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ps, ys, cs = [], [], []
    for x, y, c in loader:
        logits, _ = model(x.to(dev))
        ps.append(F.softmax(logits, 1)[:, 1].cpu().numpy())
        ys.append(np.asarray(y)); cs += list(c)
    return np.concatenate(ps), np.concatenate(ys), cs


def case_auc(prob, y, case):
    d = pd.DataFrame({"case_id": case, "label": y, "p": prob})
    g = d.groupby("case_id", sort=False).agg(label=("label", "first"), p=("p", "mean"))
    return float(roc_auc_score(g.label.values, g.p.values))


def main():
    ap = argparse.ArgumentParser("stage 1: frozen generator + translation-pair contrastive")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--manifest", default="/root/autodl-tmp/breast/da_route/da_manifest.csv")
    ap.add_argument("--target_csv", default="/root/autodl-tmp/breast/cache/fusion_eval_breast_diag.csv")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/breast/stage1/runs")
    ap.add_argument("--sealed_dir", default="/root/autodl-tmp/breast/stage1/sealed")
    ap.add_argument("--pool", choices=["attn", "gap"], default="attn")
    ap.add_argument("--lambda_a", type=float, default=1.0, help="weight on the CE of view2")
    ap.add_argument("--lambda_sup", type=float, default=1.0, help="weight on SupCon; 0 disables it")
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tail", type=int, default=5, help="average predictions over the last N epochs")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{a.arm}_{a.pool}_sup{a.lambda_sup:g}_s{a.seed}"
    run = Path(a.out_dir) / tag
    run.mkdir(parents=True, exist_ok=True)
    Path(a.sealed_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.manifest)
    tr_df, va_df = df[df.split == "src_train"], df[df.split == "src_valid"]
    cond = ARMS[a.arm]

    # three independent streams: init and batch order identical across arms for a seed
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    model = AttnPoolNet(pool=a.pool).to(dev)
    tr = DataLoader(TwoViewDataset(tr_df, cond, train=True, aug_seed=a.seed + 90000),
                    batch_size=a.batch_size, shuffle=True, num_workers=4,
                    generator=torch.Generator().manual_seed(a.seed + 50000))
    va_raw = DataLoader(EvalDataset(va_df["raw"], va_df.label, va_df.case_id), batch_size=32, num_workers=4)
    va_cond = DataLoader(EvalDataset(va_df[cond], va_df.label, va_df.case_id), batch_size=32, num_workers=4)
    tgt = pd.read_csv(a.target_csv)
    tg = DataLoader(EvalDataset(tgt["before_png"], tgt.label, tgt.case_id), batch_size=32, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    cfg = {"stage": 1, "arm": a.arm, "cond_col": cond, "pool": a.pool, "lambda_a": a.lambda_a,
           "lambda_sup": a.lambda_sup, "temp": a.temp, "seed": a.seed, "epochs": a.epochs,
           "tail": a.tail, "batch_size": a.batch_size, "lr": a.lr,
           "selection_rule": f"fixed budget, predictions averaged over the last {a.tail} epochs",
           "manifest_sha256": sha256_file(a.manifest), "target_csv_sha256": sha256_file(a.target_csv),
           "n_train": len(tr_df), "n_valid": len(va_df), "n_target": int(tgt.case_id.nunique())}

    hist, sealed, tail_probs = [], [], []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = ce_tot = sc_tot = fl_tot = 0.0
        for v1, v2, y, _ in tr:
            v1, v2, y = v1.to(dev), v2.to(dev), y.to(dev)
            opt.zero_grad()
            o1, z1 = model(v1)
            o2, z2 = model(v2)
            ce = F.cross_entropy(o1, y) + a.lambda_a * F.cross_entropy(o2, y)
            loss = ce
            sc = torch.zeros((), device=dev)
            yy = torch.cat([y, y])
            if a.lambda_sup > 0:
                sc = sup_con_loss(torch.cat([z1, z2]), yy, a.temp)
                loss = loss + a.lambda_sup * sc
            loss.backward(); opt.step()
            tot += float(loss) * len(y); ce_tot += float(ce) * len(y); sc_tot += float(sc) * len(y)
            fl_tot += supcon_floor(yy, a.temp) * len(y)

        pr, yr, cr = predict(model, va_raw, dev)
        pc, yc, cc = predict(model, va_cond, dev)
        pt, yt, ct = predict(model, tg, dev)                    # sealed: never read here
        n = len(tr_df)
        scos, csep = representation_metrics(model, va_df, cond, dev)
        row = {"epoch": ep, "loss": tot / n, "ce": ce_tot / n, "supcon": sc_tot / n,
               "supcon_floor": fl_tot / n, "style_cos": scos, "class_sep": csep,
               "val_raw_auc": case_auc(pr, yr, cr), "val_cond_auc": case_auc(pc, yc, cc),
               "target_auc_SEALED": case_auc(pt, yt, ct)}
        hist.append(row)
        sealed += [{"epoch": ep, "case_id": ci, "label": int(lb), "prob": float(p)}
                   for ci, lb, p in zip(ct, yt, pt)]
        if ep > a.epochs - a.tail:
            tail_probs.append(pd.DataFrame({"case_id": ct, "label": yt, "p": pt}))
        print(f"epoch {ep:3d} loss {row['loss']:.4f} (ce {row['ce']:.4f} "
              f"supcon {row['supcon']:.4f}/floor {row['supcon_floor']:.4f})  "
              f"style_cos {scos:.4f} class_sep {csep:.4f}  "
              f"val_raw {row['val_raw_auc']:.4f}", flush=True)

    # the pre-registered selection: average the last `tail` epochs' probabilities
    avg = pd.concat(tail_probs).groupby("case_id", sort=False).agg(
        label=("label", "first"), p=("p", "mean")).reset_index()
    final = case_auc(avg.p.values, avg.label.values, avg.case_id.tolist())
    cfg["final_target_auc"] = final

    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    seal = Path(a.sealed_dir) / f"{tag}_target_percase.csv"
    pd.DataFrame(sealed).to_csv(seal, index=False)
    avg.to_csv(run / "final_percase.csv", index=False)
    cfg["sealed_sha256"] = sha256_file(seal)
    with open(run / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{tag}] final target case-AUC (last-{a.tail} average) = {final:.4f} -> {run}")


if __name__ == "__main__":
    main()
