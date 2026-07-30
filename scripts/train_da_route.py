#!/usr/bin/env python
"""STEP 4 -- matched two-view classifier training for the augmentation route.

Every arm sees the SAME number of images and the SAME number of gradient steps:
each item contributes TWO views, view1 = raw (always) and view2 = the arm's
condition image. A0 uses raw for both, so it controls for sample count and step
count rather than acting as a half-size baseline.

    L = CE(f(v1), y) + lambda_a * CE(f(v2), y) + lambda_c * MSE(p(v1), p(v2))

lambda_a = 1.0 and lambda_c = 1.0 (on SOFTMAX probabilities, so the term is
bounded and needs no per-arm scaling) are PRE-REGISTERED. The `+C` arms differ
from their base arm ONLY by lambda_c > 0 -- same files, same data, same
selection rule.

THREE INDEPENDENT RNG STREAMS (rng_init / rng_order / rng_aug). The same seed
gives bit-identical weight init and batch order in every arm; only the
augmentation stream is consumed differently. Without this the "paired" seeds are
paired in name only.

CHECKPOINT SELECTION (pre-registered): the epoch maximising
    0.5 * AUC(source val, RAW view) + 0.5 * AUC(source val, TRANSFORMED view)
No target label, and no target image, is involved. Using both views is what makes
the rule arm-neutral: selecting on a single view leaves "which view is the
validation set" as a second uncontrolled factor that differs across arms.

The target set is scored EVERY epoch and written to a sealed directory, but never
read during training and never used for any choice. The locked 51-case BrEaST
test is not referenced anywhere in this file.

No weights are saved: 13 arms x seeds x ResNet50 would not fit the disk. Per-epoch
per-sample probabilities (a few tens of KB per run) are saved instead, which lets
every alternative selection rule be replayed offline with no extra GPU.
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


def build_post(znorm=False):
    """Final tensor normalisation.

    Default is the project's FIXED Normalize(0.5, 0.5), which passes each image's
    global intensity straight through to the network -- so the measured
    BUSI->BrEaST gap (0.87 SD brightness, 2.05 SD contrast) reaches the classifier
    intact.

    `znorm` instead standardises EACH IMAGE to zero mean and unit variance, which
    removes both of those gaps by construction. This is the three-line version of
    advisor suggestion 4 (source-statistics conditioning) and must be measured
    before any learned FiLM variant: on the knee dataset per-case normalisation
    erased the domain gap entirely.
    """
    base = [T.ToTensor(), T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t)]
    if znorm:
        return T.Compose(base + [T.Lambda(lambda t: (t - t.mean()) / (t.std() + 1e-6))])
    return T.Compose(base + [T.Normalize([0.5]*3, [0.5]*3)])


def sha256_file(p):
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class TwoViewDataset(Dataset):
    """view1 = raw, view2 = the arm's condition image, SHARING one geometric draw.

    The geometric augmentation must be shared, otherwise the consistency term also
    has to undo a flip/rotation and stops measuring appearance invariance.
    """

    def __init__(self, df, cond_col, resize=224, train=False, aug_seed=0, znorm=False):
        self.df = df.reset_index(drop=True)
        self.cond = cond_col
        self.train = train
        self.rng_aug = random.Random(aug_seed)
        self.post = build_post(znorm)
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def _load(self, p, flip, ang):
        im = Image.open(p).convert("L")
        if im.size != (SIZE, SIZE):           # every arm through the same resampling path
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if ang:
            im = im.rotate(ang, resample=Image.BILINEAR)
        return self.post(im.resize((self.resize, self.resize), Image.BILINEAR))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        flip = self.train and (self.rng_aug.random() < 0.5)
        ang = (self.rng_aug.uniform(-10, 10) if self.train else 0.0)
        return (self._load(r["raw"], flip, ang), self._load(r[self.cond], flip, ang),
                int(r["label"]), str(r["case_id"]))


class EvalDataset(Dataset):
    def __init__(self, paths, labels, cases, resize=224, znorm=False):
        self.p, self.y, self.c = list(paths), list(labels), list(cases)
        self.post = build_post(znorm)
        self.resize = resize

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = Image.open(self.p[i]).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        return self.post(im.resize((self.resize, self.resize), Image.BILINEAR)), int(self.y[i]), str(self.c[i])


def make_model(seed, device):
    """Weight init driven ONLY by rng_init, so it is bit-identical across arms."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    with torch.no_grad():                      # only the new head is random
        w = torch.empty_like(m.fc.weight)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=g)
        m.fc.weight.copy_(w)
        m.fc.bias.zero_()
    return m.to(device)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ps, ys, cs = [], [], []
    for x, y, c in loader:
        p = F.softmax(model(x.to(dev)), dim=1)[:, 1]
        ps.append(p.cpu().numpy()); ys.append(np.asarray(y)); cs += list(c)
    return np.concatenate(ps), np.concatenate(ys), cs


def case_auc(prob, y, case):
    d = pd.DataFrame({"case_id": case, "label": y, "p": prob})
    g = d.groupby("case_id", sort=False).agg(label=("label", "first"), p=("p", "mean"))
    return float(roc_auc_score(g.label.values, g.p.values))


def main():
    ap = argparse.ArgumentParser("matched DA-route training")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--consistency", action="store_true", help="the `+C` variant (lambda_c > 0)")
    ap.add_argument("--manifest", default="/root/autodl-tmp/breast/da_route/da_manifest.csv")
    ap.add_argument("--target_csv", default="/root/autodl-tmp/breast/cache/fusion_eval_breast_diag.csv")
    ap.add_argument("--src_test_csv", default="/root/autodl-tmp/breast/cache/busi_test.csv")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/breast/da_route/runs")
    ap.add_argument("--sealed_dir", default="/root/autodl-tmp/breast/da_route/sealed")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--lambda_c", type=float, default=1.0)
    ap.add_argument("--znorm", action="store_true",
                    help="per-image z-score instead of the fixed Normalize(0.5,0.5); removes the "
                         "global brightness/contrast gap by construction (the 3-line version of ④)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    lam_c = a.lambda_c if a.consistency else 0.0
    name = f"{a.arm}{'C' if a.consistency else ''}{'Z' if a.znorm else ''}_s{a.seed}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.out_dir) / name
    run.mkdir(parents=True, exist_ok=True)
    Path(a.sealed_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.manifest)
    tr_df, va_df = df[df.split == "src_train"], df[df.split == "src_valid"]
    cond = ARMS[a.arm]

    # three independent streams; init and order identical across arms for a given seed
    np.random.seed(a.seed); random.seed(a.seed)
    model = make_model(a.seed, dev)                                   # rng_init
    tr = DataLoader(TwoViewDataset(tr_df, cond, train=True, aug_seed=a.seed + 90000, znorm=a.znorm),
                    batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=False,
                    generator=torch.Generator().manual_seed(a.seed + 50000))           # rng_order

    Z = a.znorm
    va_raw = DataLoader(EvalDataset(va_df["raw"], va_df.label, va_df.case_id, znorm=Z), batch_size=32, num_workers=4)
    va_cond = DataLoader(EvalDataset(va_df[cond], va_df.label, va_df.case_id, znorm=Z), batch_size=32, num_workers=4)
    tgt = pd.read_csv(a.target_csv)
    tg = DataLoader(EvalDataset(tgt["before_png"], tgt.label, tgt.case_id, znorm=Z), batch_size=32, num_workers=4)
    ste = pd.read_csv(a.src_test_csv)
    st = DataLoader(EvalDataset(ste.image_path, ste.label, range(len(ste)), znorm=Z), batch_size=32, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    cfg = {"arm": a.arm, "consistency": bool(a.consistency), "lambda_a": a.lambda_a, "lambda_c": lam_c,
           "cond_col": cond, "znorm": bool(a.znorm), "seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
           "selection_rule": "0.5*AUC(src_val,raw) + 0.5*AUC(src_val,cond)",
           "manifest_sha256": sha256_file(a.manifest), "target_csv_sha256": sha256_file(a.target_csv),
           "n_train": len(tr_df), "n_valid": len(va_df), "n_target": len(tgt)}

    hist, tgt_rows = [], []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for v1, v2, y, _ in tr:
            v1, v2, y = v1.to(dev), v2.to(dev), y.to(dev)
            opt.zero_grad()
            o1, o2 = model(v1), model(v2)
            loss = F.cross_entropy(o1, y) + a.lambda_a * F.cross_entropy(o2, y)
            if lam_c > 0:
                loss = loss + lam_c * F.mse_loss(F.softmax(o1, 1), F.softmax(o2, 1))
            loss.backward(); opt.step()
            tot += float(loss) * len(y)

        pr, yr, cr = predict(model, va_raw, dev)
        pc, yc, cc = predict(model, va_cond, dev)
        a_raw, a_cond = case_auc(pr, yr, cr), case_auc(pc, yc, cc)
        score = 0.5 * a_raw + 0.5 * a_cond                       # THE selection score
        pt, yt, ct = predict(model, tg, dev)                     # sealed: never read here
        ps, ys, cs = predict(model, st, dev)
        hist.append({"epoch": ep, "loss": tot / len(tr_df), "val_raw_auc": a_raw,
                     "val_cond_auc": a_cond, "select_score": score,
                     "src_test_auc": case_auc(ps, ys, cs), "target_auc_SEALED": case_auc(pt, yt, ct)})
        for cid, lb, p in zip(ct, yt, pt):
            tgt_rows.append({"epoch": ep, "case_id": cid, "label": int(lb), "prob": float(p)})
        print(f"epoch {ep:3d} loss {tot/len(tr_df):.4f}  val_raw {a_raw:.4f}  val_cond {a_cond:.4f}  "
              f"score {score:.4f}", flush=True)

    h = pd.DataFrame(hist)
    best = int(h.loc[h.select_score.idxmax(), "epoch"])
    cfg["selected_epoch"] = best
    cfg["selected_score"] = float(h.select_score.max())
    h.to_csv(run / "history.csv", index=False)
    seal = Path(a.sealed_dir) / f"{name}_target_percase.csv"
    pd.DataFrame(tgt_rows).to_csv(seal, index=False)
    cfg["sealed_target_sha256"] = sha256_file(seal)
    with open(run / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{name}] selected epoch {best} (score {cfg['selected_score']:.4f}) -> {run}")
    print(f"[{name}] sealed target predictions -> {seal}")


if __name__ == "__main__":
    main()
