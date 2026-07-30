#!/usr/bin/env python
"""How much of the +0.036 oracle ceiling can SELF-TRAINING recover without labels?

The oracle that defines the ceiling fine-tunes the source classifier on target
data using TRUE labels (0.7916 -> 0.8278). Self-training is exactly that procedure
with GUESSED labels, so it is the most direct unsupervised analogue of the ceiling
and the single highest-leverage lever to test.

    baseline (source only)        0.7916
    self-training (this probe)    ?          <- how much of the gap is recoverable
    oracle (true target labels)   0.8278

Pseudo-labels come from the frozen source classifier applied to each fold's TARGET
TRAINING split only. The held-out fold's true labels are used for evaluation and
never for training. Confidence filtering is CLASS-BALANCED: the top fraction is
taken within each pseudo-class, because the source prior (32.5% positive) differs
from the target prior (38.8%) and a global threshold would systematically drop the
minority class.

Pseudo-label accuracy is reported for diagnosis only; it never drives a decision.
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, "/root/autodl-tmp/knee/code2/idea2_diffusion_baseline")
from eval_existing_classifier_on_csv import build_model  # noqa: E402

C = "/root/autodl-tmp/breast/cache"
F_DIR = f"{C}/oracle_folds"
BASELINE, ORACLE = 0.7916, 0.8278


class DS(Dataset):
    def __init__(self, df, col="before_png", label_col="label", resize=224, train=False):
        self.df = df.reset_index(drop=True)
        self.col, self.lc, self.train, self.resize = col, label_col, train, resize
        self.post = T.Compose([T.ToTensor(),
                               T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                               T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r[self.col]).convert("L")
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.rotate(random.uniform(-10, 10), resample=Image.BILINEAR)
        return self.post(im.resize((self.resize, self.resize), Image.BILINEAR)), int(r[self.lc]), str(r.case_id)


def case_auc(df):
    g = df.groupby("case_id", sort=False).agg(label=("label", "first"), p=("p", "mean"))
    return float(roc_auc_score(g.label.values, g.p.values))


@torch.no_grad()
def predict(m, df, dev, col="before_png"):
    m.eval()
    rows = []
    for x, y, c in DataLoader(DS(df, col=col), batch_size=32, num_workers=4):
        p = F.softmax(m(x.to(dev)), 1)[:, 1].cpu().numpy()
        rows += [{"case_id": ci, "label": int(yi), "p": float(pi)} for ci, yi, pi in zip(c, y, p)]
    return pd.DataFrame(rows)


def make_model(st, dev):
    m = build_model("custom_resnet50_space", num_classes=2, pretrained="imagenet", device=torch.device("cpu"))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return m.to(dev)


def estimate_target_prior_em(p, prior_src, iters=200, tol=1e-8):
    """Saerens-Latinne-Decaestecker EM estimate of the target class prior. LABEL-FREE.

    The source classifier outputs probabilities calibrated to the SOURCE prior. Under
    label shift (p(x|y) unchanged, p(y) changed) the target prior can be recovered from
    the soft predictions alone by alternating:
        E: reweight each prediction by (prior_tgt/prior_src)
        M: set prior_tgt to the mean of the reweighted posteriors
    Using the target's TRUE positive rate instead would leak label information and the
    result could not be called unsupervised.
    """
    pi = float(prior_src)
    for _ in range(iters):
        w1 = (pi / prior_src) * p
        w0 = ((1 - pi) / (1 - prior_src)) * (1 - p)
        post = w1 / np.maximum(w1 + w0, 1e-12)
        new = float(post.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi


def apply_prior_correction(p, prior_src, prior_tgt):
    """Re-calibrate probabilities from the source prior to the target prior."""
    w1 = (prior_tgt / prior_src) * p
    w0 = ((1 - prior_tgt) / (1 - prior_src)) * (1 - p)
    return w1 / np.maximum(w1 + w0, 1e-12)


PRIOR = {"src": None, "tgt": None}   # set once from the CLI; None => no correction


def select_confident(pred, frac):
    """Class-balanced confidence filter: keep the top `frac` within each pseudo-class.

    Prior correction, when enabled, is applied HERE so that every pseudo-labelling
    call (including later self-training rounds, whose teacher is a fine-tuned model)
    goes through the same calibration.
    """
    d = pred.copy()
    if PRIOR["tgt"] is not None:
        d["p"] = apply_prior_correction(d.p.values, PRIOR["src"], PRIOR["tgt"])
    d["pseudo"] = (d.p >= 0.5).astype(int)
    d["conf"] = (d.p - 0.5).abs()
    if frac >= 1.0:
        return d
    keep = []
    for cls, g in d.groupby("pseudo"):
        n = max(1, int(round(frac * len(g))))
        keep.append(g.nlargest(n, "conf"))
    return pd.concat(keep)


def main():
    ap = argparse.ArgumentParser("self-training probe against the measured oracle ceiling")
    ap.add_argument("--src_ckpt", default="/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt")
    ap.add_argument("--target_csv", default=f"{C}/fusion_eval_breast_diag.csv")
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.7, 0.5])
    ap.add_argument("--rounds", type=int, default=1, help="self-training iterations (labels refreshed each round)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--prior", choices=["none", "em", "oracle"], default="none",
                    help="none: raw source-calibrated probabilities. em: label-free Saerens EM estimate "
                         "of the target prior (the usable method). oracle: the target's TRUE positive "
                         "rate -- DIAGNOSTIC ONLY, it leaks label information and is not a UDA result.")
    ap.add_argument("--prior_src", type=float, default=None,
                    help="source positive rate; defaults to the BUSI training rate")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.src_ckpt, map_location="cpu", weights_only=False)
    st = ck.get("state_dict", ck.get("model", ck))
    src = make_model(st, dev).eval()

    tgt = pd.read_csv(a.target_csv)
    prior_src = a.prior_src if a.prior_src is not None else float(pd.read_csv(f"{C}/busi_train.csv").label.mean())
    print(f"=== BrEaST dev pool n={tgt.case_id.nunique()} | baseline {BASELINE:.4f} | "
          f"oracle ceiling {ORACLE:.4f} (headroom {ORACLE-BASELINE:+.4f}) ===")
    print(f"    prior correction: {a.prior}   (source positive rate {prior_src:.3f})\n")

    # pseudo-label quality of the source classifier (diagnosis only)
    p0 = predict(src, tgt, dev)
    print(f"raw source-classifier predictions: accuracy "
          f"{float(((p0.p>=0.5).astype(int)==p0.label).mean()):.3f}, "
          f"predicted positive rate {float((p0.p>=0.5).mean()):.3f} (true {tgt.label.mean():.3f})")
    if a.prior != "none":
        pi = (estimate_target_prior_em(p0.p.values, prior_src) if a.prior == "em"
              else float(tgt.label.mean()))
        print(f"  target prior used: {pi:.3f} "
              f"({'EM estimate, label-free' if a.prior=='em' else 'TRUE rate -- diagnostic only'};"
              f" true {tgt.label.mean():.3f})")
        PRIOR["src"], PRIOR["tgt"] = prior_src, pi
        p0 = p0.assign(p=apply_prior_correction(p0.p.values, prior_src, pi))
        print(f"  after correction: accuracy "
              f"{float(((p0.p>=0.5).astype(int)==p0.label).mean()):.3f}, "
              f"predicted positive rate {float((p0.p>=0.5).mean()):.3f}")
    acc = float(((p0.p >= 0.5).astype(int) == p0.label).mean())
    for f in a.fracs:
        s = select_confident(p0, f)
        sa = float((s.pseudo == s.label).mean())
        print(f"  keep top {f:.0%} per class -> n={len(s):3d}, pseudo-label accuracy {sa:.3f}")
    print()

    results = {}
    for frac in a.fracs:
        per_seed = []
        for seed in a.seeds:
            oof = []
            for k in range(a.nfolds):
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                tr = pd.concat([pd.read_csv(f"{F_DIR}/f{k}_train.csv"), pd.read_csv(f"{F_DIR}/f{k}_val.csv")])
                te = pd.read_csv(f"{F_DIR}/f{k}_test.csv")
                assert not (set(tr.case_id) & set(te.case_id)), "fold leak"
                m = make_model(st, dev)
                teacher = src
                for rnd in range(a.rounds):
                    pl = select_confident(predict(teacher, tr, dev), frac)
                    sub = tr.merge(pl[["case_id", "pseudo"]], on="case_id", how="inner")
                    dl = DataLoader(DS(sub, label_col="pseudo", train=True), batch_size=a.batch_size,
                                    shuffle=True, num_workers=4,
                                    generator=torch.Generator().manual_seed(seed + rnd))
                    m = make_model(st, dev)          # always fine-tune from source, not from the previous round
                    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
                    tail = []
                    for ep in range(1, a.epochs + 1):
                        m.train()
                        for x, y, _ in dl:
                            opt.zero_grad()
                            F.cross_entropy(m(x.to(dev)), y.to(dev)).backward()
                            opt.step()
                        if ep > a.epochs - 5:
                            tail.append(predict(m, te, dev))
                    teacher = m                       # next round relabels with the improved model
                oof.append(pd.concat(tail).groupby("case_id", sort=False)
                           .agg(label=("label", "first"), p=("p", "mean")).reset_index())
            pooled = pd.concat(oof)
            assert pooled.case_id.nunique() == len(pooled) == len(tgt), "coverage broken"
            per_seed.append(case_auc(pooled))
            print(f"  frac {frac:.0%} seed {seed}: {per_seed[-1]:.4f}", flush=True)
        v = np.array(per_seed)
        results[frac] = v
        got = (v.mean() - BASELINE) / (ORACLE - BASELINE) * 100
        print(f"  frac {frac:.0%}: {v.mean():.4f} +- {v.std(ddof=1):.4f}  "
              f"vs baseline {v.mean()-BASELINE:+.4f}  -> recovers {got:.0f}% of the ceiling\n")

    print("=== SUMMARY ===")
    print("  %-14s %-10s %-10s %s" % ("arm", "AUC", "vs base", "% of ceiling"))
    print("  %-14s %-10.4f %-10s %s" % ("baseline", BASELINE, "-", "0%"))
    for f, v in results.items():
        print("  %-14s %-10.4f %+-10.4f %.0f%%" % (f"self-train {f:.0%}", v.mean(), v.mean()-BASELINE,
                                                   (v.mean()-BASELINE)/(ORACLE-BASELINE)*100))
    print("  %-14s %-10.4f %+-10.4f %s" % ("oracle", ORACLE, ORACLE-BASELINE, "100%"))


if __name__ == "__main__":
    main()
