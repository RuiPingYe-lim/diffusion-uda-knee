#!/usr/bin/env python
"""How much of the breast source->target drop is RECOVERABLE?

The knee pair was just decomposed this way and turned out to be almost entirely
intrinsic difficulty: source->target dropped 0.179, but a TARGET-TRAINED
classifier only beat the source-trained one by +0.018. The same decomposition has
never been done for breast -- every comparison so far used source 0.950 vs target
0.791, which conflates domain shift with the target task simply being harder.

  recoverable headroom = AUC(target-trained) - AUC(source-trained)   [same 201 cases]

The target-trained arm uses the target's TRUE labels, so it is an ORACLE CEILING,
never a UDA result. It is estimated by 5-fold CV over the 201-case development
pool, using the same pre-built, leak-asserted folds as the earlier oracle probe;
out-of-fold predictions are pooled to cover all 201 cases, matching the set the
source-trained 0.7909 was computed on.

Selection: fixed epoch budget, predictions averaged over the last 5 epochs (the
rule adopted after the calibration showed source-val selection costs 3.6x the MCID).
No checkpoint is saved.

The LOCKED 51-case BrEaST test set is not referenced anywhere in this file.
"""
from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

sys.path.insert(0, "/root/autodl-tmp/knee/code2/idea2_diffusion_baseline")
from eval_existing_classifier_on_csv import build_model  # noqa: E402

C = "/root/autodl-tmp/breast/cache"
F_DIR = f"{C}/oracle_folds"


class DS(Dataset):
    def __init__(self, df, col="before_png", resize=224, train=False):
        self.df = df.reset_index(drop=True)
        self.col, self.train, self.resize = col, train, resize
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
        return self.post(im.resize((self.resize, self.resize), Image.BILINEAR)), int(r.label), str(r.case_id)


def case_auc(df):
    g = df.groupby("case_id", sort=False).agg(label=("label", "first"), p=("p", "mean"))
    return float(roc_auc_score(g.label.values, g.p.values))


@torch.no_grad()
def predict(m, dl, dev):
    m.eval()
    rows = []
    for x, y, c in dl:
        p = F.softmax(m(x.to(dev)), 1)[:, 1].cpu().numpy()
        rows += [{"case_id": ci, "label": int(yi), "p": float(pi)} for ci, yi, pi in zip(c, y, p)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser("breast recoverable-headroom probe")
    ap.add_argument("--src_ckpt", default="/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt")
    ap.add_argument("--target_csv", default=f"{C}/fusion_eval_breast_diag.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--init_from_source", action="store_true",
                    help="initialise each fold from the SOURCE checkpoint instead of ImageNet, then "
                         "fine-tune on target labels. This is the fairer ceiling for UDA: a UDA method "
                         "also has the source data, so the question is what target LABELS add ON TOP "
                         "of the source classifier -- not what 160 target cases achieve from scratch.")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- source-trained reference on the same 201 cases ----
    src = build_model("custom_resnet50_space", num_classes=2, pretrained="imagenet", device=torch.device("cpu"))
    ck = torch.load(a.src_ckpt, map_location="cpu", weights_only=False)
    st = ck.get("state_dict", ck.get("model", ck))
    r = src.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    bad = [k for k in list(r.missing_keys) + list(r.unexpected_keys) if "rsa" not in k.lower()]
    assert not bad, f"source checkpoint mismatch {bad[:4]}"
    src = src.to(dev).eval()
    tgt = pd.read_csv(a.target_csv)
    a_src = case_auc(predict(src, DataLoader(DS(tgt), batch_size=32, num_workers=4), dev))
    print(f"=== BrEaST development pool, n={tgt.case_id.nunique()} cases ===")
    print(f"  source-trained (BUSI, frozen)  : {a_src:.4f}   <- what UDA starts from\n")

    # ---- target-trained ceiling, 5-fold CV, multi-seed ----
    print(f"  target-trained ceiling: {a.nfolds}-fold CV x {len(a.seeds)} seeds, "
          f"{a.epochs} epochs, last-5-epoch prediction average")
    per_seed = []
    for seed in a.seeds:
        oof = []
        for k in range(a.nfolds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            tr = pd.concat([pd.read_csv(f"{F_DIR}/f{k}_train.csv"), pd.read_csv(f"{F_DIR}/f{k}_val.csv")])
            te = pd.read_csv(f"{F_DIR}/f{k}_test.csv")
            assert not (set(tr.case_id) & set(te.case_id)), "fold leak"
            m = build_model("custom_resnet50_space", num_classes=2, pretrained="imagenet",
                            device=torch.device("cpu"))
            if a.init_from_source:
                m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
            m = m.to(dev)
            dl = DataLoader(DS(tr, train=True), batch_size=a.batch_size, shuffle=True, num_workers=4,
                            generator=torch.Generator().manual_seed(seed))
            tel = DataLoader(DS(te), batch_size=32, num_workers=4)
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
            tail = []
            for ep in range(1, a.epochs + 1):
                m.train()
                for x, y, _ in dl:
                    opt.zero_grad()
                    F.cross_entropy(m(x.to(dev)), y.to(dev)).backward()
                    opt.step()
                if ep > a.epochs - 5:                      # last-5 prediction average
                    tail.append(predict(m, tel, dev))
            avg = pd.concat(tail).groupby(["case_id"], sort=False).agg(
                label=("label", "first"), p=("p", "mean")).reset_index()
            oof.append(avg)
            print(f"    seed {seed} fold {k}: n={len(avg)} fold-AUC {case_auc(avg):.4f}", flush=True)
        pooled = pd.concat(oof)
        assert pooled.case_id.nunique() == len(pooled) == len(tgt), \
            f"coverage {pooled.case_id.nunique()} != {len(tgt)}"
        per_seed.append(case_auc(pooled))
        print(f"  seed {seed}: pooled out-of-fold target-trained AUC = {per_seed[-1]:.4f}")

    v = np.array(per_seed)
    print(f"\n=== RESULT ===")
    print(f"  source-trained            : {a_src:.4f}")
    print(f"  target-trained (oracle)   : {v.mean():.4f} +- {v.std(ddof=1):.4f}  (seeds {a.seeds})")
    print(f"  RECOVERABLE HEADROOM      : {v.mean() - a_src:+.4f}")
    print(f"\n  [knee reference: source 0.7146, target-trained 0.7325, headroom +0.018]")
    print("\n=== READING ===")
    print("  headroom >> 0.05 -> real room; the contrastive/joint-training plan is worth building")
    print("  headroom ~ 0.02  -> this pair's ceiling is intrinsically low, like the knee pair;")
    print("                      re-scope the project rather than adding more method")
    print("  (the target-trained arm uses TRUE target labels: an oracle ceiling, never a UDA result)")


if __name__ == "__main__":
    main()
