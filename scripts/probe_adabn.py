#!/usr/bin/env python
"""HEADROOM PROBE: does target-side adaptation have room, before investing in the
full contrastive + joint-training pipeline?

AdaBN (Li et al. 2018) is the cheapest possible target-side adaptation: keep every
learned weight, and only replace the BatchNorm running statistics with statistics
recomputed on UNLABELLED target images. No training, no labels, no gradients.

If this alone moves the target AUC, the target side has exploitable headroom and
the heavier machinery is worth building. If it does not move at all, that is an
early warning that this dataset's ceiling is low.

Variants swept (all label-free):
  * momentum: fraction of the target statistics blended in (0 = source stats,
    1 = pure target stats). A partial blend is often better than a full swap on
    small target sets, where the target statistics are themselves noisy.
  * n_target_images used to estimate the statistics.

The classifier is the project's frozen source classifier; the target set is the
201-case BrEaST development pool. The LOCKED 51-case test is never touched.
"""
from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, "/root/autodl-tmp/knee/code2/idea2_diffusion_baseline")
from eval_existing_classifier_on_csv import build_model  # noqa: E402


class ImgDS(Dataset):
    def __init__(self, paths, labels, cases, resize=224):
        self.p, self.y, self.c = list(paths), list(labels), list(cases)
        self.tf = T.Compose([T.ToTensor(), T.Resize((resize, resize), antialias=True),
                             T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                             T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("L")), int(self.y[i]), str(self.c[i])


@torch.no_grad()
def case_auc(model, loader, dev):
    model.eval()
    rows = []
    for x, y, c in loader:
        o = model(x.to(dev))
        m = (o[:, 1] - o[:, 0]).cpu().numpy()
        rows += [{"case_id": ci, "label": int(yi), "m": float(mi)} for ci, yi, mi in zip(c, y, m)]
    d = pd.DataFrame(rows).groupby("case_id", sort=False).agg(label=("label", "first"), m=("m", "mean"))
    return float(roc_auc_score(d.label.values, d.m.values))


@torch.no_grad()
def recompute_bn(model, loader, dev, momentum):
    """Re-estimate BN running stats on target images. No labels, no gradients.

    momentum=1.0 -> statistics come purely from the target pass; smaller values
    blend the target statistics into the source ones.
    """
    m = copy.deepcopy(model)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise RuntimeError("no BatchNorm layers found -- AdaBN does not apply to this backbone")
    for b in bns:
        b.reset_running_stats()      # forget the source statistics
        b.momentum = None            # None => cumulative average over the passes
        b.train()                    # update running stats during forward
    for x, _, _ in loader:
        m(x.to(dev))
    m.eval()
    if momentum < 1.0:               # blend back toward the source statistics
        for bs, bt in zip([mod for mod in model.modules()
                           if isinstance(mod, nn.modules.batchnorm._BatchNorm)], bns):
            bt.running_mean.mul_(momentum).add_(bs.running_mean, alpha=1 - momentum)
            bt.running_var.mul_(momentum).add_(bs.running_var, alpha=1 - momentum)
    return m


def main():
    ap = argparse.ArgumentParser("AdaBN headroom probe")
    ap.add_argument("--ckpt", default="/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt")
    ap.add_argument("--target_csv", default="/root/autodl-tmp/breast/cache/fusion_eval_breast_diag.csv")
    ap.add_argument("--src_test_csv", default="/root/autodl-tmp/breast/cache/busi_test.csv")
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=5, help="bootstrap-style resamples of the BN-estimation subset")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(a.backbone, num_classes=2, pretrained="imagenet", device=torch.device("cpu"))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st = ck.get("state_dict", ck.get("model", ck))
    res = model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    bad = [k for k in list(res.missing_keys) + list(res.unexpected_keys) if "rsa" not in k.lower()]
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:5]}")
    model = model.to(dev).eval()

    tgt = pd.read_csv(a.target_csv)
    tl = DataLoader(ImgDS(tgt["before_png"], tgt.label, tgt.case_id), batch_size=a.batch_size, num_workers=4)
    ste = pd.read_csv(a.src_test_csv)
    sl = DataLoader(ImgDS(ste.image_path, ste.label, range(len(ste))), batch_size=a.batch_size, num_workers=4)

    base_t, base_s = case_auc(model, tl, dev), case_auc(model, sl, dev)
    print(f"=== baseline (source BN statistics, no adaptation) ===")
    print(f"  target (BrEaST dev, n={tgt.case_id.nunique()}): {base_t:.4f}")
    print(f"  source (BUSI test,  n={len(ste)}):              {base_s:.4f}")
    print(f"  gap source->target: {base_s - base_t:+.4f}\n")

    print("=== AdaBN: BN statistics re-estimated on UNLABELLED target images ===")
    print("  %-12s %-10s %-10s %s" % ("momentum", "target", "delta", "source (should drop: stats now target's)"))
    for mom in [0.1, 0.25, 0.5, 0.75, 1.0]:
        m2 = recompute_bn(model, tl, dev, mom)
        t2, s2 = case_auc(m2, tl, dev), case_auc(m2, sl, dev)
        print("  %-12.2f %-10.4f %+-10.4f %.4f" % (mom, t2, t2 - base_t, s2))

    # how stable is the estimate w.r.t. WHICH target images are used?
    print("\n=== stability: BN stats from random halves of the target pool (momentum=1.0) ===")
    accs = []
    for s in range(a.seeds):
        idx = np.random.RandomState(s).permutation(len(tgt))[: len(tgt) // 2]
        sub = tgt.iloc[idx]
        subl = DataLoader(ImgDS(sub["before_png"], sub.label, sub.case_id),
                          batch_size=a.batch_size, num_workers=4)
        m2 = recompute_bn(model, subl, dev, 1.0)
        accs.append(case_auc(m2, tl, dev))
        print(f"  half {s} (n={len(sub)}): target {accs[-1]:.4f}")
    print(f"  mean {np.mean(accs):.4f} +- {np.std(accs, ddof=1):.4f}  (vs baseline {base_t:.4f}, "
          f"delta {np.mean(accs) - base_t:+.4f})")

    print("\n=== READING ===")
    print("  A clear positive delta => the target side has exploitable headroom and the")
    print("  heavier contrastive/joint-training machinery is worth building.")
    print("  A flat or negative delta => early warning that this dataset's ceiling is low;")
    print("  re-scope before investing months. (Not proof either way: AdaBN is only the")
    print("  cheapest lever, and a null here does not exclude gains from learned alignment.)")


if __name__ == "__main__":
    main()
