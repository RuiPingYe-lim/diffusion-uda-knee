#!/usr/bin/env python
"""v10 -- teacher acceptance on the HELD-OUT BUSI test split (130 cases).

v9 evaluated on src_train, which is exactly the split the render-robust teacher was
fine-tuned on, so its numbers are optimistic. This file repeats the acceptance and the
margin-asymmetry check on `results_u2b_srctest`, a split neither teacher ever trained on.

Renderings: real/ (256px source render), fake_1 (=U1) and fake_5 (=U5).
Old and new teacher are scored side by side. The decisive quantity is the per-class margin
change: with the exported teacher the safety term penalises malignant cases by ~7.7 while
never firing on benign ones, i.e. it reads as "do not translate malignant cases".
Source labels only.
"""
import sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T, models
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OLD = "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt"
NEW = "/root/autodl-tmp/breast/exp/teacher_render_robust/teacher.pt"
MAN = "/root/autodl-tmp/breast/cache/u2b_rev_srctest_manifest.csv"
IMG = "/root/autodl-tmp/UNSB/results_u2b_srctest/u2b_rev_SB/test_latest/images"

TF = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                T.Normalize([0.5] * 3, [0.5] * 3)])


class SA(nn.Module):
    def __init__(s, d):
        super().__init__(); s.conv = nn.Conv2d(d, d, 1); s.soft = nn.Softmax(dim=2)

    def forward(s, x):
        a = s.conv(x); b, c, h, w = a.shape; a = s.soft(a.view(b, c, -1))
        m = a.amax(2, keepdim=True).clamp_min(1e-6); return x * (a / m).view(b, c, h, w)


class Gate(nn.Module):
    def __init__(s, ck):
        super().__init__(); m = models.resnet50(weights=None)
        s.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        s.space_attn = SA(2048); s.avgpool = nn.AdaptiveAvgPool2d(1); s.classifier = nn.Linear(2048, 2)
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        for p, mod in {"stem.": s.stem, "space_attn.": s.space_attn, "classifier.": s.classifier}.items():
            mod.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)}, strict=False)

    @torch.no_grad()
    def score(s, x):
        z = s.classifier(s.avgpool(s.space_attn(s.stem(x))).flatten(1))
        return (z[:, 1] - z[:, 0]).cpu().numpy()


def scores(net, paths):
    out = []
    for b0 in range(0, len(paths), 32):
        xb = torch.stack([TF(Image.open(p).convert("L")) for p in paths[b0:b0 + 32]]).to(DEV)
        out.append(net.score(xb))
    return np.concatenate(out)


def main():
    man = pd.read_csv(MAN)
    keys = list(man["key"]); y = man["label"].values.astype(int)
    print("held-out BUSI test: n=%d  %s" % (len(y), dict(pd.Series(y).value_counts())))
    views = {"real(src)": [f"{IMG}/real/{k}.png" for k in keys],
             "U1": [f"{IMG}/fake_1/{k}.png" for k in keys],
             "U5": [f"{IMG}/fake_5/{k}.png" for k in keys]}

    for tag, ck in (("OLD teacher (raw-only)", OLD), ("NEW teacher (render-robust)", NEW)):
        net = Gate(ck).to(DEV).eval()
        d = {v: scores(net, p) for v, p in views.items()}
        print("\n############## %s ##############" % tag)
        print("  rendering   |  AUC   |  acc   | bal-acc | corr with src (P/S)")
        for v in views:
            pred = (d[v] > 0).astype(int)
            if v == "real(src)":
                corr = "     --"
            else:
                corr = "%.3f / %.3f" % (pearsonr(d["real(src)"], d[v])[0], spearmanr(d["real(src)"], d[v]).statistic)
            print("  %-11s | %.4f | %.4f | %.4f  | %s"
                  % (v, roc_auc_score(y, d[v]), accuracy_score(y, pred), balanced_accuracy_score(y, pred), corr))
        s = 2 * y - 1
        print("  -- margin change Delta_m = m(U) - m(src),  m_y = (2y-1)d --")
        for v in ("U1", "U5"):
            dm = s * (d[v] - d["real(src)"])
            pen = np.maximum(dm * -1 - 0.10, 0)          # exactly what the safety term charges
            print("     %s: benign %+.3f | malignant %+.3f | offset %+.3f || safety charge: benign %.3f, malignant %.3f"
                  % (v, dm[y == 0].mean(), dm[y == 1].mean(), float((d[v] - d["real(src)"]).mean()),
                     pen[y == 0].mean(), pen[y == 1].mean()))
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()

    print("\nREAD: the safety charge is relu(-Delta_m - 0.10) averaged per class -- what the")
    print("      generator is actually penalised. Class-asymmetric charge = 'stop translating that class'.")


if __name__ == "__main__":
    main()
