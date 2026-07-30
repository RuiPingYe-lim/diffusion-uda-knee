#!/usr/bin/env python
"""v9 -- is the teacher's collapse on translated images a CALIBRATION shift or real damage?

Motivation. The DOSC safety term penalises a drop in the true-class margin
    m_y = (2y-1) d,   d = z1 - z0.
Under a pure score offset d' = d + b the positives GAIN b and the negatives LOSE b, so one
class is penalised even when the ranking (and hence AUC) is untouched. Whether that failure
mode is actually active here depends on how much of the teacher's accuracy collapse on
translated images (0.984 -> 0.594) is recoverable by re-calibrating d alone.

"AUC high, accuracy low" is only indirect evidence, and the earlier -9.33 -> -5.07 figure was
the DOMAIN probe's logit, not the teacher's. This file measures the teacher's own score d.

Protocol. Fit the calibration on one source split, evaluate on a DISJOINT source split:
    bias-only        d' = d + b
    temperature-only d' = a d,        a > 0
    affine (Platt)   d' = a d + b,    a > 0
Two fitting targets are reported, since they answer different questions:
    to-source : argmin_{a,b} || (a d_u + b) - d_s ||   (advisor's formulation; align the
                translated score with the same case's source score)
    to-labels : logistic regression of the label on d_u (standard Platt scaling)
Also reported: threshold-oracle accuracy (the best any pure threshold move could give),
optimal-threshold shift, ECE / Brier / NLL, and the per-case rank agreement between d on the
source image and d on its translation. Recovery to ~0.98 under bias-only means the damage is
a threshold shift; partial recovery means a shift PLUS genuine loss of separability.
Source labels only; no target data is read.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T, models
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss

import sys as _sys
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CK = (_sys.argv[1] if len(_sys.argv) > 1
      else "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt")
MANIFEST = "/root/autodl-tmp/breast/da_route/da_manifest.csv"
RENDERINGS = ["raw", "U1", "U5"]

CLF_TF = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
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
        return (z[:, 1] - z[:, 0]).cpu().numpy()          # d = z1 - z0


def scores(net, paths):
    out = []
    for b0 in range(0, len(paths), 32):
        xb = torch.stack([CLF_TF(Image.open(p).convert("L")) for p in paths[b0:b0 + 32]]).to(DEV)
        out.append(net.score(xb))
    return np.concatenate(out)


def ece(prob, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (prob > edges[i]) & (prob <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(e)


def metrics(d, y, tag):
    p = 1 / (1 + np.exp(-d)); pred = (d > 0).astype(int)
    thr = np.unique(d)
    accs = [(accuracy_score(y, (d > t).astype(int)), t) for t in thr]
    best_acc, best_t = max(accs)
    return {
        "tag": tag, "auc": roc_auc_score(y, d), "acc": accuracy_score(y, pred),
        "bacc": balanced_accuracy_score(y, pred), "oracle_acc": best_acc, "opt_thr": float(best_t),
        "ece": ece(p, y), "brier": brier_score_loss(y, p),
        "nll": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


def show(m):
    print("  %-26s AUC %.4f | acc %.4f | bal-acc %.4f | oracle-thr acc %.4f (thr %+.2f) | ECE %.3f Brier %.3f NLL %.3f"
          % (m["tag"], m["auc"], m["acc"], m["bacc"], m["oracle_acc"], m["opt_thr"], m["ece"], m["brier"], m["nll"]))


def fit_to_source(du, ds, mode):
    """argmin over the requested family of || (a du + b) - ds ||^2."""
    if mode == "bias":
        return 1.0, float(np.mean(ds - du))
    if mode == "temp":
        f = lambda a: np.mean((a * du - ds) ** 2)
        r = minimize_scalar(f, bounds=(1e-3, 100.0), method="bounded")
        return float(r.x), 0.0
    A = np.stack([du, np.ones_like(du)], 1)
    sol, *_ = np.linalg.lstsq(A, ds, rcond=None)
    return float(max(sol[0], 1e-3)), float(sol[1])


def main():
    net = Gate(CK).to(DEV).eval()
    man = pd.read_csv(MANIFEST)
    splits = man["split"].unique().tolist()
    print("splits in manifest:", splits)
    fit_split = "src_valid" if "src_valid" in splits else splits[0]
    test_split = "src_train" if "src_train" in splits else splits[-1]
    fit = man[man["split"] == fit_split].reset_index(drop=True)
    tst = man[man["split"] == test_split].reset_index(drop=True)
    print("fit on %s (n=%d), evaluate on %s (n=%d)" % (fit_split, len(fit), test_split, len(tst)))

    D = {}
    for part, df in (("fit", fit), ("test", tst)):
        D[part] = {"y": df.label.values.astype(int)}
        for r in RENDERINGS:
            D[part][r] = scores(net, list(df[r]))
        print("scored %s" % part, flush=True)

    y_t = D["test"]["y"]
    print("\n############## RAW (reference) ##############")
    show(metrics(D["test"]["raw"], y_t, "raw"))

    for r in ["U1", "U5"]:
        print("\n############## %s -- uncalibrated, then recalibrated ##############" % r)
        show(metrics(D["test"][r], y_t, "%s uncalibrated" % r))
        for mode, name in [("bias", "bias-only  d+b"), ("temp", "temperature a*d"), ("affine", "affine  a*d+b")]:
            a, b = fit_to_source(D["fit"][r], D["fit"]["raw"], mode)
            show(metrics(a * D["test"][r] + b, y_t, "%s [to-source] (a=%.2f b=%+.2f)" % (name, a, b)))
        # Platt against labels, fitted on the fit split
        lr = LogisticRegression(max_iter=2000).fit(D["fit"][r].reshape(-1, 1), D["fit"]["y"])
        a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        show(metrics(a * D["test"][r] + b, y_t, "Platt [to-labels] (a=%.2f b=%+.2f)" % (a, b)))
        pr = pearsonr(D["test"]["raw"], D["test"][r])[0]
        sr = spearmanr(D["test"]["raw"], D["test"][r]).statistic
        print("  per-case agreement with the raw score: Pearson %.3f | Spearman %.3f" % (pr, sr))

    print("\n############## MARGIN ASYMMETRY (the failure mode the safety term has) ##############")
    print("  Delta_m = m(U) - m(raw), m_y = (2y-1)*d.  A pure offset b gives +b for one class, -b for the other.")
    for r in ["U1", "U5"]:
        s = 2 * y_t - 1
        dm = s * (D["test"][r] - D["test"]["raw"])
        print("  %s : benign mean Dm %+.3f | malignant mean Dm %+.3f | offset b_hat %+.3f"
              % (r, dm[y_t == 0].mean(), dm[y_t == 1].mean(), float(np.mean(D["test"][r] - D["test"]["raw"]))))
    print("\nREAD: if bias-only restores accuracy to near the raw value, the damage is a threshold shift;")
    print("      partial recovery means a shift PLUS genuine loss of separability.")


if __name__ == "__main__":
    main()
