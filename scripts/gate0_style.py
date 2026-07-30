#!/usr/bin/env python
"""GATE 0 -- is there any target STYLE to condition on, beyond first/second moments?

The proposed innovation conditions the translator on a style code s_ik = E_s(x_t^{j_k})
taken from DIFFERENT real BrEaST individuals, to create K candidates. Two premises must
hold, and both are cheap to test WITHOUT training anything:

 (P1) Target individuals must genuinely differ in shallow style, and must STILL differ
      after per-image moment normalisation. If the difference vanishes under moment
      normalisation, exemplar conditioning is just moment matching re-skinned -- and that
      axis is already exhausted here (fixed moment intervention delta = -0.013).
      Metric: F-ratio = var_across_individuals / var_within_individual, where the
      within-individual floor comes from label-preserving augmentations of the SAME image.
      F ~ 1 => individuals are indistinguishable from augmentation noise.

 (P2) The style code must NOT carry benign/malignant information, otherwise conditioning
      on a target exemplar copies its pathology into the source case. Measured with a
      5-fold logistic probe on the style codes (target labels used POST-HOC only).
      AUC ~ 0.5 => the adversarial class-orthogonality term is unnecessary decoration.
      AUC >> 0.5 => that term is load-bearing and must be implemented.

Also reports how many style clusters the target actually supports (silhouette, k=2..6),
which sets a ceiling on a sensible K.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms as T, models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.cluster import KMeans

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TGT_CSV = "/root/autodl-tmp/breast/cache/breast_diag_cid.csv"   # image_path,label (target)
SRC_CSV = "/root/autodl-tmp/breast/cache/busi_train.csv"        # image_path,label (source)
SIZE = 256
NAUG = 4          # augmented copies per image -> within-individual floor
EPS = 1e-6


class ShallowEnc(nn.Module):
    """Frozen ImageNet ResNet50, shallow only: this is what a style encoder E_s sees."""
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def style(self, x01):
        """x01 in [0,1]. Style code = [mu, log sigma] over channels of stem and layer1."""
        x = (x01 - self.mean) / self.std
        s = self.stem(x); a1 = self.l1(s)
        out = []
        for f in (s, a1):
            out.append(f.mean(dim=(2, 3)))
            out.append(torch.log(f.std(dim=(2, 3)) + EPS))
        return torch.cat(out, dim=1)          # [B, 64*2 + 256*2 = 640]


def load01(p):
    im = Image.open(p).convert("L")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
    t = T.functional.to_tensor(im)
    return t.repeat(3, 1, 1) if t.shape[0] == 1 else t


def augment(t, k, rng):
    """Label-preserving augmentation: the within-individual noise floor."""
    if k == 0:
        return t
    x = t
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[2])
    ang = rng.uniform(-10, 10)
    x = T.functional.rotate(x.unsqueeze(0), ang).squeeze(0)
    return x


def moment_norm(t):
    """Per-image z-normalisation: removes global intensity AND contrast by construction."""
    return (t - t.mean()) / (t.std() + EPS)


def collect(paths, enc, moment_normalised):
    """Return [N, NAUG, D] style codes."""
    rng = np.random.RandomState(0)
    out = np.zeros((len(paths), NAUG, 640), dtype=np.float32)
    B = 16
    buf, idx = [], []
    for i, p in enumerate(paths):
        base = load01(p)
        for a in range(NAUG):
            x = augment(base, a, rng)
            if moment_normalised:
                x = moment_norm(x)
                x = (x - x.min()) / (x.max() - x.min() + EPS)   # back to [0,1] for the encoder
            buf.append(x); idx.append((i, a))
            if len(buf) == B:
                z = enc.style(torch.stack(buf).to(DEV)).cpu().numpy()
                for j, (ii, aa) in enumerate(idx):
                    out[ii, aa] = z[j]
                buf, idx = [], []
    if buf:
        z = enc.style(torch.stack(buf).to(DEV)).cpu().numpy()
        for j, (ii, aa) in enumerate(idx):
            out[ii, aa] = z[j]
    return out


def f_ratio(S):
    """S: [N,A,D]. Per-dim var_across_individuals / var_within_individual."""
    within = S.var(axis=1).mean(axis=0)            # [D]
    across = S.mean(axis=1).var(axis=0)            # [D]
    keep = within > 1e-10
    r = across[keep] / within[keep]
    return float(np.median(r)), float(np.percentile(r, 25)), float(np.percentile(r, 75)), int(keep.sum())


def probe_auc(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def main():
    enc = ShallowEnc().to(DEV).eval()
    tdf = pd.read_csv(TGT_CSV)
    tcol = "image_path" if "image_path" in tdf.columns else tdf.columns[0]
    tpaths = list(tdf[tcol]); ty = tdf["label"].astype(int).values
    sdf = pd.read_csv(SRC_CSV)
    spaths = list(sdf["image_path"])[:201]
    print("target %d (%s), source %d" % (len(tpaths), dict(pd.Series(ty).value_counts()), len(spaths)))

    res = {}
    for tag, mn in [("RAW", False), ("MOMENT-NORMALISED", True)]:
        St = collect(tpaths, enc, mn)
        Ss = collect(spaths, enc, mn)
        med, q1, q3, nd = f_ratio(St)
        meds, _, _, _ = f_ratio(Ss)
        Zt = St.mean(axis=1)
        auc = probe_auc(Zt, ty)
        # cross-domain separability of style (source vs target)
        Zs = Ss.mean(axis=1)
        Xd = np.concatenate([Zs, Zt]); yd = np.concatenate([np.zeros(len(Zs)), np.ones(len(Zt))])
        auc_dom = probe_auc(Xd, yd.astype(int))
        # how many style clusters does the target support
        sc = StandardScaler().fit_transform(Zt)
        sil = {}
        for k in range(2, 7):
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(sc)
            sil[k] = float(silhouette_score(sc, lab))
        res[tag] = dict(med=med, q1=q1, q3=q3, nd=nd, meds=meds, auc=auc, auc_dom=auc_dom, sil=sil)
        print("collected %s" % tag, flush=True)

    print("\n################ GATE 0: is there target style to condition on? ################")
    print("\n(P1) STYLE SPREAD  F = var(across individuals) / var(within individual, augmentation floor)")
    print("     F~1 => individuals indistinguishable from augmentation noise => nothing to encode")
    print("  condition            | target F (median [IQR])        | source F (median)")
    for tag in ("RAW", "MOMENT-NORMALISED"):
        r = res[tag]
        print("  %-20s | %8.2f  [%.2f, %.2f]        | %8.2f" % (tag, r["med"], r["q1"], r["q3"], r["meds"]))
    keep = res["MOMENT-NORMALISED"]["med"] / max(res["RAW"]["med"], 1e-9)
    print("\n  --> fraction of style spread SURVIVING moment normalisation: %.1f%%" % (100 * keep))
    print("      (low => exemplar conditioning is moment matching re-skinned; that axis is exhausted)")

    print("\n(P2) DOES THE STYLE CODE LEAK benign/malignant?  (5-fold logistic probe, target labels post-hoc)")
    for tag in ("RAW", "MOMENT-NORMALISED"):
        print("  %-20s style->label AUC = %.3f" % (tag, res[tag]["auc"]))
    print("      AUC~0.5 => the class-adversarial term is decoration; AUC>>0.5 => it is load-bearing")

    print("\n(aux) style->domain (source vs target) AUC, and how many style clusters the target supports:")
    for tag in ("RAW", "MOMENT-NORMALISED"):
        r = res[tag]
        best = max(r["sil"], key=r["sil"].get)
        print("  %-20s domain AUC = %.3f | silhouette %s | best k = %d" %
              (tag, r["auc_dom"], {k: round(v, 3) for k, v in r["sil"].items()}, best))


if __name__ == "__main__":
    main()
