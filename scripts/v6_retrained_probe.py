#!/usr/bin/env python
"""STEP 2 (correct): retrained-probe analysis of U1, valid ON the translated rendering.

The frozen gate clf is OOD on before_png/fake (acc 0.58). So we probe with RENDERING-ROBUST
ImageNet-frozen pooled features + a light linear probe fit on source labels (5-fold), which is
valid on both x0=before_png and U1=G(before_png).

Outputs:
 (A) content preservation:  5-fold AUC(probe on x0), 5-fold AUC(probe on U1),
     cross AUC(train x0 -> test U1).  U1_AUC ~ x0_AUC => content kept (prior "0.023" analog).
 (B) diagnostic margin:     with the x0-trained (out-of-fold) probe, Delta_m = m(U1)-m(x0),
     per class mean/median + frac(Delta_m<0).
 (C) class-conditional domain progression: dom_cover_c, SNR_domain_c for Delta_trans.
 (D) domain probe P(target) for x0 vs U1 (source-vs-target logistic on the same features).
"""
import os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T, models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/autodl-tmp/UNSB")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GEN_CKPT = "/root/autodl-tmp/UNSB/checkpoints/u2b_rev_SB/latest_net_G.pth"
SRC_CSV = "/root/autodl-tmp/breast/cache/fusion_train_busi.csv"   # before_png (generator input rendering)
TGT_CSV = "/root/autodl-tmp/breast/cache/breast_diag_cid.csv"     # image_path,label
NPER = 64
R = 8
NIMG_B = 8
SIZE = 256
TAU = 0.01


def _parse_opt(path):
    o = {}
    for ln in open(path):
        if ":" not in ln or ln.strip().startswith("-"):
            continue
        k, v = ln.split(":", 1); k = k.strip(); v = v.split("[default")[0].strip()
        if v in ("True", "False"):
            val = v == "True"
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        o[k] = val
    return o


def build_generator(ckpt, device):
    from models import networks
    d = _parse_opt(os.path.join(os.path.dirname(ckpt), "train_opt.txt")); d["gpu_ids"] = []

    class O:
        pass
    o = O(); o.__dict__.update(d)
    G = networks.define_G(d.get("input_nc", 3), d.get("output_nc", 3), d.get("ngf", 64),
                          d.get("netG", "resnet_9blocks_cond"), d.get("normG", "instance"),
                          not d.get("no_dropout", True), d.get("init_type", "xavier"),
                          d.get("init_gain", 0.02), d.get("no_antialias", False),
                          d.get("no_antialias_up", False), [], o)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    G.load_state_dict(sd, strict=False)
    return G.to(device).eval(), int(d.get("ngf", 64))


@torch.no_grad()
def gen_U1(G, x0, ngf, gen):
    ti = torch.zeros(x0.shape[0], device=x0.device).long()
    z = torch.randn((x0.shape[0], 4 * ngf), device=x0.device, generator=gen)
    return G(x0, ti, z)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def gapz(self, x_m11):
        x = (x_m11 + 1) / 2
        x = (x - self.mean) / self.std
        return self.seq(x).mean(dim=(2, 3))   # [B,2048] pooled


TF = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                T.Normalize([0.5] * 3, [0.5] * 3)])


def load(p):
    im = Image.open(p).convert("L")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
    return TF(im)


def main():
    torch.manual_seed(0)
    G, ngf = build_generator(GEN_CKPT, DEV)
    enc = Enc().to(DEV).eval()

    df = pd.read_csv(SRC_CSV)
    rng = np.random.RandomState(0)
    sdf = pd.concat([df[df.label == c].iloc[rng.permutation((df.label == c).sum())[:NPER]] for c in [0, 1]]) \
        .sample(frac=1, random_state=1).reset_index(drop=True)
    y = sdf.label.values.astype(int)
    print("source", len(sdf), dict(sdf.label.value_counts()))

    zx0 = np.zeros((len(sdf), 2048), dtype=np.float32)
    zu1 = np.zeros((len(sdf), R, 2048), dtype=np.float32)
    for b0 in range(0, len(sdf), NIMG_B):
        rows = sdf.iloc[b0:b0 + NIMG_B]; ni = len(rows)
        x0 = torch.stack([load(p) for p in rows.before_png]).to(DEV)
        zx0[b0:b0 + ni] = enc.gapz(x0).cpu().numpy()
        for r in range(R):
            gen = torch.Generator(device=DEV).manual_seed(11 + b0 * 50 + r)
            zu1[b0:b0 + ni, r] = enc.gapz(gen_U1(G, x0, ngf, gen)).cpu().numpy()
        print("gen batch %d done" % b0, flush=True)
    zu1m = zu1.mean(1)   # [N,2048]

    # target features
    tdf = pd.read_csv(TGT_CSV); tcol = "image_path" if "image_path" in tdf.columns else "before_png"
    zt = np.zeros((len(tdf), 2048), dtype=np.float32); yt = tdf.label.values.astype(int)
    for b0 in range(0, len(tdf), 32):
        xb = torch.stack([load(p) for p in tdf[tcol].iloc[b0:b0 + 32]]).to(DEV)
        zt[b0:b0 + 32] = enc.gapz(xb).cpu().numpy()

    # ---- (A) content preservation via 5-fold linear probe ----
    def cv_auc(Z, lab, Ztest=None):
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        oof = np.zeros(len(lab)); oof_cross = np.zeros(len(lab)) if Ztest is not None else None
        margins = np.zeros(len(lab)); margins_cross = np.zeros(len(lab)) if Ztest is not None else None
        for tr, te in skf.split(Z, lab):
            sc = StandardScaler().fit(Z[tr])
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Z[tr]), lab[tr])
            oof[te] = lr.predict_proba(sc.transform(Z[te]))[:, 1]
            margins[te] = lr.decision_function(sc.transform(Z[te]))
            if Ztest is not None:
                oof_cross[te] = lr.predict_proba(sc.transform(Ztest[te]))[:, 1]
                margins_cross[te] = lr.decision_function(sc.transform(Ztest[te]))
        return roc_auc_score(lab, oof), oof_cross, margins, margins_cross

    auc_x0, oof_u1_from_x0, m_x0, m_u1 = cv_auc(zx0, y, Ztest=zu1m)
    auc_u1, _, _, _ = cv_auc(zu1m, y)
    auc_cross = roc_auc_score(y, oof_u1_from_x0)
    print("\n############## v6 retrained-probe (ImageNet feats, %d src, R=%d) ##############" % (len(sdf), R))
    print("\n(A) CONTENT PRESERVATION (5-fold linear probe AUC):")
    print("   probe(x0)->x0 : %.4f" % auc_x0)
    print("   probe(U1)->U1 : %.4f   (self-recoverable content in U1)" % auc_u1)
    print("   probe(x0)->U1 : %.4f   (x0 diagnostic axis applied to U1; drop = U1 moved off it)" % auc_cross)

    # ---- (B) diagnostic margin with x0-trained probe ----
    sm_x0 = (2 * y - 1) * m_x0; sm_u1 = (2 * y - 1) * m_u1; dm = sm_u1 - sm_x0
    print("\n(B) DIAGNOSTIC MARGIN (x0-probe, signed):  Delta_m = m(U1)-m(x0)")
    for c, nm in [(0, "benign"), (1, "malignant")]:
        a = dm[y == c]
        print("   %-9s mean Dm=%+.3f  median=%+.3f  frac(Dm<0)=%.2f" % (nm, a.mean(), np.median(a), float((a < 0).mean())))
    print("   OVERALL   mean Dm=%+.3f  frac(Dm<0)=%.2f" % (dm.mean(), float((dm < 0).mean())))

    # ---- (C) class-conditional domain progression ----
    print("\n(C) CLASS-CONDITIONAL DOMAIN PROGRESSION (dom_cover>0 = toward target-class centroid):")
    for c, nm in [(0, "benign"), (1, "malignant")]:
        mus = zx0[y == c].mean(0); mut = zt[yt == c].mean(0)
        gv = mut - mus; gn = np.linalg.norm(gv) + 1e-8; dhat = gv / gn
        idx = np.where(y == c)[0]
        covers, snrs = [], []
        for i in idx:
            q = (zu1[i] - zx0[i]) @ dhat          # [R]
            covers.append(q.mean() / gn); snrs.append(q.mean() / (q.std() + 1e-8))
        print("   %-9s ||mu_t-mu_s||=%.2f  dom_cover=%+.3f  SNR_domain=%+.2f" %
              (nm, gn, float(np.mean(covers)), float(np.mean(snrs))))

    # ---- (D) domain probe ----
    Xd = np.concatenate([zx0, zt]); yd = np.concatenate([np.zeros(len(zx0)), np.ones(len(zt))])
    sc = StandardScaler().fit(Xd)
    probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xd), yd)
    px0 = probe.predict_proba(sc.transform(zx0))[:, 1]; pu1 = probe.predict_proba(sc.transform(zu1m))[:, 1]
    print("\n(D) DOMAIN PROBE P(target)  (1=BrEaST):  x0=%.3f  U1=%.3f  delta=%+.3f" %
          (px0.mean(), pu1.mean(), (pu1 - px0).mean()))
    print("\nREAD: content kept if probe(U1)->U1 ~ probe(x0)->x0; domain shift real if dom_cover>0 & P(target) rises.")


if __name__ == "__main__":
    main()
