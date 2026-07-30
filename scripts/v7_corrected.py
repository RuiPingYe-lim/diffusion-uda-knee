#!/usr/bin/env python
"""v7 -- CORRECTED operand analysis. Supersedes deltak_precheck v1/v2/v3, v5 and v6.

THE ERROR BEING FIXED
---------------------
v1-v6 read x0 from ``fusion_train_busi.csv:before_png``. That column does NOT hold the
source image: it points at ``results_u2b_rev/.../fake_5/``, i.e. the BUSI image ALREADY
translated 5 bridge steps toward BrEaST (that manifest is a cycle-audit table whose
``fake_*`` columns are the cycled-BACK images, and those files no longer exist). So every
"U1 = G(x0)" measured there was a SECOND translation pass applied to an already-translated
image, which is exactly the regime where a further pass should add little. The pessimistic
readings that followed -- "translation covers only 2-3% of the domain gap", "steps 2-5 are
noise", "DA-BRF is only perturbation repair" -- are therefore not established.

Verified-correct source of truth used here: ``da_route/da_manifest.csv``
    src_path : raw BUSI PNG              128px, mean 64.50 / std 55.81
    raw      : 256px source rendering    mean 64.52 / std 55.70  <-- the true x0
    U1, U5   : UNSB translations         mean 51.87 / 55.87
Unaffected by the bug and left standing: the 0.8181 training result (it trains from
``raw``) and gate0_style.py (it reads raw ``image_path`` on both domains).

WHAT IS MEASURED
----------------
A. Teacher validity, with the control v1-v6 never had. Scoring the frozen gate classifier
   on src_path / raw / U1 / U5 separates a RESOLUTION-and-rendering effect (128 -> 256) from
   a TRANSLATION effect. Only the latter justifies retraining the teacher.
B. The DA-BRF operand done right: Delta_trans = F(U1) - F(x0) with x0 = raw, U1 regenerated
   over R trajectories so per-step SNR and leave-one-out direction consistency are real.
C. Class-conditional domain progression: signed projection on d_c = (mu_t,c - mu_s,c)/||.||,
   plus a source-vs-target domain probe read as P(target) for x0 and for U1.
D. Content preservation via a probe retrained on each rendering (the frozen classifier is
   not admissible on translated images).
"""
import os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T, models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, "/root/autodl-tmp/UNSB")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GEN_CKPT = "/root/autodl-tmp/UNSB/checkpoints/u2b_rev_SB/latest_net_G.pth"
TASK_CKPT = "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt"
MANIFEST = "/root/autodl-tmp/breast/da_route/da_manifest.csv"
TGT_CSV = "/root/autodl-tmp/breast/cache/breast_diag_cid.csv"
NPER, R, NIMG_B, GSIZE, CSIZE = 64, 8, 8, 256, 224


# ---------------- generator ----------------
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

    class O: pass
    o = O(); o.__dict__.update(d)
    G = networks.define_G(d.get("input_nc", 3), d.get("output_nc", 3), d.get("ngf", 64),
                          d.get("netG", "resnet_9blocks_cond"), d.get("normG", "instance"),
                          not d.get("no_dropout", True), d.get("init_type", "xavier"),
                          d.get("init_gain", 0.02), d.get("no_antialias", False),
                          d.get("no_antialias_up", False), [], o)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    miss, unexp = G.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    print("[gen load] missing=%d unexpected=%d" % (len([k for k in miss if 'num_batches' not in k]), len(unexp)))
    return G.to(device).eval(), int(d.get("ngf", 64))


@torch.no_grad()
def gen_U1(G, x, ngf, gen):
    ti = torch.zeros(x.shape[0], device=x.device).long()
    z = torch.randn((x.shape[0], 4 * ngf), device=x.device, generator=gen)
    return G(x, ti, z)


# ---------------- task classifier (validated pipeline) ----------------
class SA(nn.Module):
    def __init__(s, d):
        super().__init__(); s.conv = nn.Conv2d(d, d, 1); s.soft = nn.Softmax(dim=2)

    def forward(s, x):
        a = s.conv(x); b, c, h, w = a.shape
        a = s.soft(a.view(b, c, -1)); m = a.amax(2, keepdim=True).clamp_min(1e-6)
        return x * (a / m).view(b, c, h, w)


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
    def forward(s, x):
        return s.classifier(s.avgpool(s.space_attn(s.stem(x))).flatten(1))


CLF_TF = T.Compose([T.ToTensor(), T.Resize((CSIZE, CSIZE), antialias=True),
                    T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                    T.Normalize([0.5] * 3, [0.5] * 3)])
GEN_TF = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                    T.Normalize([0.5] * 3, [0.5] * 3)])


def load_clf(p):
    return CLF_TF(Image.open(p).convert("L"))


def load_gen(p):
    im = Image.open(p).convert("L")
    if im.size != (GSIZE, GSIZE):
        im = im.resize((GSIZE, GSIZE), Image.BICUBIC)
    return GEN_TF(im)


def to_clf(u256):
    u01 = (u256 + 1) / 2
    u01 = F.interpolate(u01, size=(CSIZE, CSIZE), mode="bilinear", align_corners=False, antialias=True)
    return u01 * 2 - 1


class Enc(nn.Module):
    """Rendering-robust frozen ImageNet encoder (pooled)."""
    def __init__(s):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        s.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def gapz(s, x_m11):
        x = (x_m11 + 1) / 2
        return s.seq((x - s.mean) / s.std).mean(dim=(2, 3))


def cv_probe(Z, y, Ztest=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y)); cross = np.zeros(len(y)) if Ztest is not None else None
    for tr, te in skf.split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(Z[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(Z[te]))[:, 1]
        if Ztest is not None:
            cross[te] = lr.predict_proba(sc.transform(Ztest[te]))[:, 1]
    return roc_auc_score(y, oof), (roc_auc_score(y, cross) if Ztest is not None else None)


def main():
    torch.manual_seed(0)
    man = pd.read_csv(MANIFEST)
    man = man[man["split"] == "src_train"].reset_index(drop=True)
    rng = np.random.RandomState(0)
    sdf = pd.concat([man[man.label == c].iloc[rng.permutation((man.label == c).sum())[:NPER]]
                     for c in [0, 1]]).sample(frac=1, random_state=1).reset_index(drop=True)
    y = sdf.label.values.astype(int)
    print("source sample %d %s  (from da_manifest, split=src_train)" % (len(sdf), dict(sdf.label.value_counts())))

    gate = Gate(TASK_CKPT).to(DEV).eval()

    # ---------- A. teacher validity: resolution effect vs translation effect ----------
    print("\n############## A. FROZEN GATE TEACHER, per rendering ##############")
    print("  (src_path=raw 128 | raw=256 source rendering | U1,U5 = translations)")
    for col, tag in [("src_path", "src_path  raw128"), ("raw", "raw       src256"),
                     ("U1", "U1        transl"), ("U5", "U5        transl")]:
        P = []
        for b0 in range(0, len(sdf), 32):
            xb = torch.stack([load_clf(p) for p in sdf[col].iloc[b0:b0 + 32]]).to(DEV)
            P.append(F.softmax(gate(xb), 1)[:, 1].cpu().numpy())
        p = np.concatenate(P); pred = (p >= 0.5).astype(int)
        print("  %-20s AUC %.4f  acc %.4f" % (tag, roc_auc_score(y, p), accuracy_score(y, pred)))

    # ---------- B/C/D with the CORRECT x0 ----------
    G, ngf = build_generator(GEN_CKPT, DEV)
    enc = Enc().to(DEV).eval()

    zx0 = np.zeros((len(sdf), 2048), np.float32)
    zu1 = np.zeros((len(sdf), R, 2048), np.float32)
    zu1_file = np.zeros((len(sdf), 2048), np.float32)
    snr_l, loo_l = [], []
    for b0 in range(0, len(sdf), NIMG_B):
        rows = sdf.iloc[b0:b0 + NIMG_B]; ni = len(rows)
        x0 = torch.stack([load_gen(p) for p in rows["raw"]]).to(DEV)          # TRUE x0
        zx0[b0:b0 + ni] = enc.gapz(x0).cpu().numpy()
        zu1_file[b0:b0 + ni] = enc.gapz(torch.stack([load_gen(p) for p in rows["U1"]]).to(DEV)).cpu().numpy()
        with torch.no_grad():
            feats = []
            for r in range(R):
                gen = torch.Generator(device=DEV).manual_seed(31 + b0 * 100 + r)
                u = gen_U1(G, x0, ngf, gen)
                zu1[b0:b0 + ni, r] = enc.gapz(u).cpu().numpy()
                feats.append(enc.seq((((u + 1) / 2) - enc.mean) / enc.std))
            Fs = torch.stack(feats, 1)                                        # [ni,R,C,h,w]
            f0 = enc.seq((((x0 + 1) / 2) - enc.mean) / enc.std).unsqueeze(1)
            d = Fs - f0
            md = d.mean(1)
            Sk = md.flatten(1).norm(dim=1)
            Nk = ((d - md.unsqueeze(1)).flatten(2).norm(dim=2) ** 2).mean(1).clamp_min(1e-12).sqrt()
            s = d.sum(1, keepdim=True); loo = (s - d) / (R - 1)
            snr_l.append((Sk / Nk).cpu().numpy())
            loo_l.append(F.cosine_similarity(d.flatten(2), loo.flatten(2), dim=2).mean(1).cpu().numpy())
        print("  batch %d-%d" % (b0, b0 + ni), flush=True)
    zu1m = zu1.mean(1)

    tdf = pd.read_csv(TGT_CSV); yt = tdf.label.values.astype(int)
    zt = np.zeros((len(tdf), 2048), np.float32)
    for b0 in range(0, len(tdf), 32):
        zt[b0:b0 + 32] = enc.gapz(torch.stack([load_gen(p) for p in tdf.image_path.iloc[b0:b0 + 32]]).to(DEV)).cpu().numpy()

    print("\n############## B. Delta_trans = F(U1) - F(x0), x0 = TRUE source ##############")
    print("  SNR (over R=%d traj) = %.2f     LOO dir cos = %.3f" %
          (R, float(np.mean(np.concatenate(snr_l))), float(np.mean(np.concatenate(loo_l)))))
    rel = np.linalg.norm(zu1m - zx0, axis=1) / np.linalg.norm(zx0, axis=1)
    relf = np.linalg.norm(zu1_file - zx0, axis=1) / np.linalg.norm(zx0, axis=1)
    print("  ||dz||/||z0||  regenerated U1 = %.4f | precomputed U1 file = %.4f  (sanity: should agree)"
          % (rel.mean(), relf.mean()))

    print("\n############## C. CLASS-CONDITIONAL DOMAIN PROGRESSION ##############")
    for c, nm in [(0, "benign"), (1, "malignant")]:
        mus = zx0[y == c].mean(0); mut = zt[yt == c].mean(0)
        gv = mut - mus; gn = np.linalg.norm(gv) + 1e-8; dh = gv / gn
        idx = np.where(y == c)[0]
        cov = [((zu1[i] - zx0[i]) @ dh).mean() / gn for i in idx]
        sd = [((zu1[i] - zx0[i]) @ dh).mean() / (((zu1[i] - zx0[i]) @ dh).std() + 1e-8) for i in idx]
        print("  %-9s ||mu_t-mu_s||=%.2f  dom_cover=%+.3f  SNR_domain=%+.2f" %
              (nm, gn, float(np.mean(cov)), float(np.mean(sd))))

    Xd = np.concatenate([zx0, zt]); yd = np.concatenate([np.zeros(len(zx0)), np.ones(len(zt))])
    sc = StandardScaler().fit(Xd)
    dp = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xd), yd)
    px0 = dp.predict_proba(sc.transform(zx0))[:, 1]; pu1 = dp.predict_proba(sc.transform(zu1m))[:, 1]
    lx0 = dp.decision_function(sc.transform(zx0)); lu1 = dp.decision_function(sc.transform(zu1m))
    print("  domain probe  P(target): x0=%.3f -> U1=%.3f  (delta %+.3f)" % (px0.mean(), pu1.mean(), (pu1 - px0).mean()))
    print("  domain probe  logit    : x0=%+.2f -> U1=%+.2f  (delta %+.2f)  [logit avoids saturation]"
          % (lx0.mean(), lu1.mean(), (lu1 - lx0).mean()))

    print("\n############## D. CONTENT PRESERVATION (retrained probe) ##############")
    a_x0, a_cross = cv_probe(zx0, y, Ztest=zu1m)
    a_u1, _ = cv_probe(zu1m, y)
    print("  probe(x0)->x0 %.4f | probe(U1)->U1 %.4f | probe(x0)->U1 %.4f" % (a_x0, a_u1, a_cross))


if __name__ == "__main__":
    main()
