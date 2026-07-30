#!/usr/bin/env python
"""v8 -- the multi-step / PathFuse premise, re-tested from the CORRECT x0.

v2 ran this from `fusion_train_busi.csv:before_png`, which is `results_u2b_rev/.../fake_5/`
-- an ALREADY 5-step-translated image. Its verdict ("steps 2-5 are noise, the multi-step
premise fails") is therefore not established. Here x0 = `da_manifest.csv:raw`, the verified
256px source rendering, and the whole ordered path is rebuilt from it.

Both objects are measured separately, since they are NOT the same thing:
  states  x0 -> X_t1 -> ... : the true bridge states (noise-injected)
  preds   U1 -> U2 -> ...   : endpoint predictions -- what the classifier actually consumes

Per step k, over R independent trajectories of the SAME image:
  cum_cover    class-conditional domain-gap fraction covered by U_k relative to x0
               (the number that matters: does the path keep MOVING toward the target?)
  inc_SNR      ||mean_r increment|| / sqrt(mean_r ||increment - mean||^2)  -- is step k signal?
  inc_LOOcos   leave-one-out direction consistency of that increment
  dom_logit    source-vs-target probe logit (unsaturated) at step k
  probe_AUC    benign/malignant AUC of a probe RETRAINED on step k (content preservation)
A precomputed reference path (results_u2b_rev real/ + fake_1..5) is scored alongside as a
sanity check that the regenerated path matches what UNSB actually wrote to disk.
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
MANIFEST = "/root/autodl-tmp/breast/da_route/da_manifest.csv"
TGT_CSV = "/root/autodl-tmp/breast/cache/breast_diag_cid.csv"
PRECOMP = "/root/autodl-tmp/UNSB/results_u2b_rev/u2b_rev_SB/test_latest/images"
NPER, R, NIMG_B, GSIZE, STEPS, TAU = 64, 8, 8, 256, 5, 0.01


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
    return G.to(device).eval(), int(d.get("ngf", 64)), int(d.get("num_timesteps", 5))


def sb_times(n, device):
    incs = np.array([0] + [1 / (i + 1) for i in range(n - 1)])
    t = np.cumsum(incs); t = t / t[-1]; t = 0.5 * t[-1] + 0.5 * t
    return torch.tensor(np.concatenate([np.zeros(1), t])).float().to(device)


@torch.no_grad()
def sb_path(G, x, steps, times, ngf, gen):
    states, preds = [x], []
    Xt, prev = x, None
    for t in range(steps):
        if t > 0:
            dl = times[t] - times[t - 1]; dn = times[-1] - times[t - 1]
            inter = (dl / dn).reshape(-1, 1, 1, 1); sc = (dl * (1 - dl / dn)).reshape(-1, 1, 1, 1)
            Xt = (1 - inter) * Xt + inter * prev + (sc * TAU).sqrt() * torch.randn(
                Xt.shape, device=Xt.device, generator=gen)
            states.append(Xt)
        ti = (t * torch.ones(x.shape[0], device=x.device)).long()
        z = torch.randn((x.shape[0], 4 * ngf), device=x.device, generator=gen)
        prev = G(Xt, ti, z)
        preds.append(prev)
    return states, preds


class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        s.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def spatial(s, x_m11):
        x = (x_m11 + 1) / 2
        return s.seq((x - s.mean) / s.std)


TF = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                T.Normalize([0.5] * 3, [0.5] * 3)])


def load(p):
    im = Image.open(p).convert("L")
    if im.size != (GSIZE, GSIZE):
        im = im.resize((GSIZE, GSIZE), Image.BICUBIC)
    return TF(im)


def cv_auc(Z, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(Z, y):
        sc = StandardScaler().fit(Z[tr])
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(Z[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(Z[te]))[:, 1]
    return roc_auc_score(y, oof)


def main():
    torch.manual_seed(0)
    man = pd.read_csv(MANIFEST)
    man = man[man["split"] == "src_train"].reset_index(drop=True)
    rng = np.random.RandomState(0)
    sdf = pd.concat([man[man.label == c].iloc[rng.permutation((man.label == c).sum())[:NPER]]
                     for c in [0, 1]]).sample(frac=1, random_state=1).reset_index(drop=True)
    y = sdf.label.values.astype(int)
    print("source %d %s (da_manifest src_train, x0 = 'raw' column)" % (len(sdf), dict(sdf.label.value_counts())))

    G, ngf, nts = build_generator(GEN_CKPT, DEV)
    times = sb_times(nts, DEV)
    enc = Enc().to(DEV).eval()

    N = len(sdf)
    NSTATE = STEPS                                          # states = [x0, X_t1 ... X_t{STEPS-1}]
    z_state = np.zeros((N, NSTATE, 2048), np.float32)       # pooled, seed-mean
    z_pred = np.zeros((N, STEPS, 2048), np.float32)
    snr_s = [[] for _ in range(NSTATE - 1)]; loo_s = [[] for _ in range(NSTATE - 1)]
    snr_p = [[] for _ in range(STEPS - 1)]; loo_p = [[] for _ in range(STEPS - 1)]
    z_pre = np.zeros((N, STEPS + 1, 2048), np.float32)     # precomputed reference path

    for b0 in range(0, N, NIMG_B):
        rows = sdf.iloc[b0:b0 + NIMG_B]; ni = len(rows)
        x0 = torch.stack([load(p) for p in rows["raw"]]).to(DEV)
        with torch.no_grad():
            # precomputed reference path from disk
            keys = list(rows["key"])
            ref = [torch.stack([load(f"{PRECOMP}/real/{k}.png") for k in keys]).to(DEV)]
            for s in range(1, STEPS + 1):
                ref.append(torch.stack([load(f"{PRECOMP}/fake_{s}/{k}.png") for k in keys]).to(DEV))
            for s, t in enumerate(ref):
                z_pre[b0:b0 + ni, s] = enc.spatial(t).mean(dim=(2, 3)).cpu().numpy()

            allS = [[] for _ in range(NSTATE)]; allP = [[] for _ in range(STEPS)]
            for r in range(R):
                gen = torch.Generator(device=DEV).manual_seed(77 + b0 * 100 + r)
                states, preds = sb_path(G, x0, STEPS, times, ngf, gen)
                for s, t in enumerate(states):
                    allS[s].append(enc.spatial(t))
                for s, t in enumerate(preds):
                    allP[s].append(enc.spatial(t))
            S = [torch.stack(v, 1) for v in allS]          # each [ni,R,C,h,w]
            P = [torch.stack(v, 1) for v in allP]
            for s in range(len(S)):
                z_state[b0:b0 + ni, s] = S[s].mean(1).mean(dim=(2, 3)).cpu().numpy()
            for s in range(len(P)):
                z_pred[b0:b0 + ni, s] = P[s].mean(1).mean(dim=(2, 3)).cpu().numpy()

            def incr(a, b, snr_list, loo_list, idx):
                d = a - b
                md = d.mean(1)
                Sk = md.flatten(1).norm(dim=1)
                Nk = ((d - md.unsqueeze(1)).flatten(2).norm(dim=2) ** 2).mean(1).clamp_min(1e-12).sqrt()
                s_ = d.sum(1, keepdim=True); loo = (s_ - d) / (R - 1)
                snr_list[idx].append((Sk / Nk).cpu().numpy())
                loo_list[idx].append(F.cosine_similarity(d.flatten(2), loo.flatten(2), dim=2).mean(1).cpu().numpy())

            for s in range(len(S) - 1):                     # states: x0->Xt1, Xt1->Xt2 ...
                incr(S[s + 1], S[s], snr_s, loo_s, s)
            for s in range(STEPS - 1):                      # preds: U1->U2 ...
                incr(P[s + 1], P[s], snr_p, loo_p, s)
        print("  batch %d-%d" % (b0, b0 + ni), flush=True)

    # target features + class-conditional domain axes
    tdf = pd.read_csv(TGT_CSV); yt = tdf.label.values.astype(int)
    zt = np.zeros((len(tdf), 2048), np.float32)
    with torch.no_grad():
        for b0 in range(0, len(tdf), 32):
            xb = torch.stack([load(p) for p in tdf.image_path.iloc[b0:b0 + 32]]).to(DEV)
            zt[b0:b0 + 32] = enc.spatial(xb).mean(dim=(2, 3)).cpu().numpy()

    x0z = z_state[:, 0]
    axes = {}
    for c in [0, 1]:
        gv = zt[yt == c].mean(0) - x0z[y == c].mean(0)
        gn = np.linalg.norm(gv) + 1e-8
        axes[c] = (gv / gn, gn)

    def cover(zk):
        out = {}
        for c in [0, 1]:
            dh, gn = axes[c]; idx = y == c
            out[c] = float((((zk[idx] - x0z[idx]) @ dh) / gn).mean())
        return out

    Xd = np.concatenate([x0z, zt]); yd = np.concatenate([np.zeros(len(x0z)), np.ones(len(zt))])
    sc = StandardScaler().fit(Xd)
    dp = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xd), yd)

    def logit(zk):
        return float(dp.decision_function(sc.transform(zk)).mean())

    M = lambda x: float(np.mean(np.concatenate(x)))

    print("\n########## ENDPOINT PREDICTIONS U1..U5  (what the classifier consumes) ##########")
    print(" step | cum_cover ben | cum_cover mal | dom_logit | probe_AUC | inc_SNR | inc_LOOcos")
    print("  x0  |     0.000     |     0.000     |  %+7.2f  |   %.3f   |    -    |    -" %
          (logit(x0z), cv_auc(x0z, y)))
    for s in range(STEPS):
        cv = cover(z_pred[:, s])
        inc = (" %6.2f  |   %.3f" % (M(snr_p[s - 1]), M(loo_p[s - 1]))) if s >= 1 else "    -    |    -"
        print("  U%d  |    %+.3f     |    %+.3f     |  %+7.2f  |   %.3f   |%s" %
              (s + 1, cv[0], cv[1], logit(z_pred[:, s]), cv_auc(z_pred[:, s], y), inc))

    print("\n########## PRECOMPUTED PATH ON DISK (real/ + fake_1..5) -- sanity check ##########")
    print(" step | cum_cover ben | cum_cover mal | dom_logit | probe_AUC")
    for s in range(STEPS + 1):
        cv = cover(z_pre[:, s])
        name = "real" if s == 0 else f"fake{s}"
        print("  %-5s|    %+.3f     |    %+.3f     |  %+7.2f  |   %.3f" %
              (name, cv[0], cv[1], logit(z_pre[:, s]), cv_auc(z_pre[:, s], y)))

    print("\n########## TRUE BRIDGE STATES x0 -> X_t1 -> ... (PathFuse's path premise) ##########")
    print(" step | cum_cover ben | cum_cover mal | dom_logit | inc_SNR | inc_LOOcos")
    for s in range(NSTATE):
        cv = cover(z_state[:, s])
        inc = (" %6.2f  |   %.3f" % (M(snr_s[s - 1]), M(loo_s[s - 1]))) if s >= 1 else "   -     |    -"
        print("  %-5s|    %+.3f     |    %+.3f     |  %+7.2f  |%s" %
              ("x0" if s == 0 else f"Xt{s}", cv[0], cv[1], logit(z_state[:, s]), inc))

    print("\nREAD: PathFuse needs later steps to KEEP raising cum_cover / dom_logit with inc_SNR > ~2.")


if __name__ == "__main__":
    main()
