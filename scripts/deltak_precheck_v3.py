#!/usr/bin/env python
"""RETRACTED -- kept for provenance. DO NOT cite its numbers. See scripts/v7_corrected.py.

x0 is read from ``cache/fusion_train_busi.csv:before_png``, which is NOT the source image but
``results_u2b_rev/.../fake_5/`` -- BUSI already translated five bridge steps toward BrEaST.
So ``Delta_trans = F(U1) - F(x0)`` as computed here is really F(G(fake_5)) - F(fake_5): a
second translation pass on an already-translated image. The true source rendering is
``da_route/da_manifest.csv:raw`` (mean 64.52 / std 55.70 vs before_png's 55.87 / 34.93).

Every domain-progression number below is wrong by roughly an order of magnitude. Corrected
(v7_corrected.py): class-conditional dom_cover +0.187 (benign) / +0.315 (malignant) rather
than +0.016 / +0.018; SNR_domain +38.6 / +83.5 rather than +4 / +11; domain-probe P(target)
0.001 -> 0.107 rather than unmoved; ||dz||/||z0|| 0.437; Delta_trans SNR 34.8 with
leave-one-out direction cosine 0.999. The residual really is a large, highly consistent,
content-preserving domain transport, which is the opposite of what this file reported.

--- original docstring below ---

Delta_k pre-check v3 -- targets DA-BRF's ACTUAL operand and fixes 'toward target'.

Measures three DISTINCT residuals (v2 never measured the first one):
   Delta_trans   = F(U1) - F(x0)          <- the residual DA-BRF would repair
   Delta_refine_k= F(U_{k+1}) - F(U_k)     <- endpoint-prediction refinements (U1..U5)
   Delta_state_k = F(X_{t_k}) - F(X_{t_{k-1}})  <- true bridge states (noise-injected)

Per residual, over R independent trajectories of the SAME image:
   SNR        = ||mean_r d|| / sqrt(mean_r ||d-mean||^2)
   LOO_cos    = mean_r cos(d_r, mean_{r'!=r} d_r')     (leave-one-out; no self-inclusion bias)
   dom_cover  = mean_r <GAP(d_r), dhat> / ||mu_t-mu_s|| (SIGNED fraction of the source->target
                gap covered along the domain axis dhat=(mu_t-mu_s)/||.||; >0 = toward target)
   SNR_dom    = mean_r q / std_r q,  q=<GAP(d_r), dhat>
Split benign/malignant. Plus the source-label MARGIN change x0->U1 (diagnostic safety).
Encoders: task backbone (Normalize(0.5), PRIMARY) + ImageNet (AUX). Layers: layer3, layer4.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T, models

sys.path.insert(0, "/root/autodl-tmp/UNSB")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 256
CKPT = "/root/autodl-tmp/UNSB/checkpoints/u2b_rev_SB/latest_net_G.pth"
TASK_CKPT = "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt"
SRC_CSV = "/root/autodl-tmp/breast/cache/fusion_train_busi.csv"
TGT_CSV = "/root/autodl-tmp/breast/cache/breast_train.csv"
NPER = 32
R = 8
NIMG_B = 8
STEPS = 5
TAU = 0.01
LAYERS = ["layer3", "layer4"]


def _parse_train_opt(path):
    opt = {}
    for line in open(path):
        if ":" not in line or line.strip().startswith("-"):
            continue
        k, v = line.split(":", 1)
        k = k.strip(); v = v.split("[default")[0].strip()
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
        opt[k] = val
    return opt


def build_generator(ckpt, device):
    from models import networks
    d = _parse_train_opt(os.path.join(os.path.dirname(ckpt), "train_opt.txt"))
    d["gpu_ids"] = []

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
    miss, unexp = G.load_state_dict(sd, strict=False)
    print("[generator load] missing=%d unexpected=%d" % (len([k for k in miss if 'num_batches' not in k]), len(unexp)))
    return G.to(device).eval(), int(d.get("ngf", 64)), int(d.get("num_timesteps", 5))


def sb_times(T_steps, device):
    incs = np.array([0] + [1 / (i + 1) for i in range(T_steps - 1)])
    t = np.cumsum(incs); t = t / t[-1]; t = 0.5 * t[-1] + 0.5 * t
    t = np.concatenate([np.zeros(1), t])
    return torch.tensor(t).float().to(device)


@torch.no_grad()
def sb_path_both(G, x, steps, times, ngf, gen):
    states = [x]; preds = []
    Xt = x; prev = None
    for t in range(steps):
        if t > 0:
            delta = times[t] - times[t - 1]; denom = times[-1] - times[t - 1]
            inter = (delta / denom).reshape(-1, 1, 1, 1)
            scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
            noise = torch.randn(Xt.shape, device=Xt.device, generator=gen)
            Xt = (1 - inter) * Xt + inter * prev + (scale * TAU).sqrt() * noise
            states.append(Xt)
        ti = (t * torch.ones(x.shape[0], device=x.device)).long()
        z = torch.randn((x.shape[0], 4 * ngf), device=x.device, generator=gen)
        prev = G(Xt, ti, z)
        preds.append(prev)
    return states, preds


class ImagenetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def feats(self, x_m11):
        x = (x_m11 + 1) / 2
        x = (x - self.mean) / self.std
        h = self.seq[0:4](x); a1 = self.seq[4](h); a2 = self.seq[5](a1)
        a3 = self.seq[6](a2); a4 = self.seq[7](a3)
        return {"layer3": a3, "layer4": a4}, None


class SpaceAttention(nn.Module):
    """Exact copy of the trained model's SpaceAttention (per-channel spatial gating)."""
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv2d(d, d, 1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        att = self.conv(x); b, c, h, w = att.shape
        att = self.soft(att.view(b, c, -1))
        m = att.amax(dim=2, keepdim=True).clamp_min(1e-6)
        att = (att / m).view(b, c, h, w)
        return x * att


class TaskModel(nn.Module):
    """Reconstructs HybridResNet(method='resnet50_space'): stem -> space_attn -> avgpool -> classifier."""
    def __init__(self, ckpt):
        super().__init__()
        m = models.resnet50(weights=None)
        self.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.space_attn = SpaceAttention(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 2)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        stem_sd = {k[len("stem."):]: v for k, v in sd.items() if k.startswith("stem.")}
        m1, u1 = self.seq.load_state_dict(stem_sd, strict=False)
        sa = {k[len("space_attn."):]: v for k, v in sd.items() if k.startswith("space_attn.")}
        m2, u2 = self.space_attn.load_state_dict(sa, strict=False)
        cl = {k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.")}
        m3, u3 = self.classifier.load_state_dict(cl, strict=False)
        print("[task load] stem_mapped=%d stem_missing=%d | space_attn_missing=%d | clf_missing=%d | unexpected=%s" %
              (len(stem_sd), len([k for k in m1 if 'num_batches' not in k]), len(m2), len(m3),
               (list(u1)[:2] + list(u2)[:2] + list(u3)[:2])))

    @torch.no_grad()
    def feats(self, x_m11):
        h = self.seq[0:4](x_m11); a1 = self.seq[4](h); a2 = self.seq[5](a1)
        a3 = self.seq[6](a2); a4 = self.seq[7](a3)
        z = self.avgpool(self.space_attn(a4)).flatten(1)          # [B,2048]
        logits = self.classifier(z)
        return {"layer3": a3, "layer4": a4}, logits


def gap(x):
    return x.mean(dim=(2, 3))


def per_image_metrics(fk, fkm1, dhat, gapnorm):
    """fk,fkm1: [ni,R,C,H,W]. Returns dict of per-image [ni] arrays."""
    d = fk - fkm1                                    # [ni,R,C,H,W]
    md = d.mean(1)                                   # [ni,C,H,W]
    Sk = md.flatten(1).norm(dim=1)
    Nk = ((d - md.unsqueeze(1)).flatten(2).norm(dim=2) ** 2).mean(1).clamp_min(1e-12).sqrt()
    snr = (Sk / Nk).cpu().numpy()
    # LOO direction consistency
    s = d.sum(1, keepdim=True)                       # [ni,1,C,H,W]
    loo = (s - d) / (d.shape[1] - 1)                 # [ni,R,C,H,W] mean of the others
    loo_cos = F.cosine_similarity(d.flatten(2), loo.flatten(2), dim=2).mean(1).cpu().numpy()
    # signed domain projection on GAP
    gd = gap(d.reshape(-1, *d.shape[2:])).view(d.shape[0], d.shape[1], -1)  # [ni,R,Cp]
    q = (gd * dhat.view(1, 1, -1)).sum(2)            # [ni,R]
    cover = (q.mean(1) / gapnorm).cpu().numpy()      # signed fraction of domain gap per step
    snr_dom = (q.mean(1) / q.std(1).clamp_min(1e-8)).cpu().numpy()
    return {"snr": snr, "loo": loo_cos, "cover": cover, "snr_dom": snr_dom}


def main():
    torch.manual_seed(0)
    G, ngf, nts = build_generator(CKPT, DEV)
    times = sb_times(nts, DEV)
    task = TaskModel(TASK_CKPT).to(DEV).eval()
    imag = ImagenetEnc().to(DEV).eval()
    encs = {"task": task, "imagenet": imag}

    df = pd.read_csv(SRC_CSV)
    rng = np.random.RandomState(0)
    picks = [df[df.label == lab].iloc[rng.permutation((df.label == lab).sum())[:NPER]] for lab in sorted(df.label.unique())]
    sdf = pd.concat(picks).sample(frac=1, random_state=1).reset_index(drop=True)
    print("source sample %d %s" % (len(sdf), dict(sdf.label.value_counts())))

    tf = T.Compose([T.ToTensor(), T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                    T.Normalize([0.5] * 3, [0.5] * 3)])

    def load(p):
        im = Image.open(p).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        return tf(im)

    # domain centroids mu_s (BUSI src) and mu_t (BrEaST) per encoder/layer (GAP space)
    mu = {e: {L: {"s": [], "t": []} for L in LAYERS} for e in encs}
    with torch.no_grad():
        for b0 in range(0, len(sdf), 16):
            xb = torch.stack([load(p) for p in sdf.before_png.iloc[b0:b0 + 16]]).to(DEV)
            for e, enc in encs.items():
                fe, _ = enc.feats(xb)
                for L in LAYERS:
                    mu[e][L]["s"].append(gap(fe[L]))
        tdf = pd.read_csv(TGT_CSV)
        pcol = "image_path" if "image_path" in tdf.columns else ("before_png" if "before_png" in tdf.columns else tdf.columns[0])
        tdf = tdf.head(96)
        for b0 in range(0, len(tdf), 16):
            xb = torch.stack([load(p) for p in tdf[pcol].iloc[b0:b0 + 16]]).to(DEV)
            for e, enc in encs.items():
                fe, _ = enc.feats(xb)
                for L in LAYERS:
                    mu[e][L]["t"].append(gap(fe[L]))
    dvec = {}
    for e in encs:
        dvec[e] = {}
        for L in LAYERS:
            ms = torch.cat(mu[e][L]["s"]).mean(0); mt = torch.cat(mu[e][L]["t"]).mean(0)
            gapv = (mt - ms); gn = gapv.norm().clamp_min(1e-8)
            dvec[e][L] = (gapv / gn, gn)
            print("[domain axis] %s %s  ||mu_t-mu_s||=%.3f" % (e, L, float(gn)))

    RES = ["trans", "refine1", "refine_mean", "state1"]
    acc = {e: {L: {rs: {mk: {c: [] for c in ["all", 0, 1]} for mk in ["snr", "loo", "cover", "snr_dom"]}
                   for rs in RES} for L in LAYERS} for e in encs}
    marg = {"x0": {0: [], 1: []}, "u1": {0: [], 1: []}}

    for b0 in range(0, len(sdf), NIMG_B):
        rows = sdf.iloc[b0:b0 + NIMG_B]
        labs = torch.tensor(rows.label.values)
        imgs = torch.stack([load(p) for p in rows.before_png]).to(DEV)
        ni = imgs.shape[0]
        x0 = imgs.repeat_interleave(R, dim=0)
        gen = torch.Generator(device=DEV).manual_seed(4242 + b0)
        states, preds = sb_path_both(G, x0, STEPS, times, ngf, gen)

        for e, enc in encs.items():
            def FE(t):
                f, lg = enc.feats(t)
                return {L: f[L].view(ni, R, *f[L].shape[1:]) for L in LAYERS}, lg
            f_x0, lg_x0 = FE(x0)
            f_u1, lg_u1 = FE(preds[0])
            f_u2, _ = FE(preds[1])
            f_umean_hi = None
            f_st1, _ = FE(states[1])
            # margins (task only)
            if e == "task":
                lx = lg_x0.view(ni, R, 2).mean(1); lu = lg_u1.view(ni, R, 2).mean(1)
                for i in range(ni):
                    y = int(labs[i]); mx = float(lx[i, y] - lx[i, 1 - y]); mu1 = float(lu[i, y] - lu[i, 1 - y])
                    marg["x0"][y].append(mx); marg["u1"][y].append(mu1)
            for L in LAYERS:
                dhat, gn = dvec[e][L]
                defs = {"trans": (f_u1[L], f_x0[L]), "refine1": (f_u2[L], f_u1[L]),
                        "state1": (f_st1[L], f_x0[L])}
                for rs, (fk, fkm1) in defs.items():
                    r = per_image_metrics(fk, fkm1, dhat, gn)
                    for mk in ["snr", "loo", "cover", "snr_dom"]:
                        acc[e][L][rs][mk]["all"].append(r[mk])
                        for c in [0, 1]:
                            sel = (labs.numpy() == c)
                            if sel.any():
                                acc[e][L][rs][mk][c].append(r[mk][sel])
                # refine_mean over k=1..4
                rr = {mk: [] for mk in ["snr", "loo", "cover", "snr_dom"]}
                for k in range(1, STEPS):
                    fa, _ = FE(preds[k]); fb, _ = FE(preds[k - 1])
                    r = per_image_metrics(fa[L], fb[L], dhat, gn)
                    for mk in rr:
                        rr[mk].append(r[mk])
                for mk in rr:
                    acc[e][L]["refine_mean"][mk]["all"].append(np.mean(rr[mk], axis=0))
        torch.cuda.empty_cache()
        print("batch %d-%d done" % (b0, b0 + ni), flush=True)

    def M(x):
        x = [a for a in x if len(a)]
        return float(np.mean(np.concatenate(x))) if x else float("nan")

    print("\n############## v3: DA-BRF operand & domain-directed signal (R=%d, %d imgs) ##############" % (R, len(sdf)))
    for e in encs:
        print("\n============ ENCODER %s ============" % e.upper())
        for L in LAYERS:
            print(" [%s]  (dom_cover>0 = moves TOWARD target; SNR_dom = signed-projection SNR)" % L)
            print("  residual      |  SNR  | LOO_cos | dom_cover | SNR_dom")
            for rs in RES:
                print("  %-12s  | %5.2f |  %5.2f  |  %+.3f   |  %+.2f" %
                      (rs, M(acc[e][L][rs]["snr"]["all"]), M(acc[e][L][rs]["loo"]["all"]),
                       M(acc[e][L][rs]["cover"]["all"]), M(acc[e][L][rs]["snr_dom"]["all"])))
            # class split for trans only
            print("   -- Delta_trans by class (benign=0 / malignant=1):")
            for c in [0, 1]:
                print("      class %d: SNR %.2f  dom_cover %+.3f" %
                      (c, M(acc[e][L]["trans"]["snr"][c]), M(acc[e][L]["trans"]["cover"][c])))

    print("\n============ SOURCE-LABEL MARGIN  x0 -> U1  (task classifier) ============")
    for y, nm in [(0, "benign"), (1, "malignant")]:
        mx = np.mean(marg["x0"][y]); mu1 = np.mean(marg["u1"][y])
        print("  %-9s: margin x0=%+.3f  U1=%+.3f  delta=%+.3f  (neg delta = translation HURTS discriminability)"
              % (nm, mx, mu1, mu1 - mx))
    print("\nREAD: DA-BRF needs Delta_trans with SNR>~2, LOO_cos high, dom_cover>0 (toward target),")
    print("      and margin delta ~0 (translation keeps the lesion discriminable).")


if __name__ == "__main__":
    main()
