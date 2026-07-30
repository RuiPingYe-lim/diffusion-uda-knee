#!/usr/bin/env python
"""SUPERSEDED by deltak_precheck_v2.py -- kept only for provenance. DO NOT cite its numbers.

Two methodological errors were found in this version:
  1. `noise_floor` is the CUMULATIVE divergence between two trajectories up to step k,
     compared against a SINGLE-step increment. The baseline therefore grows with k and
     the ratio is guaranteed to fall below 1 for later steps -- it cannot show that those
     steps are noise.
  2. It captures `prev = G(Xt, t, z)`, which is the ENDPOINT PREDICTION, not the bridge
     state Xt. The "source -> target path" object was never actually measured here.
v2 captures both sequences and computes per-step SNR over R trajectories of the same image.

--- original docstring below ---

Delta_k operand pre-check for DA-BRF / PathFuse.

Generates the ORDERED UNSB bridge path myself (capturing prev at every step of ONE
trajectory) so Delta_k = A^(k) - A^(k-1) is a true same-trajectory increment, not a
mix of trajectory noise. Two generator seeds give the same-step noise floor.

Reports, per encoder depth (ImageNet ResNet50 layer1/layer3/layer4) and per step:
  rel_mag        ||Delta_k|| / ||A^(k-1)||           -- is the step big enough to act on?
  cum_rel        ||A^(5)-A^(0)|| / ||A^(0)||          -- total feature translation
  svar_frac      spatially-varying energy / total    -- 1=structured, 0=global per-channel shift
  step_corr      mean pairwise corr of change-maps    -- do different steps move DIFFERENT regions?
  noise_floor    same-step ||seedA-seedB|| / ||A||    -- rel_mag must beat this to be real
Image level: uniform_frac = 1 - svar_frac(Delta_img)  -- how much of a step is global brightness.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T, models

sys.path.insert(0, "/root/autodl-tmp/UNSB")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 256
CKPT = "/root/autodl-tmp/UNSB/checkpoints/u2b_rev_SB/latest_net_G.pth"
SRC_CSV = "/root/autodl-tmp/breast/cache/fusion_train_busi.csv"   # before_png = BUSI source
N = 128
BATCH = 16
STEPS = 5
TAU = 0.01


# ---- UNSB generator (copied verbatim from train_stage2.py) ----
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


def build_unsb_generator(ckpt, device, ngf=64):
    from models import networks
    opt_txt = os.path.join(os.path.dirname(ckpt), "train_opt.txt")
    d = _parse_train_opt(opt_txt)
    d["gpu_ids"] = []

    class O:
        pass
    o = O(); o.__dict__.update(d)
    G = networks.define_G(d.get("input_nc", 3), d.get("output_nc", 3), d.get("ngf", ngf),
                          d.get("netG", "resnet_9blocks_cond"), d.get("normG", "instance"),
                          not d.get("no_dropout", True), d.get("init_type", "xavier"),
                          d.get("init_gain", 0.02), d.get("no_antialias", False),
                          d.get("no_antialias_up", False), [], o)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    G.load_state_dict(sd, strict=False)
    return G.to(device).eval(), int(d.get("ngf", ngf)), int(d.get("num_timesteps", 5))


def sb_times(T_steps, device):
    incs = np.array([0] + [1 / (i + 1) for i in range(T_steps - 1)])
    t = np.cumsum(incs); t = t / t[-1]; t = 0.5 * t[-1] + 0.5 * t
    t = np.concatenate([np.zeros(1), t])
    return torch.tensor(t).float().to(device)


@torch.no_grad()
def sb_path(G, x, steps, times, ngf, tau=TAU, generator=None):
    """Return [x^(1),...,x^(steps)] captured along ONE trajectory."""
    outs = []; Xt = x; prev = None
    for t in range(steps):
        if t > 0:
            delta = times[t] - times[t - 1]; denom = times[-1] - times[t - 1]
            inter = (delta / denom).reshape(-1, 1, 1, 1)
            scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
            noise = torch.randn(Xt.shape, device=Xt.device, generator=generator)
            Xt = (1 - inter) * Xt + inter * prev + (scale * tau).sqrt() * noise
        ti = (t * torch.ones(x.shape[0], device=x.device)).long()
        z = torch.randn((x.shape[0], 4 * ngf), device=x.device, generator=generator)
        prev = G(Xt, ti, z)
        outs.append(prev)
    return outs


# ---- encoder: ImageNet ResNet50, features at layer1/layer3/layer4 ----
class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, x_m11):
        # x_m11 in [-1,1] -> [0,1] -> ImageNet norm
        x = (x_m11 + 1) / 2
        x = (x - self.mean.to(x)) / self.std.to(x)
        s = self.stem(x); a1 = self.l1(s); a2 = self.l2(a1); a3 = self.l3(a2); a4 = self.l4(a3)
        return {"layer1": a1, "layer3": a3, "layer4": a4}


# ---- metrics (all per-sample, then averaged) ----
def relnorm(delta, base):
    dn = delta.flatten(1).norm(dim=1); bn = base.flatten(1).norm(dim=1).clamp_min(1e-8)
    return (dn / bn).cpu().numpy()


def svar_frac(delta):
    m = delta.mean(dim=(2, 3), keepdim=True)
    varying = delta - m
    num = varying.flatten(1).norm(dim=1) ** 2
    den = (delta.flatten(1).norm(dim=1).clamp_min(1e-10)) ** 2
    return (num / den).cpu().numpy()


def changemap(delta):
    return delta.norm(dim=1)  # [B,H,W]


def pairwise_corr(maps):
    def z(m):
        f = m.flatten(1)
        return (f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True).clamp_min(1e-8)
    fs = [z(m) for m in maps]; K = len(fs)
    tot = torch.zeros(fs[0].shape[0], device=fs[0].device); cnt = 0
    for i in range(K):
        for j in range(i + 1, K):
            tot += (fs[i] * fs[j]).mean(1); cnt += 1
    return (tot / cnt).cpu().numpy()


def main():
    G, ngf, nts = build_unsb_generator(CKPT, DEV)
    times = sb_times(nts, DEV)
    enc = Enc().to(DEV).eval()
    df = pd.read_csv(SRC_CSV).head(N)
    tot = T.Compose([T.ToTensor(),
                     T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                     T.Normalize([0.5] * 3, [0.5] * 3)])

    layers = ["layer1", "layer3", "layer4"]
    acc = {L: {"rel": [[] for _ in range(STEPS)], "svar": [[] for _ in range(STEPS)],
               "noise": [[] for _ in range(STEPS)], "corr": [], "cum": []} for L in layers}
    img_rel = [[] for _ in range(STEPS)]; img_unif = [[] for _ in range(STEPS)]
    done = 0
    for b0 in range(0, len(df), BATCH):
        rows = df.iloc[b0:b0 + BATCH]
        ims = []
        for _, r in rows.iterrows():
            p = r["before_png"]
            im = Image.open(p).convert("L")
            if im.size != (SIZE, SIZE):
                im = im.resize((SIZE, SIZE), Image.BICUBIC)
            ims.append(tot(im))
        x0 = torch.stack(ims).to(DEV)                        # [B,3,256,256] in [-1,1]

        gA = torch.Generator(device=DEV).manual_seed(1000 + b0)
        gB = torch.Generator(device=DEV).manual_seed(9000 + b0)
        pathA = [x0] + sb_path(G, x0, STEPS, times, ngf, generator=gA)   # 6 tensors
        pathB = [x0] + sb_path(G, x0, STEPS, times, ngf, generator=gB)

        # image-level per step
        for k in range(1, STEPS + 1):
            dimg = pathA[k] - pathA[k - 1]
            img_rel[k - 1].append(relnorm(dimg, pathA[k - 1]))
            img_unif[k - 1].append(1.0 - svar_frac(dimg))

        featA = [enc(v) for v in pathA]
        featB = [enc(v) for v in pathB]
        for L in layers:
            A = [f[L] for f in featA]; Bf = [f[L] for f in featB]
            deltas = [A[k] - A[k - 1] for k in range(1, STEPS + 1)]
            for k in range(STEPS):
                acc[L]["rel"][k].append(relnorm(deltas[k], A[k]))
                acc[L]["svar"][k].append(svar_frac(deltas[k]))
                acc[L]["noise"][k].append(relnorm(A[k + 1] - Bf[k + 1], A[k + 1]))
            acc[L]["cum"].append(relnorm(A[STEPS] - A[0], A[0]))
            acc[L]["corr"].append(pairwise_corr([changemap(d) for d in deltas]))
        done += len(rows)
        print("processed %d/%d" % (done, len(df)), flush=True)

    def m(x):
        return float(np.mean(np.concatenate(x)))

    print("\n================ Delta_k OPERAND PRE-CHECK (u2b_rev_SB, %d BUSI src, %d steps) ================" % (len(df), STEPS))
    print("\n--- IMAGE LEVEL (per step k) ---")
    print("step   rel_change   uniform_frac(=global brightness/contrast share)")
    for k in range(STEPS):
        print("  %d      %.4f       %.3f" % (k + 1, m(img_rel[k]), m(img_unif[k])))

    for L in layers:
        print("\n--- FEATURE LEVEL @ %s ---" % L)
        print("  cumulative ||A5-A0||/||A0|| = %.4f" % m(acc[L]["cum"]))
        print("  cross-step change-map corr  = %.3f   (high=all steps hit SAME regions -> PathFuse weak)" % m(acc[L]["corr"]))
        print("  step | rel_mag | noise_floor | rel/noise | svar_frac(1=structured,0=global shift)")
        for k in range(STEPS):
            rm = m(acc[L]["rel"][k]); nf = m(acc[L]["noise"][k]); sv = m(acc[L]["svar"][k])
            print("   %d   |  %.4f |   %.4f    |  %5.1fx  |  %.3f" % (k + 1, rm, nf, rm / max(nf, 1e-6), sv))
    print("\nREAD: DA-BRF needs rel_mag >> noise_floor AND svar_frac not ~0.")
    print("      PathFuse needs LOW cross-step corr (steps move different regions) AND non-trivial svar_frac.")


if __name__ == "__main__":
    main()
