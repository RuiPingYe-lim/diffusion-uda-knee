#!/usr/bin/env python
"""RETRACTED -- kept for provenance. DO NOT cite its numbers. See scripts/v8_path_corrected.py.

x0 is read from ``cache/fusion_train_busi.csv:before_png``, which is NOT the source image but
``results_u2b_rev/.../fake_5/`` -- BUSI already translated five bridge steps toward BrEaST.
The whole "path" measured below therefore starts from an already-translated image. The true
source rendering is ``da_route/da_manifest.csv:raw``; check pixel statistics, not column
names (raw: mean 64.52 / std 55.70; before_png: 55.87 / 34.93).

This file's headline verdict -- "steps 2-5 sit at SNR<=1 and do NOT progress toward the
target, so the multi-step premise fails" -- is RETRACTED. Re-run from the correct x0
(v8_path_corrected.py), the endpoint sequence progresses MONOTONICALLY on both domain
measures: class-conditional coverage 0.19 -> 0.39 (benign) and 0.31 -> 0.52 (malignant),
domain-probe logit -9.33 -> -2.09. What is really there is a monotone trade-off, deeper
steps buying domain coverage at the cost of recoverable content (probe AUC 0.782 -> 0.686).

--- original docstring below ---

Delta_k operand pre-check v2 -- corrected after the endpoint-vs-state critique.

FIXES over v1:
 1. Capture BOTH the true bridge states Xt AND the endpoint predictions prev=G(Xt,.).
    Test the two sequences separately (v1 conflated them and only kept `prev`).
 2. Per-step SNR the RIGHT way: over R independent trajectories of the SAME image,
    S_k=||mean_r d_k||, N_k=sqrt(mean_r||d_k-mean||^2), SNR=S/N, plus direction
    consistency C_k=mean_r cos(d_k, mean). (v1 compared a single-step increment to a
    CUMULATIVE two-trajectory divergence -- an unfair, k-growing baseline.)
 3. Stratified label-balanced sampling (v1 used df.head, which is order-biased).
 4. Print load_state_dict missing/unexpected keys.
 5. Two encoders: ImageNet ResNet50 (ImageNet-normed, AUX) and the trained breast
    task backbone (fed in its own Normalize(0.5) space, PRIMARY).
 6. Image residual decomposed into brightness (per-channel DC) + contrast (projection
    onto centered previous image) + genuine structure. (v1's 1-svar mislabelled the
    DC-only part as brightness/contrast.)
 7. Per-step cosine of pooled features to the BrEaST target centroid (does the
    sequence actually move toward the target domain?).
"""
import os, sys, re
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
TASK_CKPT = "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt"
SRC_CSV = "/root/autodl-tmp/breast/cache/fusion_train_busi.csv"     # before_png,label (BUSI source)
TGT_CSV = "/root/autodl-tmp/breast/cache/breast_train.csv"          # unlabeled target (BrEaST)
NPER = 32          # per class -> 64 source images
R = 8             # trajectories per image
NIMG_B = 8        # distinct images per batch (batch = NIMG_B*R)
STEPS = 5
TAU = 0.01
LAYERS = ["layer1", "layer3", "layer4"]


# ---------- UNSB generator ----------
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
    miss = [k for k in miss if "num_batches" not in k]
    print("[generator load] missing=%d unexpected=%d" % (len(miss), len(unexp)))
    if miss:
        print("   missing sample:", miss[:4])
    if unexp:
        print("   unexpected sample:", list(unexp)[:4])
    return G.to(device).eval(), int(d.get("ngf", 64)), int(d.get("num_timesteps", 5))


def sb_times(T_steps, device):
    incs = np.array([0] + [1 / (i + 1) for i in range(T_steps - 1)])
    t = np.cumsum(incs); t = t / t[-1]; t = 0.5 * t[-1] + 0.5 * t
    t = np.concatenate([np.zeros(1), t])
    return torch.tensor(t).float().to(device)


@torch.no_grad()
def sb_path_both(G, x, steps, times, ngf, gen):
    """Return (states, preds). states=[x0,X_t1,...,X_t{steps-1}] (true bridge states),
       preds=[xhat1^(1..steps)] (endpoint predictions -- what U1..U5 actually feed)."""
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


# ---------- encoders ----------
class ImagenetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x_m11):
        x = (x_m11 + 1) / 2
        x = (x - self.mean) / self.std
        h = self.seq[0:4](x)
        a1 = self.seq[4](h); a2 = self.seq[5](a1); a3 = self.seq[6](a2); a4 = self.seq[7](a3)
        return {"layer1": a1, "layer3": a3, "layer4": a4}


class TaskEnc(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        m = models.resnet50(weights=None)
        self.seq = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        stem_sd = {k[len("stem."):]: v for k, v in sd.items() if k.startswith("stem.")}
        miss, unexp = self.seq.load_state_dict(stem_sd, strict=False)
        miss = [k for k in miss if "num_batches" not in k]
        print("[task backbone load] mapped=%d  missing=%d unexpected=%d" % (len(stem_sd), len(miss), len(unexp)))
        if miss:
            print("   missing sample:", miss[:4])
        if unexp:
            print("   unexpected sample:", list(unexp)[:4])

    @torch.no_grad()
    def forward(self, x_m11):
        # task backbone trained with Normalize(0.5) => x_m11 is already its input space
        h = self.seq[0:4](x_m11)
        a1 = self.seq[4](h); a2 = self.seq[5](a1); a3 = self.seq[6](a2); a4 = self.seq[7](a3)
        return {"layer1": a1, "layer3": a3, "layer4": a4}


# ---------- image-level brightness / contrast / structure ----------
def decomp_img(delta, prev):
    """delta,prev: [B,3,H,W]. Returns (b_frac,c_frac,s_frac) per sample."""
    HW = delta.shape[2] * delta.shape[3]
    m = delta.mean(dim=(2, 3), keepdim=True)                      # per-channel DC (brightness)
    bright = m.expand_as(delta)
    r1 = delta - bright
    xc = prev - prev.mean(dim=(2, 3), keepdim=True)               # centered previous image
    coef = (r1 * xc).sum(dim=(2, 3), keepdim=True) / (xc * xc).sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    contrast = coef * xc                                          # scaling of existing pattern
    struct = r1 - contrast
    tot = (delta * delta).sum(dim=(1, 2, 3)).clamp_min(1e-10)
    eb = (bright * bright).sum(dim=(1, 2, 3))
    ec = (contrast * contrast).sum(dim=(1, 2, 3))
    es = (struct * struct).sum(dim=(1, 2, 3))
    return (eb / tot).cpu().numpy(), (ec / tot).cpu().numpy(), (es / tot).cpu().numpy()


def gap(x):
    return x.mean(dim=(2, 3))  # [B,C]


def main():
    torch.manual_seed(0)
    G, ngf, nts = build_generator(CKPT, DEV)
    times = sb_times(nts, DEV)
    encs = {"imagenet": ImagenetEnc().to(DEV).eval(), "task": TaskEnc(TASK_CKPT).to(DEV).eval()}

    # stratified source sample
    df = pd.read_csv(SRC_CSV)
    rng = np.random.RandomState(0)
    picks = []
    for lab in sorted(df["label"].unique()):
        sub = df[df["label"] == lab]
        picks.append(sub.iloc[rng.permutation(len(sub))[:NPER]])
    sdf = pd.concat(picks).sample(frac=1, random_state=1).reset_index(drop=True)
    print("source sample: %d (%s)" % (len(sdf), dict(sdf["label"].value_counts())))

    tot = T.Compose([T.ToTensor(),
                     T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                     T.Normalize([0.5] * 3, [0.5] * 3)])

    def load_img(p):
        im = Image.open(p).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        return tot(im)

    # target centroid (BrEaST) per encoder per layer
    tgt_mu = {e: {L: [] for L in LAYERS} for e in encs}
    try:
        tdf = pd.read_csv(TGT_CSV)
        pcol = "image_path" if "image_path" in tdf.columns else ("before_png" if "before_png" in tdf.columns else tdf.columns[0])
        tdf = tdf.head(64)
        for b0 in range(0, len(tdf), 16):
            xb = torch.stack([load_img(p) for p in tdf[pcol].iloc[b0:b0 + 16]]).to(DEV)
            for e, enc in encs.items():
                f = enc(xb)
                for L in LAYERS:
                    tgt_mu[e][L].append(gap(f[L]))
        tgt_mu = {e: {L: torch.cat(v).mean(0) for L, v in d.items()} for e, d in tgt_mu.items()}
        print("target centroid built from %d BrEaST imgs (%s)" % (len(tdf), pcol))
    except Exception as ex:
        print("WARN target centroid skipped:", ex)
        tgt_mu = None

    seqs = ["states", "preds"]
    # accumulators
    snr = {e: {sq: {L: [[] for _ in range(STEPS)] for L in LAYERS} for sq in seqs} for e in encs}
    con = {e: {sq: {L: [[] for _ in range(STEPS)] for L in LAYERS} for sq in seqs} for e in encs}
    rel = {e: {sq: {L: [[] for _ in range(STEPS)] for L in LAYERS} for sq in seqs} for e in encs}
    tcos = {e: {sq: {L: [[] for _ in range(STEPS + 1)] for L in LAYERS} for sq in seqs} for e in encs}
    bcs = {sq: [[[], [], []] for _ in range(STEPS)] for sq in seqs}   # image-level, per step

    for b0 in range(0, len(sdf), NIMG_B):
        rows = sdf.iloc[b0:b0 + NIMG_B]
        imgs = torch.stack([load_img(p) for p in rows["before_png"]]).to(DEV)  # [ni,3,H,W]
        ni = imgs.shape[0]
        x0 = imgs.repeat_interleave(R, dim=0)                                  # [ni*R,...]
        gen = torch.Generator(device=DEV).manual_seed(4242 + b0)
        states, preds = sb_path_both(G, x0, STEPS, times, ngf, gen)
        seqdata = {"states": states, "preds": preds}

        # image-level decomposition (per step, seed-pooled)
        for sq in seqs:
            S = seqdata[sq]
            for k in range(1, len(S)):
                bf, cf, sf = decomp_img(S[k] - S[k - 1], S[k - 1])
                bcs[sq][k - 1][0].append(bf); bcs[sq][k - 1][1].append(cf); bcs[sq][k - 1][2].append(sf)

        for e, enc in encs.items():
            for sq in seqs:
                S = seqdata[sq]
                feats = [enc(v) for v in S]                     # each dict of layers
                for L in LAYERS:
                    fl = [f[L].view(ni, R, *f[L].shape[1:]) for f in feats]   # [ni,R,C,H,W] per step
                    # cosine of seed-mean pooled feature to target centroid
                    if tgt_mu is not None:
                        for k in range(len(fl)):
                            pooled = gap(fl[k].reshape(ni * R, *fl[k].shape[2:])).view(ni, R, -1).mean(1)  # [ni,D]
                            mu = tgt_mu[e][L]
                            c = torch.nn.functional.cosine_similarity(pooled, mu.unsqueeze(0), dim=1)
                            tcos[e][sq][L][k].append(c.cpu().numpy())
                    # per-step SNR / consistency / rel signal
                    for k in range(1, len(fl)):
                        d = fl[k] - fl[k - 1]                    # [ni,R,C,H,W]
                        md = d.mean(1)                           # [ni,C,H,W] consistent part
                        Sk = md.flatten(1).norm(dim=1)           # [ni]
                        var = (d - md.unsqueeze(1)).flatten(2).norm(dim=2) ** 2  # [ni,R]
                        Nk = var.mean(1).clamp_min(1e-12).sqrt()  # [ni]
                        base = fl[k - 1].mean(1).flatten(1).norm(dim=1).clamp_min(1e-8)
                        # direction consistency
                        dfl = d.flatten(2)                       # [ni,R,CHW]
                        mfl = md.flatten(1).unsqueeze(1)         # [ni,1,CHW]
                        cc = torch.nn.functional.cosine_similarity(dfl, mfl, dim=2).mean(1)  # [ni]
                        snr[e][sq][L][k - 1].append((Sk / Nk).cpu().numpy())
                        con[e][sq][L][k - 1].append(cc.cpu().numpy())
                        rel[e][sq][L][k - 1].append((Sk / base).cpu().numpy())
                del feats
            torch.cuda.empty_cache()
        print("batch done imgs %d-%d" % (b0, b0 + ni), flush=True)

    def m(x):
        return float(np.mean(np.concatenate(x))) if x and len(np.concatenate(x)) else float("nan")

    print("\n############## Delta_k PRE-CHECK v2  (R=%d traj, %d src imgs, %d steps) ##############" % (R, len(sdf), STEPS))
    for e in encs:
        norm = "ImageNet-norm" if e == "imagenet" else "Normalize(0.5), trained-on-task"
        print("\n=================== ENCODER: %s  (%s) ===================" % (e.upper(), norm))
        for sq in seqs:
            title = "TRUE BRIDGE STATES  x0->X_t1->...  (PathFuse-path premise)" if sq == "states" \
                    else "ENDPOINT PREDICTIONS  U1->U2->... (what the classifier actually uses)"
            print("\n  --- sequence: %s ---" % title)
            for L in LAYERS:
                print("   [%s]" % L)
                if tgt_mu is not None:
                    tc = [m(tcos[e][sq][L][k]) for k in range(STEPS + 1)]
                    print("     cos-to-BrEaST-centroid by index: " + " ".join("%.3f" % v for v in tc) + "   (should rise if moving to target)")
                print("     step |  SNR  | dir_cos | rel_signal   (SNR>~2 & dir_cos>0.5 => usable operand)")
                for k in range(STEPS - 1):
                    print("      %d   | %5.2f |  %5.2f  |  %.4f" %
                          (k + 1, m(snr[e][sq][L][k]), m(con[e][sq][L][k]), m(rel[e][sq][L][k])))
    print("\n=================== IMAGE-LEVEL residual decomposition (fractions of energy) ===================")
    for sq in seqs:
        print("  sequence %s:" % sq)
        print("   step | brightness | contrast | structure")
        for k in range(STEPS - 1):
            print("    %d   |   %.3f    |  %.3f   |  %.3f" %
                  (k + 1, m(bcs[sq][k][0]), m(bcs[sq][k][1]), m(bcs[sq][k][2])))
    print("\nGATE: PathFuse needs, on the RELEVANT sequence, later steps with SNR>~2, dir_cos>0.5,")
    print("      monotone rise in cos-to-target, and non-trivial structure fraction. Else -> not a path.")


if __name__ == "__main__":
    main()
