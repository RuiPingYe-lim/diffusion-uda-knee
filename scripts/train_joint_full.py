#!/usr/bin/env python
"""FULL joint training: UNSB's own objectives run alongside the classifier's.

Stage 2 unfroze the generator but ran none of UNSB's own losses -- only an L1
anchor toward the pretrained output. That anchor is not an objective: it says
"do not move", never "move toward the target style". Two consequences, both
raised by the user and both correct:

  * it is a single-objective fine-tune with a leash, not a balanced joint
    optimisation of two objectives;
  * stage 3 ("make the style change harder while preserving the lesion") is
    structurally impossible, because no loss term represents style or content
    and there is therefore no knob to turn.

This file restores UNSB's three objectives (all four networks are available as
trained checkpoints) and adds the classifier on top:

    D step:  GAN discriminator, on (fake_B, real_B)
    E step:  the Schrodinger-bridge potential
    G step:  lambda_GAN * GAN + lambda_SB * SB + lambda_NCE * PatchNCE   <- UNSB's own
             + lambda_task * (CE + lambda_sup * SupCon)                  <- the task

lambda_task is the ONE new balance knob. lambda_task=0 reproduces plain UNSB
training; the classifier's gradient never reaches G, which is the control arm.

The discriminator needs real TARGET images. Unlabelled BrEaST training images are
used for that -- no target label is read anywhere.

Faithfulness: options are read from the saved train_opt.txt, and the bridge walk,
the noisy endpoints and the identity branch replicate models/sb_model.py exactly.
netF builds its MLPs lazily, so it is primed with one forward pass before its
weights are loaded.

The LOCKED 51-case BrEaST test set is not referenced anywhere in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, "/root/autodl-tmp/breast")
sys.path.insert(0, "/root/autodl-tmp/UNSB")
from train_stage1 import AttnPoolNet, EvalDataset, case_auc, predict, sup_con_loss  # noqa: E402
from train_stage2 import _parse_train_opt, apply_geom, sb_times  # noqa: E402

SIZE = 256
UNSB = "/root/autodl-tmp/UNSB"


def sha256_file(p):
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Opt:
    pass


def load_opt(ckpt_dir, **over):
    d = _parse_train_opt(os.path.join(ckpt_dir, "train_opt.txt"))
    d.update(over)
    d["gpu_ids"] = []
    d["isTrain"] = True
    d["phase"] = "train"
    o = Opt()
    o.__dict__.update(d)
    o.nce_layers_list = [int(i) for i in str(d["nce_layers"]).split(",")]
    return o


def build_unsb_stack(ckpt_dir, dev, opt):
    """Rebuild G, D, E, F with the trained weights. Returns them plus the criteria."""
    from models import networks
    from models.patchnce import PatchNCELoss

    G = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                          not opt.no_dropout, opt.init_type, opt.init_gain,
                          opt.no_antialias, opt.no_antialias_up, [], opt).to(dev)
    D = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                          opt.init_type, opt.init_gain, opt.no_antialias, [], opt).to(dev)
    E = networks.define_D(opt.output_nc * 4, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                          opt.init_type, opt.init_gain, opt.no_antialias, [], opt).to(dev)
    Fnet = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                             opt.init_type, opt.init_gain, opt.no_antialias, [], opt).to(dev)

    def load(net, name, strict=True):
        p = os.path.join(ckpt_dir, f"latest_net_{name}.pth")
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        miss, unexp = net.load_state_dict(sd, strict=False)
        bad = [k for k in list(miss) + list(unexp) if "num_batches" not in k]
        if strict and bad:
            raise RuntimeError(f"net{name} load mismatch: {bad[:4]}")
        return len(bad)

    load(G, "G"); load(D, "D"); load(E, "E")
    # netF builds its MLPs lazily on the first forward, and PatchSampleF.create_mlp
    # only moves them to the GPU when self.gpu_ids is non-empty (networks.py:556).
    # define_F was called with gpu_ids=[] to keep the module off DataParallel, so
    # the MLPs would be created on the CPU and immediately mismatch the GPU feats.
    # Set gpu_ids just for the priming pass, then prime, then load the weights.
    Fnet.gpu_ids = [0] if dev.type == "cuda" else []
    with torch.no_grad():
        dummy = torch.zeros(2, 3, SIZE, SIZE, device=dev)
        ti = torch.zeros(2, device=dev).long()
        z = torch.zeros(2, 4 * opt.ngf, device=dev)
        feats = G(dummy, ti, z, opt.nce_layers_list, encode_only=True)
        Fnet(feats, opt.num_patches, None)
    Fnet = Fnet.to(dev)
    load(Fnet, "F")

    critGAN = networks.GANLoss(opt.gan_mode).to(dev)
    critNCE = [PatchNCELoss(opt).to(dev) for _ in opt.nce_layers_list]
    return G, D, E, Fnet, critGAN, critNCE


class SBJoint:
    """Replicates models/sb_model.py's forward + the three optimisation steps."""

    def __init__(self, G, D, E, Fnet, critGAN, critNCE, opt, dev, lr_gen):
        self.G, self.D, self.E, self.F = G, D, E, Fnet
        self.critGAN, self.critNCE, self.opt, self.dev = critGAN, critNCE, opt, dev
        self.times = sb_times(opt.num_timesteps, dev)
        b1, b2 = getattr(opt, "beta1", 0.5), getattr(opt, "beta2", 0.999)
        self.opt_G = torch.optim.Adam(G.parameters(), lr=lr_gen, betas=(b1, b2))
        self.opt_D = torch.optim.Adam(D.parameters(), lr=lr_gen, betas=(b1, b2))
        self.opt_E = torch.optim.Adam(E.parameters(), lr=lr_gen, betas=(b1, b2))
        self.opt_F = torch.optim.Adam(Fnet.parameters(), lr=lr_gen, betas=(b1, b2))

    def forward(self, real_A, real_B):
        """Walk the bridge to a random timestep, then produce fake_B / fake_B2 / idt_B."""
        o, T = self.opt, self.opt.num_timesteps
        times, tau, dev = self.times, o.tau, self.dev
        bs = real_A.size(0)
        self.time_idx = (torch.randint(T, size=[1], device=dev)
                         * torch.ones(1, device=dev)).long()
        with torch.no_grad():
            self.G.eval()
            Xt = Xt2 = XtB = None
            Xt_1 = Xt_12 = Xt_1B = None
            for t in range(self.time_idx.int().item() + 1):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                ti = (t * torch.ones(bs, device=dev)).long()
                Xt = real_A if t == 0 else (1 - inter) * Xt + inter * Xt_1.detach() \
                    + (scale * tau).sqrt() * torch.randn_like(Xt)
                Xt_1 = self.G(Xt, ti, torch.randn(bs, 4 * o.ngf, device=dev))
                Xt2 = real_A if t == 0 else (1 - inter) * Xt2 + inter * Xt_12.detach() \
                    + (scale * tau).sqrt() * torch.randn_like(Xt2)
                Xt_12 = self.G(Xt2, ti, torch.randn(bs, 4 * o.ngf, device=dev))
                if o.nce_idt:
                    XtB = real_B if t == 0 else (1 - inter) * XtB + inter * Xt_1B.detach() \
                        + (scale * tau).sqrt() * torch.randn_like(XtB)
                    Xt_1B = self.G(XtB, ti, torch.randn(bs, 4 * o.ngf, device=dev))
            self.real_A_noisy, self.real_A_noisy2 = Xt.detach(), Xt2.detach()
            self.XtB = XtB.detach() if o.nce_idt else None

        self.G.train()
        self.real_A, self.real_B = real_A, real_B
        realt = torch.cat([self.real_A_noisy, self.XtB], 0) if o.nce_idt else self.real_A_noisy
        z_in = torch.randn(realt.size(0), 4 * o.ngf, device=dev)
        z_in2 = torch.randn(bs, 4 * o.ngf, device=dev)
        self.fake = self.G(realt, self.time_idx, z_in)
        self.fake_B2 = self.G(self.real_A_noisy2, self.time_idx, z_in2)
        self.fake_B = self.fake[:bs]
        self.idt_B = self.fake[bs:] if o.nce_idt else None
        return self.fake_B

    def nce(self, src, tgt):
        o = self.opt
        z = torch.randn(self.real_A.size(0), 4 * o.ngf, device=self.dev)
        fq = self.G(tgt, self.time_idx * 0, z, o.nce_layers_list, encode_only=True)
        fk = self.G(src, self.time_idx * 0, z, o.nce_layers_list, encode_only=True)
        fk_pool, ids = self.F(fk, o.num_patches, None)
        fq_pool, _ = self.F(fq, o.num_patches, ids)
        tot = 0.0
        for q, k, crit in zip(fq_pool, fk_pool, self.critNCE):
            tot = tot + crit(q, k).mean() * o.lambda_NCE
        return tot / len(self.critNCE)

    def step_D_E(self):
        """UNSB's discriminator and bridge-potential updates (G held fixed)."""
        o = self.opt
        for p in self.D.parameters():
            p.requires_grad_(True)
        self.opt_D.zero_grad()
        lD = (self.critGAN(self.D(self.fake_B.detach(), self.time_idx), False).mean()
              + self.critGAN(self.D(self.real_B, self.time_idx), True).mean()) * 0.5
        lD.backward(); self.opt_D.step()

        for p in self.E.parameters():
            p.requires_grad_(True)
        self.opt_E.zero_grad()
        X1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], 1)
        X2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], 1)
        tmp = torch.logsumexp(self.E(X1, self.time_idx, X2).reshape(-1), 0).mean()
        lE = -self.E(X1, self.time_idx, X1).mean() + tmp + tmp ** 2
        lE.backward(); self.opt_E.step()
        return float(lD), float(lE)

    def g_loss(self):
        """UNSB's own generator objective: GAN + SB + PatchNCE."""
        o = self.opt
        for p in list(self.D.parameters()) + list(self.E.parameters()):
            p.requires_grad_(False)
        gan = self.critGAN(self.D(self.fake_B, self.time_idx), True).mean() * o.lambda_GAN
        X1 = torch.cat([self.real_A_noisy, self.fake_B], 1)
        X2 = torch.cat([self.real_A_noisy2, self.fake_B2], 1)
        ET = self.E(X1, self.time_idx, X1).mean() \
            - torch.logsumexp(self.E(X1, self.time_idx, X2).reshape(-1), 0)
        sb = -(o.num_timesteps - self.time_idx[0]) / o.num_timesteps * o.tau * ET \
            + o.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)
        nce = self.nce(self.real_A, self.fake_B)
        if o.nce_idt:
            nce = (nce + self.nce(self.real_B, self.idt_B)) * 0.5
        return gan + o.lambda_SB * sb + o.lambda_NCE * nce, float(gan), float(sb), float(nce)


@torch.no_grad()
def style_closure(G, times, opt, src_x, tgt_stats, dev, steps=1, chunk=32):
    """Is the generator still translating TOWARD the target domain?

    Stage 2 monitored only the translation MAGNITUDE |G(x)-x|, which says the output
    still differs from the input but not that it differs in the target's DIRECTION.
    A generator can drift while its magnitude holds. This closes that gap: it
    reports the fraction of the source->target brightness gap that the CURRENT
    generator's output closes, on the same pooled-between-image SD scale used
    throughout (the pretrained generator closes 36.4%).

    PROBE SIZE MATTERS. Measured noise floor of this metric: z-sampling contributes
    only 0.50 pp, but the choice of probe images contributes 15.2 pp at n=32 (batches
    of 32 gave 20.5% to 59.1% for the SAME frozen generator). n=256 brings it to
    2.0 pp. Use >=256, and evaluate in chunks -- a single 256-image forward exceeds
    the conv kernel's 32-bit indexing limit.

    A falling closure with a rising drift means the task loss is pulling the
    generator off the domain, and any AUC gain cannot be called domain translation.
    """
    G.eval()
    outs, srcs = [], []
    for i in range(0, src_x.shape[0], chunk):
        x = src_x[i:i + chunk]
        Xt, prev = x, None
        for t in range(steps):
            ti = (t * torch.ones(x.shape[0], device=dev)).long()
            prev = G(Xt, ti, torch.randn(x.shape[0], 4 * opt.ngf, device=dev))
        outs.append(((prev.clamp(-1, 1) + 1) / 2).mean(dim=(1, 2, 3)).cpu().numpy())
        srcs.append(((x + 1) / 2).mean(dim=(1, 2, 3)).cpu().numpy())
    G.train()
    m_out, m_src = np.concatenate(outs), np.concatenate(srcs)
    t_mean, t_sd = tgt_stats

    def gap(a):
        sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(a) - 1) * t_sd ** 2) / (2 * len(a) - 2))
        return (t_mean - a.mean()) / max(sd, 1e-9)

    return float(100 * (1 - abs(gap(m_out)) / max(abs(gap(m_src)), 1e-9)))


class SrcDataset(Dataset):
    def __init__(self, df, train=False, aug_seed=0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.rng = random.Random(aug_seed)
        self.to_t = T.Compose([T.ToTensor(),
                               T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                               T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r["raw"]).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        flip = 1.0 if (self.train and self.rng.random() < 0.5) else 0.0
        ang = self.rng.uniform(-10, 10) if self.train else 0.0
        return self.to_t(im), float(flip), float(ang), int(r["label"]), str(r["case_id"])


class TgtImgDataset(Dataset):
    """Unlabelled target images -- the discriminator's `real_B`. No label is read."""

    def __init__(self, paths):
        self.p = list(paths)
        self.to_t = T.Compose([T.ToTensor(),
                               T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                               T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        im = Image.open(self.p[i]).convert("L")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.BICUBIC)
        return self.to_t(im)


def main():
    ap = argparse.ArgumentParser("full joint training: UNSB objectives + classifier")
    ap.add_argument("--manifest", default="/root/autodl-tmp/breast/da_route/da_manifest.csv")
    ap.add_argument("--target_csv", default="/root/autodl-tmp/breast/cache/fusion_eval_breast_diag.csv")
    ap.add_argument("--tgt_train_csv", default="/root/autodl-tmp/breast/cache/breast_train.csv",
                    help="unlabelled target images for the discriminator")
    ap.add_argument("--ckpt_dir", default=f"{UNSB}/checkpoints/u2b_rev_SB")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/breast/joint/runs")
    ap.add_argument("--sealed_dir", default="/root/autodl-tmp/breast/joint/sealed")
    ap.add_argument("--lambda_task", type=float, default=1.0,
                    help="0 reproduces plain UNSB training (the control): no task gradient reaches G")
    ap.add_argument("--lambda_sup", type=float, default=0.0)
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--lr_gen", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pool", choices=["attn", "gap"], default="attn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss_mode", choices=["unsb", "anchor"], default="unsb",
                    help="the generator's OWN objective. unsb: GAN + bridge + PatchNCE (the real "
                         "thing). anchor: an L1 leash toward the frozen generator, which is what "
                         "stage 2 used -- not an objective at all, it only says 'do not move'. "
                         "Both modes share the identical forward pass, data, optimiser and "
                         "hyper-parameters, so the contrast isolates the loss structure alone; "
                         "comparing the two SCRIPTS instead would also confound the bridge walk.")
    ap.add_argument("--lambda_anchor", type=float, default=1.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--cal_epochs", type=int, default=3)
    ap.add_argument("--probe_n", type=int, default=256,
                    help="images in the style-closure probe; 32 has a 15.2 pp noise floor, 256 has 2.0 pp")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = (f"full_task{a.lambda_task:g}_sup{a.lambda_sup:g}_s{a.seed}" if a.loss_mode == "unsb"
           else f"anchor_task{a.lambda_task:g}_sup{a.lambda_sup:g}_s{a.seed}")
    run = Path(a.out_dir) / tag
    run.mkdir(parents=True, exist_ok=True)
    Path(a.sealed_dir).mkdir(parents=True, exist_ok=True)

    opt = load_opt(a.ckpt_dir, batch_size=a.batch_size)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    G, D, E, Fnet, cG, cN = build_unsb_stack(a.ckpt_dir, dev, opt)
    import copy
    G_ref = copy.deepcopy(G).eval()
    for p in G_ref.parameters():
        p.requires_grad_(False)
    sb = SBJoint(G, D, E, Fnet, cG, cN, opt, dev, a.lr_gen)
    clf = AttnPoolNet(pool=a.pool).to(dev)
    opt_clf = torch.optim.AdamW(clf.parameters(), lr=a.lr, weight_decay=1e-4)

    df = pd.read_csv(a.manifest)
    tr_df = df[df.split == "src_train"]
    tr = DataLoader(SrcDataset(tr_df, True, a.seed + 90000), batch_size=a.batch_size,
                    shuffle=True, num_workers=4, drop_last=True,
                    generator=torch.Generator().manual_seed(a.seed + 50000))
    tgt_paths = pd.read_csv(a.tgt_train_csv).image_path.tolist()
    tgt_dl = DataLoader(TgtImgDataset(tgt_paths), batch_size=a.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    tgt = pd.read_csv(a.target_csv)
    tg = DataLoader(EvalDataset(tgt["before_png"], tgt.label, tgt.case_id, a.resize),
                    batch_size=32, num_workers=4)
    # SOURCE validation (BUSI valid, raw + translated views) -- logged every epoch so
    # any hyper-parameter (e.g. the classifier lr) can be selected WITHOUT touching a
    # target label. Uses the same manifest's src_valid split.
    va_df = df[df.split == "src_valid"]
    va_raw = DataLoader(EvalDataset(va_df["raw"], va_df.label, va_df.case_id, a.resize),
                        batch_size=32, num_workers=4)
    va_u1 = DataLoader(EvalDataset(va_df["U1"], va_df.label, va_df.case_id, a.resize),
                       batch_size=32, num_workers=4)

    def to_clf(x):
        return F.interpolate(x, (a.resize, a.resize), mode="bilinear", align_corners=False)

    cfg = {"stage": "full_joint", "loss_mode": a.loss_mode, "lambda_anchor": a.lambda_anchor,
           "lambda_task": a.lambda_task, "lambda_sup": a.lambda_sup,
           "lr_gen": a.lr_gen, "lr": a.lr, "pool": a.pool, "seed": a.seed, "epochs": a.epochs,
           "batch_size": a.batch_size, "tail": a.tail,
           "unsb_losses": {"lambda_GAN": opt.lambda_GAN, "lambda_NCE": opt.lambda_NCE,
                           "lambda_SB": opt.lambda_SB, "num_timesteps": opt.num_timesteps},
           "selection_rule": f"fixed budget, predictions averaged over the last {a.tail} epochs",
           "gen_ckpt_sha256": sha256_file(f"{a.ckpt_dir}/latest_net_G.pth"),
           "n_train": len(tr_df), "n_target": int(tgt.case_id.nunique())}

    # fixed probe batch + target statistics for the style-closure monitor.
    # 256 images: at 32 the metric's probe-selection noise is 15.2 pp, at 256 it is
    # 2.0 pp -- the difference between a readable trend and an unreadable one.
    _sd = SrcDataset(tr_df)
    probe = torch.stack([_sd[i][0] for i in range(min(a.probe_n, len(tr_df)))]).to(dev)
    _tg_imgs = TgtImgDataset(tgt_paths)
    _tm = np.array([((_tg_imgs[i] + 1) / 2).mean().item() for i in range(len(_tg_imgs))])
    tgt_stats = (float(_tm.mean()), float(_tm.std(ddof=1)))
    print(f"target brightness {tgt_stats[0]:.4f} +- {tgt_stats[1]:.4f}  "
          f"(pretrained generator closes 36.4% of the gap)")

    hist, sealed, tail_probs = [], [], []
    tgt_iter = iter(tgt_dl)
    for ep in range(1, a.epochs + 1):
        clf.train()
        n = ce_t = sc_t = gan_t = sb_t = nce_t = d_t = drift_t = ref_t = 0
        for x, flip, ang, y, _ in tr:
            x, y = x.to(dev), y.to(dev)
            flip, ang = flip.to(dev).float(), ang.to(dev).float()
            try:
                rb = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_dl); rb = next(tgt_iter)
            rb = rb.to(dev)

            fake = sb.forward(x, rb)                       # bridge walk + fake_B (identical in both modes)
            if a.loss_mode == "unsb":
                lD, lE = sb.step_D_E()                     # UNSB's D and E updates
            else:
                lD = lE = 0.0                              # anchor mode trains no discriminator

            sb.opt_G.zero_grad(); sb.opt_F.zero_grad(); opt_clf.zero_grad()
            if a.loss_mode == "unsb":
                g_own, gan, sbl, nce = sb.g_loss()         # UNSB's own generator objective
            else:
                with torch.no_grad():                      # the frozen generator's output, same input
                    ref = G_ref(sb.real_A_noisy, sb.time_idx,
                                torch.randn(x.size(0), 4 * opt.ngf, device=dev))
                anchor = (fake - ref).abs().mean()
                g_own = a.lambda_anchor * anchor
                gan, sbl, nce = float(anchor), 0.0, 0.0    # logged in the `gan` column for comparability
            # Route the task gradient: the forward value is unchanged
            # (lam*f + (1-lam)*f.detach() == f), but only `lam` of the gradient
            # reaches G. The CLASSIFIER always receives the full gradient, so its
            # training is identical in both arms and the single isolated variable is
            # "does the task gradient reach the generator". Weighting the whole task
            # term by lambda_task instead would silently stop training the classifier
            # at lambda_task=0 and make the control arm a random model.
            fake_in = a.lambda_task * fake + (1.0 - a.lambda_task) * fake.detach()
            xa, fa = apply_geom(x, flip, ang), apply_geom(fake_in, flip, ang)
            o1, z1 = clf(to_clf(xa))
            o2, z2 = clf(to_clf(fa))
            ce = F.cross_entropy(o1, y) + a.lambda_a * F.cross_entropy(o2, y)
            task = ce
            sc = torch.zeros((), device=dev)
            if a.lambda_sup > 0:
                sc = sup_con_loss(torch.cat([z1, z2]), torch.cat([y, y]), a.temp)
                task = task + a.lambda_sup * sc
            (g_own + task).backward()      # task weight is 1 for the classifier in BOTH arms
            sb.opt_G.step(); sb.opt_F.step(); opt_clf.step()

            with torch.no_grad():
                ref = sb.G_ref_out = None
                r = G_ref(sb.real_A_noisy, sb.time_idx,
                          torch.randn(x.size(0), 4 * opt.ngf, device=dev))
                ref_t += float((fake.detach() - r).abs().mean()) * len(y)
            b = len(y); n += b
            ce_t += float(ce) * b; sc_t += float(sc) * b; gan_t += gan * b
            sb_t += sbl * b; nce_t += nce * b; d_t += lD * b
            drift_t += float((fake.detach() - x).abs().mean()) * b

        clo = style_closure(sb.G, sb.times, opt, probe, tgt_stats, dev)
        pvr, yvr, cvr = predict(clf, va_raw, dev)
        pvu, yvu, cvu = predict(clf, va_u1, dev)
        src_val = 0.5 * case_auc(pvr, yvr, cvr) + 0.5 * case_auc(pvu, yvu, cvu)  # legal selector
        row = {"epoch": ep, "ce": ce_t / n, "supcon": sc_t / n, "gan": gan_t / n,
               "sb": sb_t / n, "nce": nce_t / n, "loss_D": d_t / n,
               "translate_dist": drift_t / n, "gen_drift": ref_t / n,
               "style_closure_pct": clo, "src_val_auc": src_val}
        if not a.calibrate:
            pt, yt, ct = predict(clf, tg, dev)
            row["target_auc_SEALED"] = case_auc(pt, yt, ct)
            sealed += [{"epoch": ep, "case_id": ci, "label": int(lb), "prob": float(p)}
                       for ci, lb, p in zip(ct, yt, pt)]
            if ep > a.epochs - a.tail:
                tail_probs.append(pd.DataFrame({"case_id": ct, "label": yt, "p": pt}))
        hist.append(row)
        print(f"ep {ep:3d} ce {row['ce']:.4f} gan {row['gan']:.4f} sb {row['sb']:.4f} "
              f"nce {row['nce']:.4f} D {row['loss_D']:.4f} | drift {row['gen_drift']:.4f} "
              f"({row['gen_drift']/0.0035:4.1f}x) |G(x)-x| {row['translate_dist']:.4f} "
              f"| 风格关闭 {clo:5.1f}%", flush=True)
        if a.calibrate and ep >= a.cal_epochs:
            print(f"\n[calibrate] lambda_task={a.lambda_task:g} lr_gen={a.lr_gen:g} -> "
                  f"drift {row['gen_drift']:.4f} ({row['gen_drift']/0.0035:.1f}x noise), "
                  f"style closure {clo:.1f}% (pretrained 36.4%)")
            return

    if tail_probs:
        avg = pd.concat(tail_probs).groupby("case_id", sort=False).agg(
            label=("label", "first"), p=("p", "mean")).reset_index()
        cfg["final_target_auc"] = case_auc(avg.p.values, avg.label.values, avg.case_id.tolist())
        avg.to_csv(run / "final_percase.csv", index=False)
        seal = Path(a.sealed_dir) / f"{tag}_target_percase.csv"
        pd.DataFrame(sealed).to_csv(seal, index=False)
        cfg["sealed_sha256"] = sha256_file(seal)
    h = pd.DataFrame(hist)
    cfg["gen_drift_end"] = float(h.gen_drift.tail(5).mean())
    cfg["translate_dist_end"] = float(h.translate_dist.tail(5).mean())
    cfg["collapse_flag"] = bool(cfg["translate_dist_end"] < 0.5 * float(h.translate_dist.iloc[0]))
    h.to_csv(run / "history.csv", index=False)
    with open(run / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{tag}] target AUC {cfg.get('final_target_auc', float('nan')):.4f}  "
          f"G drift {cfg['gen_drift_end']:.4f}  |G(x)-x| {cfg['translate_dist_end']:.4f}")


if __name__ == "__main__":
    main()
