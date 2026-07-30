#!/usr/bin/env python
"""STAGE 2 -- joint training: the UNSB generator is UNFROZEN and co-adapts.

Stage 1 fixed the translator. It had been optimised, separately and much earlier,
for "look like the target domain" as judged by a discriminator -- nobody ever told
it "produce images that help tell benign from malignant". Stage 2 opens that
channel: the classifier's loss back-propagates into the generator, so the
translator learns what kind of translation actually helps the task.

    x_s --G(theta_G)--> x_t' --classifier--> loss --grad--> BOTH the classifier AND G

WHY THIS MIGHT MATTER HERE. The stage-1 diagnostic measured the features of an
image and its translation at cosine 0.906 with the contrastive term switched off,
and lambda barely moved it: the two views are already nearly identical, so the
contrastive term has almost no distance to close. That "too similar" is the frozen
generator's own choice. Unfreezing lets it find a translation that changes
appearance more while keeping the lesion -- something a fixed translator cannot do.

THE FAILURE MODE, AND THE GUARD. Given freedom, the generator's easiest way to
lower a classification loss is to stop translating at all: if G(x) ~= x the
classifier sees familiar images and the loss drops, but style invariance becomes
vacuous. Two defences, both always on:
  * ANCHOR: an L1 penalty pulling G's output toward the FROZEN pretrained
    generator's output for the same input, so G stays a translator.
  * MONITOR: mean |G(x) - x| is logged every epoch. If it collapses toward the
    frozen generator's baseline distance, the run is flagged -- a number that
    improved by ceasing to translate is not evidence for translation.

VALIDITY. The headline contrast is Stage2 - Stage1 at the SAME lambda and the SAME
seed. That paired comparison is internally valid whatever lambda was chosen, since
both sides share it. The absolute AUC is NOT quotable as a UDA result when lambda
came from target peeking.

The LOCKED 51-case BrEaST test set is not referenced anywhere in this file.
"""
from __future__ import annotations

import argparse
import copy
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
from train_stage1 import (AttnPoolNet, EvalDataset, case_auc, predict,  # noqa: E402
                          representation_metrics, sup_con_loss, supcon_floor)

UNSB = "/root/autodl-tmp/UNSB"
SIZE = 256


def sha256_file(p):
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- generator
def _parse_train_opt(path):
    """Read UNSB's saved train_opt.txt so the generator is rebuilt with the EXACT
    options it was trained under -- hand-written defaults silently give a different
    architecture (n_mlp, embedding sizes, antialias flags)."""
    opt = {}
    for line in open(path):
        if ":" not in line or line.strip().startswith("-"):
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.split("[default")[0].strip()
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
    """Rebuild UNSB's conditional generator and load the trained weights."""
    from models import networks

    opt_txt = os.path.join(os.path.dirname(ckpt), "train_opt.txt")
    if not os.path.isfile(opt_txt):
        raise FileNotFoundError(f"need {opt_txt} to rebuild the generator faithfully")
    d = _parse_train_opt(opt_txt)
    d["gpu_ids"] = []

    class O:
        pass
    o = O()
    o.__dict__.update(d)
    G = networks.define_G(d.get("input_nc", 3), d.get("output_nc", 3), d.get("ngf", ngf),
                          d.get("netG", "resnet_9blocks_cond"), d.get("normG", "instance"),
                          not d.get("no_dropout", True), d.get("init_type", "xavier"),
                          d.get("init_gain", 0.02), d.get("no_antialias", False),
                          d.get("no_antialias_up", False), [], o)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = G.load_state_dict(sd, strict=False)
    if unexpected or [k for k in missing if "num_batches" not in k]:
        raise RuntimeError(f"generator load mismatch: missing={list(missing)[:4]} "
                           f"unexpected={list(unexpected)[:4]}")
    return G.to(device)


def sb_times(T_steps, device):
    """UNSB's Schrodinger-bridge time grid (copied from models/sb_model.py::forward)."""
    incs = np.array([0] + [1 / (i + 1) for i in range(T_steps - 1)])
    t = np.cumsum(incs)
    t = t / t[-1]
    t = 0.5 * t[-1] + 0.5 * t
    t = np.concatenate([np.zeros(1), t])
    return torch.tensor(t).float().to(device)


def sb_translate(G, x, steps, times, tau=0.01, ngf=64, grad=True, generator=None):
    """Run the bridge for `steps` steps. steps=1 reproduces the fake_1 used by U1."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        Xt = x
        prev = None
        for t in range(steps):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).reshape(-1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                noise = torch.randn(Xt.shape, device=Xt.device, generator=generator)
                Xt = (1 - inter) * Xt + inter * prev + (scale * tau).sqrt() * noise
            ti = (t * torch.ones(x.shape[0], device=x.device)).long()
            z = torch.randn((x.shape[0], 4 * ngf), device=x.device, generator=generator)
            prev = G(Xt, ti, z)
        return prev


# ---------------------------------------------------------------- data
class RawDataset(Dataset):
    """Returns the UN-AUGMENTED raw image plus the geometric draw to apply later.

    ORDER MATTERS. Stage 1 translated the un-rotated image offline and applied the
    geometric augmentation afterwards. Augmenting first and then translating feeds
    the generator rotated images, which it never saw during its own training --
    measured to change the translation by 0.090, i.e. 26x the generator's own
    sampling noise (0.0035) and comparable to the translation magnitude itself
    (0.118). So the augmentation is deferred and applied on GPU AFTER translation,
    identically to both views.
    """

    def __init__(self, df, train=False, aug_seed=0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.rng_aug = random.Random(aug_seed)
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
        flip = 1.0 if (self.train and self.rng_aug.random() < 0.5) else 0.0
        ang = self.rng_aug.uniform(-10, 10) if self.train else 0.0
        return self.to_t(im), float(flip), float(ang), int(r["label"]), str(r["case_id"])


def apply_geom(x, flip, ang):
    """Apply ONE geometric draw to a batch. Called with the same (flip, ang) for both
    views so they stay pixel-aligned, exactly as stage 1's shared-draw dataset did."""
    if flip.any():
        x = torch.where(flip.view(-1, 1, 1, 1) > 0.5, torch.flip(x, dims=[3]), x)
    th = ang * np.pi / 180.0
    cos, sin = torch.cos(th), torch.sin(th)
    zero = torch.zeros_like(cos)
    mat = torch.stack([torch.stack([cos, -sin, zero], 1),
                       torch.stack([sin, cos, zero], 1)], 1)          # [B,2,3]
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def main():
    ap = argparse.ArgumentParser("stage 2: joint classifier + generator training")
    ap.add_argument("--manifest", default="/root/autodl-tmp/breast/da_route/da_manifest.csv")
    ap.add_argument("--target_csv", default="/root/autodl-tmp/breast/cache/fusion_eval_breast_diag.csv")
    ap.add_argument("--gen_ckpt", default=f"{UNSB}/checkpoints/u2b_rev_SB/latest_net_G.pth")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/breast/stage2/runs")
    ap.add_argument("--sealed_dir", default="/root/autodl-tmp/breast/stage2/sealed")
    ap.add_argument("--steps", type=int, default=1, help="bridge steps; 1 == the U1 arm")
    ap.add_argument("--freeze_gen", action="store_true", help="stage-1 equivalent, for a same-code control")
    ap.add_argument("--lr_gen", type=float, default=1e-6, help="deliberately small: G only nudges")
    ap.add_argument("--lambda_anchor", type=float, default=10.0,
                    help="L1 pull toward the FROZEN generator's output; the anti-collapse guard")
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--lambda_sup", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--pool", choices=["attn", "gap"], default="attn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true",
                    help="parameter calibration only: report how far G's output actually moves, "
                         "against the frozen generator's own sampling noise (0.0035). No AUC is "
                         "read -- if G does not move, nothing about joint training is being tested.")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "frozen" if a.freeze_gen else "joint"
    tag = f"{mode}_k{a.steps}_sup{a.lambda_sup:g}_s{a.seed}"
    run = Path(a.out_dir) / tag
    run.mkdir(parents=True, exist_ok=True)
    Path(a.sealed_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.manifest)
    tr_df, va_df = df[df.split == "src_train"], df[df.split == "src_valid"]

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    clf = AttnPoolNet(pool=a.pool).to(dev)
    G = build_unsb_generator(a.gen_ckpt, dev)
    G_frozen = copy.deepcopy(G).eval()                 # the anchor; never updated
    for p in G_frozen.parameters():
        p.requires_grad_(False)
    if a.freeze_gen:
        G.eval()
        for p in G.parameters():
            p.requires_grad_(False)
    times = sb_times(5, dev)

    tr = DataLoader(RawDataset(tr_df, train=True, aug_seed=a.seed + 90000),
                    batch_size=a.batch_size, shuffle=True, num_workers=4,
                    generator=torch.Generator().manual_seed(a.seed + 50000))
    va_raw = DataLoader(EvalDataset(va_df["raw"], va_df.label, va_df.case_id, a.resize),
                        batch_size=32, num_workers=4)
    tgt = pd.read_csv(a.target_csv)
    tg = DataLoader(EvalDataset(tgt["before_png"], tgt.label, tgt.case_id, a.resize),
                    batch_size=32, num_workers=4)

    params = [{"params": clf.parameters(), "lr": a.lr}]
    if not a.freeze_gen:
        params.append({"params": G.parameters(), "lr": a.lr_gen})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)

    def to_clf(x):
        return F.interpolate(x, size=(a.resize, a.resize), mode="bilinear", align_corners=False)

    cfg = {"stage": 2, "mode": mode, "steps": a.steps, "pool": a.pool, "lambda_a": a.lambda_a,
           "lambda_sup": a.lambda_sup, "lambda_anchor": a.lambda_anchor, "lr": a.lr,
           "lr_gen": a.lr_gen, "temp": a.temp, "seed": a.seed, "epochs": a.epochs,
           "tail": a.tail, "batch_size": a.batch_size,
           "selection_rule": f"fixed budget, predictions averaged over the last {a.tail} epochs",
           "gen_ckpt_sha256": sha256_file(a.gen_ckpt), "manifest_sha256": sha256_file(a.manifest),
           "n_train": len(tr_df), "n_target": int(tgt.case_id.nunique())}

    hist, sealed, tail_probs = [], [], []
    for ep in range(1, a.epochs + 1):
        clf.train()
        if not a.freeze_gen:
            G.train()
        n = tot = ce_t = sc_t = anc_t = drift_t = 0
        for x, flip, ang, y, _ in tr:
            x, y = x.to(dev), y.to(dev)
            flip, ang = flip.to(dev).float(), ang.to(dev).float()
            opt.zero_grad()
            # translate the UN-augmented image (the generator stays in-distribution) ...
            xt = sb_translate(G, x, a.steps, times, grad=not a.freeze_gen)
            with torch.no_grad():
                xt_ref = sb_translate(G_frozen, x, a.steps, times, grad=False)
            # ... then apply ONE shared geometric draw to both views
            xa, xta = apply_geom(x, flip, ang), apply_geom(xt, flip, ang)
            o1, z1 = clf(to_clf(xa))
            o2, z2 = clf(to_clf(xta))
            ce = F.cross_entropy(o1, y) + a.lambda_a * F.cross_entropy(o2, y)
            loss = ce
            sc = torch.zeros((), device=dev)
            if a.lambda_sup > 0:
                sc = sup_con_loss(torch.cat([z1, z2]), torch.cat([y, y]), a.temp)
                loss = loss + a.lambda_sup * sc
            anc = torch.zeros((), device=dev)
            if not a.freeze_gen and a.lambda_anchor > 0:
                anc = (xt - xt_ref).abs().mean()        # stay a translator
                loss = loss + a.lambda_anchor * anc
            loss.backward(); opt.step()
            b = len(y); n += b
            tot += float(loss) * b; ce_t += float(ce) * b; sc_t += float(sc) * b
            anc_t += float(anc) * b
            drift_t += float((xt - x).abs().mean()) * b   # collapse monitor

        if a.calibrate:
            print(f"epoch {ep:3d}  |G - G_frozen| {anc_t/n:.5f}  (noise floor 0.0035, "
                  f"ratio {anc_t/n/0.0035:5.1f}x)   |G(x)-x| {drift_t/n:.4f}", flush=True)
            if ep >= 5:
                moved = anc_t / n
                print(f"\n[calibrate] lr_gen={a.lr_gen:g} lambda_anchor={a.lambda_anchor:g}  "
                      f"-> G moved {moved:.5f} = {moved/0.0035:.1f}x the sampling noise  "
                      f"{'MOVES' if moved > 3 * 0.0035 else 'DOES NOT MOVE'}")
                return
            continue

        pr, yr, cr = predict(clf, va_raw, dev)
        pt, yt, ct = predict(clf, tg, dev)                # sealed
        row = {"epoch": ep, "loss": tot / n, "ce": ce_t / n, "supcon": sc_t / n,
               "anchor": anc_t / n, "translate_dist": drift_t / n,
               "val_raw_auc": case_auc(pr, yr, cr), "target_auc_SEALED": case_auc(pt, yt, ct)}
        hist.append(row)
        sealed += [{"epoch": ep, "case_id": ci, "label": int(lb), "prob": float(p)}
                   for ci, lb, p in zip(ct, yt, pt)]
        if ep > a.epochs - a.tail:
            tail_probs.append(pd.DataFrame({"case_id": ct, "label": yt, "p": pt}))
        print(f"epoch {ep:3d} loss {row['loss']:.4f} (ce {row['ce']:.4f} supcon {row['supcon']:.4f} "
              f"anchor {row['anchor']:.4f})  |G(x)-x| {row['translate_dist']:.4f}  "
              f"val_raw {row['val_raw_auc']:.4f}", flush=True)

    avg = pd.concat(tail_probs).groupby("case_id", sort=False).agg(
        label=("label", "first"), p=("p", "mean")).reset_index()
    cfg["final_target_auc"] = case_auc(avg.p.values, avg.label.values, avg.case_id.tolist())

    h = pd.DataFrame(hist)
    d0, d1 = float(h.translate_dist.iloc[0]), float(h.translate_dist.tail(5).mean())
    cfg["translate_dist_start"], cfg["translate_dist_end"] = d0, d1
    cfg["collapse_flag"] = bool(d1 < 0.5 * d0)
    h.to_csv(run / "history.csv", index=False)
    seal = Path(a.sealed_dir) / f"{tag}_target_percase.csv"
    pd.DataFrame(sealed).to_csv(seal, index=False)
    avg.to_csv(run / "final_percase.csv", index=False)
    with open(run / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{tag}] target case-AUC = {cfg['final_target_auc']:.4f}   "
          f"|G(x)-x| {d0:.4f} -> {d1:.4f}"
          f"{'   *** COLLAPSE: stopped translating, result not usable ***' if cfg['collapse_flag'] else ''}")


if __name__ == "__main__":
    main()
