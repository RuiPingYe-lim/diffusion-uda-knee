#!/usr/bin/env python
"""STEP 3 -- build the 7 matched image sets for the augmentation route.

Seven base conditions, ONE manifest (13 training arms reuse these files and differ
only by a loss switch, so no files are duplicated):

    raw   : source image resized to 256x256 (the common base for EVERY arm)
    P1,P5 : photometric control -- per-image affine onto fake_k's (mean,std)
    F1,F5 : FDA control -- low-frequency amplitude blended toward BrEaST train
    U1,U5 : UNSB G_s->t step k (referenced in place, not copied)

Arms:  A0 = raw+raw | P_k = raw+P_k | F_k = raw+F_k | U_k = raw+U_k, and the same
six P/F/U arms again with the consistency loss enabled.

RESAMPLING: every arm must reach the classifier through the same interpolation.
UNSB wrote 256x256 outputs, so `raw` is a BICUBIC resize to 256x256 -- verified to
reproduce UNSB's own saved `real/` images to 0.0017 mean abs error (~0.4 grey
levels, i.e. rounding).

STYLE MATCHING: P_k matches fake_k's first two moments by construction; F_k's
blend weight alpha is swept to match fake_k's brightness-gap closure. beta is NOT
usable as the knob -- even a 1-pixel window contains the DC term, so any beta
transfers ~90-99% of the gap regardless of size.

FDA amplitudes come from BrEaST TRAIN (175, unlabelled). The locked 51-case
BrEaST test is never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

C = "/root/autodl-tmp/breast/cache"
FWD = "/root/autodl-tmp/UNSB/results_u2b_rev/u2b_rev_SB/test_latest/images"
OUT = "/root/autodl-tmp/breast/da_route"
SIZE = 256
BETA = 0.05
STEPS = [1, 5]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def raw256(p):
    return np.asarray(Image.open(p).convert("L").resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float64) / 255.0


def load(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def save(a, p):
    Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8)).save(p)


def pooled_gap(a, b):
    sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (b.mean() - a.mean()) / sd


def fda_alpha(src, ref, alpha, beta=BETA):
    Fs, Fr = np.fft.fft2(src), np.fft.fft2(ref)
    ph = np.angle(Fs)
    As, Ar = np.fft.fftshift(np.abs(Fs)), np.fft.fftshift(np.abs(Fr))
    h, w = src.shape
    b = max(1, int(np.floor(min(h, w) * beta / 2)))
    cy, cx = h // 2, w // 2
    As[cy-b:cy+b, cx-b:cx+b] = (1-alpha)*As[cy-b:cy+b, cx-b:cx+b] + alpha*Ar[cy-b:cy+b, cx-b:cx+b]
    return np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(As) * np.exp(1j*ph))), 0, 1)


def main():
    ap = argparse.ArgumentParser("build matched DA-route datasets")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tol", type=float, default=2.0, help="max closure mismatch, percentage points")
    a = ap.parse_args()

    man = pd.read_csv(f"{C}/u2b_rev_srcapply_manifest.csv")
    tgt = pd.read_csv(f"{C}/breast_train.csv")
    dirs = {n: os.path.join(OUT, "images", n) for n in ["raw", "P1", "P5", "F1", "F5"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"source {len(man)} images ({dict(man.split.value_counts())}) | "
          f"FDA amplitude donors: BrEaST train {len(tgt)} (locked 51 untouched)")

    srcs = [raw256(p) for p in man.image_path]
    tgts = [raw256(p) for p in tgt.image_path]
    tm = np.array([x.mean() for x in tgts])
    g0 = pooled_gap(np.array([x.mean() for x in srcs]), tm)
    print(f"original brightness gap: {g0:+.4f} SD")

    cfg = {"seed": a.seed, "size": SIZE, "fda_beta": BETA, "resize": "BICUBIC",
           "gap_original_sd": float(g0), "steps": STEPS,
           "src_manifest_sha256": sha256_file(f"{C}/u2b_rev_srcapply_manifest.csv"),
           "fda_donor_manifest_sha256": sha256_file(f"{C}/breast_train.csv")}

    # ---- write raw ----
    for x, k in zip(srcs, man.key):
        save(x, f"{dirs['raw']}/{k}.png")

    rng = np.random.RandomState(a.seed)
    for step in STEPS:
        fk = [load(f"{FWD}/fake_{step}/{k}.png") for k in man.key]
        gk = pooled_gap(np.array([x.mean() for x in fk]), tm)
        target = 100 * (1 - abs(gk) / abs(g0))
        print(f"\n--- step {step}: U{step} closes {target:.2f}% of the gap ---")

        # P_k: per-image affine onto THE SAME IMAGE's translated moments. Pairing by
        # identity (not a random draw) makes P_k exactly "the global intensity change
        # UNSB applied to this image, reproduced as a pure affine map"; a random draw
        # leaves a several-point mismatch once the [0,1] clip bites.
        fm, fs = np.array([x.mean() for x in fk]), np.array([x.std() for x in fk])
        P = [np.clip((x - x.mean())/(x.std()+1e-9)*fs[t] + fm[t], 0, 1)
             for t, x in enumerate(srcs)]
        gp = pooled_gap(np.array([x.mean() for x in P]), tm)
        cp = 100 * (1 - abs(gp) / abs(g0))
        for x, k in zip(P, man.key):
            save(x, f"{dirs[f'P{step}']}/{k}.png")
        print(f"  P{step}: closure {cp:.2f}%  (mismatch {cp-target:+.2f} pp)")

        # F_k: sweep alpha to match the closure
        don = rng.randint(0, len(tgts), size=len(srcs))
        lo, hi, best = 0.0, 1.0, None
        for _ in range(12):                       # bisection on a monotone knob
            mid = (lo + hi) / 2
            out = [fda_alpha(x, tgts[t], mid) for x, t in zip(srcs, don)]
            cl = 100 * (1 - abs(pooled_gap(np.array([x.mean() for x in out]), tm)) / abs(g0))
            if best is None or abs(cl - target) < abs(best[1] - target):
                best = (mid, cl, out)
            if cl < target:
                lo = mid
            else:
                hi = mid
        alpha, cf, out = best
        for x, k in zip(out, man.key):
            save(x, f"{dirs[f'F{step}']}/{k}.png")
        print(f"  F{step}: alpha {alpha:.4f} -> closure {cf:.2f}%  (mismatch {cf-target:+.2f} pp)")
        cfg[f"U{step}_closure_pct"] = float(target)
        cfg[f"P{step}_closure_pct"] = float(cp)
        cfg[f"F{step}_closure_pct"] = float(cf)
        cfg[f"F{step}_alpha"] = float(alpha)
        if abs(cp - target) > a.tol or abs(cf - target) > a.tol:
            raise SystemExit(f"FATAL: style matching outside tolerance {a.tol} pp for step {step}; "
                             "the controls would not be information-matched")

    # ---- manifest ----
    rows = []
    for _, r in man.iterrows():
        row = {"key": r["key"], "split": r["split"], "case_id": r["key"], "label": int(r["label"]),
               "src_path": r["image_path"], "raw": f"{dirs['raw']}/{r['key']}.png"}
        for s in STEPS:
            row[f"P{s}"] = f"{dirs[f'P{s}']}/{r['key']}.png"
            row[f"F{s}"] = f"{dirs[f'F{s}']}/{r['key']}.png"
            row[f"U{s}"] = f"{FWD}/fake_{s}/{r['key']}.png"
        rows.append(row)
    df = pd.DataFrame(rows)

    # ---- hard checks (reviewer step 3.2) ----
    cols = ["raw"] + [f"{p}{s}" for s in STEPS for p in ("P", "F", "U")]
    tr, va = set(df[df.split == "src_train"].case_id), set(df[df.split == "src_valid"].case_id)
    assert not (tr & va), "train/valid case overlap"
    assert df.case_id.nunique() == len(df), "case_id not unique"
    for c in cols:
        assert df[c].map(os.path.isfile).all(), f"missing files in {c}"
        assert df[c].nunique() == len(df), f"{c} has duplicate paths"
    probe = df.sample(min(25, len(df)), random_state=0)
    for _, r in probe.iterrows():
        for c in cols:
            im = Image.open(r[c])
            assert im.size == (SIZE, SIZE), f"{c} size {im.size} != {(SIZE, SIZE)}"
            arr = np.asarray(im.convert("L"))
            assert arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255, f"{c} grey range"
    print(f"\nchecks passed: {len(tr)} train / {len(va)} valid cases, disjoint; "
          f"{len(cols)} conditions x {len(df)} images all present, 256x256 uint8")

    os.makedirs(OUT, exist_ok=True)
    out_csv = f"{OUT}/da_manifest.csv"
    df.to_csv(out_csv, index=False)
    cfg["manifest_sha256"] = sha256_file(out_csv)
    cfg["n_train"], cfg["n_valid"] = len(tr), len(va)
    with open(f"{OUT}/da_build_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"-> {out_csv}\n-> {OUT}/da_build_config.json")
    print(json.dumps({k: v for k, v in cfg.items() if "closure" in k or "alpha" in k}, indent=2))


if __name__ == "__main__":
    main()
