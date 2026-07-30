#!/usr/bin/env python
"""Build the UNSB input set for the CYCLE SEMANTIC AUDIT (reviewer priority 2).

Round trip:  x_s  --G_s->t(step k)-->  x_k  --G_t->s-->  x_k_cycled

The audit asks, using SOURCE data only, whether a round trip preserves class
semantics and structure, and picks the forward step k* on the SOURCE VALIDATION
set. That is legitimate under strict UDA: no target image and no target label is
involved in the choice.

testA = the already-generated G_s->t outputs for BUSI valid (65) and BUSI test
(130), all five forward steps -> 5 x 195 = 975 images, keyed
`<split>__k<k>__<idx>` so every cycled image joins back to its source label.

  * valid  -> used to CHOOSE k*
  * test   -> used to REPORT k*'s behaviour on data the choice did not touch

testB is a placeholder (the unaligned loader needs a B side; unused for AtoB).
It is drawn from BUSI train, i.e. the b2u_SB output domain -- no BrEaST image of
any split is referenced here, so the locked 51 stay untouched.
"""
from __future__ import annotations

import os

import pandas as pd

UNSB = "/root/autodl-tmp/UNSB"
C = "/root/autodl-tmp/breast/cache"
FWD_TRVAL = f"{UNSB}/results_u2b_rev/u2b_rev_SB/test_latest/images"       # BUSI train+valid
FWD_TEST = f"{UNSB}/results_u2b_srctest/u2b_rev_SB/test_latest/images"    # BUSI test
STEPS = [1, 2, 3, 4, 5]


def _mk(p):
    os.makedirs(p, exist_ok=True)
    return p


def _link(src, dst):
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    root = _mk(f"{UNSB}/datasets/cycle_audit")
    dtA, dtB = _mk(f"{root}/testA"), _mk(f"{root}/testB")
    _mk(f"{root}/trainA"); _mk(f"{root}/trainB")

    val = pd.read_csv(f"{C}/u2b_rev_srcapply_manifest.csv")
    val = val[val.split == "src_valid"].reset_index(drop=True)
    test = pd.read_csv(f"{C}/u2b_rev_srctest_manifest.csv").reset_index(drop=True)

    rows = []
    for tag, man, fwd in [("valid", val, FWD_TRVAL), ("test", test, FWD_TEST)]:
        for _, r in man.iterrows():
            for k in STEPS:
                key = f"{tag}__k{k}__{r['key']}"
                _link(f"{fwd}/fake_{k}/{r['key']}.png", f"{dtA}/{key}.png")
                rows.append({"key": key, "audit_split": tag, "step": k,
                             "src_key": r["key"], "orig_path": r["image_path"],
                             "fwd_path": f"{fwd}/fake_{k}/{r['key']}.png",
                             "label": int(r["label"])})

    # placeholder B side: BUSI train (b2u_SB's own output domain). No BrEaST here.
    for i, r in pd.read_csv(f"{C}/busi_train.csv").head(30).iterrows():
        _link(r["image_path"], f"{dtB}/b_{i:03d}.png")

    out = f"{C}/cycle_audit_manifest.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"cycle_audit: testA={len(os.listdir(dtA))} testB={len(os.listdir(dtB))} -> {out}")
    print(df.groupby(["audit_split", "step"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
