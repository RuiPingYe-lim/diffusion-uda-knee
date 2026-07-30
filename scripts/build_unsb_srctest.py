#!/usr/bin/env python
"""Build a UNSB test dataset for the SOURCE TEST split (BUSI test, 130 images).

Gate A-2 of the augmentation protocol asks whether malignancy content survives
G_s->t. That must be measured on images the linear probe was NOT fitted on, and
the existing u2b_rev manifest only covers BUSI train+valid (452+65). The frozen
source classifier fits the 452 to 0.93-0.97, so evaluating there would be
meaningless.

testA = BUSI test (130), named src_test__NNNNN so outputs join back to labels.
testB is a placeholder: the `unaligned` loader requires a B side but the B
images are never used when translating AtoB.

The LOCKED BrEaST test (51) is not referenced here. The placeholder B side is
drawn from BrEaST TRAIN, the same split UNSB already trained on.
"""
from __future__ import annotations

import os

import pandas as pd

UNSB = "/root/autodl-tmp/UNSB"
C = "/root/autodl-tmp/breast/cache"


def _mk(p):
    os.makedirs(p, exist_ok=True)
    return p


def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    root = _mk(f"{UNSB}/datasets/u2b_srctest")
    dtA, dtB = _mk(f"{root}/testA"), _mk(f"{root}/testB")
    # the SB model builds train dirs too; give it empty-but-present ones
    _mk(f"{root}/trainA"); _mk(f"{root}/trainB")

    rows = []
    for i, r in pd.read_csv(f"{C}/busi_test.csv").iterrows():
        key = f"src_test__{i:05d}"
        _link(r["image_path"], f"{dtA}/{key}.png")
        rows.append({"key": key, "split": "src_test",
                     "image_path": r["image_path"], "label": int(r["label"])})

    # placeholder B side (never used for AtoB); BrEaST TRAIN only -- never the locked test
    for i, r in pd.read_csv(f"{C}/breast_train.csv").head(30).iterrows():
        _link(r["image_path"], f"{dtB}/b_{i:03d}.png")

    out = f"{C}/u2b_rev_srctest_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"u2b_srctest: testA={len(os.listdir(dtA))} testB={len(os.listdir(dtB))} -> {out}")
    print(f"labels: {pd.DataFrame(rows).label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
