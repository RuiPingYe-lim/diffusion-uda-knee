#!/usr/bin/env python
"""Build 5-fold CV splits over the TARGET diagnostic set for the SUPERVISED-ORACLE probe.

The oracle asks: if the adapter is ALLOWED to see target labels, can the extra
translation views buy anything over the frozen source margin? If not even an
oracle can, the translation views carry no information the frozen classifier is
missing -- a boundary condition, not a tuning failure.

Protocol (leak-free):
  * the 201 BrEaST train+valid cases are the diagnostic pool; the LOCKED 51-case
    test set is NOT touched here and must stay closed;
  * stratified 5-fold by case label (this CSV has exactly one row per case, so
    row-level == case-level);
  * inside each fold the 4 training folds are split 80/20 into inner train/val;
    epoch selection uses ONLY the inner val, never the held-out fold;
  * predictions on the held-out folds are concatenated into one out-of-fold set
    covering all 201 cases -- directly comparable to the `direct` reference,
    which is evaluated on the same 201.

Both arms (`repeat_before`, `true_fakes`) read the SAME CSVs; repeat_before
rebuilds its `others` in code, so no separate files are needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_folds(labels, n_folds, rng):
    """Return an array of fold ids, balanced within each label."""
    fold = np.empty(len(labels), dtype=int)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def main():
    ap = argparse.ArgumentParser("build oracle CV folds on the target diagnostic set")
    ap.add_argument("--src_csv", default="/root/autodl-tmp/breast/cache/fusion_eval_breast_diag.csv")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/breast/cache/oracle_folds")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--inner_val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    df = pd.read_csv(a.src_csv)
    if df["case_id"].duplicated().any():
        raise ValueError("src_csv has >1 row per case; fold assignment must be done at case level")
    y = df["label"].to_numpy()
    rng = np.random.RandomState(a.seed)
    fold = stratified_folds(y, a.n_folds, rng)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"pool: {len(df)} cases  labels={dict(pd.Series(y).value_counts())}")
    for k in range(a.n_folds):
        te = df[fold == k].reset_index(drop=True)
        rest = df[fold != k].reset_index(drop=True)
        # inner train/val, stratified, disjoint from the held-out fold
        vmask = np.zeros(len(rest), dtype=bool)
        rng_k = np.random.RandomState(a.seed + 1000 + k)
        for lab in np.unique(rest["label"].to_numpy()):
            idx = np.where(rest["label"].to_numpy() == lab)[0]
            rng_k.shuffle(idx)
            n_v = max(1, int(round(a.inner_val_frac * len(idx))))
            vmask[idx[:n_v]] = True
        tr, va = rest[~vmask].reset_index(drop=True), rest[vmask].reset_index(drop=True)

        assert not (set(te.case_id) & set(tr.case_id)), "test/train case leak"
        assert not (set(te.case_id) & set(va.case_id)), "test/val case leak"
        assert not (set(tr.case_id) & set(va.case_id)), "train/val case leak"
        assert len(te) + len(tr) + len(va) == len(df)

        tr.to_csv(out / f"f{k}_train.csv", index=False)
        va.to_csv(out / f"f{k}_val.csv", index=False)
        te.to_csv(out / f"f{k}_test.csv", index=False)
        print(f"fold {k}: train={len(tr)} (pos {int(tr.label.sum())})  "
              f"val={len(va)} (pos {int(va.label.sum())})  test={len(te)} (pos {int(te.label.sum())})")

    # coverage check: the held-out folds must tile the pool exactly once
    cover = pd.concat([pd.read_csv(out / f"f{k}_test.csv") for k in range(a.n_folds)])
    assert len(cover) == len(df) and cover.case_id.nunique() == len(df), "folds do not tile the pool"
    print(f"OK: {a.n_folds} folds tile all {len(df)} cases exactly once -> {out}")


if __name__ == "__main__":
    main()
