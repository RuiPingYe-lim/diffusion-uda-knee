"""Build a UNSB dataset for the REVERSE (source->target) generator G_s->t.

Review-mandated matched-fusion protocol needs a source->target translator so the
training 'before' distribution (G_s->t(source), target-styled) matches the test
'before' distribution (real target). This mirrors build_breast_unsb_dataset.py
but with A/B SWAPPED so AtoB = source->target.

  A = SOURCE domain (trainA), B = TARGET domain (trainB); UNSB translates A->B.
  testA = source train+valid images (to be translated -> training 'before').

Usage (edit the two blocks below or import build_reverse):
  breast: source=BUSI, target=BrEaST  -> datasets/u2b_rev  (BUSI->BrEaST)
  knee  : source=MRNet, target=KneeMRI -> datasets/m2k_rev (MRNet->KneeMRI)
"""
import os
import pandas as pd

UNSB = "/root/autodl-tmp/UNSB"


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d


def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build_reverse(name, src_train_csv, src_valid_csv, tgt_train_csv, tgt_placeholder_csv, manifest_csv):
    """A=source, B=target; testA = source train+valid (named by split__idx)."""
    root = _mk(f"{UNSB}/datasets/{name}")
    dA = _mk(f"{root}/trainA"); dB = _mk(f"{root}/trainB")
    dtA = _mk(f"{root}/testA"); dtB = _mk(f"{root}/testB")

    # trainA = source, trainB = target (unpaired)
    for i, r in pd.read_csv(src_train_csv).iterrows():
        _link(r["image_path"], f"{dA}/a_{i:05d}.png")
    for i, r in pd.read_csv(tgt_train_csv).iterrows():
        _link(r["image_path"], f"{dB}/b_{i:05d}.png")

    # testA = source train + valid, named by split__idx (to map back for training 'before')
    rows = []
    for split, csv in [("src_train", src_train_csv), ("src_valid", src_valid_csv)]:
        df = pd.read_csv(csv)
        for i, r in df.iterrows():
            key = f"{split}__{i:05d}"
            _link(r["image_path"], f"{dtA}/{key}.png")
            rows.append({"key": key, "split": split, "image_path": r["image_path"], "label": int(r["label"])})
    # placeholder testB (unaligned loader needs a B side)
    for i, r in pd.read_csv(tgt_placeholder_csv).head(30).iterrows():
        _link(r["image_path"], f"{dtB}/b_{i:03d}.png")

    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    print(f"{name}: trainA={len(os.listdir(dA))} trainB={len(os.listdir(dB))} "
          f"testA={len(os.listdir(dtA))} -> {manifest_csv}")


if __name__ == "__main__":
    C = "/root/autodl-tmp/breast/cache"
    M = "/root/autodl-tmp/meanproj_stage"
    # breast reverse: BUSI(source) -> BrEaST(target)
    build_reverse("u2b_rev", f"{C}/busi_train.csv", f"{C}/busi_valid.csv",
                  f"{C}/breast_train.csv", f"{C}/breast_train.csv", f"{C}/u2b_rev_srcapply_manifest.csv")
    # knee reverse: MRNet(source) -> KneeMRI(target)
    build_reverse("m2k_rev", f"{M}/mrnet_train.csv", f"{M}/mrnet_valid.csv",
                  f"{M}/kneemri_train.csv", f"{M}/kneemri_train.csv", f"{M}/m2k_rev_srcapply_manifest.csv")
