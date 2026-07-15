"""Build fusion CSVs from UNSB b2u translations.

Structure (consistent train/test): before = raw image, others = its b2u output
(source-styled) at bridge steps fake_1..fake_5.

  train/val: source BUSI  (before = busi raw, fakes = b2u(busi), label = busi)
  eval     : target BrEaST diagnostic (before = breast raw, fakes = b2u(breast),
             label = breast, case_id) -- LOCKED breast_test excluded.

Emits columns: before_png, fake_1..fake_5, label, case_id, split.
"""
import os, glob, pandas as pd

C = "/root/autodl-tmp/breast/cache"
UNSB = "/root/autodl-tmp/UNSB"
FAKES = ["fake_1", "fake_2", "fake_3", "fake_4", "fake_5"]


def find_img(images_dir, arm, stem):
    # test.py may append suffixes; match by stem
    cands = glob.glob(os.path.join(images_dir, arm, f"{stem}.png")) or \
            glob.glob(os.path.join(images_dir, arm, f"{stem}*.png"))
    return cands[0] if cands else None


def build(manifest_rows, images_dir, key_to_stem, out_csv, with_case=False):
    rows = []
    miss = 0
    for m in manifest_rows:
        stem = key_to_stem(m)
        rec = {"before_png": m["image_path"], "label": int(m["label"])}
        ok = True
        for a in FAKES:
            p = find_img(images_dir, a, stem)
            if p is None:
                ok = False; break
            rec[a] = p
        if not ok:
            miss += 1; continue
        if with_case:
            rec["case_id"] = m["case_id"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(df)} rows, {miss} missing")
    return df


# --- eval: target diagnostic (breast_b2u results) ---
diag = pd.read_csv(f"{C}/breast_diag_cid.csv")
diag_imgs = f"{UNSB}/results/b2u_SB/test_latest/images"
eval_rows = [{"image_path": r["image_path"], "label": int(r["label"]), "case_id": r["case_id"]}
             for _, r in diag.iterrows()]
build(eval_rows, diag_imgs, lambda m: m["case_id"], f"{C}/fusion_eval_breast_diag.csv", with_case=True)

# --- train/val: source BUSI (b2u_srcapply results) ---
man = pd.read_csv(f"{C}/b2u_srcapply_manifest.csv")
src_imgs = f"{UNSB}/results_srcapply/b2u_SB/test_latest/images"
for split, out in [("busi_train", "fusion_train_busi.csv"), ("busi_valid", "fusion_val_busi.csv")]:
    rows = [{"image_path": r["image_path"], "label": int(r["label"]), "key": r["key"]}
            for _, r in man[man["split"] == split].iterrows()]
    build(rows, src_imgs, lambda m: m["key"], f"{C}/{out}", with_case=False)
