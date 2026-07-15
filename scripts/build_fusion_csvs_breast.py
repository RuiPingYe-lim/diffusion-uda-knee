"""Build breast fusion CSVs from UNSB b2u translations.

FIXED (review P0): EXACT unique-key join (no glob prefix match -> no "12"/"120"
collision); a missing translation or a duplicate case_id is a HARD ERROR, never
a silent drop. Structure: before = raw image, others = its b2u output (source-
styled) at bridge steps fake_1..fake_5.
  train/val: source BUSI (before=busi raw, fakes=b2u(busi), label=busi)
  eval     : target BrEaST diagnostic (before=breast raw, fakes=b2u(breast),
             label=breast, case_id) -- LOCKED breast_test excluded.
"""
import os
import pandas as pd

C = "/root/autodl-tmp/breast/cache"
UNSB = "/root/autodl-tmp/UNSB"
FAKES = ["fake_1", "fake_2", "fake_3", "fake_4", "fake_5"]


def exact_path(images_dir, arm, stem):
    """EXACT match only: <images_dir>/<arm>/<stem>.png must exist (fail-loud)."""
    p = os.path.join(images_dir, arm, f"{stem}.png")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"missing translation: {p}")
    return p


def build(rows, images_dir, key_to_stem, out_csv, with_case=False):
    stems = [key_to_stem(m) for m in rows]
    dup = {s for s in stems if stems.count(s) > 1}
    if dup:
        raise ValueError(f"duplicate keys in {out_csv}: {sorted(dup)[:8]}")
    out = []
    for m in rows:
        stem = key_to_stem(m)
        rec = {"before_png": m["image_path"], "label": int(m["label"])}
        for a in FAKES:
            rec[a] = exact_path(images_dir, a, stem)   # raises if missing
        if with_case:
            rec["case_id"] = m["case_id"]
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(df)} rows (0 missing, 0 dup)")


# eval: target BrEaST diagnostic (translations named by case_id)
diag = pd.read_csv(f"{C}/breast_diag_cid.csv")
build([{"image_path": r["image_path"], "label": int(r["label"]), "case_id": str(r["case_id"])}
       for _, r in diag.iterrows()],
      f"{UNSB}/results/b2u_SB/test_latest/images", lambda m: m["case_id"],
      f"{C}/fusion_eval_breast_diag.csv", with_case=True)

# train/val: source BUSI (translations named by key)
man = pd.read_csv(f"{C}/b2u_srcapply_manifest.csv")
for split, out in [("busi_train", "fusion_train_busi.csv"), ("busi_valid", "fusion_val_busi.csv")]:
    rows = [{"image_path": r["image_path"], "label": int(r["label"]), "key": str(r["key"])}
            for _, r in man[man["split"] == split].iterrows()]
    build(rows, f"{UNSB}/results_srcapply/b2u_SB/test_latest/images", lambda m: m["key"], f"{C}/{out}")
