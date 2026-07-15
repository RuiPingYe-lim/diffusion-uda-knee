"""Build KNEE fusion CSVs from k2m UNSB translations (mirror of the breast builder).

FIXED (review P0): EXACT unique-key join (no glob prefix "12"/"120" collision);
missing translation or duplicate case_id -> HARD ERROR, never a silent drop.
Knee case_ids can be short numeric strings, so the exact join matters most here.
"""
import os
import pandas as pd

M = "/root/autodl-tmp/meanproj_stage"
UNSB = "/root/autodl-tmp/UNSB"
FAKES = ["fake_1", "fake_2", "fake_3", "fake_4", "fake_5"]


def exact_path(images_dir, arm, stem):
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


# eval: kneemri_test (target), translations named by case_id
kt = pd.read_csv(f"{M}/kneemri_test.csv")
build([{"image_path": r["image_path"], "label": int(r["label"]), "case_id": str(r["case_id"])}
       for _, r in kt.iterrows()],
      f"{UNSB}/results/k2m_SB/test_latest/images", lambda m: m["case_id"],
      f"{M}/knee_fusion_eval_kneemri.csv", with_case=True)

# train/val: MRNet (source), translations named by key
man = pd.read_csv(f"{M}/k2m_srcapply_manifest.csv")
for split, out in [("mrnet_train", "knee_fusion_train_mrnet.csv"), ("mrnet_valid", "knee_fusion_val_mrnet.csv")]:
    rows = [{"image_path": r["image_path"], "label": int(r["label"]), "key": str(r["key"])}
            for _, r in man[man["split"] == split].iterrows()]
    build(rows, f"{UNSB}/results_k2m_srcapply/k2m_SB/test_latest/images", lambda m: m["key"], f"{M}/{out}")
