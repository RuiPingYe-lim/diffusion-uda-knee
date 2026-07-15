"""Build KNEE fusion CSVs from k2m UNSB translations (mirror of the breast builder)."""
import os, glob, pandas as pd
M = "/root/autodl-tmp/meanproj_stage"
UNSB = "/root/autodl-tmp/UNSB"
FAKES = ["fake_1", "fake_2", "fake_3", "fake_4", "fake_5"]


def find_img(images_dir, arm, stem):
    c = glob.glob(os.path.join(images_dir, arm, f"{stem}.png")) or \
        glob.glob(os.path.join(images_dir, arm, f"{stem}*.png"))
    return c[0] if c else None


def build(rows, images_dir, key_to_stem, out_csv, with_case=False):
    out, miss = [], 0
    for m in rows:
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
        out.append(rec)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(out)} rows, {miss} missing")


# eval: kneemri_test (target), translations named by case_id
kt = pd.read_csv(f"{M}/kneemri_test.csv")
eval_imgs = f"{UNSB}/results/k2m_SB/test_latest/images"
build([{"image_path": r["image_path"], "label": int(r["label"]), "case_id": str(r["case_id"])} for _, r in kt.iterrows()],
      eval_imgs, lambda m: m["case_id"], f"{M}/knee_fusion_eval_kneemri.csv", with_case=True)

# train/val: MRNet (source), translations named by key
man = pd.read_csv(f"{M}/k2m_srcapply_manifest.csv")
src_imgs = f"{UNSB}/results_k2m_srcapply/k2m_SB/test_latest/images"
for split, out in [("mrnet_train", "knee_fusion_train_mrnet.csv"), ("mrnet_valid", "knee_fusion_val_mrnet.csv")]:
    rows = [{"image_path": r["image_path"], "label": int(r["label"]), "key": r["key"]}
            for _, r in man[man["split"] == split].iterrows()]
    build(rows, src_imgs, lambda m: m["key"], f"{M}/{out}", with_case=False)
