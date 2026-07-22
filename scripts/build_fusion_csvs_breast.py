"""Build breast fusion CSVs with the CORRECT matched pairing (review P0 fix).

CORRECT UDA pairing (was WRONG before: train fed raw BUSI into G_t->s = OOD):
  train/val (labeled = source BUSI):
      before = G_s->t(x_s)          # u2b_rev output (target-styled source), fake_5
      fake_1..5 = G_t->s(before)    # b2u cycle-back of that target-styled image
      label = BUSI source label
  eval (target BrEaST diagnostic):
      before = x_t (raw BrEaST), fake_1..5 = G_t->s(x_t) = b2u(BrEaST)   [already correct]

Prereq translation passes (run before this script):
  1) UNSB test u2b_rev on BUSI (u2b_rev/testA) -> results_u2b_rev/.../fake_5/<key>.png
  2) build cycleback dataset (testA = those fake_5) + UNSB test b2u
     -> results_cycleback/.../fake_1..5/<key>.png

EXACT unique-key join; missing/duplicate -> HARD ERROR (no silent drops).
"""
import os
import pandas as pd

C = "/root/autodl-tmp/breast/cache"
UNSB = "/root/autodl-tmp/UNSB"
FAKES = ["fake_1", "fake_2", "fake_3", "fake_4", "fake_5"]

REV_DIR = f"{UNSB}/results_u2b_rev/u2b_rev_SB/test_latest/images"      # G_s->t(BUSI): befores
CYC_DIR = f"{UNSB}/results_cycleback/b2u_SB/test_latest/images"        # G_t->s(before): others
EVAL_DIR = f"{UNSB}/results/b2u_SB/test_latest/images"                 # G_t->s(BrEaST): eval others


def exact(images_dir, arm, stem):
    p = os.path.join(images_dir, arm, f"{stem}.png")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"missing translation: {p}")
    return p


def build_train(manifest_csv, split, out_csv):
    """before = u2b_rev fake_5 (target-styled source); others = b2u cycle-back fake_1..5."""
    man = pd.read_csv(manifest_csv)
    rows = man[man["split"] == split]
    keys = [str(k) for k in rows["key"]]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys in {split}")
    out = []
    for _, r in rows.iterrows():
        key = str(r["key"])
        rec = {"before_png": exact(REV_DIR, "fake_5", key), "label": int(r["label"]), "key": key}
        for a in FAKES:
            rec[a] = exact(CYC_DIR, a, key)
        out.append(rec)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(out)} rows (correct pairing: before=G_s->t, others=cycle-back)")


def build_eval(diag_csv, out_csv):
    """before = raw BrEaST (target); others = b2u(BrEaST). Already the correct test pairing."""
    diag = pd.read_csv(diag_csv)
    if not diag["case_id"].is_unique:
        raise ValueError("duplicate case_id in diagnostic set")
    out = []
    for _, r in diag.iterrows():
        cid = str(r["case_id"])
        rec = {"before_png": r["image_path"], "label": int(r["label"]), "case_id": cid}
        for a in FAKES:
            rec[a] = exact(EVAL_DIR, a, cid)
        out.append(rec)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(out)} rows (eval: before=raw target, others=b2u(target))")


if __name__ == "__main__":
    man = f"{C}/u2b_rev_srcapply_manifest.csv"    # key,split,image_path,label (from build_unsb_reverse_dataset)
    build_train(man, "src_train", f"{C}/fusion_train_busi.csv")
    build_train(man, "src_valid", f"{C}/fusion_val_busi.csv")
    build_eval(f"{C}/breast_diag_cid.csv", f"{C}/fusion_eval_breast_diag.csv")
