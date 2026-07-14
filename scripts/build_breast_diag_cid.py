import pandas as pd, os
man = "/root/autodl-tmp/breast/BrEaST/csv"
cache = "/root/autodl-tmp/breast/cache"
out = os.path.join(cache, "breast_diag_cid.csv")

parts = []
for s, cf in [("train", "breast_train"), ("valid", "breast_valid")]:
    m = pd.read_csv(os.path.join(man, s + "_0.csv"))
    c = pd.read_csv(os.path.join(cache, cf + ".csv"))
    assert len(m) == len(c), (s, len(m), len(c))
    assert m["label"].tolist() == c["label"].tolist(), f"label order mismatch in {s}"
    part = pd.DataFrame({"case_id": m["case_id"].values,
                         "image_path": c["image_path"].values,
                         "label": c["label"].values,
                         "split": s})
    parts.append(part)

diag = pd.concat(parts, ignore_index=True)
assert diag["case_id"].is_unique, "duplicate case_id in diagnostic set!"
diag.to_csv(out, index=False)
print("wrote", out, "rows", len(diag), "unique_cases", diag["case_id"].nunique(),
      "labels", dict(diag["label"].value_counts().sort_index()))
print("sample:", diag.iloc[0].to_dict())
