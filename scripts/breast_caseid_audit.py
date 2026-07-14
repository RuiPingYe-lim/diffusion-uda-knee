import pandas as pd, os
man = "/root/autodl-tmp/breast/BrEaST/csv"
cache = "/root/autodl-tmp/breast/cache"
busi_man = "/root/autodl-tmp/breast/BUSI/csv"

def load(p):
    return pd.read_csv(p) if os.path.isfile(p) else None

print("=== BrEaST manifest vs cache: counts + positional label alignment ===")
splits = {"train": "breast_train", "valid": "breast_valid", "test": "breast_test"}
cid = {}
for s, cf in splits.items():
    m = load(os.path.join(man, s + "_0.csv"))
    c = load(os.path.join(cache, cf + ".csv"))
    if m is None or c is None:
        print(s, "MISSING", m is None, c is None); continue
    same_n = len(m) == len(c)
    lbl_align = same_n and (m["label"].tolist() == c["label"].tolist())
    uniq = bool(m["case_id"].is_unique)
    ipc = len(m) / m["case_id"].nunique()
    cid[s] = set(m["case_id"])
    print("%-6s man=%4d cache=%4d nMatch=%s labelOrderMatch=%s caseUnique=%s imgs/case=%.2f"
          % (s, len(m), len(c), same_n, lbl_align, uniq, ipc))

print("=== BrEaST cross-split case_id overlap (expect all 0) ===")
if len(cid) == 3:
    print("train&valid:", len(cid["train"] & cid["valid"]),
          " train&test:", len(cid["train"] & cid["test"]),
          " valid&test:", len(cid["valid"] & cid["test"]))

print("=== BUSI source: 1 img/case? ===")
for s in ["train", "valid", "test"]:
    m = load(os.path.join(busi_man, s + "_0.csv"))
    if m is not None:
        print("busi_%-6s n=%4d caseUnique=%s imgs/case=%.2f"
              % (s, len(m), bool(m["case_id"].is_unique), len(m) / m["case_id"].nunique()))
