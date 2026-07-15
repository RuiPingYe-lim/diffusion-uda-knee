import os, pandas as pd
C = "/root/autodl-tmp/breast/cache"
root = "/root/autodl-tmp/UNSB/datasets/breast_b2u"  # A=BrEaST(target), B=BUSI(source); AtoB = target->source

def mk(d):
    os.makedirs(d, exist_ok=True)
    return d

def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)

# trainA = BrEaST train (target, 175), trainB = BUSI train (source, 452)
trainA = pd.read_csv(f"{C}/breast_train.csv")
trainB = pd.read_csv(f"{C}/busi_train.csv")
# testA = BrEaST DIAGNOSTIC (train+valid, 201), named by case_id (locked test EXCLUDED)
testA = pd.read_csv(f"{C}/breast_diag_cid.csv")
# testB = BUSI valid (placeholder so the unaligned loader has a B side at test)
testB = pd.read_csv(f"{C}/busi_valid.csv")

dA = mk(f"{root}/trainA"); dB = mk(f"{root}/trainB")
dtA = mk(f"{root}/testA"); dtB = mk(f"{root}/testB")

for i, r in trainA.iterrows():
    link(r["image_path"], f"{dA}/a_{i:04d}.png")
for i, r in trainB.iterrows():
    link(r["image_path"], f"{dB}/b_{i:04d}.png")
for _, r in testA.iterrows():
    link(r["image_path"], f"{dtA}/{r['case_id']}.png")  # filename = case_id for label mapping
for i, r in testB.iterrows():
    link(r["image_path"], f"{dtB}/b_{i:04d}.png")

print("trainA", len(os.listdir(dA)), "trainB", len(os.listdir(dB)),
      "testA", len(os.listdir(dtA)), "testB", len(os.listdir(dtB)))
print("sample testA:", sorted(os.listdir(dtA))[:2])
