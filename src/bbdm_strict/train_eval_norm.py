"""
用 NMF baseline 的预处理(逐图 p1-p99 百分位归一化 + 逐域 mean/std 标准化)
重训 KneeMRI 源分类器, 再测反向直接迁移(目标=MRNet)。
验证: 正确归一化能否把直接迁移拉到 ~0.8; 以及矩匹配是否还有增益。
"""
import argparse, sys
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model


def intensity_norm01(arr):
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
    if vmax > vmin:
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr.astype(np.float32)


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


def vol_auc(cases, probs, ys):
    d = pd.DataFrame({"case": cases, "p": probs, "y": ys})
    g = d.groupby("case").agg(p=("p", "mean"), y=("y", "first"))
    return auc(g.y.values, g.p.values)


def domain_stats(paths, size, n=1200):
    rng = np.random.RandomState(0)
    sel = [paths[i] for i in rng.choice(len(paths), min(n, len(paths)), replace=False)]
    s = ss = c = 0.0
    for p in sel:
        a = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32)
        a = intensity_norm01(a)
        s += a.sum(); ss += (a ** 2).sum(); c += a.size
    m = s / c; return float(m), float(np.sqrt(max(ss / c - m ** 2, 1e-8)))


class DS(Dataset):
    def __init__(self, csv, size, mean, std, train=False):
        self.df = pd.read_csv(csv)
        self.df = self.df[pd.to_numeric(self.df["label"], errors="coerce").isin([0, 1])].reset_index(drop=True)
        self.size = size
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if train else []
        self.tf = T.Compose(aug + [T.ToTensor(), T.Resize((size, size), antialias=True),
                                   T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                                   T.Normalize([mean] * 3, [std] * 3)])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = np.asarray(Image.open(r["image_path"]).convert("L"), dtype=np.float32)
        a = intensity_norm01(a)
        img = Image.fromarray((a * 255).astype(np.uint8), "L")
        return self.tf(img), int(float(r["label"])), str(r["case_id"])


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); P, Y, C = [], [], []
    for x, y, c in dl:
        p = torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().numpy()
        P += list(p); Y += list(y.numpy()); C += list(c)
    return vol_auc(C, P, Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_train", required=True); ap.add_argument("--src_val", required=True)
    ap.add_argument("--tgt", required=True); ap.add_argument("--tgt2", default="")
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")

    srcp = pd.read_csv(a.src_train)["image_path"].tolist()
    tgtp = pd.read_csv(a.tgt)["image_path"].tolist()
    sm, ss = domain_stats(srcp, a.size); tm, ts = domain_stats(tgtp, a.size)
    print("逐域统计(百分位归一化后): 源KneeMRI mean=%.3f std=%.3f | 目标MRNet mean=%.3f std=%.3f" % (sm, ss, tm, ts))

    tr = DataLoader(DS(a.src_train, a.size, sm, ss, train=True), batch_size=a.bs, shuffle=True, num_workers=8)
    sval = DataLoader(DS(a.src_val, a.size, sm, ss), batch_size=a.bs, num_workers=8)
    # 目标用"逐域"= 目标自己的 stats
    tval = DataLoader(DS(a.tgt, a.size, tm, ts), batch_size=a.bs, num_workers=8)

    model = build_model("custom_resnet50_space", 2, "imagenet", dev).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y, _ in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        sv = evaluate(model, sval, dev)
        if sv > best:
            best = sv; torch.save({"model": model.state_dict()}, a.out)
        print("epoch %d  src_val(vol)=%.4f  best=%.4f" % (ep, sv, best))

    # 用最优权重测目标(反向直接迁移)
    model.load_state_dict(torch.load(a.out, map_location=dev)["model"]); model.eval()
    print("\n===== 结果(体级 AUC) =====")
    print("源域 in-domain (KneeMRI test):  %.3f" % best)
    print("反向直接迁移 (目标=MRNet, 逐域归一化): %.3f" % evaluate(model, tval, dev))
    if a.tgt2:
        t2 = DataLoader(DS(a.tgt2, a.size, tm, ts), batch_size=a.bs, num_workers=8)
        print("反向直接迁移 (第二目标集): %.3f" % evaluate(model, t2, dev))


if __name__ == "__main__":
    main()
