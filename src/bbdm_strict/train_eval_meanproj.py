"""
在 paper1 的 mean-projection 表示上训 source-only 分类器并测目标迁移。
图像已是 intensity_norm01 后的均值投影(每 case 一张)。transform=Normalize(0.5)。
每 case 一个预测, AUC 直接按 case 算。目标: 复现 Source Only 0.740(M->K)/0.807(K->M)。
"""
import argparse, sys
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


class DS(Dataset):
    def __init__(self, csv, size=224, train=False):
        self.df = pd.read_csv(csv)
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if train else []
        self.tf = T.Compose(aug + [T.ToTensor(), T.Resize((size, size), antialias=True),
                                   T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                                   T.Normalize([0.5] * 3, [0.5] * 3)])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["image_path"]).convert("L")), int(r["label"])


@torch.no_grad()
def ev(model, dl, dev):
    model.eval(); P, Y = [], []
    for x, y in dl:
        P += list(torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().numpy()); Y += list(y.numpy())
    return auc(Y, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_train", required=True); ap.add_argument("--src_val", required=True)
    ap.add_argument("--tgt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=4e-4); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    tr = DataLoader(DS(a.src_train, train=True), batch_size=a.bs, shuffle=True, num_workers=8)
    sv = DataLoader(DS(a.src_val), batch_size=a.bs, num_workers=8)
    tg = DataLoader(DS(a.tgt), batch_size=a.bs, num_workers=8)
    model = build_model("custom_resnet50_space", 2, "imagenet", dev).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(); scaler = torch.cuda.amp.GradScaler()
    best = -1; best_tgt = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        vs = ev(model, sv, dev)
        if vs > best:
            best = vs; best_tgt = ev(model, tg, dev)
            torch.save({"model": model.state_dict()}, a.out)
        if ep % 5 == 0 or ep == a.epochs:
            print("[%s] epoch %d src_val=%.4f (best src_val=%.4f -> tgt=%.4f)" % (a.tag, ep, vs, best, best_tgt))
    print("===== [%s] 结果: 源in-domain=%.4f  |  目标直接迁移(Source Only)=%.4f =====" % (a.tag, best, best_tgt))


if __name__ == "__main__":
    main()
