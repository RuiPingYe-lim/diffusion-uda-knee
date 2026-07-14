"""
建议 5：交叉注意力多图融合分类器。

融合输入（不止两张）：
  - before：翻译前的原图（目标域，细节完整）—— 作为 query 锚点
  - translated：确定性翻译图（源域风格）
  - sample_0..sample_{K-1}：多次随机采样得到的源域风格图（每次结果不同）

做法：每张图共享主干提特征 -> 把 before 当 query，去交叉注意力融合
      "翻译图 + 所有采样图"的特征 -> 残差+池化 -> 分类头。
让分类器同时利用 原图细节 + 多个源域风格变体（类似注意力版的多采样集成）。

包含：模型 + Dataset(读 before 列 + 多张其它图列) + 训练/评估。

用法示例：
  评估（目标域）:
    python fusion_classifier.py --mode eval --weights .../best.pt \
      --test_csv pairs.csv --before_col before_png \
      --other_cols translated_png,sample_0,sample_1,sample_2 \
      --label_col target_label
  训练（源域构造的配对，见底部说明）:
    python fusion_classifier.py --mode train --train_csv ... --val_csv ... \
      --before_col before_png --other_cols translated_png,sample_0,sample_1 \
      --label_col label --out_dir .../fusion_run
"""
from __future__ import annotations
import argparse, os, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# ---------------- 数据：before + 任意多张其它图 ----------------
class MultiImageDataset(Dataset):
    def __init__(self, csv, before_col, other_cols, label_col, resize=224, is_train=False):
        self.df = pd.read_csv(csv)
        self.df = self.df[pd.to_numeric(self.df[label_col], errors="coerce").isin([0, 1])].reset_index(drop=True)
        self.bc = before_col
        self.others = [c.strip() for c in other_cols if c.strip()]
        self.lc = label_col
        self.is_train = bool(is_train)
        # Deterministic transform. Geometric augmentation (horizontal flip) is applied
        # in __getitem__ with a SINGLE shared decision so all views (before + others)
        # stay geometrically aligned -- otherwise cross-attention attends across
        # mirror-mismatched token grids (fix for the independent-flip bug).
        self.tf = T.Compose([
            T.ToTensor(), T.Resize((resize, resize), antialias=True),
            T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self): return len(self.df)

    def _load(self, p, do_flip=False):
        im = Image.open(p).convert("L")
        if do_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(im)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        do_flip = self.is_train and (random.random() < 0.5)  # one decision, shared by all views
        before = self._load(r[self.bc], do_flip)
        others = torch.stack([self._load(r[c], do_flip) for c in self.others])  # [K, 3, H, W]
        case = str(r["case_id"]) if "case_id" in self.df.columns else str(i)
        return before, others, int(float(r[self.lc])), case


# ---------------- 模型 ----------------
class CrossAttnFusionClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dim=256, heads=4, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=w); feat_dim = 2048
        else:
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=w); feat_dim = 512
        self.backbone = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
        self.proj = nn.Conv2d(feat_dim, dim, 1)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(dim, num_classes))

    def _tokens(self, x):                       # x: [B,3,H,W] -> [B, h*w, dim]
        f = self.proj(self.backbone(x))
        return f.flatten(2).transpose(1, 2)

    def forward(self, before, others):
        # before: [B,3,H,W]; others: [B,K,3,H,W]
        B, K = others.shape[0], others.shape[1]
        q = self._tokens(before)                                  # [B, N, dim]  原图 query
        oth = self._tokens(others.reshape(B * K, *others.shape[2:]))  # [B*K, N, dim]
        N = oth.shape[1]
        kv = oth.reshape(B, K * N, oth.shape[2])                  # [B, K*N, dim] 所有风格图的 token 拼一起
        fused, _ = self.attn(q, kv, kv)                           # before 去 query 全部风格图
        fused = self.norm(q + fused)
        return self.head(fused.mean(dim=1))


# ---------------- 训练/评估 ----------------
def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); ps, ys = [], []
    for before, others, y, _ in loader:
        p = torch.softmax(model(before.to(dev), others.to(dev)), 1)[:, 1].cpu().numpy()
        ps.append(p); ys.append(y.numpy())
    return auc(np.concatenate(ys), np.concatenate(ps))


@torch.no_grad()
def evaluate_volume(model, loader, dev):
    """切片级 + 体级(按 case_id 聚合 mean/max/top5) AUC。"""
    model.eval(); ps, ys, cs = [], [], []
    for before, others, y, case in loader:
        p = torch.softmax(model(before.to(dev), others.to(dev)), 1)[:, 1].cpu().numpy()
        ps += list(p); ys += list(y.numpy()); cs += list(case)
    df = pd.DataFrame({"case": cs, "p": ps, "y": ys})
    slice_auc = auc(df.y.values, df.p.values)
    def tk(s, k=5):
        s = np.sort(s.values)[::-1]; return float(s[:max(1, min(k, len(s)))].mean())
    gm = df.groupby("case").agg(p=("p", "mean"), y=("y", "first"))
    gx = df.groupby("case").agg(p=("p", "max"), y=("y", "first"))
    gt = df.groupby("case").apply(lambda d: pd.Series({"p": tk(d.p), "y": d.y.iloc[0]}))
    return slice_auc, auc(gm.y.values, gm.p.values), auc(gx.y.values, gx.p.values), auc(gt.y.values, gt.p.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--train_csv"); ap.add_argument("--val_csv"); ap.add_argument("--test_csv")
    ap.add_argument("--before_col", default="before_png")
    ap.add_argument("--other_cols", default="translated_png",
                    help="逗号分隔：translated_png,sample_0,sample_1,...")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="./fusion_run")
    ap.add_argument("--weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet18"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    others = a.other_cols.split(",")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttnFusionClassifier(backbone=a.backbone).to(dev)

    if a.mode == "eval":
        ck = torch.load(a.weights, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        dl = DataLoader(MultiImageDataset(a.test_csv, a.before_col, others, a.label_col, a.resize),
                        batch_size=a.batch_size, num_workers=4)
        sa, vm, vx, vt = evaluate_volume(model, dl, dev)
        print("fusion eval  slice_AUC=%.3f  (K=%d 风格图 + before)" % (sa, len(others)))
        print("    volume_AUC:  mean=%.3f  max=%.3f  top5=%.3f" % (vm, vx, vt))
        return

    os.makedirs(a.out_dir, exist_ok=True)
    tr = DataLoader(MultiImageDataset(a.train_csv, a.before_col, others, a.label_col, a.resize, True),
                    batch_size=a.batch_size, shuffle=True, num_workers=4)
    va = DataLoader(MultiImageDataset(a.val_csv, a.before_col, others, a.label_col, a.resize),
                    batch_size=a.batch_size, num_workers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for before, oth, y, _ in tr:
            opt.zero_grad()
            loss = crit(model(before.to(dev), oth.to(dev)), y.to(dev))
            loss.backward(); opt.step()
        v = evaluate(model, va, dev)
        print(f"epoch {ep} val_auc={v:.4f}")
        if v > best:
            best = v; torch.save({"model": model.state_dict(), "val_auc": v}, os.path.join(a.out_dir, "best.pt"))
    print("best val AUC =", best)


if __name__ == "__main__":
    main()


# ============================ 配套：怎么得到多张采样图 ============================
# 用 BBDM 的随机采样（reverse_eta>0，例如 0.35）对每张目标图跑 K 次，得到 K 张不同的源域风格图，
# 连同 before(原图) 和 translated(确定性 eta=0) 一起写进同一个 csv 的多列：
#   before_png, translated_png, sample_0, sample_1, ..., target_label
# 评估时 --other_cols translated_png,sample_0,sample_1,sample_2。
# 训练配对（需标签->源域）方案见上一版说明：源域图 + 其 BBDM 风格化的多张采样作为 others。
# 采样脚本我可以在 sample_strict_bbdm/sample_guided 基础上加个 --num_samples_per_image K 来批量生成。
