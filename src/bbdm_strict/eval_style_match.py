"""
针对性风格变换 vs 直接迁移。
只改目标图的强度分布(保留全部结构细节)，再用源分类器分类，体级AUC。
变体：
  direct       原图直接分类(对照, 应≈0.742)
  moment       全局矩匹配：把每张目标图的均值/对比度对齐到源域全局均值/std
  histogram    直方图匹配：把每张目标图的强度CDF映射到源域参考CDF
  hist+sharpen 直方图匹配 + 轻度锐化(补源域更锐的高频)
作用在原图上，不过 VAE、不过 bridge。
"""
import argparse, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


def vol_aucs(df):
    def tk(s, k=5):
        s = np.sort(s.values)[::-1]; return float(s[:max(1, min(k, len(s)))].mean())
    gm = df.groupby("case").agg(p=("p", "mean"), y=("y", "first"))
    gx = df.groupby("case").agg(p=("p", "max"), y=("y", "first"))
    gt = df.groupby("case").apply(lambda d: pd.Series({"p": tk(d.p), "y": d.y.iloc[0]}))
    return auc(df.y.values, df.p.values), auc(gm.y.values, gm.p.values), auc(gx.y.values, gx.p.values), auc(gt.y.values, gt.p.values)


def build_source_ref(paths, size):
    hist = np.zeros(256, dtype=np.float64)
    s = ss = n = 0.0
    for p in paths:
        g = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32)
        hist += np.bincount(g.astype(int).ravel(), minlength=256)
        gg = g / 255.0
        s += gg.sum(); ss += (gg ** 2).sum(); n += gg.size
    cdf = np.cumsum(hist); cdf /= cdf[-1]
    mean = s / n; std = float(np.sqrt(max(ss / n - mean ** 2, 1e-8)))
    return cdf, float(mean), std


def hist_match(u8, ref_cdf):
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    c = np.cumsum(hist); c /= c[-1]
    lut = np.searchsorted(ref_cdf, c).clip(0, 255).astype(np.uint8)
    return lut[u8]


class TgtCSV(Dataset):
    def __init__(self, csv, ref_cdf, smean, sstd):
        self.df = pd.read_csv(csv)
        self.df = self.df[pd.to_numeric(self.df["label"], errors="coerce").isin([0, 1])].reset_index(drop=True)
        self.ref, self.sm, self.ss = ref_cdf, smean, sstd
        self.norm = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                               T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                               T.Normalize([0.5]*3, [0.5]*3)])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r["image_path"]).convert("L")
        u8 = np.asarray(im, dtype=np.uint8)
        g = u8.astype(np.float32) / 255.0
        # moment match
        m, s = g.mean(), g.std() + 1e-6
        gm = np.clip((g - m) / s * self.ss + self.sm, 0, 1)
        im_moment = Image.fromarray((gm * 255).astype(np.uint8), "L")
        # histogram match
        hm = hist_match(u8, self.ref)
        im_hist = Image.fromarray(hm, "L")
        im_hist_sharp = im_hist.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=1))
        y = int(float(r["label"])); c = str(r["case_id"])
        return (self.norm(im), self.norm(im_moment), self.norm(im_hist), self.norm(im_hist_sharp), y, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--n_ref", type=int, default=1500)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")

    clf = build_model("custom_resnet50_space", 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    srcdf = pd.read_csv(a.source_csv)
    rng = np.random.RandomState(0)
    spaths = srcdf.iloc[rng.choice(len(srcdf), min(a.n_ref, len(srcdf)), replace=False)]["image_path"].tolist()
    ref_cdf, smean, sstd = build_source_ref(spaths, a.size)
    print("source ref: mean=%.3f std=%.3f" % (smean, sstd))

    dl = DataLoader(TgtCSV(a.target_csv, ref_cdf, smean, sstd), batch_size=a.batch_size, shuffle=False, num_workers=6)
    P = {k: [] for k in ["direct", "moment", "histogram", "hist+sharpen"]}
    Y, C = [], []
    with torch.no_grad():
        for xd, xm, xh, xhs, y, c in dl:
            for key, xb in [("direct", xd), ("moment", xm), ("histogram", xh), ("hist+sharpen", xhs)]:
                P[key] += list(torch.softmax(clf(xb.to(dev)), 1)[:, 1].cpu().numpy())
            Y += list(y.numpy()); C += list(c)

    print("\n%-14s %-7s %-7s %-7s %-7s" % ("variant", "slice", "mean", "max", "top5"))
    for key in ["direct", "moment", "histogram", "hist+sharpen"]:
        d = pd.DataFrame({"case": C, "p": P[key], "y": Y})
        s, m, x, t = vol_aucs(d)
        print("%-14s %-7.3f %-7.3f %-7.3f %-7.3f" % (key, s, m, x, t))


if __name__ == "__main__":
    main()
