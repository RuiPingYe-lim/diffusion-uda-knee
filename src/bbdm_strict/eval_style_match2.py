"""
把膝关节矩匹配榨到最高。诊断=差异在低频，故重点试"只对齐低频、保留高频细节"。
变体(全部作用在原图, 确定性, 只用源域全局统计 -> 无监督)：
  direct                原图
  moment_global         全局矩匹配(已知 0.795 锚)
  lowfreq_moment_rN     拆低频(高斯r=N)只对齐低频均值/对比度, 高频原样保留
  blendA                α*moment_global + (1-α)*原图
  gamma_mean            gamma 校正对齐亮度(非线性)
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


def source_stats(paths, size):
    s = ss = n = 0.0
    for p in paths:
        g = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32) / 255.0
        s += g.sum(); ss += (g ** 2).sum(); n += g.size
    m = s / n; return float(m), float(np.sqrt(max(ss / n - m ** 2, 1e-8)))


def gauss_lf(g, radius):
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(im, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--n_ref", type=int, default=1500)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=48)
    a = ap.parse_args()
    dev = torch.device("cuda")

    clf = build_model("custom_resnet50_space", 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    srcdf = pd.read_csv(a.source_csv)
    rng = np.random.RandomState(0)
    sp = srcdf.iloc[rng.choice(len(srcdf), min(a.n_ref, len(srcdf)), replace=False)]["image_path"].tolist()
    SM, SS = source_stats(sp, a.size)
    print("source ref: mean=%.3f std=%.3f" % (SM, SS))

    def moment(g):
        m, s = g.mean(), g.std() + 1e-6
        return np.clip((g - m) / s * SS + SM, 0, 1)

    def lowfreq_moment(g, r):
        lf = gauss_lf(g, r); hf = g - lf
        m, s = lf.mean(), lf.std() + 1e-6
        lf2 = (lf - m) / s * SS + SM
        return np.clip(lf2 + hf, 0, 1)

    def blend(g, al):
        return np.clip(al * moment(g) + (1 - al) * g, 0, 1)

    def gamma_mean(g):
        tm = max(g.mean(), 1e-3)
        gm = np.log(max(SM, 1e-3)) / np.log(tm)
        return np.clip(g ** gm, 0, 1)

    VAR = [("direct", lambda g: g),
           ("moment_global", moment),
           ("lowfreq_moment_r4", lambda g: lowfreq_moment(g, 4)),
           ("lowfreq_moment_r8", lambda g: lowfreq_moment(g, 8)),
           ("lowfreq_moment_r16", lambda g: lowfreq_moment(g, 16)),
           ("blend_0.5", lambda g: blend(g, 0.5)),
           ("blend_0.75", lambda g: blend(g, 0.75)),
           ("gamma_mean", gamma_mean)]

    norm = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                      T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                      T.Normalize([0.5]*3, [0.5]*3)])

    class DS(Dataset):
        def __init__(self, csv):
            self.df = pd.read_csv(csv)
            self.df = self.df[pd.to_numeric(self.df["label"], errors="coerce").isin([0, 1])].reset_index(drop=True)
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            g = np.asarray(Image.open(r["image_path"]).convert("L"), dtype=np.float32) / 255.0
            xs = torch.stack([norm(Image.fromarray((fn(g) * 255).astype(np.uint8), "L")) for _, fn in VAR])
            return xs, int(float(r["label"])), str(r["case_id"])

    dl = DataLoader(DS(a.target_csv), batch_size=a.batch_size, shuffle=False, num_workers=6)
    P = [[] for _ in VAR]; Y, C = [], []
    with torch.no_grad():
        for xs, y, c in dl:  # xs: [B, V, 3, 224, 224]
            B, V = xs.shape[0], xs.shape[1]
            out = torch.softmax(clf(xs.view(B * V, *xs.shape[2:]).to(dev)), 1)[:, 1].view(B, V).cpu().numpy()
            for v in range(V): P[v] += list(out[:, v])
            Y += list(y.numpy()); C += list(c)

    print("\n%-20s %-7s %-7s %-7s %-7s" % ("variant", "slice", "mean", "max", "top5"))
    for v, (name, _) in enumerate(VAR):
        d = pd.DataFrame({"case": C, "p": P[v], "y": Y})
        s, m, x, t = vol_aucs(d)
        print("%-20s %-7.3f %-7.3f %-7.3f %-7.3f" % (name, s, m, x, t))


if __name__ == "__main__":
    main()
