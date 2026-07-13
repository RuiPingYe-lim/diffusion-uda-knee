"""
晚融合(决策级)集成：矩匹配分类器 与 扩散融合分类器 的预测概率按权重混合。
两个模型互不干扰，若互补则集成 > 单模型。扫权重 alpha(矩匹配所占比重)。
"""
import argparse, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model
from fusion_classifier import CrossAttnFusionClassifier


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


def vol_mean_auc(cases, probs, ys):
    d = pd.DataFrame({"case": cases, "p": probs, "y": ys})
    g = d.groupby("case").agg(p=("p", "mean"), y=("y", "first"))
    return auc(g.y.values, g.p.values)


NORM = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                  T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                  T.Normalize([0.5]*3, [0.5]*3)])


class DS(Dataset):
    def __init__(self, csv, others):
        self.df = pd.read_csv(csv)
        self.df = self.df[pd.to_numeric(self.df["label"], errors="coerce").isin([0, 1])].reset_index(drop=True)
        self.others = others
    def __len__(self): return len(self.df)
    def _ld(self, p): return NORM(Image.open(p).convert("L"))
    def __getitem__(self, i):
        r = self.df.iloc[i]
        moment = self._ld(r["moment_png"])
        before = self._ld(r["before_png"])
        oth = torch.stack([self._ld(r[c]) for c in self.others])
        return moment, before, oth, int(float(r["label"])), str(r["case_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_clf", required=True)
    ap.add_argument("--fusion_ckpt", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--other_cols", default="translated_png,sample_0,sample_1")
    ap.add_argument("--batch_size", type=int, default=48)
    a = ap.parse_args()
    dev = torch.device("cuda")
    others = a.other_cols.split(",")

    clf = build_model("custom_resnet50_space", 2, "none", dev)
    ck = torch.load(a.src_clf, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    fus = CrossAttnFusionClassifier(backbone="resnet50").to(dev)
    fus.load_state_dict(torch.load(a.fusion_ckpt, map_location=dev, weights_only=False)["model"]); fus.eval()

    dl = DataLoader(DS(a.pairs_csv, others), batch_size=a.batch_size, shuffle=False, num_workers=6)
    Pm, Pf, Y, C = [], [], [], []
    with torch.no_grad():
        for moment, before, oth, y, c in dl:
            pm = torch.softmax(clf(moment.to(dev)), 1)[:, 1].cpu().numpy()
            pf = torch.softmax(fus(before.to(dev), oth.to(dev)), 1)[:, 1].cpu().numpy()
            Pm += list(pm); Pf += list(pf); Y += list(y.numpy()); C += list(c)
    Pm, Pf = np.array(Pm), np.array(Pf)

    print("单模型: 矩匹配 mean=%.3f  |  扩散融合 mean=%.3f" % (vol_mean_auc(C, Pm, Y), vol_mean_auc(C, Pf, Y)))
    print("\nalpha(矩匹配权重)  体级AUC(mean)")
    best = (-1, None)
    for al in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        ens = al * Pm + (1 - al) * Pf
        m = vol_mean_auc(C, ens, Y)
        print("  %.1f              %.3f" % (al, m))
        if m > best[0]: best = (m, al)
    print("\n最佳: alpha=%.1f  mean=%.3f" % (best[1], best[0]))


if __name__ == "__main__":
    main()
