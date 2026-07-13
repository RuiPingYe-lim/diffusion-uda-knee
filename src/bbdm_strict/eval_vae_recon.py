"""
诊断：VAE 纯重建(encode->decode, 不做bridge翻译) 对分类 AUC 的影响。
对目标测试集每张切片同时算：
  direct = 原图直接分类(224)
  recon  = 只过 KL-VAE 编解码后再分类
体级聚合 mean/max/top5，两者并排输出，隔离"自编码器瓶颈"。
"""
import argparse, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model
from ae_frontend import load_ae_model, ae_encode_latent, ae_decode_latent


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


class SliceCSV(Dataset):
    def __init__(self, csv, size):
        self.df = pd.read_csv(csv)
        self.df = self.df[pd.to_numeric(self.df["label"], errors="coerce").isin([0, 1])].reset_index(drop=True)
        self.size = size
        self.tf = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                             T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                             T.Normalize([0.5]*3, [0.5]*3)])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r["image_path"]).convert("L")
        x224 = self.tf(im)
        a = np.asarray(im.resize((self.size, self.size)), dtype=np.float32) / 255.0 * 2 - 1
        x128 = torch.from_numpy(a)[None]
        return x224, x128, int(float(r["label"])), str(r["case_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--slice_csv", required=True)
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--ae_config", required=True)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")

    clf = build_model(a.backbone, 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    ae = load_ae_model(a.ae_ckpt, a.ae_config, device=dev, freeze=True)

    dl = DataLoader(SliceCSV(a.slice_csv, a.image_size), batch_size=a.batch_size, shuffle=False, num_workers=6)
    pd_, pr_, ys, cs = [], [], [], []
    with torch.no_grad():
        for x224, x128, y, cid in dl:
            p_direct = torch.softmax(clf(x224.to(dev)), 1)[:, 1].cpu().numpy()
            z = ae_encode_latent(ae, x128.to(dev))["latent"]
            rec = ae_decode_latent(ae, z)["recon"]
            xin = TF.resize(rec, [224, 224], antialias=True).repeat(1, 3, 1, 1)
            p_recon = torch.softmax(clf(xin), 1)[:, 1].cpu().numpy()
            pd_ += list(p_direct); pr_ += list(p_recon); ys += list(y.numpy()); cs += list(cid)

    d_direct = pd.DataFrame({"case": cs, "p": pd_, "y": ys})
    d_recon = pd.DataFrame({"case": cs, "p": pr_, "y": ys})
    for name, d in [("direct(原图)", d_direct), ("VAE-recon(仅编解码)", d_recon)]:
        s, m, x, t = vol_aucs(d)
        print("%-22s slice=%.3f  volume: mean=%.3f max=%.3f top5=%.3f" % (name, s, m, x, t))


if __name__ == "__main__":
    main()
