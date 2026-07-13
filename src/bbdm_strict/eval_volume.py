"""
体级聚合评估：对切片分类后按 case_id 取均值得到体级预测，算体级 AUC。
支持 direct(直接分类原切片) 与 translate(先 BBDM 翻译再分类)。
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
        # 直接分类用 224 张量；翻译模式另取 128 的 [-1,1]
        x224 = self.tf(im)
        a = np.asarray(im.resize((self.size, self.size)), dtype=np.float32) / 255.0 * 2 - 1
        x128 = torch.from_numpy(a)[None]
        return x224, x128, int(float(r["label"])), str(r["case_id"])


def rev_indices(n, T_):
    raw = torch.linspace(T_, 0, steps=max(n, 2)).round().long().tolist()
    idx = []
    for v in raw:
        iv = int(max(0, min(T_, v)))
        if not idx or idx[-1] != iv: idx.append(iv)
    if idx[0] != T_: idx = [T_] + idx
    if idx[-1] != 0: idx.append(0)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--slice_csv", required=True)
    ap.add_argument("--mode", choices=["direct", "translate"], required=True)
    ap.add_argument("--bbdm_config"); ap.add_argument("--bbdm_ckpt")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--topk", type=int, default=5, help="top-k 切片平均聚合的 k")
    a = ap.parse_args()
    dev = torch.device("cuda")

    clf = build_model(a.backbone, 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    ae = bridge = model = t_list = None
    if a.mode == "translate":
        from ae_frontend import load_ae_model, ae_encode_latent, ae_decode_latent
        from bridge_scheduler import LinearBrownianBridgeScheduler
        from models_strict_bbdm import StrictBridgeUNet
        cfg = json.load(open(a.bbdm_config, encoding="utf-8"))
        Tt = int(cfg["num_train_timesteps"])
        ae = load_ae_model(cfg["ae_ckpt"], cfg.get("ae_config"), device=dev, freeze=True)
        with torch.no_grad():
            lat = ae_encode_latent(ae, torch.zeros((1,1,a.image_size,a.image_size), device=dev))["latent"]
        cin, ss = int(lat.shape[1]), int(lat.shape[-1])
        model = StrictBridgeUNet(image_size=ss, base_channels=int(cfg["base_channels"]), in_channels=cin, out_channels=cin)
        model.load_state_dict(torch.load(a.bbdm_ckpt, map_location=dev, weights_only=False)["model"]); model.to(dev).eval()
        bridge = LinearBrownianBridgeScheduler(num_steps=Tt, bridge_sigma=float(cfg["bridge_sigma"]))
        t_list = rev_indices(a.num_inference_steps, Tt)
        globals().update(ae_encode_latent=ae_encode_latent, ae_decode_latent=ae_decode_latent)

    @torch.no_grad()
    def reverse(z):
        x = z.clone(); B = x.shape[0]
        for i, t in enumerate(t_list[:-1]):
            s = int(t_list[i+1]); tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
            pred = model(x, tt)
            if s == 0:
                x = bridge.recover_xa_from_bridge_target(x_t=x, bridge_target_hat=pred); break
            st = torch.full((B,), s, device=dev, dtype=torch.long)
            m, _ = bridge.reverse_mean_variance_from_bridge_target(x_t=x, x_b=z, bridge_target_hat=pred, t_index=tt, s_index=st)
            x = m
        return x

    dl = DataLoader(SliceCSV(a.slice_csv, a.image_size), batch_size=a.batch_size, shuffle=False, num_workers=6)
    probs, labels, cases = [], [], []
    with torch.no_grad():
        for x224, x128, y, cid in dl:
            if a.mode == "direct":
                xin = x224.to(dev)
            else:
                z = ae_encode_latent(ae, x128.to(dev))["latent"]
                rec = ae_decode_latent(ae, reverse(z))["recon"]
                xin = TF.resize(rec, [224, 224], antialias=True).repeat(1, 3, 1, 1)
            p = torch.softmax(clf(xin), 1)[:, 1].cpu().numpy()
            probs += list(p); labels += list(y.numpy()); cases += list(cid)
    df = pd.DataFrame({"case": cases, "p": probs, "y": labels})
    slice_auc = auc(df.y.values, df.p.values)
    # 保存每张切片分数，便于以后换聚合方式而不必重跑分类
    out_csv = a.slice_csv.rsplit("/", 1)[0] + f"/slice_probs_{a.mode}.csv"
    df.to_csv(out_csv, index=False)

    def topk_mean(s, k):
        s = np.sort(s.values)[::-1]
        return float(s[:max(1, min(k, len(s)))].mean())

    gmean = df.groupby("case").agg(p=("p", "mean"), y=("y", "first"))
    gmax = df.groupby("case").agg(p=("p", "max"), y=("y", "first"))
    gtk = df.groupby("case").apply(lambda d: pd.Series({"p": topk_mean(d.p, a.topk), "y": d.y.iloc[0]}))
    name = a.slice_csv.split("/")[-2]
    print(f"[{a.mode}] {name}  n_vol={len(gmean)}  slice_AUC={slice_auc:.3f}")
    print(f"    volume_AUC:  mean={auc(gmean.y.values, gmean.p.values):.3f}  "
          f"max={auc(gmax.y.values, gmax.p.values):.3f}  top{a.topk}={auc(gtk.y.values, gtk.p.values):.3f}")
    print(f"    (每切片分数已存: {out_csv})")


if __name__ == "__main__":
    main()
