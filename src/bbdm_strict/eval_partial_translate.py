"""
部分翻译 t0 扫描：反向桥接从中间时刻 t0 起步(而非 t=T)，
t0 越小越保留原内容。t0=0 即纯VAE重建(上限锚)，t0=T 即全翻译(下限锚)。
对每个 t0 算目标测试集体级 AUC，找"改够风格又不毁细节"的甜点。
"""
import argparse, json, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model
from ae_frontend import load_ae_model, ae_encode_latent, ae_decode_latent
from bridge_scheduler import LinearBrownianBridgeScheduler
from models_strict_bbdm import StrictBridgeUNet


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


def rev_indices(n, T_):
    raw = torch.linspace(T_, 0, steps=max(n, 2)).round().long().tolist()
    idx = []
    for v in raw:
        iv = int(max(0, min(T_, v)))
        if not idx or idx[-1] != iv: idx.append(iv)
    if idx[0] != T_: idx = [T_] + idx
    if idx[-1] != 0: idx.append(0)
    return idx


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
        a = np.asarray(im.resize((self.size, self.size)), dtype=np.float32) / 255.0 * 2 - 1
        return torch.from_numpy(a)[None], int(float(r["label"])), str(r["case_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--slice_csv", required=True)
    ap.add_argument("--bbdm_config", required=True)
    ap.add_argument("--bbdm_ckpt", required=True)
    ap.add_argument("--start_ts", default="0,100,200,300,400,600,800,1000")
    ap.add_argument("--nsteps", type=int, default=25)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = json.load(open(a.bbdm_config, encoding="utf-8"))
    Tt = int(cfg["num_train_timesteps"])

    clf = build_model("custom_resnet50_space", 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    ae = load_ae_model(cfg["ae_ckpt"], cfg.get("ae_config"), device=dev, freeze=True)
    with torch.no_grad():
        lat = ae_encode_latent(ae, torch.zeros((1, 1, a.image_size, a.image_size), device=dev))["latent"]
    cin, ss = int(lat.shape[1]), int(lat.shape[-1])
    model = StrictBridgeUNet(image_size=ss, base_channels=int(cfg["base_channels"]), in_channels=cin, out_channels=cin)
    model.load_state_dict(torch.load(a.bbdm_ckpt, map_location=dev, weights_only=False)["model"]); model.to(dev).eval()
    bridge = LinearBrownianBridgeScheduler(num_steps=Tt, bridge_sigma=float(cfg["bridge_sigma"]))

    @torch.no_grad()
    def reverse(z, start_t):
        if start_t <= 0:
            return z
        tl = rev_indices(a.nsteps, int(start_t))
        x = z.clone(); B = x.shape[0]
        for i, t in enumerate(tl[:-1]):
            s = int(tl[i + 1]); tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
            pred = model(x, tt)
            if s == 0:
                x = bridge.recover_xa_from_bridge_target(x_t=x, bridge_target_hat=pred); break
            st = torch.full((B,), s, device=dev, dtype=torch.long)
            m, _ = bridge.reverse_mean_variance_from_bridge_target(x_t=x, x_b=z, bridge_target_hat=pred, t_index=tt, s_index=st)
            x = m
        return x

    ds = SliceCSV(a.slice_csv, a.image_size)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=6)
    start_ts = [int(v) for v in a.start_ts.split(",")]

    # 预先缓存编码，避免每个 t0 重复 encode
    print("caching encodings + labels ...")
    Z, Y, C = [], [], []
    with torch.no_grad():
        for x128, y, cid in dl:
            Z.append(ae_encode_latent(ae, x128.to(dev))["latent"].cpu())
            Y += list(y.numpy()); C += list(cid)
    print("n_slices=%d  n_vol=%d" % (len(Y), len(set(C))))

    print("\n%-8s %-7s %-7s %-7s %-7s" % ("t0", "slice", "mean", "max", "top5"))
    for st in start_ts:
        probs = []
        with torch.no_grad():
            for zc in Z:
                z = zc.to(dev)
                rec = ae_decode_latent(ae, reverse(z, st))["recon"]
                xin = TF.resize(rec, [224, 224], antialias=True).repeat(1, 3, 1, 1)
                probs += list(torch.softmax(clf(xin), 1)[:, 1].cpu().numpy())
        d = pd.DataFrame({"case": C, "p": probs, "y": Y})
        s, m, x, t = vol_aucs(d)
        tag = "  <recon" if st == 0 else ("  <full" if st >= Tt else "")
        print("%-8d %-7.3f %-7.3f %-7.3f %-7.3f%s" % (st, s, m, x, t, tag))


if __name__ == "__main__":
    main()
