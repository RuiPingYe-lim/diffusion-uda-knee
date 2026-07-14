"""
moment_self 诊断的 Oracle 评估:在完全相同的 128 / VAE / BBDM 管线上,
一次算出四个参照点的目标域体级 AUC + BBDM 到端点的保真度。

四个参照点(目标 KneeMRI test):
  1. direct_128    : 目标原图(128->224)直接分类
  2. moment_oracle : 目标图矩匹配到源域风格(128->224),不过 VAE
  3. moment_vae    : 上者再经 VAE 编解码(BBDM 在该端点上的可达上限)
  4. moment_bbdm   : 完整 BBDM 输出(encode(target)->反向桥->decode)

正确判读(而非直接对比历史 0.795):
  moment_bbdm ≈ moment_vae            -> 桥已到达端点; 之前的失败是随机配对/损失冲突
  moment_bbdm << moment_vae           -> 桥实现仍有问题(latent scale / t=T / 反向采样)
  moment_vae  << moment_oracle        -> 信息损失来自 VAE / 128 分辨率
  moment_bbdm 追平 moment_vae 但≈direct -> 翻译不额外提供判别信息, 转向源->目标增强

保真度(BBDM 解码输出 vs 矩匹配端点, [0,1] 128):MAE、|mean 差|、|std 差|。

用法:
  python eval_moment_self_oracle.py --clf_ckpt <cls.pt> \
     --slice_csv <target_test.csv> --source_csv <source_train.csv> \
     --bbdm_config <moment_self_pure.json> --bbdm_ckpt <bbdm latest.pt>
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model
from ae_frontend import load_ae_model, ae_encode_latent, ae_decode_latent
from bridge_scheduler import LinearBrownianBridgeScheduler
from models_strict_bbdm import StrictBridgeUNet
from source_style_stats import get_source_stats, moment_match


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


def rev_indices(n, T_):
    raw = torch.linspace(T_, 0, steps=max(n, 2)).round().long().tolist()
    idx = []
    for v in raw:
        iv = int(max(0, min(T_, v)))
        if not idx or idx[-1] != iv: idx.append(iv)
    if idx[0] != T_: idx = [T_] + idx
    if idx[-1] != 0: idx.append(0)
    return idx


class DS(Dataset):
    def __init__(self, csv, size, smean, sstd):
        self.df = pd.read_csv(csv)
        self.df = self.df[pd.to_numeric(self.df["label"], errors="coerce").isin([0, 1])].reset_index(drop=True)
        self.size = size; self.sm = smean; self.ss = sstd

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = np.asarray(Image.open(r["image_path"]).convert("L").resize((self.size, self.size)),
                       dtype=np.float32) / 255.0
        gm = moment_match(g, self.sm, self.ss)
        # tensors in [-1,1], shape [1,H,W]
        xb = torch.from_numpy(g)[None] * 2 - 1        # target original
        xm = torch.from_numpy(gm)[None] * 2 - 1       # moment-matched
        return xb, xm, int(float(r["label"])), str(r["case_id"])


def to224(x_m1p1):  # [-1,1] [B,1,H,W] -> classifier input [B,3,224,224]
    return TF.resize(x_m1p1, [224, 224], antialias=True).repeat(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--slice_csv", required=True)
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--bbdm_config", required=True)
    ap.add_argument("--bbdm_ckpt", required=True)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_ref", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = json.load(open(a.bbdm_config, encoding="utf-8"))
    T = int(cfg["num_train_timesteps"])

    smean, sstd = get_source_stats(a.source_csv, image_size=a.image_size, n=a.n_ref, seed=a.seed)
    print("source stats (shared): mean=%.4f std=%.4f" % (smean, sstd))

    clf = build_model(a.backbone, 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False); clf.to(dev).eval()

    ae = load_ae_model(cfg["ae_ckpt"], cfg.get("ae_config"), device=dev, freeze=True)
    with torch.no_grad():
        lat = ae_encode_latent(ae, torch.zeros((1, 1, a.image_size, a.image_size), device=dev))["latent"]
    cin, ss = int(lat.shape[1]), int(lat.shape[-1])
    model = StrictBridgeUNet(image_size=ss, base_channels=int(cfg["base_channels"]), in_channels=cin, out_channels=cin)
    model.load_state_dict(torch.load(a.bbdm_ckpt, map_location=dev, weights_only=False)["model"]); model.to(dev).eval()
    bridge = LinearBrownianBridgeScheduler(num_steps=T, bridge_sigma=float(cfg["bridge_sigma"]))
    t_list = rev_indices(a.num_inference_steps, T)

    @torch.no_grad()
    def reverse(z):
        x = z.clone(); B = x.shape[0]
        for i, t in enumerate(t_list[:-1]):
            s = int(t_list[i + 1]); tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
            pred = model(x, tt)
            if s == 0:
                x = bridge.recover_xa_from_bridge_target(x_t=x, bridge_target_hat=pred); break
            st = torch.full((B,), s, device=dev, dtype=torch.long)
            m, _ = bridge.reverse_mean_variance_from_bridge_target(x_t=x, x_b=z, bridge_target_hat=pred, t_index=tt, s_index=st)
            x = m
        return x

    dl = DataLoader(DS(a.slice_csv, a.image_size, smean, sstd), batch_size=a.batch_size, shuffle=False, num_workers=6)
    P = {k: [] for k in ["direct_128", "moment_oracle", "moment_vae", "moment_bbdm"]}
    Y, C = [], []
    fid_mae, fid_dmu, fid_dsd = [], [], []
    with torch.no_grad():
        for xb, xm, y, c in dl:
            xb, xm = xb.to(dev), xm.to(dev)
            P["direct_128"] += list(torch.softmax(clf(to224(xb)), 1)[:, 1].cpu().numpy())
            P["moment_oracle"] += list(torch.softmax(clf(to224(xm)), 1)[:, 1].cpu().numpy())
            rec_m = ae_decode_latent(ae, ae_encode_latent(ae, xm)["latent"])["recon"]
            P["moment_vae"] += list(torch.softmax(clf(to224(rec_m)), 1)[:, 1].cpu().numpy())
            rec_b = ae_decode_latent(ae, reverse(ae_encode_latent(ae, xb)["latent"]))["recon"]
            P["moment_bbdm"] += list(torch.softmax(clf(to224(rec_b)), 1)[:, 1].cpu().numpy())
            # fidelity: BBDM decoded vs moment-matched endpoint, both -> [0,1]
            b01 = (rec_b.clamp(-1, 1) + 1) / 2
            m01 = (xm + 1) / 2
            fid_mae += list((b01 - m01).abs().mean(dim=(1, 2, 3)).cpu().numpy())
            fid_dmu += list((b01.mean(dim=(1, 2, 3)) - m01.mean(dim=(1, 2, 3))).abs().cpu().numpy())
            fid_dsd += list((b01.std(dim=(1, 2, 3)) - m01.std(dim=(1, 2, 3))).abs().cpu().numpy())
            Y += list(y.numpy()); C += list(c)

    print("\n===== 体级 AUC (mean 聚合, 目标 KneeMRI test) =====")
    for k in ["direct_128", "moment_oracle", "moment_vae", "moment_bbdm"]:
        print("  %-14s %.3f" % (k, vol_mean_auc(C, P[k], Y)))
    print("\n===== BBDM 到端点的保真度 (越小越好) =====")
    print("  MAE(bbdm, moment)      = %.4f" % float(np.mean(fid_mae)))
    print("  |mean(bbdm)-mean(mom)| = %.4f" % float(np.mean(fid_dmu)))
    print("  |std(bbdm)-std(mom)|   = %.4f" % float(np.mean(fid_dsd)))
    print("\n判读: 比较 moment_bbdm 与 moment_vae(不是历史 0.795)。")


if __name__ == "__main__":
    main()
