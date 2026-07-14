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
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--equiv_margin", type=float, default=0.02, help="|Δbridge| CI within this = bbdm≈vae")
    ap.add_argument("--out_csv", default="", help="save per-case probs + fidelity")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = json.load(open(a.bbdm_config, encoding="utf-8"))
    T = int(cfg["num_train_timesteps"])

    smean, sstd = get_source_stats(a.source_csv, image_size=a.image_size, n=a.n_ref, seed=a.seed)
    print("source stats (shared): mean=%.4f std=%.4f" % (smean, sstd))

    clf = build_model(a.backbone, 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    res = clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    # fail-fast: only the (disabled) rsa module keys may be ignored; anything else means the
    # classifier head/backbone did not load -> AUCs would be meaningless.
    bad_missing = [k for k in res.missing_keys if "rsa" not in k.lower()]
    bad_unexpected = [k for k in res.unexpected_keys if "rsa" not in k.lower()]
    print("[clf load] missing=%d unexpected=%d (rsa keys ignored)" % (len(res.missing_keys), len(res.unexpected_keys)))
    if bad_missing or bad_unexpected:
        raise RuntimeError("classifier weights did not load cleanly: "
                           f"missing(non-rsa)={bad_missing[:8]} unexpected(non-rsa)={bad_unexpected[:8]}")
    clf.to(dev).eval()

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
    # fidelity decomposed: pure-bridge (latent & image) vs VAE vs total
    fid_bridge_lat, fid_bridge_img, fid_vae_img, fid_total_img = [], [], [], []
    with torch.no_grad():
        for xb, xm, y, c in dl:
            xb, xm = xb.to(dev), xm.to(dev)
            P["direct_128"] += list(torch.softmax(clf(to224(xb)), 1)[:, 1].cpu().numpy())
            P["moment_oracle"] += list(torch.softmax(clf(to224(xm)), 1)[:, 1].cpu().numpy())
            z_moment = ae_encode_latent(ae, xm)["latent"]              # target moment endpoint latent
            rec_m = ae_decode_latent(ae, z_moment)["recon"]            # moment through VAE (oracle)
            P["moment_vae"] += list(torch.softmax(clf(to224(rec_m)), 1)[:, 1].cpu().numpy())
            z_bbdm = reverse(ae_encode_latent(ae, xb)["latent"])       # BBDM reverse-bridge latent
            rec_b = ae_decode_latent(ae, z_bbdm)["recon"]              # BBDM output
            P["moment_bbdm"] += list(torch.softmax(clf(to224(rec_b)), 1)[:, 1].cpu().numpy())
            # ---- fidelity, decomposed (isolate bridge from VAE) ----
            B = xb.shape[0]
            zb, zm = z_bbdm.reshape(B, -1), z_moment.reshape(B, -1)
            fid_bridge_lat += list(((zb - zm).norm(dim=1) / (zm.norm(dim=1) + 1e-8)).cpu().numpy())
            b01, m01, v01 = (rec_b.clamp(-1, 1) + 1) / 2, (xm + 1) / 2, (rec_m.clamp(-1, 1) + 1) / 2
            fid_bridge_img += list((b01 - v01).abs().mean(dim=(1, 2, 3)).cpu().numpy())   # rec_b vs rec_m (both VAE'd)
            fid_vae_img += list((v01 - m01).abs().mean(dim=(1, 2, 3)).cpu().numpy())      # VAE error alone
            fid_total_img += list((b01 - m01).abs().mean(dim=(1, 2, 3)).cpu().numpy())    # total
            Y += list(y.numpy()); C += list(c)

    # ---- per-case aggregation (mean over slices) ----
    df = pd.DataFrame({"case": C, "y": Y,
                       "p_direct": P["direct_128"], "p_moment": P["moment_oracle"],
                       "p_moment_vae": P["moment_vae"], "p_moment_bbdm": P["moment_bbdm"],
                       "bridge_latent_rel": fid_bridge_lat, "bridge_image_mae": fid_bridge_img,
                       "vae_image_mae": fid_vae_img, "total_image_mae": fid_total_img})
    g = df.groupby("case").agg(
        y=("y", "first"),
        p_direct=("p_direct", "mean"), p_moment=("p_moment", "mean"),
        p_moment_vae=("p_moment_vae", "mean"), p_moment_bbdm=("p_moment_bbdm", "mean"),
        bridge_latent_rel=("bridge_latent_rel", "mean"), bridge_image_mae=("bridge_image_mae", "mean"),
        vae_image_mae=("vae_image_mae", "mean"), total_image_mae=("total_image_mae", "mean"),
    ).reset_index()
    yv = g["y"].values
    cols = {"direct_128": "p_direct", "moment_oracle": "p_moment",
            "moment_vae": "p_moment_vae", "moment_bbdm": "p_moment_bbdm"}
    base = {k: auc(yv, g[v].values) for k, v in cols.items()}

    # ---- PAIRED case-level bootstrap (shared indices; skip single-class resamples) ----
    rng = np.random.RandomState(a.seed); n = len(g)
    boot = {k: [] for k in cols}
    d_bridge, d_vae, d_gain = [], [], []
    kept = 0
    for _ in range(a.n_boot):
        idx = rng.randint(0, n, n)
        yb = yv[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        kept += 1
        av = {k: auc(yb, g[v].values[idx]) for k, v in cols.items()}
        for k in cols:
            boot[k].append(av[k])
        d_bridge.append(av["moment_bbdm"] - av["moment_vae"])
        d_vae.append(av["moment_vae"] - av["moment_oracle"])
        d_gain.append(av["moment_bbdm"] - av["direct_128"])

    def ci(x):
        x = np.asarray(x, dtype=float)
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    print("\n===== 体级 AUC (mean 聚合, n_vol=%d, %d/%d bootstrap 有效) =====" % (n, kept, a.n_boot))
    for k in cols:
        lo, hi = ci(boot[k])
        print("  %-14s %.3f  [%.3f, %.3f]" % (k, base[k], lo, hi))

    print("\n===== 配对 ΔAUC (同一 bootstrap 重采样, 95% CI) =====")
    for name, d, desc in [("Δbridge = bbdm-vae", d_bridge, "桥是否到达 VAE 端点"),
                          ("Δvae    = vae-oracle", d_vae, "VAE 造成的损失"),
                          ("Δgain   = bbdm-direct", d_gain, "整体相对直接迁移")]:
        lo, hi = ci(d)
        print("  %-22s %+.3f  [%+.3f, %+.3f]   %s" % (name, float(np.mean(d)), lo, hi, desc))
    lo_b, hi_b = ci(d_bridge)
    equiv = (lo_b >= -a.equiv_margin) and (hi_b <= a.equiv_margin)
    print("  等价判定 (|Δbridge| 95%%CI ⊂ [-%.2f, %.2f]): %s"
          % (a.equiv_margin, a.equiv_margin, "是 -> 桥≈VAE端点" if equiv else "否"))

    print("\n===== 保真度 (拆开桥 vs VAE, 病例级均值, 越小越好) =====")
    print("  bridge_latent_rel  (z_bbdm vs z_moment) = %.4f   <- 纯桥 latent 误差(最关键)" % float(g["bridge_latent_rel"].mean()))
    print("  bridge_image_MAE   (rec_b vs rec_m)      = %.4f   <- 纯桥图像误差(去 VAE)" % float(g["bridge_image_mae"].mean()))
    print("  vae_image_MAE      (rec_m vs moment)     = %.4f   <- 单独 VAE 误差" % float(g["vae_image_mae"].mean()))
    print("  total_image_MAE    (rec_b vs moment)     = %.4f   <- 总误差" % float(g["total_image_mae"].mean()))

    if a.out_csv:
        g.to_csv(a.out_csv, index=False)
        print("\n病例级结果已保存:", a.out_csv)
    print("\n判读: 用 Δbridge 的 CI(不是历史 0.795); Δbridge≈0 且 bridge_latent_rel 小 -> 桥到达端点。")


if __name__ == "__main__":
    main()
