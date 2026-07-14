"""
为融合分类器(#5)生成多采样配对数据。

对输入 csv 里每张图，用训练好的 BBDM(潜空间)生成：
  - before     : 原图（直接拷贝/重存）
  - translated : 确定性翻译(eta=0)
  - sample_0..sample_{K-1} : K 张随机采样翻译(eta>0, 每次不同)
输出多列 csv： before_png, translated_png, sample_0..sample_{K-1}, label

源域(BUSI/MRNet train,val 带标签) -> 训练融合分类器
目标域(BrEaST/KneeMRI test)       -> 评估
两边用同一个 BBDM、同样的处理。

⚠️ 已知问题 (C5, 分布不对称):
  BBDM 是 target->source 桥, 训练时反向起点 x_B = 目标域 latent。
  这里对源域图也用同一个反向桥来生成"融合训练视图", 但源图作为反向起点
  属于 off-distribution -> 若 BBDM 用随机端点(label_random)训练, 源域"翻译视图"
  会退化/异常, 与测试时真正的目标->源翻译分布不同, 融合分类器于是学会忽略翻译视图
  (这正是"融合提升几乎全来自原图"的原因)。
  缓解: 必须搭配 pair_mode='moment_self' 训练的 CONTENT-PRESERVING 桥(见
  datasets_strict_bbdm.py), 它学的是"保内容、只改风格"的映射, 对源图近似恒等、
  在分布内。根本解决需改为"源->目标 任务保持增强"(训练/测试同分布)。

用法：
  python gen_fusion_pairs.py --config <bbdm_cfg.json> --checkpoint <bbdm latest.pt> \
    --input_csv <带image_path,label的csv> --out_dir <输出目录> \
    --num_samples 3 --reverse_eta 0.35
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader

from ae_frontend import load_ae_model, ae_encode_latent, ae_decode_latent
from bridge_scheduler import LinearBrownianBridgeScheduler
from datasets_strict_bbdm import SingleDomainSliceDataset
from models_strict_bbdm import StrictBridgeUNet


def build_reverse_indices(num_steps, T):
    raw = torch.linspace(T, 0, steps=max(num_steps, 2)).round().long().tolist()
    idx = []
    for v in raw:
        iv = int(max(0, min(T, v)))
        if not idx or idx[-1] != iv:
            idx.append(iv)
    if idx[0] != T: idx = [T] + idx
    if idx[-1] != 0: idx.append(0)
    return idx


@torch.no_grad()
def reverse_sample(model, bridge, x_b, t_list, eta):
    bsz = x_b.shape[0]
    x = x_b.clone()
    for i, t in enumerate(t_list[:-1]):
        s = int(t_list[i + 1])
        tt = torch.full((bsz,), int(t), device=x_b.device, dtype=torch.long)
        pred = model(x, tt)
        if s == 0:
            x = bridge.recover_xa_from_bridge_target(x_t=x, bridge_target_hat=pred)
            break
        st = torch.full((bsz,), s, device=x_b.device, dtype=torch.long)
        if eta > 0:
            noise = torch.randn_like(x)
            x = bridge.step_stochastic_from_bridge_target(x_t=x, x_b=x_b, bridge_target_hat=pred,
                                                          t_index=tt, s_index=st, noise=noise, eta=eta)
        else:
            mean, _ = bridge.reverse_mean_variance_from_bridge_target(
                x_t=x, x_b=x_b, bridge_target_hat=pred, t_index=tt, s_index=st)
            x = mean
    return x


def save_png(t, path):
    a = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
    Image.fromarray(a, mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=3, help="K 张随机采样")
    ap.add_argument("--reverse_eta", type=float, default=0.35)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import json
    cfg = json.load(open(a.config, encoding="utf-8"))
    dev = torch.device("cuda")
    T = int(cfg["num_train_timesteps"])
    torch.manual_seed(a.seed)

    ae = load_ae_model(cfg["ae_ckpt"], cfg.get("ae_config"), device=dev, freeze=True)
    with torch.no_grad():
        dummy = torch.zeros((1, 1, int(cfg["image_size"]), int(cfg["image_size"])), device=dev)
        lat = ae_encode_latent(ae, dummy)["latent"]
    cin, ssize = int(lat.shape[1]), int(lat.shape[-1])
    model = StrictBridgeUNet(image_size=ssize, base_channels=int(cfg["base_channels"]), in_channels=cin, out_channels=cin)
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    bridge = LinearBrownianBridgeScheduler(num_steps=T, bridge_sigma=float(cfg["bridge_sigma"]))
    t_list = build_reverse_indices(int(cfg.get("num_inference_steps", 50)), T)

    ds = SingleDomainSliceDataset(csv_path=a.input_csv, image_size=int(cfg["image_size"]), root_dir=None)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=2)

    dirs = {k: os.path.join(a.out_dir, k) for k in ["before", "translated"] + [f"sample_{i}" for i in range(a.num_samples)]}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    rows, c = [], 0
    with torch.no_grad():
        for batch in loader:
            x_img = batch["image"].to(dev)
            labels = batch["label"].tolist()
            x_b = ae_encode_latent(ae, x_img)["latent"]
            # 确定性 + K 随机
            variants = {"translated": reverse_sample(model, bridge, x_b, t_list, 0.0)}
            for i in range(a.num_samples):
                variants[f"sample_{i}"] = reverse_sample(model, bridge, x_b, t_list, a.reverse_eta)
            dec = {k: ae_decode_latent(ae, v)["recon"] for k, v in variants.items()}
            for j in range(x_img.shape[0]):
                sid = f"{c:06d}"
                row = {"label": int(labels[j])}
                save_png(x_img[j], os.path.join(dirs["before"], sid + ".png"))
                row["before_png"] = os.path.join(dirs["before"], sid + ".png")
                for k, im in dec.items():
                    p = os.path.join(dirs[k], sid + ".png"); save_png(im[j], p); row[k + "_png" if k == "translated" else k] = p
                rows.append(row); c += 1
    cols = ["before_png", "translated_png"] + [f"sample_{i}" for i in range(a.num_samples)] + ["label"]
    # 按行号补 case_id（loader shuffle=False，顺序一致），供后续体级聚合
    src_df = pd.read_csv(a.input_csv)
    if "case_id" in src_df.columns and len(src_df) == len(rows):
        for k in range(len(rows)):
            rows[k]["case_id"] = str(src_df["case_id"].iloc[k])
        cols = cols + ["case_id"]
    pd.DataFrame(rows)[cols].to_csv(os.path.join(a.out_dir, "fusion_pairs.csv"), index=False)
    print(f"done: {c} 张, 每张 before + translated + {a.num_samples} samples")
    print("csv:", os.path.join(a.out_dir, "fusion_pairs.csv"))


if __name__ == "__main__":
    main()
