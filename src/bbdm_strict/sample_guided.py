"""
Classifier-guided BBDM sampling (latent).

Reverse bridge from target latent -> source-style latent, but at each step nudge
the latent so the DECODED image keeps the discriminative content of the ORIGINAL
target image (measured by a frozen source-classifier feature map). This fights the
over-smoothing that erases lesions in plain BBDM translation.

Run with PYTHONPATH=<code2 root>, cwd anywhere. Self-contained.
"""
import argparse, sys, os, json
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from ae_frontend import load_ae_model, ae_encode_latent, ae_decode_latent
from bridge_scheduler import LinearBrownianBridgeScheduler
from datasets_strict_bbdm import SingleDomainSliceDataset
from models_strict_bbdm import StrictBridgeUNet
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model


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


class FeatHook:
    """Grab one intermediate conv feature map from the classifier."""
    def __init__(self, model, layer_name):
        self.model = model
        self.feat = None
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self._hook)
    def _hook(self, m, i, o):
        self.feat = o[0] if isinstance(o, (tuple, list)) else o
    def __call__(self, img_1ch):
        # img_1ch in [-1,1], [B,1,H,W]; classifier wants [B,3,224,224] (already in [-1,1] = Normalize(0.5))
        x = TF.resize(img_1ch, [224, 224], antialias=True).repeat(1, 3, 1, 1)
        _ = self.model(x)
        return self.feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--guidance_scale", type=float, default=0.5)
    ap.add_argument("--feat_layer", type=str, default="stem.5")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    dev = torch.device("cuda")
    T = int(cfg["num_train_timesteps"])

    # --- AE (KL-VAE) ---
    ae = load_ae_model(cfg["ae_ckpt"], cfg.get("ae_config"), device=dev, freeze=True)

    # --- bridge model ---
    with torch.no_grad():
        dummy = torch.zeros((1, 1, int(cfg["image_size"]), int(cfg["image_size"])), device=dev)
        lat = ae_encode_latent(ae, dummy)["latent"]
    cin = int(lat.shape[1]); ssize = int(lat.shape[-1])
    model = StrictBridgeUNet(image_size=ssize, base_channels=int(cfg["base_channels"]), in_channels=cin, out_channels=cin)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.to(dev).eval()

    # --- frozen source classifier (feature extractor) ---
    clf = build_model("custom_resnet50_space", 2, "none", dev)
    cck = torch.load(args.clf_ckpt, map_location=dev, weights_only=False)
    sd = cck["state_dict"] if "state_dict" in cck else cck
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    clf.to(dev).eval()
    for p in clf.parameters(): p.requires_grad_(False)
    feat = FeatHook(clf, args.feat_layer)

    bridge = LinearBrownianBridgeScheduler(num_steps=T, bridge_sigma=float(cfg["bridge_sigma"]))
    t_list = build_reverse_indices(args.num_inference_steps, T)

    ds = SingleDomainSliceDataset(csv_path=cfg["target_test_csv"], image_size=int(cfg["image_size"]), root_dir=cfg.get("target_test_root"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    bdir = os.path.join(args.out_dir, "before"); tdir = os.path.join(args.out_dir, "translated")
    os.makedirs(bdir, exist_ok=True); os.makedirs(tdir, exist_ok=True)

    def to_png(t, path):
        a = ((t.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).squeeze(0).numpy()
        Image.fromarray(a, mode="L").save(path)

    rows = []; counter = 0
    for batch in loader:
        x_b_img = batch["image"].to(dev)
        labels = batch["label"].tolist()
        with torch.no_grad():
            x_b = ae_encode_latent(ae, x_b_img)["latent"]
            f_orig = feat(ae_decode_latent(ae, x_b)["recon"]).detach()
        x_cur = x_b.clone()
        bsz = x_b.shape[0]
        for i, t in enumerate(t_list[:-1]):
            s = int(t_list[i + 1])
            tt = torch.full((bsz,), int(t), device=dev, dtype=torch.long)
            with torch.no_grad():
                pred = model(x_cur, tt)
                if s == 0:
                    x_next = bridge.recover_xa_from_bridge_target(x_t=x_cur, bridge_target_hat=pred)
                else:
                    st = torch.full((bsz,), s, device=dev, dtype=torch.long)
                    mean, _ = bridge.reverse_mean_variance_from_bridge_target(
                        x_t=x_cur, x_b=x_b, bridge_target_hat=pred, t_index=tt, s_index=st)
                    x_next = mean  # eta=0 deterministic
            # --- classifier-guidance: keep content close to original target ---
            if args.guidance_scale > 0:
                z = x_next.detach().requires_grad_(True)
                with torch.enable_grad():
                    img = ae_decode_latent(ae, z)["recon"]
                    loss = F.mse_loss(feat(img), f_orig)
                    g = torch.autograd.grad(loss, z)[0]
                g = g / (g.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1))
                x_next = x_next - args.guidance_scale * g
            x_cur = x_next
        with torch.no_grad():
            translated = ae_decode_latent(ae, x_cur)["recon"]
        for j in range(bsz):
            sid = f"{counter:05d}"
            to_png(x_b_img[j], os.path.join(bdir, sid + ".png"))
            to_png(translated[j], os.path.join(tdir, sid + ".png"))
            rows.append({"before_png": os.path.join(bdir, sid + ".png"),
                         "translated_png": os.path.join(tdir, sid + ".png"),
                         "target_label": int(labels[j])})
            counter += 1
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "sample_pairs.csv"), index=False)
    print(f"guided sampling done: {counter} samples, scale={args.guidance_scale}, layer={args.feat_layer}")
    print("out:", args.out_dir)


if __name__ == "__main__":
    main()
