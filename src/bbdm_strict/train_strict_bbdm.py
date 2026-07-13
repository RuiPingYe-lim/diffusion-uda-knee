from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ae_frontend import ae_decode_latent, ae_encode_features, ae_encode_latent, ae_reconstruct, load_ae_model
from bridge_scheduler import LinearBrownianBridgeScheduler
from contrastive import SupConLoss
from datasets_strict_bbdm import StrictBBDMPairedDataset
from models_strict_bbdm import StrictBridgeUNet

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model, load_weights
from idea2_diffusion_baseline.vqgan_pretrain.models_vqgan import AEUNet


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pseudo-paired strict-style BBDM training")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--source_csv", type=str, default=None)
    ap.add_argument("--target_csv", type=str, default=None)
    ap.add_argument("--source_root", type=str, default=None)
    ap.add_argument("--target_root", type=str, default=None)
    ap.add_argument("--pair_mode", type=str, default=None, choices=["label_random", "paired_csv"])
    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--num_train_timesteps", type=int, default=None)
    ap.add_argument("--bridge_sigma", type=float, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base_channels", type=int, default=None)
    ap.add_argument("--run_root", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--use_pair_cache", type=int, default=None)
    ap.add_argument("--reshuffle_pairs_each_epoch", type=int, default=None)
    ap.add_argument("--loss_weighting", type=str, default=None, choices=["none", "inv_delta"])

    ap.add_argument("--lambda_edge", type=float, default=None)
    ap.add_argument("--lambda_content", type=float, default=None)
    ap.add_argument("--lambda_feat_content", type=float, default=None)

    ap.add_argument("--lambda_feat_src", type=float, default=None)
    ap.add_argument("--lambda_style_src", type=float, default=None)
    ap.add_argument("--lambda_self_l1", type=float, default=None)
    ap.add_argument("--lambda_self_ssim", type=float, default=None)
    ap.add_argument("--lambda_self_edge", type=float, default=None)

    ap.add_argument("--feat_weights", type=str, default=None)
    ap.add_argument("--feat_backbone", type=str, default=None)
    ap.add_argument("--feat_pretrained", type=str, default=None)
    ap.add_argument("--feat_layers", type=str, default=None)

    ap.add_argument("--use_ae_frontend", type=int, default=None)
    ap.add_argument("--ae_ckpt", type=str, default=None)
    ap.add_argument("--ae_config", type=str, default=None)
    ap.add_argument("--ae_freeze", type=int, default=None)
    ap.add_argument("--ae_input_mode", type=str, default=None, choices=["raw", "recon", "latent"])
    ap.add_argument("--lambda_ae_feat", type=float, default=None)
    ap.add_argument("--lambda_ae_edge", type=float, default=None)
    ap.add_argument("--lambda_identity", type=float, default=None)
    ap.add_argument("--lambda_source_recon", type=float, default=None)
    ap.add_argument("--lambda_delta_reg", type=float, default=None)
    ap.add_argument("--save_ae_vis", type=int, default=None)
    ap.add_argument("--lambda_latent_recon", type=float, default=None)
    return ap.parse_args()


def _default_config() -> Dict[str, Any]:
    return {
        "exp_name": "bbdm_kneemri_to_mrnet_strict",
        "source_csv": "/home/mnnu/Standford2KennMRIData1/experiments/idea2_diffusion/cache_mrnet_train/train_0_cached_slices.csv",
        "target_csv": "/home/mnnu/Standford2KennMRIData1/experiments/idea2_diffusion/cache_kneemri_train/train_0_cached_slices.csv",
        "source_root": None,
        "target_root": None,
        "pair_mode": "label_random",
        "image_size": 128,
        "batch_size": 8,
        "epochs": 5,
        "max_steps": 0,
        "lr": 1e-4,
        "num_workers": 2,
        "num_train_timesteps": 1000,
        "bridge_sigma": 1.0,
        "save_every": 1,
        "seed": 42,
        "base_channels": 64,
        "run_root": "/home/mnnu/Standford2KennMRIData1/experiments/bbdm_strict_runs",
        "device": "cuda",
        "use_pair_cache": 1,
        "reshuffle_pairs_each_epoch": 1,
        "loss_weighting": "none",

        "lambda_edge": 0.0,
        "lambda_content": 0.0,
        "lambda_feat_content": 0.0,

        "lambda_feat_src": 0.05,
        "lambda_style_src": 0.10,
        "lambda_self_l1": 1.00,
        "lambda_self_ssim": 0.30,
        "lambda_self_edge": 0.20,

        "feat_weights": None,
        "feat_backbone": "custom_resnet50_space",
        "feat_pretrained": "none",
        "feat_layers": "stem.4",

        "use_ae_frontend": 0,
        "ae_ckpt": "/home/mnnu/Standford2KennMRIData1/experiments/ae_unet_knee_pretrain/checkpoints/best.pt",
        "ae_config": None,
        "ae_freeze": 1,
        "ae_input_mode": "raw",
        "lambda_ae_feat": 0.0,
        "lambda_ae_edge": 0.0,
        "lambda_identity": 0.0,
        "lambda_source_recon": 0.0,
        "lambda_delta_reg": 0.0,
        "save_ae_vis": 0,
        "lambda_latent_recon": 1.0,
    }


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v
    return cfg


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(v)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_serializable(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _sample_train_t_indices(batch_size: int, num_train_timesteps: int, device: torch.device) -> torch.Tensor:
    if num_train_timesteps < 3:
        raise ValueError("num_train_timesteps must be >= 3 for stable pseudo-paired BBDM-style training")
    return torch.randint(low=1, high=num_train_timesteps, size=(batch_size,), device=device, dtype=torch.long)


def _gradient_maps(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    return gx, gy


def _edge_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay = _gradient_maps(a)
    bx, by = _gradient_maps(b)
    return (ax - bx).abs().mean() + (ay - by).abs().mean()


def _ssim_loss(x: torch.Tensor, y: torch.Tensor, window: int = 7) -> torch.Tensor:
    pad = window // 2
    mu_x = F.avg_pool2d(x, kernel_size=window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=window, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(x * x, kernel_size=window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=window, stride=1, padding=pad) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = (num / (den + 1e-6)).clamp(-1.0, 1.0)
    return 1.0 - ssim.mean()


def _tensor_to_u8(x_chw: torch.Tensor) -> np.ndarray:
    x = x_chw.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 0.5 * 255.0).round().to(torch.uint8).squeeze(0).numpy()
    return x


def _edge_map_u8(x_chw: torch.Tensor) -> np.ndarray:
    x = x_chw.detach().float().cpu().unsqueeze(0)
    gx, gy = _gradient_maps(x)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8).squeeze(0)
    mag = mag / (mag.max() + 1e-6)
    return (mag.squeeze(0).numpy() * 255.0).astype(np.uint8)


def _save_ae_vis_panel(path: Path, before: torch.Tensor, ae_recon: torch.Tensor, translated: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b = _tensor_to_u8(before)
    r = _tensor_to_u8(ae_recon)
    t = _tensor_to_u8(translated)
    resid = np.abs(t.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    edge = _edge_map_u8(translated)
    canvas = np.concatenate([b, r, t, resid, edge], axis=1)
    Image.fromarray(canvas, mode="L").save(path)


class FrozenFeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, layer_names: List[str]) -> None:
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.layers = {name: self.model.get_submodule(name) for name in layer_names}
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        handles = []

        for name, layer in self.layers.items():
            def _hook(_m, _inp, out, key=name):
                outputs[key] = out[0] if isinstance(out, (tuple, list)) else out

            handles.append(layer.register_forward_hook(_hook))

        _ = self.model(x)

        for h in handles:
            h.remove()

        return [outputs[n] for n in self.layer_names]


def _to_3ch(x_1ch: torch.Tensor) -> torch.Tensor:
    return x_1ch.repeat(1, 3, 1, 1)


def _feature_instance_l1(
    extractor: FrozenFeatureExtractor,
    x_pred_1ch: torch.Tensor,
    x_src_1ch: torch.Tensor,
) -> torch.Tensor:
    pred_feats = extractor(_to_3ch(x_pred_1ch))
    with torch.no_grad():
        src_feats = extractor(_to_3ch(x_src_1ch))

    losses = []
    for pf, sf in zip(pred_feats, src_feats):
        losses.append(F.l1_loss(pf, sf))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=x_pred_1ch.device)


def _style_stats_loss(
    extractor: FrozenFeatureExtractor,
    x_pred_1ch: torch.Tensor,
    x_src_1ch: torch.Tensor,
) -> torch.Tensor:
    pred_feats = extractor(_to_3ch(x_pred_1ch))
    with torch.no_grad():
        src_feats = extractor(_to_3ch(x_src_1ch))

    losses = []
    for pf, sf in zip(pred_feats, src_feats):
        mu_p = pf.mean(dim=(2, 3))
        mu_s = sf.mean(dim=(2, 3))
        std_p = pf.std(dim=(2, 3), unbiased=False)
        std_s = sf.std(dim=(2, 3), unbiased=False)
        losses.append(F.l1_loss(mu_p, mu_s) + F.l1_loss(std_p, std_s))

    return torch.stack(losses).mean() if losses else torch.zeros((), device=x_pred_1ch.device)


def _resolve_effective_lambdas(cfg: Dict[str, Any]) -> Dict[str, float]:
    lam_feat_src = float(cfg.get("lambda_feat_src", 0.05))
    lam_style_src = float(cfg.get("lambda_style_src", 0.10))
    lam_self_l1 = float(cfg.get("lambda_self_l1", 1.00))
    lam_self_ssim = float(cfg.get("lambda_self_ssim", 0.30))
    lam_self_edge = float(cfg.get("lambda_self_edge", 0.20))

    lam_self_edge += float(cfg.get("lambda_edge", 0.0))
    lam_self_l1 += float(cfg.get("lambda_content", 0.0))
    lam_feat_src += float(cfg.get("lambda_feat_content", 0.0))

    return {
        "lambda_feat_src": lam_feat_src,
        "lambda_style_src": lam_style_src,
        "lambda_self_l1": lam_self_l1,
        "lambda_self_ssim": lam_self_ssim,
        "lambda_self_edge": lam_self_edge,
    }


def _ae_feat_consistency_loss(ae_model: AEUNet, x_pred: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ref = ae_encode_features(ae_model, x_ref)
    pred = ae_encode_features(ae_model, x_pred)

    losses = [F.l1_loss(pred["bottleneck"], ref["bottleneck"]) ]
    p_list = pred.get("encoder_features", [])
    r_list = ref.get("encoder_features", [])
    for pf, rf in zip(p_list, r_list):
        losses.append(F.l1_loss(pf, rf))

    if not losses:
        return torch.zeros((), device=x_pred.device)
    return torch.stack(losses).mean()


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))

    run_dir = Path(cfg["run_root"]) / str(cfg["exp_name"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eff = _resolve_effective_lambdas(cfg)
    cfg.update(eff)

    use_ae_frontend = _as_bool(cfg.get("use_ae_frontend", 0))
    ae_freeze = _as_bool(cfg.get("ae_freeze", 1))
    ae_input_mode = str(cfg.get("ae_input_mode", "raw")).lower()
    save_ae_vis = _as_bool(cfg.get("save_ae_vis", 0))

    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(cfg), f, indent=2, ensure_ascii=False)

    device = _resolve_device(str(cfg["device"]))

    dataset = StrictBBDMPairedDataset(
        source_csv=str(cfg["source_csv"]),
        target_csv=str(cfg["target_csv"]),
        image_size=int(cfg["image_size"]),
        pair_mode=str(cfg["pair_mode"]),
        source_root=cfg.get("source_root"),
        target_root=cfg.get("target_root"),
        seed=int(cfg["seed"]),
        use_pair_cache=bool(int(cfg.get("use_pair_cache", 1))),
        reshuffle_pairs_each_epoch=bool(int(cfg.get("reshuffle_pairs_each_epoch", 1))),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    feat_extractor = None
    need_feat = (cfg["lambda_feat_src"] > 0.0) or (cfg["lambda_style_src"] > 0.0)
    if need_feat:
        feat_weights = cfg.get("feat_weights")
        if not feat_weights:
            raise ValueError("Feature/style source losses require --feat_weights")
        feat_model = build_model(
            backbone=str(cfg.get("feat_backbone", "custom_resnet50_space")),
            num_classes=2,
            pretrained=str(cfg.get("feat_pretrained", "none")),
            device=device,
        )
        load_weights(feat_model, Path(str(feat_weights)), device=device)
        feat_layers = [s.strip() for s in str(cfg.get("feat_layers", "stem.4")).split(",") if s.strip()]
        if not feat_layers:
            raise ValueError("feat_layers is empty")
        feat_extractor = FrozenFeatureExtractor(feat_model, feat_layers).to(device)

    ae_model = None
    if use_ae_frontend:
        ae_ckpt = cfg.get("ae_ckpt")
        if not ae_ckpt:
            raise ValueError("use_ae_frontend=true requires --ae_ckpt")
        ae_model = load_ae_model(
            ae_ckpt=str(ae_ckpt),
            ae_config=cfg.get("ae_config"),
            device=device,
            freeze=ae_freeze,
        )

    bridge_in_channels = 1
    bridge_sample_size = int(cfg["image_size"])
    if use_ae_frontend and ae_input_mode == "latent":
        if ae_model is None:
            raise ValueError("ae_input_mode=latent requires use_ae_frontend=true and valid AE model")
        with torch.no_grad():
            dummy = torch.zeros((1, 1, int(cfg["image_size"]), int(cfg["image_size"])), device=device)
            latent_dummy = ae_encode_latent(ae_model, dummy)["latent"]
        bridge_in_channels = int(latent_dummy.shape[1])
        bridge_sample_size = int(latent_dummy.shape[-1])

    model = StrictBridgeUNet(
        image_size=bridge_sample_size,
        base_channels=int(cfg["base_channels"]),
        in_channels=bridge_in_channels,
        out_channels=bridge_in_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    bridge = LinearBrownianBridgeScheduler(
        num_steps=int(cfg["num_train_timesteps"]), bridge_sigma=float(cfg["bridge_sigma"])
    )
    supcon = SupConLoss(temperature=0.07)  # 建议2：监督对比损失

    max_steps = int(cfg["max_steps"]) if int(cfg["max_steps"]) > 0 else None
    global_step = 0

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,step,loss_total,loss_bb,loss_latent_recon,loss_feat_src,loss_style_src,"
            "loss_self_l1,loss_self_ssim,loss_self_edge,"
            "loss_ae_feat,loss_ae_edge,loss_identity,loss_source_recon,loss_delta_reg,"
            "objective,pair_mode,pair_epoch,ae_input_mode,ae_frozen,use_ae_frontend,note\n"
        )

    vis_dir = run_dir / "ae_vis"
    if save_ae_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch - 1)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            x_a_img = batch["x_A"].to(device)
            x_b_raw = batch["x_B"].to(device)
            bsz = x_a_img.shape[0]

            x_b_ae_recon = x_b_raw
            x_a_bridge = x_a_img
            x_b_bridge = x_b_raw
            x_b_in = x_b_raw
            x_b_lat_feats = None

            if ae_model is not None and ae_input_mode == "latent":
                with torch.no_grad():
                    a_pack = ae_encode_latent(ae_model, x_a_img)
                    b_pack = ae_encode_latent(ae_model, x_b_raw)
                    x_b_ae_recon = ae_reconstruct(ae_model, x_b_raw)["recon"]
                x_a_bridge = a_pack["latent"]
                x_b_bridge = b_pack["latent"]
                x_b_lat_feats = b_pack["encoder_features"]
                x_b_in = x_b_raw
            elif ae_model is not None and ae_input_mode == "recon":
                with torch.no_grad():
                    ae_b = ae_reconstruct(ae_model, x_b_raw)
                x_b_in = ae_b["recon"]
                x_b_ae_recon = ae_b["recon"]
                x_a_bridge = x_a_img
                x_b_bridge = x_b_in
            elif ae_model is not None:
                with torch.no_grad():
                    ae_b = ae_reconstruct(ae_model, x_b_raw)
                x_b_ae_recon = ae_b["recon"]
                x_a_bridge = x_a_img
                x_b_bridge = x_b_raw

            t_idx = _sample_train_t_indices(
                batch_size=bsz,
                num_train_timesteps=int(cfg["num_train_timesteps"]),
                device=device,
            )
            eps = torch.randn_like(x_a_bridge)

            x_t = bridge.sample_xt(x_a=x_a_bridge, x_b=x_b_bridge, t_index=t_idx, noise=eps)
            target_bb = bridge.make_bridge_target(x_a=x_a_bridge, x_b=x_b_bridge, t_index=t_idx, noise=eps)
            pred_bb = model(x_t=x_t, timesteps=t_idx)

            per_pixel = (pred_bb - target_bb) ** 2
            per_sample = per_pixel.mean(dim=(1, 2, 3))
            w = bridge.loss_weight(t_idx, mode=str(cfg.get("loss_weighting", "none"))).to(device)
            loss_bb = (per_sample * w).sum() / torch.clamp(w.sum(), min=1.0)

            x_a_hat_bridge = bridge.recover_xa_from_bridge_target(x_t=x_t, bridge_target_hat=pred_bb)

            # 建议2：对预测的源域风格潜表示按类别做监督对比，保住类别判别结构
            loss_supcon = torch.zeros((), device=device)
            if float(cfg.get("lambda_supcon", 0.0)) > 0.0 and "label" in batch:
                y_sup = batch["label"].to(device)
                feat_sup = x_a_hat_bridge.mean(dim=(2, 3))  # [B, C] 潜表示全局池化
                loss_supcon = supcon(feat_sup, y_sup)

            loss_latent_recon = torch.zeros((), device=device)
            if ae_model is not None and ae_input_mode == "latent":
                loss_latent_recon = F.l1_loss(x_a_hat_bridge, x_a_bridge)
                x_a_hat = ae_decode_latent(ae_model, latent=x_a_hat_bridge, encoder_features=x_b_lat_feats)["recon"]
            else:
                x_a_hat = x_a_hat_bridge

            loss_feat_src = (
                _feature_instance_l1(feat_extractor, x_a_hat, x_a_img)
                if (cfg["lambda_feat_src"] > 0.0 and feat_extractor is not None)
                else torch.zeros((), device=device)
            )
            loss_style_src = (
                _style_stats_loss(feat_extractor, x_a_hat, x_a_img)
                if (cfg["lambda_style_src"] > 0.0 and feat_extractor is not None)
                else torch.zeros((), device=device)
            )

            loss_self_l1 = F.l1_loss(x_a_hat, x_b_in) if cfg["lambda_self_l1"] > 0.0 else torch.zeros((), device=device)
            loss_self_ssim = _ssim_loss(x_a_hat, x_b_in) if cfg["lambda_self_ssim"] > 0.0 else torch.zeros((), device=device)
            loss_self_edge = _edge_l1(x_a_hat, x_b_in) if cfg["lambda_self_edge"] > 0.0 else torch.zeros((), device=device)

            loss_ae_feat = torch.zeros((), device=device)
            loss_ae_edge = torch.zeros((), device=device)
            loss_identity = torch.zeros((), device=device)
            loss_source_recon = torch.zeros((), device=device)
            loss_delta_reg = torch.zeros((), device=device)

            if ae_model is not None:
                if float(cfg.get("lambda_ae_feat", 0.0)) > 0.0:
                    loss_ae_feat = _ae_feat_consistency_loss(ae_model, x_pred=x_a_hat, x_ref=x_b_in)

                if float(cfg.get("lambda_ae_edge", 0.0)) > 0.0:
                    loss_ae_edge = _edge_l1(x_a_hat, x_b_in)

                if float(cfg.get("lambda_identity", 0.0)) > 0.0:
                    loss_identity = F.l1_loss(x_a_hat, x_b_in)

                if float(cfg.get("lambda_source_recon", 0.0)) > 0.0:
                    if ae_input_mode == "latent":
                        with torch.no_grad():
                            x_src_lat = ae_encode_latent(ae_model, x_a_img)["latent"]
                        eps_src = torch.randn_like(x_src_lat)
                        x_t_src = bridge.sample_xt(x_a=x_src_lat, x_b=x_src_lat, t_index=t_idx, noise=eps_src)
                        target_src = bridge.make_bridge_target(x_a=x_src_lat, x_b=x_src_lat, t_index=t_idx, noise=eps_src)
                        pred_src = model(x_t=x_t_src, timesteps=t_idx)
                        x_src_hat = bridge.recover_xa_from_bridge_target(x_t=x_t_src, bridge_target_hat=pred_src)
                        loss_source_recon = F.l1_loss(x_src_hat, x_src_lat) + 0.1 * F.mse_loss(pred_src, target_src)
                    else:
                        if ae_input_mode == "recon":
                            with torch.no_grad():
                                x_src_in = ae_reconstruct(ae_model, x_a_img)["recon"]
                        else:
                            x_src_in = x_a_img
                        eps_src = torch.randn_like(x_a_img)
                        x_t_src = bridge.sample_xt(x_a=x_a_img, x_b=x_src_in, t_index=t_idx, noise=eps_src)
                        target_src = bridge.make_bridge_target(x_a=x_a_img, x_b=x_src_in, t_index=t_idx, noise=eps_src)
                        pred_src = model(x_t=x_t_src, timesteps=t_idx)
                        x_src_hat = bridge.recover_xa_from_bridge_target(x_t=x_t_src, bridge_target_hat=pred_src)
                        loss_source_recon = F.l1_loss(x_src_hat, x_a_img) + 0.1 * F.mse_loss(pred_src, target_src)

                if float(cfg.get("lambda_delta_reg", 0.0)) > 0.0:
                    ae_hat = ae_reconstruct(ae_model, x_a_hat)
                    loss_delta_reg = ae_hat["delta"].abs().mean()

            loss_total = (
                loss_bb
                + float(cfg.get("lambda_latent_recon", 1.0)) * loss_latent_recon
                + cfg["lambda_feat_src"] * loss_feat_src
                + cfg["lambda_style_src"] * loss_style_src
                + cfg["lambda_self_l1"] * loss_self_l1
                + cfg["lambda_self_ssim"] * loss_self_ssim
                + cfg["lambda_self_edge"] * loss_self_edge
                + float(cfg.get("lambda_ae_feat", 0.0)) * loss_ae_feat
                + float(cfg.get("lambda_ae_edge", 0.0)) * loss_ae_edge
                + float(cfg.get("lambda_identity", 0.0)) * loss_identity
                + float(cfg.get("lambda_source_recon", 0.0)) * loss_source_recon
                + float(cfg.get("lambda_delta_reg", 0.0)) * loss_delta_reg
                + float(cfg.get("lambda_supcon", 0.0)) * loss_supcon
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()

            global_step += 1
            n_batches += 1
            epoch_loss += float(loss_total.item())

            if save_ae_vis and (n_batches == 1):
                vis_path = vis_dir / f"epoch_{epoch:04d}_step_{global_step:07d}.png"
                _save_ae_vis_panel(vis_path, x_b_raw[0], x_b_ae_recon[0], x_a_hat[0])

            pair_epoch = int(batch.get("pair_epoch", torch.tensor([-1]))[0]) if "pair_epoch" in batch else -1
            note = "pseudo-paired,label_random; source weak style guide; self/ae/latent losses preserve structure"

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{global_step},{loss_total.item():.8f},{loss_bb.item():.8f},{loss_latent_recon.item():.8f},"
                    f"{loss_feat_src.item():.8f},{loss_style_src.item():.8f},"
                    f"{loss_self_l1.item():.8f},{loss_self_ssim.item():.8f},{loss_self_edge.item():.8f},"
                    f"{loss_ae_feat.item():.8f},{loss_ae_edge.item():.8f},{loss_identity.item():.8f},"
                    f"{loss_source_recon.item():.8f},{loss_delta_reg.item():.8f},"
                    f"bbdm_bridge_target,{cfg['pair_mode']},{pair_epoch},{ae_input_mode},"
                    f"{int(ae_freeze)},{int(use_ae_frontend)},{note}\n"
                )

            if max_steps is not None and global_step >= max_steps:
                break

        avg_loss = epoch_loss / max(1, n_batches)
        print(
            f"[Epoch {epoch}] avg_loss_total={avg_loss:.6f} steps={global_step} "
            f"ae(use={int(use_ae_frontend)},mode={ae_input_mode},freeze={int(ae_freeze)})"
        )

        if epoch % int(cfg["save_every"]) == 0 or epoch == int(cfg["epochs"]):
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": _to_serializable(cfg),
                "note": "pseudo-paired BBDM-style checkpoint + optional AE frontend regularization (raw/recon/latent)",
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:04d}.pt")
            torch.save(ckpt, ckpt_dir / "latest.pt")

        if max_steps is not None and global_step >= max_steps:
            break

    print(f"Pseudo-paired strict-style BBDM training finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
