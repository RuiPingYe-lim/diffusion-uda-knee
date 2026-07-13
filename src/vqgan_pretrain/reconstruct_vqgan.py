from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_vqgan import VQGANSliceDataset
from models_vqgan import AEUNet, AEUNetConfig, VQGAN, VQGANConfig
from utils_vqgan import compute_recon_metrics, mean_metrics, save_json, save_single_visualization, tensor_to_01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct images with pretrained VQ / AE-UNet checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config JSON (for fallback model settings)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--root_dir", type=Path, default=None)
    parser.add_argument("--image_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_ch_mult(cfg: Dict[str, Any]) -> tuple[int, ...]:
    ch_mult = cfg.get("ch_mult", [1, 2, 4, 4])
    if isinstance(ch_mult, str):
        ch_mult = [int(x.strip()) for x in ch_mult.split(",") if x.strip()]
    return tuple(int(x) for x in ch_mult)


def _resolve_model_type(ckpt: Dict[str, Any], fallback_cfg: Dict[str, Any]) -> str:
    if "model_type" in ckpt:
        return str(ckpt["model_type"]).lower()
    train_cfg = ckpt.get("train_config", {})
    if isinstance(train_cfg, dict) and "model_type" in train_cfg:
        return str(train_cfg["model_type"]).lower()
    return str(fallback_cfg.get("model_type", "vq")).lower()


def _build_model_from_ckpt(ckpt: Dict[str, Any], fallback_cfg: Dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    model_type = _resolve_model_type(ckpt, fallback_cfg)
    model_cfg_dict = ckpt.get("model_config", None)

    if model_type == "vq":
        if model_cfg_dict is None:
            model_cfg_dict = {
                "in_channels": int(fallback_cfg.get("in_channels", 1)),
                "out_channels": int(fallback_cfg.get("out_channels", 1)),
                "base_channels": int(fallback_cfg.get("base_channels", 128)),
                "ch_mult": _parse_ch_mult(fallback_cfg),
                "num_res_blocks": int(fallback_cfg.get("num_res_blocks", 2)),
                "z_channels": int(fallback_cfg.get("z_channels", 256)),
                "embed_dim": int(fallback_cfg.get("embed_dim", 256)),
                "num_embeddings": int(fallback_cfg.get("num_embeddings", 1024)),
                "commitment_weight": float(fallback_cfg.get("commitment_weight", 0.25)),
            }
        else:
            model_cfg_dict = dict(model_cfg_dict)
            model_cfg_dict["ch_mult"] = tuple(model_cfg_dict["ch_mult"])

        model = VQGAN(VQGANConfig(**model_cfg_dict)).to(device)

    elif model_type == "ae_unet":
        if model_cfg_dict is None:
            model_cfg_dict = {
                "in_channels": int(fallback_cfg.get("in_channels", 1)),
                "out_channels": int(fallback_cfg.get("out_channels", 1)),
                "base_channels": int(fallback_cfg.get("base_channels", 128)),
                "ch_mult": _parse_ch_mult(fallback_cfg),
                "num_res_blocks": int(fallback_cfg.get("num_res_blocks", 2)),
                "residual_out": bool(fallback_cfg.get("residual_out", True)),
            }
        else:
            model_cfg_dict = dict(model_cfg_dict)
            model_cfg_dict["ch_mult"] = tuple(model_cfg_dict["ch_mult"])

        model = AEUNet(AEUNetConfig(**model_cfg_dict)).to(device)

    else:
        raise ValueError(f"Unsupported model_type in checkpoint/config: {model_type}")

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, model_type


def _to_device(args: argparse.Namespace, cfg: Dict[str, Any]) -> torch.device:
    device_str = args.device or cfg.get("device", "cuda")
    if str(device_str) == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = _load_json(args.config)
    device = _to_device(args, cfg)

    ckpt = torch.load(args.ckpt, map_location=device)
    model, model_type = _build_model_from_ckpt(ckpt=ckpt, fallback_cfg=cfg, device=device)

    image_size = int(args.image_size or cfg.get("image_size", 128))
    ds = VQGANSliceDataset(
        csv_path=args.csv,
        root_dir=args.root_dir,
        image_col=args.image_col or cfg.get("image_col"),
        label_col=args.label_col or cfg.get("label_col"),
        image_size=image_size,
        normalize_to_neg_one_one=True,
        return_metadata=True,
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    output_dir = args.output_dir
    recon_dir = output_dir / "reconstructed_images"
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, float]] = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Reconstruct]", ncols=120):
            x = batch["image"].to(device, non_blocking=True)
            out = model(x)
            recon = out["recon"]
            delta = out.get("delta", None)

            x01 = tensor_to_01(x).numpy()
            recon01 = tensor_to_01(recon).numpy()
            labels = batch["label"]
            resolved_paths = batch.get("resolved_image_path", [""] * x.shape[0])

            delta_abs_mean_np = None
            if delta is not None:
                delta_abs_mean_np = delta.detach().abs().mean(dim=(1, 2, 3)).cpu().numpy()

            for i in range(x.shape[0]):
                img = x01[i, 0]
                rec = recon01[i, 0]
                metrics = compute_recon_metrics(img, rec)
                all_metrics.append(metrics)

                out_name = f"sample_{sample_idx:06d}.png"
                recon_path = recon_dir / out_name
                vis_path = vis_dir / out_name
                Image.fromarray((np.clip(rec, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(recon_path)
                save_single_visualization(img, rec, vis_path)

                row: Dict[str, Any] = {
                    "index": sample_idx,
                    "model_type": model_type,
                    "source_image_path": str(resolved_paths[i]) if isinstance(resolved_paths, list) else "",
                    "recon_image_path": str(recon_path),
                    "vis_image_path": str(vis_path),
                    "label": int(labels[i]) if torch.is_tensor(labels) else labels[i],
                    "l1": metrics["l1"],
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                }
                if delta_abs_mean_np is not None:
                    row["delta_abs_mean"] = float(delta_abs_mean_np[i])

                rows.append(row)
                sample_idx += 1

    df = pd.DataFrame(rows)
    csv_out = output_dir / "reconstruction_results.csv"
    df.to_csv(csv_out, index=False)

    mean_stat = mean_metrics(all_metrics)
    mean_stat["model_type"] = model_type
    save_json(output_dir / "metrics_summary.json", mean_stat)

    print(f"[Done] Model type: {model_type}")
    print(f"[Done] Reconstructed {len(rows)} samples.")
    print(f"[Done] CSV: {csv_out}")
    print(f"[Done] Mean metrics: {mean_stat}")


if __name__ == "__main__":
    main()
