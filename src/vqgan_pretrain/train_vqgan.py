from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_vqgan import VQGANSliceDataset
from losses_vqgan import PerceptualLoss, build_recon_loss, hinge_d_loss, ssim_loss_torch
from models_vqgan import AEUNet, AEUNetConfig, PatchDiscriminator, VQGAN, VQGANConfig
from utils_vqgan import (
    append_csv_row,
    ensure_dir,
    maybe_to_float,
    save_comparison_grid,
    save_json,
    to_serializable_config,
)

_GLOBAL_WORKER_BASE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQ / AE-UNet pretraining for 2D medical slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON config path")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--model_type", type=str, choices=["vq", "ae_unet"], default=None)
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--val_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--lr_d", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--disc_start_epoch", type=int, default=None)
    parser.add_argument("--use_gan", type=int, default=None, help="1 enable, 0 disable")
    parser.add_argument("--use_amp", action="store_true", help="Override config and enable AMP")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_worker_base_seed(seed: int) -> None:
    global _GLOBAL_WORKER_BASE_SEED
    _GLOBAL_WORKER_BASE_SEED = int(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = int(_GLOBAL_WORKER_BASE_SEED + worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
    import json

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    override_keys = [
        "model_type",
        "train_csv",
        "val_csv",
        "output_dir",
        "batch_size",
        "epochs",
        "num_workers",
        "lr_g",
        "lr_d",
        "image_size",
        "disc_start_epoch",
        "device",
    ]
    for k in override_keys:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = str(v) if isinstance(v, Path) else v

    if args.use_gan is not None:
        cfg["use_gan"] = bool(int(args.use_gan))

    if args.use_amp:
        cfg["use_amp"] = True

    cfg.setdefault("model_type", "vq")
    cfg.setdefault("use_gan", str(cfg["model_type"]).lower() == "vq")
    return cfg


def _to_device(cfg: Dict[str, Any]) -> torch.device:
    device_str = str(cfg.get("device", "cuda"))
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_ch_mult(cfg: Dict[str, Any]) -> Tuple[int, ...]:
    ch_mult = cfg.get("ch_mult", [1, 2, 4, 4])
    if isinstance(ch_mult, str):
        ch_mult = [int(x.strip()) for x in ch_mult.split(",") if x.strip()]
    return tuple(int(x) for x in ch_mult)


def _build_vq_model_config(cfg: Dict[str, Any]) -> VQGANConfig:
    return VQGANConfig(
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 128)),
        ch_mult=_parse_ch_mult(cfg),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        z_channels=int(cfg.get("z_channels", 256)),
        embed_dim=int(cfg.get("embed_dim", 256)),
        num_embeddings=int(cfg.get("num_embeddings", 1024)),
        commitment_weight=float(cfg.get("commitment_weight", 0.25)),
    )


def _build_ae_model_config(cfg: Dict[str, Any]) -> AEUNetConfig:
    return AEUNetConfig(
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 128)),
        ch_mult=_parse_ch_mult(cfg),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        residual_out=bool(cfg.get("residual_out", True)),
    )


def _build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    seed = int(cfg.get("seed", 42))
    _set_worker_base_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = VQGANSliceDataset(
        csv_path=Path(cfg["train_csv"]),
        root_dir=Path(cfg["train_root_dir"]) if cfg.get("train_root_dir") else None,
        image_col=cfg.get("image_col"),
        label_col=cfg.get("label_col"),
        image_size=int(cfg.get("image_size", 128)),
        normalize_to_neg_one_one=True,
        return_metadata=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    val_loader = None
    if cfg.get("val_csv"):
        val_ds = VQGANSliceDataset(
            csv_path=Path(cfg["val_csv"]),
            root_dir=Path(cfg["val_root_dir"]) if cfg.get("val_root_dir") else None,
            image_col=cfg.get("image_col"),
            label_col=cfg.get("label_col"),
            image_size=int(cfg.get("image_size", 128)),
            normalize_to_neg_one_one=True,
            return_metadata=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.get("val_batch_size", cfg.get("batch_size", 16))),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=_seed_worker,
        )
    return train_loader, val_loader


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    l1_list = []
    mse_list = []
    ssim_list = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        out = model(x)
        recon = out["recon"]
        l1 = torch.mean(torch.abs(recon - x))
        mse = torch.mean((recon - x) ** 2)
        x_01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        recon_01 = ((recon + 1.0) / 2.0).clamp(0.0, 1.0)
        ssim_val = 1.0 - ssim_loss_torch(recon_01, x_01)
        l1_list.append(float(l1.detach().cpu().item()))
        mse_list.append(float(mse.detach().cpu().item()))
        ssim_list.append(float(ssim_val.detach().cpu().item()))
    return {
        "val_recon_l1": float(np.mean(l1_list)) if l1_list else 0.0,
        "val_recon_mse": float(np.mean(mse_list)) if mse_list else 0.0,
        "val_ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
    }


def _model_cfg_dict(model: nn.Module) -> Dict[str, Any]:
    cfg_obj = getattr(model, "cfg", None)
    if cfg_obj is None:
        return {}
    cfg_dict = dict(cfg_obj.__dict__)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, tuple):
            cfg_dict[k] = list(v)
    return cfg_dict


def save_ckpt(
    path: Path,
    epoch: int,
    best_metric: float,
    model: nn.Module,
    model_type: str,
    use_gan: bool,
    disc: Optional[PatchDiscriminator],
    opt_g: torch.optim.Optimizer,
    opt_d: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_state": model.state_dict(),
            "model_type": str(model_type),
            "use_gan": bool(use_gan),
            "disc_state": disc.state_dict() if disc is not None else None,
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict() if opt_d is not None else None,
            "scaler_state": scaler.state_dict(),
            "model_config": _model_cfg_dict(model),
            "train_config": to_serializable_config(cfg),
        },
        path,
    )


def maybe_resume(
    resume_path: Optional[Path],
    device: torch.device,
    model: nn.Module,
    disc: Optional[PatchDiscriminator],
    opt_g: torch.optim.Optimizer,
    opt_d: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
) -> Tuple[int, float]:
    if resume_path is None:
        return 1, float("inf")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    disc_state = ckpt.get("disc_state", None)
    if disc is not None and disc_state is not None:
        disc.load_state_dict(disc_state, strict=True)

    opt_g.load_state_dict(ckpt["opt_g_state"])

    opt_d_state = ckpt.get("opt_d_state", None)
    if opt_d is not None and opt_d_state is not None:
        opt_d.load_state_dict(opt_d_state)

    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("inf")))
    return start_epoch, best_metric


def main() -> None:
    args = parse_args()
    cfg = _load_config(args)
    _set_seed(int(cfg.get("seed", 42)))
    device = _to_device(cfg)

    model_type = str(cfg.get("model_type", "vq")).lower()
    if model_type not in {"vq", "ae_unet"}:
        raise ValueError("model_type must be 'vq' or 'ae_unet'")

    use_gan = bool(cfg.get("use_gan", model_type == "vq"))

    output_dir = ensure_dir(Path(cfg["output_dir"]))
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    sample_dir = ensure_dir(output_dir / "samples")
    log_dir = ensure_dir(output_dir / "logs")
    save_json(output_dir / "train_config_used.json", to_serializable_config(cfg))

    train_loader, val_loader = _build_dataloaders(cfg)

    if model_type == "vq":
        model: nn.Module = VQGAN(_build_vq_model_config(cfg)).to(device)
    else:
        model = AEUNet(_build_ae_model_config(cfg)).to(device)

    disc: Optional[PatchDiscriminator] = None
    if use_gan:
        disc = PatchDiscriminator(
            in_channels=int(cfg.get("in_channels", 1)),
            base_channels=int(cfg.get("disc_base_channels", 64)),
            n_layers=int(cfg.get("disc_n_layers", 3)),
        ).to(device)

    opt_g = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr_g", 2e-4)),
        betas=(float(cfg.get("beta1", 0.5)), float(cfg.get("beta2", 0.9))),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    opt_d: Optional[torch.optim.Optimizer] = None
    if disc is not None:
        opt_d = torch.optim.AdamW(
            disc.parameters(),
            lr=float(cfg.get("lr_d", 2e-4)),
            betas=(float(cfg.get("beta1", 0.5)), float(cfg.get("beta2", 0.9))),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    perceptual = PerceptualLoss(
        enabled=bool(cfg.get("use_perceptual", False)),
        net=str(cfg.get("perceptual_net", "vgg")),
    ).to(device)

    weights = {
        "recon_l1": float(cfg.get("w_recon_l1", 1.0)),
        "ssim": float(cfg.get("w_ssim", 0.5)),
        "edge": float(cfg.get("w_edge", 1.0)),
        "residual_reg": float(cfg.get("w_residual_reg", 0.05)),
        "perceptual": float(cfg.get("w_perceptual", 0.0)),
        "codebook": float(cfg.get("w_codebook", 1.0)),
        "commitment": float(cfg.get("w_commitment", 1.0)),
        "generator_adv": float(cfg.get("w_gan", 0.1)),
        "discriminator": float(cfg.get("w_disc", 1.0)),
    }

    start_epoch, best_metric = maybe_resume(
        resume_path=args.resume,
        device=device,
        model=model,
        disc=disc,
        opt_g=opt_g,
        opt_d=opt_d,
        scaler=scaler,
    )

    epochs = int(cfg.get("epochs", 100))
    disc_start_epoch = int(cfg.get("disc_start_epoch", 6))
    grad_clip = float(cfg.get("grad_clip", 1.0))  # 0 disables; fixes NaN divergence
    save_every = int(cfg.get("save_every", 5))
    sample_every = int(cfg.get("sample_every", 1))
    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if disc is not None:
            disc.train()
        disc_enabled = use_gan and disc is not None and (epoch >= disc_start_epoch)

        running = {
            "g_total": 0.0,
            "recon_l1": 0.0,
            "ssim": 0.0,
            "edge": 0.0,
            "residual_reg": 0.0,
            "perceptual": 0.0,
            "codebook": 0.0,
            "commitment": 0.0,
            "adv_g": 0.0,
            "d_total": 0.0,
            "perplexity": 0.0,
        }
        num_batches = 0
        last_batch_x = None
        last_batch_recon = None

        pbar = tqdm(train_loader, desc=f"[Train] epoch {epoch}/{epochs}", ncols=120)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            num_batches += 1
            global_step += 1

            opt_g.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                out = model(x)
                recon = out["recon"]
                logits_fake_for_g = disc(recon) if disc_enabled and disc is not None else None
                g_losses = build_recon_loss(
                    model_type=model_type,
                    x=x,
                    recon=recon,
                    model_out=out,
                    logits_fake=logits_fake_for_g,
                    perceptual_loss_fn=perceptual,
                    weights=weights,
                )
            scaler.scale(g_losses["total"]).backward()
            if grad_clip > 0:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt_g)

            d_loss_value = x.new_zeros(())
            if disc_enabled and disc is not None and opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits_real = disc(x.detach())
                    logits_fake = disc(recon.detach())
                    d_core = hinge_d_loss(logits_real, logits_fake)
                    d_loss = weights["discriminator"] * d_core
                scaler.scale(d_loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
                scaler.step(opt_d)
                d_loss_value = d_loss.detach()

            scaler.update()

            running["g_total"] += maybe_to_float(g_losses["total"])
            running["recon_l1"] += maybe_to_float(g_losses["recon_l1"])
            running["ssim"] += maybe_to_float(g_losses["ssim"])
            running["edge"] += maybe_to_float(g_losses["edge"])
            running["residual_reg"] += maybe_to_float(g_losses["residual_reg"])
            running["perceptual"] += maybe_to_float(g_losses["perceptual"])
            running["codebook"] += maybe_to_float(g_losses["codebook"])
            running["commitment"] += maybe_to_float(g_losses["commitment"])
            running["adv_g"] += maybe_to_float(g_losses["adv_g"])
            running["d_total"] += maybe_to_float(d_loss_value)
            running["perplexity"] += maybe_to_float(out.get("perplexity", x.new_zeros(())))

            last_batch_x = x.detach()
            last_batch_recon = recon.detach()
            pbar.set_postfix(
                recon_l1=f"{running['recon_l1']/num_batches:.4f}",
                ssim=f"{running['ssim']/num_batches:.4f}",
                edge=f"{running['edge']/num_batches:.4f}",
                g_total=f"{running['g_total']/num_batches:.4f}",
                disc_on=int(disc_enabled),
            )

        for k in running:
            running[k] = running[k] / max(1, num_batches)

        val_stats = {}
        if val_loader is not None:
            val_stats = validate(model, val_loader, device=device)

        metric_for_best = (
            float(val_stats["val_recon_l1"]) if "val_recon_l1" in val_stats else float(running["recon_l1"])
        )
        is_best = metric_for_best < best_metric
        if is_best:
            best_metric = metric_for_best

        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "model_type": model_type,
            "disc_enabled": int(disc_enabled),
            "train_g_total": running["g_total"],
            "train_recon_l1": running["recon_l1"],
            "train_ssim": running["ssim"],
            "train_edge": running["edge"],
            "train_residual_reg": running["residual_reg"],
            "train_perceptual": running["perceptual"],
            "train_codebook": running["codebook"],
            "train_commitment": running["commitment"],
            "train_adv_g": running["adv_g"],
            "train_d_total": running["d_total"],
            "train_perplexity": running["perplexity"],
            "val_recon_l1": float(val_stats.get("val_recon_l1", 0.0)),
            "val_recon_mse": float(val_stats.get("val_recon_mse", 0.0)),
            "val_ssim": float(val_stats.get("val_ssim", 0.0)),
            "best_metric": best_metric,
        }
        append_csv_row(log_dir / "train_log.csv", epoch_row)

        save_ckpt(
            path=ckpt_dir / "latest.pt",
            epoch=epoch,
            best_metric=best_metric,
            model=model,
            model_type=model_type,
            use_gan=use_gan,
            disc=disc,
            opt_g=opt_g,
            opt_d=opt_d,
            scaler=scaler,
            cfg=cfg,
        )

        if is_best:
            save_ckpt(
                path=ckpt_dir / "best.pt",
                epoch=epoch,
                best_metric=best_metric,
                model=model,
                model_type=model_type,
                use_gan=use_gan,
                disc=disc,
                opt_g=opt_g,
                opt_d=opt_d,
                scaler=scaler,
                cfg=cfg,
            )

        if save_every > 0 and (epoch % save_every == 0):
            save_ckpt(
                path=ckpt_dir / f"epoch_{epoch:04d}.pt",
                epoch=epoch,
                best_metric=best_metric,
                model=model,
                model_type=model_type,
                use_gan=use_gan,
                disc=disc,
                opt_g=opt_g,
                opt_d=opt_d,
                scaler=scaler,
                cfg=cfg,
            )

        if sample_every > 0 and (epoch % sample_every == 0) and last_batch_x is not None and last_batch_recon is not None:
            save_comparison_grid(last_batch_x, last_batch_recon, sample_dir / f"epoch_{epoch:04d}.png", max_items=8)

    print(f"[Done] Training finished. Best metric (lower is better): {best_metric:.6f}")
    print(f"[Done] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
