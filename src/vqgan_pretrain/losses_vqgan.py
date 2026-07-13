from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, enabled: bool = False, net: str = "vgg") -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.lpips_model = None
        if self.enabled:
            try:
                import lpips  # type: ignore

                self.lpips_model = lpips.LPIPS(net=net)
                self.lpips_model.eval()
                for p in self.lpips_model.parameters():
                    p.requires_grad = False
            except Exception:
                self.enabled = False
                self.lpips_model = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.lpips_model is None:
            return pred.new_zeros(())
        pred_3 = pred.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)
        return self.lpips_model(pred_3, target_3).mean()


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -logits_fake.mean()


def _sobel_kernels(device: torch.device, dtype: torch.dtype, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)
    return kx.repeat(channels, 1, 1, 1), ky.repeat(channels, 1, 1, 1)


def ssim_loss_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    pad = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=window_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(x * x, kernel_size=window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=window_size, stride=1, padding=pad) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = (num / (den + 1e-6)).clamp(-1.0, 1.0)
    return 1.0 - ssim.mean()


def edge_gradient_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    channels = int(x.shape[1])
    kx, ky = _sobel_kernels(device=x.device, dtype=x.dtype, channels=channels)
    gx_x = F.conv2d(x, kx, padding=1, groups=channels)
    gy_x = F.conv2d(x, ky, padding=1, groups=channels)
    gx_r = F.conv2d(recon, kx, padding=1, groups=channels)
    gy_r = F.conv2d(recon, ky, padding=1, groups=channels)
    return F.l1_loss(gx_r, gx_x) + F.l1_loss(gy_r, gy_x)


def residual_regularization_loss(delta: torch.Tensor) -> torch.Tensor:
    return delta.abs().mean()


def build_recon_loss(
    model_type: str,
    x: torch.Tensor,
    recon: torch.Tensor,
    model_out: Dict[str, torch.Tensor],
    logits_fake: torch.Tensor | None,
    perceptual_loss_fn: PerceptualLoss,
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    model_type = str(model_type).lower()

    rec_l1 = F.l1_loss(recon, x)
    perc = perceptual_loss_fn(recon, x)
    adv = hinge_g_loss(logits_fake) if logits_fake is not None else x.new_zeros(())

    if model_type == "vq":
        codebook = model_out.get("codebook_loss", x.new_zeros(()))
        commitment = model_out.get("commitment_loss", x.new_zeros(()))
        ssim = x.new_zeros(())
        edge = x.new_zeros(())
        residual_reg = x.new_zeros(())
        total = (
            float(weights.get("recon_l1", 1.0)) * rec_l1
            + float(weights.get("perceptual", 0.0)) * perc
            + float(weights.get("codebook", 1.0)) * codebook
            + float(weights.get("commitment", 1.0)) * commitment
            + float(weights.get("generator_adv", 0.0)) * adv
        )
    elif model_type == "kl":
        x_01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        recon_01 = ((recon + 1.0) / 2.0).clamp(0.0, 1.0)
        ssim = ssim_loss_torch(recon_01, x_01)
        edge = edge_gradient_loss(x, recon)
        kl = model_out.get("kl_loss", x.new_zeros(()))
        codebook = x.new_zeros(())
        commitment = x.new_zeros(())
        residual_reg = x.new_zeros(())
        total = (
            float(weights.get("recon_l1", 1.0)) * rec_l1
            + float(weights.get("ssim", 0.5)) * ssim
            + float(weights.get("edge", 1.0)) * edge
            + float(weights.get("perceptual", 0.0)) * perc
            + float(weights.get("kl", 1e-6)) * kl
            + float(weights.get("generator_adv", 0.0)) * adv
        )
    elif model_type == "ae_unet":
        x_01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        recon_01 = ((recon + 1.0) / 2.0).clamp(0.0, 1.0)
        ssim = ssim_loss_torch(recon_01, x_01)
        edge = edge_gradient_loss(x, recon)
        delta = model_out.get("delta", recon.new_zeros(recon.shape))
        residual_reg = residual_regularization_loss(delta)
        codebook = x.new_zeros(())
        commitment = x.new_zeros(())
        total = (
            float(weights.get("recon_l1", 1.0)) * rec_l1
            + float(weights.get("ssim", 0.5)) * ssim
            + float(weights.get("edge", 1.0)) * edge
            + float(weights.get("residual_reg", 0.05)) * residual_reg
            + float(weights.get("perceptual", 0.0)) * perc
            + float(weights.get("generator_adv", 0.0)) * adv
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return {
        "total": total,
        "recon_l1": rec_l1,
        "ssim": ssim,
        "edge": edge,
        "residual_reg": residual_reg,
        "perceptual": perc,
        "codebook": codebook,
        "commitment": commitment,
        "adv_g": adv,
    }
