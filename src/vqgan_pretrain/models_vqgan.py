from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 32
    while num_channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        blocks: List[nn.Module] = []
        in_ch = base_channels
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i != len(ch_mult) - 1:
                blocks.append(Downsample(in_ch))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = _group_norm(in_ch)
        self.conv_out = nn.Conv2d(in_ch, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.blocks(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
    ) -> None:
        super().__init__()
        rev_mult = list(ch_mult)[::-1]
        in_ch = base_channels * rev_mult[0]
        self.conv_in = nn.Conv2d(z_channels, in_ch, kernel_size=3, stride=1, padding=1)

        blocks: List[nn.Module] = []
        for i, mult in enumerate(rev_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i != len(rev_mult) - 1:
                blocks.append(Upsample(in_ch))
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = _group_norm(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.blocks(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 1024, embedding_dim: int = 256, commitment_weight: float = 0.25) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_weight = float(commitment_weight)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 量化关键步骤：把连续 latent 映射到最近的 codebook 向量
        b, c, h, w = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, c)

        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * (z @ self.embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment_weight * F.mse_loss(z_q.detach(), z_e)
        z_q_st = z_e + (z_q - z_e).detach()

        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        return {
            "quantized": z_q_st,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "indices": indices.view(b, h, w),
            "perplexity": perplexity,
        }


@dataclass
class VQGANConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 256
    embed_dim: int = 256
    num_embeddings: int = 1024
    commitment_weight: float = 0.25


class VQGAN(nn.Module):
    def __init__(self, cfg: VQGANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            ch_mult=cfg.ch_mult,
            num_res_blocks=cfg.num_res_blocks,
            z_channels=cfg.z_channels,
        )
        self.quant_conv = nn.Conv2d(cfg.z_channels, cfg.embed_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embed_dim,
            commitment_weight=cfg.commitment_weight,
        )
        self.post_quant_conv = nn.Conv2d(cfg.embed_dim, cfg.z_channels, kernel_size=1)
        self.decoder = Decoder(
            out_channels=cfg.out_channels,
            base_channels=cfg.base_channels,
            ch_mult=cfg.ch_mult,
            num_res_blocks=cfg.num_res_blocks,
            z_channels=cfg.z_channels,
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.quant_conv(self.encoder(x))
        q_out = self.quantizer(z_e)
        return {"z_e": z_e, **q_out}

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z_q))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(x)
        recon = self.decode(enc["quantized"])
        return {"recon": recon, **enc}


@dataclass
class KLVAEConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 4  # continuous latent channels (LDM-style, small)


class KLVAE(nn.Module):
    """Continuous KL-regularized autoencoder (no vector quantization).

    Skip-free compressive encoder -> Gaussian latent (mean/logvar) -> decoder.
    This is the LDM-style first stage; avoids VQ codebook collapse.
    """

    def __init__(self, cfg: KLVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder emits 2*z_channels (mean and logvar stacked).
        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            ch_mult=cfg.ch_mult,
            num_res_blocks=cfg.num_res_blocks,
            z_channels=2 * cfg.z_channels,
        )
        self.decoder = Decoder(
            out_channels=cfg.out_channels,
            base_channels=cfg.base_channels,
            ch_mult=cfg.ch_mult,
            num_res_blocks=cfg.num_res_blocks,
            z_channels=cfg.z_channels,
        )

    def encode(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        # KL to standard normal, averaged over batch (summed over latent dims).
        kl = 0.5 * torch.mean(
            torch.sum(mean.pow(2) + logvar.exp() - 1.0 - logvar, dim=[1, 2, 3])
        )
        return {"z": z, "mean": mean, "logvar": logvar, "kl_loss": kl}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(x, sample=self.training)
        recon = self.decode(enc["z"])
        return {"recon": recon, **enc}


@dataclass
class AEUNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    residual_out: bool = True


class AEUNet(nn.Module):
    """
    Minimal U-Net style autoencoder:
    - no vector quantization
    - skip connections between encoder / decoder
    - optional residual output head
    """

    def __init__(self, cfg: AEUNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.residual_out = bool(cfg.residual_out)

        self.conv_in = nn.Conv2d(cfg.in_channels, cfg.base_channels, kernel_size=3, stride=1, padding=1)

        self.down_blocks: nn.ModuleList[nn.ModuleList] = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: List[int] = []

        in_ch = cfg.base_channels
        for i, mult in enumerate(cfg.ch_mult):
            out_ch = cfg.base_channels * int(mult)
            stage = nn.ModuleList()
            for _ in range(int(cfg.num_res_blocks)):
                stage.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            self.down_blocks.append(stage)
            self.skip_channels.append(in_ch)
            if i != len(cfg.ch_mult) - 1:
                self.downsamples.append(Downsample(in_ch))

        self.mid_block1 = ResBlock(in_ch, in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch)

        self.up_blocks: nn.ModuleList[nn.ModuleList] = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(cfg.ch_mult))):
            skip_ch = self.skip_channels[i]
            out_ch = cfg.base_channels * int(cfg.ch_mult[i])
            stage = nn.ModuleList()
            stage.append(ResBlock(in_ch + skip_ch, out_ch))
            in_ch = out_ch
            for _ in range(max(int(cfg.num_res_blocks) - 1, 0)):
                stage.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            self.up_blocks.append(stage)
            if i != 0:
                self.upsamples.append(Upsample(in_ch))

        self.norm_out = _group_norm(in_ch)
        self.conv_out = nn.Conv2d(in_ch, cfg.out_channels, kernel_size=3, stride=1, padding=1)

    def encode_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        h = self.conv_in(x)
        skips: List[torch.Tensor] = []

        for i, stage in enumerate(self.down_blocks):
            for block in stage:
                h = block(h)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid_block1(h)
        h = self.mid_block2(h)
        return {"bottleneck": h, "encoder_features": skips}

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        return self.encode_features(x)

    def decode_from_bottleneck(
        self,
        bottleneck: torch.Tensor,
        encoder_features: List[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        h = bottleneck
        skips = encoder_features if encoder_features is not None else [None] * len(self.up_blocks)
        for i, stage in enumerate(self.up_blocks):
            skip = skips[-1 - i]
            first_block = stage[0]
            expected_in = first_block.norm1.num_channels if isinstance(first_block, ResBlock) else h.shape[1]

            if skip is None:
                # Fallback path for latent-only decoding: synthesize zero skip.
                skip_ch = max(int(expected_in - int(h.shape[1])), 0)
                skip = h.new_zeros((h.shape[0], skip_ch, h.shape[2], h.shape[3]))
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in stage:
                h = block(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)
        delta = torch.tanh(self.conv_out(F.silu(self.norm_out(h))))
        return {"delta": delta}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encode_features(x)
        dec = self.decode_from_bottleneck(
            bottleneck=feat["bottleneck"],
            encoder_features=feat["encoder_features"],
        )
        delta = dec["delta"]
        if self.residual_out:
            recon = torch.clamp(x + delta, min=-1.0, max=1.0)
        else:
            recon = delta
        return {"recon": recon, "delta": delta, "bottleneck": feat["bottleneck"], "encoder_features": feat["encoder_features"]}


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c_in = base_channels
        for i in range(1, n_layers):
            c_out = min(base_channels * (2 ** i), 512)
            layers.extend(
                [
                    nn.Conv2d(c_in, c_out, kernel_size=4, stride=2 if i < n_layers - 1 else 1, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            c_in = c_out
        layers.append(nn.Conv2d(c_in, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
