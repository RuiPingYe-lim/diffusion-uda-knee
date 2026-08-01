"""Diagnosis-aware bridge residual repair utilities for TRSC U1 views."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


RepairMode = Literal["identity", "fixed_scale", "norm_clip", "learned"]


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _validate_image_pair(
    source: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    if source.ndim != 4 or candidate.ndim != 4:
        raise ValueError(
            "source and candidate must be BCHW tensors, got "
            f"{tuple(source.shape)} and {tuple(candidate.shape)}"
        )
    if source.shape != candidate.shape:
        raise ValueError(
            "source and candidate shapes must match, got "
            f"{tuple(source.shape)} and {tuple(candidate.shape)}"
        )


def project_residual_radius(
    residual: torch.Tensor,
    reference_residual: torch.Tensor,
    max_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project each residual onto an L2 ball defined by the raw U1 residual."""

    if residual.shape != reference_residual.shape or residual.ndim != 4:
        raise ValueError("residual tensors must have matching BCHW shapes")
    if float(max_ratio) <= 0.0:
        raise ValueError("max_ratio must be positive")

    residual_norm = residual.flatten(1).norm(dim=1)
    reference_norm = reference_residual.flatten(1).norm(dim=1)
    radius = float(max_ratio) * reference_norm
    scale = torch.minimum(
        torch.ones_like(residual_norm),
        radius / residual_norm.clamp_min(float(eps)),
    )
    projected = residual * scale[:, None, None, None]
    projected_norm = projected.flatten(1).norm(dim=1)
    ratio = projected_norm / reference_norm.clamp_min(float(eps))
    ratio = torch.where(reference_norm > float(eps), ratio, torch.zeros_like(ratio))
    return projected, ratio, scale


class DiagnosisAwareBridgeResidualRepair(nn.Module):
    """Repair a U1 image by gating its residual relative to the source image.

    The learned path can attenuate a residual locally but cannot inject an
    unrelated target-reference texture directly. The final L2 projection is a
    hard safety bound and keeps the repaired residual inside the configured
    radius of the original U1 residual.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 32,
        mode: RepairMode = "learned",
        fixed_scale: float = 0.8,
        max_radius_ratio: float = 1.0,
        gate_floor: float = 0.0,
        gate_init: float = 0.95,
    ) -> None:
        super().__init__()
        if int(input_channels) < 1:
            raise ValueError("input_channels must be positive")
        if int(hidden_channels) < 4:
            raise ValueError("hidden_channels must be at least four")
        if mode not in ("identity", "fixed_scale", "norm_clip", "learned"):
            raise ValueError(f"Unsupported DA-BRF mode: {mode}")
        if not 0.0 <= float(fixed_scale) <= 1.0:
            raise ValueError("fixed_scale must be in [0, 1]")
        if float(max_radius_ratio) <= 0.0:
            raise ValueError("max_radius_ratio must be positive")
        if not 0.0 <= float(gate_floor) < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        if not float(gate_floor) < float(gate_init) < 1.0:
            raise ValueError("gate_init must satisfy gate_floor < gate_init < 1")

        self.input_channels = int(input_channels)
        self.mode: RepairMode = mode
        self.fixed_scale = float(fixed_scale)
        self.max_radius_ratio = float(max_radius_ratio)
        self.gate_floor = float(gate_floor)

        self.gate_network: nn.Sequential | None = None
        if self.mode == "learned":
            hidden_channels = int(hidden_channels)
            groups = _group_count(hidden_channels)
            self.gate_network = nn.Sequential(
                nn.Conv2d(3 * self.input_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(groups, hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(groups, hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, self.input_channels, 1),
            )
            output_layer = self.gate_network[-1]
            nn.init.zeros_(output_layer.weight)
            normalized_init = (
                (float(gate_init) - self.gate_floor) / (1.0 - self.gate_floor)
            )
            nn.init.constant_(
                output_layer.bias,
                math.log(normalized_init / (1.0 - normalized_init)),
            )

    def forward(
        self,
        source: torch.Tensor,
        candidate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_image_pair(source, candidate)
        if source.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {source.shape[1]}"
            )

        raw_residual = candidate - source
        if self.mode == "identity":
            gate = torch.ones_like(raw_residual)
            repaired_residual = raw_residual
            radius_ratio = torch.where(
                raw_residual.flatten(1).norm(dim=1) > 1e-8,
                torch.ones(source.shape[0], device=source.device, dtype=source.dtype),
                torch.zeros(source.shape[0], device=source.device, dtype=source.dtype),
            )
            radius_scale = torch.ones_like(radius_ratio)
        elif self.mode == "fixed_scale":
            gate = torch.full_like(raw_residual, self.fixed_scale)
            repaired_residual = self.fixed_scale * raw_residual
            radius_ratio = torch.full(
                (source.shape[0],),
                self.fixed_scale,
                device=source.device,
                dtype=source.dtype,
            )
            radius_scale = torch.ones_like(radius_ratio)
        elif self.mode == "norm_clip":
            repaired_residual, radius_ratio, radius_scale = project_residual_radius(
                raw_residual,
                raw_residual,
                self.max_radius_ratio,
            )
            gate = radius_scale[:, None, None, None].expand_as(raw_residual)
        else:
            if self.gate_network is None:
                raise RuntimeError("Learned DA-BRF mode requires a gate network")
            gate_logits = self.gate_network(
                torch.cat([source, candidate, raw_residual], dim=1)
            )
            gate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(
                gate_logits
            )
            gated_residual = gate * raw_residual
            repaired_residual, radius_ratio, radius_scale = project_residual_radius(
                gated_residual,
                raw_residual,
                self.max_radius_ratio,
            )

        # Keep the identity arm bitwise equal to the historical K3 candidate.
        repaired = candidate if self.mode == "identity" else source + repaired_residual
        return {
            "repaired": repaired,
            "raw_residual": raw_residual,
            "repaired_residual": repaired_residual,
            "gate": gate,
            "gate_mean": gate.mean(),
            "radius_ratio": radius_ratio,
            "radius_scale": radius_scale,
        }


def multiscale_style_descriptor(
    image: torch.Tensor,
    pooling_scales: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Describe intensity, contrast, and local texture without a learned probe."""

    if image.ndim != 4:
        raise ValueError(f"image must be BCHW, got {tuple(image.shape)}")
    descriptors = []
    for scale in pooling_scales:
        scale = int(scale)
        if scale < 1:
            raise ValueError("pooling scales must be positive")
        feature = image if scale == 1 else F.avg_pool2d(image, scale, scale)
        mean = feature.mean(dim=(2, 3))
        std = torch.sqrt(feature.var(dim=(2, 3), unbiased=False) + 1e-6)
        dx = feature[:, :, :, 1:] - feature[:, :, :, :-1]
        dy = feature[:, :, 1:, :] - feature[:, :, :-1, :]
        dx_energy = (
            torch.sqrt(dx.square().mean(dim=(2, 3)) + 1e-6)
            if dx.shape[3] > 0
            else torch.zeros_like(mean)
        )
        dy_energy = (
            torch.sqrt(dy.square().mean(dim=(2, 3)) + 1e-6)
            if dy.shape[2] > 0
            else torch.zeros_like(mean)
        )
        descriptors.extend([mean, std, dx_energy, dy_energy])
    return torch.cat(descriptors, dim=1)


def target_style_progress_loss(
    source: torch.Tensor,
    candidate: torch.Tensor,
    repaired: torch.Tensor,
    target_reference: torch.Tensor,
    minimum_retention: float = 0.8,
    minimum_gain: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Retain a fraction of the candidate's reference-style distance reduction."""

    _validate_image_pair(source, candidate)
    _validate_image_pair(source, repaired)
    _validate_image_pair(source, target_reference)
    if not 0.0 <= float(minimum_retention) <= 1.0:
        raise ValueError("minimum_retention must be in [0, 1]")
    if float(minimum_gain) < 0.0:
        raise ValueError("minimum_gain must be non-negative")

    source_style = multiscale_style_descriptor(source)
    candidate_style = multiscale_style_descriptor(candidate)
    repaired_style = multiscale_style_descriptor(repaired)
    reference_style = multiscale_style_descriptor(target_reference)
    source_distance = (source_style - reference_style).norm(dim=1)
    candidate_distance = (candidate_style - reference_style).norm(dim=1)
    repaired_distance = (repaired_style - reference_style).norm(dim=1)
    candidate_gain = source_distance - candidate_distance
    repaired_gain = source_distance - repaired_distance
    valid = candidate_gain > float(minimum_gain)

    target_gain = float(minimum_retention) * candidate_gain.detach()
    denominator = candidate_gain.detach().abs().clamp_min(
        float(minimum_gain) if float(minimum_gain) > 0.0 else 1e-8
    )
    normalized_charge = F.relu(target_gain - repaired_gain) / denominator
    if torch.any(valid):
        loss = normalized_charge[valid].mean()
        retention = (
            repaired_gain[valid] / candidate_gain.detach()[valid].clamp_min(1e-8)
        ).mean()
    else:
        loss = repaired.sum() * 0.0
        retention = repaired.new_tensor(1.0)
    return {
        "loss": loss,
        "candidate_gain": candidate_gain.mean().detach(),
        "repaired_gain": repaired_gain.mean().detach(),
        "retention": retention.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }


def residual_radius_loss(
    source: torch.Tensor,
    candidate: torch.Tensor,
    repaired: torch.Tensor,
    max_ratio: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Measure and penalize repaired residuals outside the configured radius."""

    _validate_image_pair(source, candidate)
    _validate_image_pair(source, repaired)
    if float(max_ratio) <= 0.0:
        raise ValueError("max_ratio must be positive")
    raw_norm = (candidate - source).flatten(1).norm(dim=1)
    repaired_norm = (repaired - source).flatten(1).norm(dim=1)
    ratio = repaired_norm / raw_norm.clamp_min(1e-8)
    ratio = torch.where(raw_norm > 1e-8, ratio, torch.zeros_like(ratio))
    charge = F.relu(ratio - float(max_ratio))
    return {
        "loss": charge.mean(),
        "ratio": ratio.mean().detach(),
        "maximum_ratio": ratio.max().detach(),
    }
