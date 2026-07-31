"""Classifier and loss utilities for multi-reference TRSC joint training."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ViewWeighting = Literal["equal_groups", "equal_views"]


class SpaceAttention(nn.Module):
    """Spatial attention used by the validated source-only classifier."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        attention = self.conv(feature_map)
        batch, channels, height, width = attention.shape
        attention = self.soft(attention.view(batch, channels, -1))
        attention = attention.view(batch, channels, height, width)
        maximum = (
            attention.view(batch, channels, -1)
            .amax(dim=2, keepdim=True)
            .clamp_min(1e-6)
        )
        attention = (
            attention.view(batch, channels, -1) / maximum
        ).view(batch, channels, height, width)
        return feature_map * attention


class SourceWarmStartResNet50(nn.Module):
    """Exact ``custom_resnet50_space`` architecture used by source-only runs."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        from torchvision import models

        backbone = models.resnet50(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        feature_dim = 2048
        self.space_attn = SpaceAttention(feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = self.space_attn(self.stem(image))
        return self.avgpool(feature_map).flatten(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(image))


def route_task_gradient(
    translated: torch.Tensor,
    generator_scale: float,
) -> torch.Tensor:
    """Keep the forward value while scaling only the translator-side gradient."""

    scale = float(generator_scale)
    if scale < 0.0:
        raise ValueError("generator_scale must be non-negative")
    return scale * translated + (1.0 - scale) * translated.detach()


def multi_reference_cross_entropy(
    raw_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    weighting: ViewWeighting = "equal_groups",
) -> dict[str, torch.Tensor]:
    """Compute raw-plus-candidate CE without filtering any candidate.

    ``candidate_logits`` is shaped ``[B, K, C]``. ``equal_groups`` assigns half
    of the task weight to the raw view and half to the mean over all K target-
    reference candidates, so increasing K does not silently increase the total
    weight of one source case. ``equal_views`` reproduces ordinary dataset
    concatenation, where every one of the K+1 images has equal weight.
    """

    if raw_logits.ndim != 2:
        raise ValueError(f"raw_logits must be [B,C], got {tuple(raw_logits.shape)}")
    if candidate_logits.ndim != 3:
        raise ValueError(
            "candidate_logits must be [B,K,C], "
            f"got {tuple(candidate_logits.shape)}"
        )
    batch_size, candidate_count, class_count = candidate_logits.shape
    if candidate_count < 1:
        raise ValueError("At least one translated candidate is required")
    if raw_logits.shape != (batch_size, class_count):
        raise ValueError(
            "Raw and candidate logits are incompatible: "
            f"{tuple(raw_logits.shape)} vs {tuple(candidate_logits.shape)}"
        )
    if labels.shape != (batch_size,):
        raise ValueError(
            f"labels must have shape {(batch_size,)}, got {tuple(labels.shape)}"
        )

    raw_ce = F.cross_entropy(raw_logits, labels)
    repeated_labels = labels[:, None].expand(-1, candidate_count).reshape(-1)
    candidate_ce = F.cross_entropy(
        candidate_logits.reshape(-1, class_count),
        repeated_labels,
    )

    if weighting == "equal_groups":
        total = 0.5 * (raw_ce + candidate_ce)
    elif weighting == "equal_views":
        all_logits = torch.cat([raw_logits[:, None], candidate_logits], dim=1)
        all_labels = labels[:, None].expand(-1, candidate_count + 1).reshape(-1)
        total = F.cross_entropy(all_logits.reshape(-1, class_count), all_labels)
    else:
        raise ValueError(f"Unsupported view weighting: {weighting}")

    return {
        "total": total,
        "raw": raw_ce,
        "candidate": candidate_ce,
    }
