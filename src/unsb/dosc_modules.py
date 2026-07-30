"""Diagnostic-orthogonal target-style conditioning modules for UNSB.

This file is intentionally self-contained so it can be copied into the upstream
UNSB ``models`` package by ``scripts/install_dosc_unsb_overlay.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


def gradient_reverse(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Keep the forward value and reverse its encoder gradient."""

    return _GradientReverse.apply(x, float(strength))


class MultiScaleMomentStyleEncoder(nn.Module):
    """Encode shallow texture statistics instead of global image semantics.

    The descriptor concatenates channel-wise means and standard deviations from
    several convolutional scales. This keeps speckle and acquisition-pattern
    information that survives global intensity normalization.
    """

    def __init__(
        self,
        input_channels: int = 3,
        style_dim: int = 128,
        widths: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("widths must contain at least one stage")
        blocks = []
        in_channels = int(input_channels)
        for width in widths:
            width = int(width)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = width
        self.blocks = nn.ModuleList(blocks)
        descriptor_dim = 2 * sum(int(width) for width in widths)
        hidden_dim = max(int(style_dim) * 2, 256)
        self.project = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(style_dim)),
        )

    def forward(
        self,
        image: torch.Tensor,
        return_descriptor: bool = False,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got shape={tuple(image.shape)}")
        feature = image
        moments = []
        for block in self.blocks:
            feature = block(feature)
            mean = feature.mean(dim=(2, 3))
            variance = feature.var(dim=(2, 3), unbiased=False)
            moments.extend([mean, torch.sqrt(variance + 1e-6)])
        descriptor = torch.cat(moments, dim=1)
        style = self.project(descriptor)
        if return_descriptor:
            return style, descriptor
        return style


class DiagnosticSubspaceProjector(nn.Module):
    """Remove the EMA class-mean subspace from style embeddings.

    For binary diagnosis this removes the single direction connecting the two
    source class centroids. For C classes it removes at most C-1 directions.
    The basis is updated from labeled source images only and stored as buffers,
    so target labels are never required.
    """

    def __init__(
        self,
        style_dim: int,
        num_classes: int = 2,
        momentum: float = 0.95,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.style_dim = int(style_dim)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.eps = float(eps)
        max_rank = self.num_classes - 1
        self.register_buffer("class_means", torch.zeros(self.num_classes, self.style_dim))
        self.register_buffer("class_initialized", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("basis", torch.zeros(max_rank, self.style_dim))
        self.register_buffer("basis_rank", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, style: torch.Tensor, labels: torch.Tensor) -> None:
        if style.ndim != 2 or style.shape[1] != self.style_dim:
            raise ValueError(
                f"Expected style shape [B,{self.style_dim}], got {tuple(style.shape)}"
            )
        labels = labels.reshape(-1).long()
        if labels.shape[0] != style.shape[0]:
            raise ValueError("style and labels must have the same batch size")

        valid = (labels >= 0) & (labels < self.num_classes)
        for class_index in range(self.num_classes):
            mask = valid & (labels == class_index)
            if not torch.any(mask):
                continue
            current_mean = style[mask].mean(dim=0)
            if bool(self.class_initialized[class_index]):
                self.class_means[class_index].mul_(self.momentum).add_(
                    current_mean, alpha=1.0 - self.momentum
                )
            else:
                self.class_means[class_index].copy_(current_mean)
                self.class_initialized[class_index] = True

        initialized_means = self.class_means[self.class_initialized]
        self.basis.zero_()
        self.basis_rank.zero_()
        if initialized_means.shape[0] < 2:
            return

        centered = initialized_means - initialized_means.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        numerical_rank = int((singular_values > self.eps).sum().item())
        rank = min(numerical_rank, self.basis.shape[0])
        if rank > 0:
            self.basis[:rank].copy_(vh[:rank])
            self.basis_rank.fill_(rank)

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        rank = int(self.basis_rank.item())
        if rank == 0:
            return style
        active_basis = self.basis[:rank]
        diagnostic_component = (style @ active_basis.t()) @ active_basis
        return style - diagnostic_component

    def removed_energy_ratio(self, style: torch.Tensor) -> torch.Tensor:
        projected = self(style)
        removed = style - projected
        numerator = removed.square().sum(dim=1)
        denominator = style.square().sum(dim=1).clamp_min(self.eps)
        return (numerator / denominator).mean()


class _MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden_dim = max(int(input_dim), 128)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReferenceStyleQueue(nn.Module):
    """Memory queue used by reference-instance contrastive learning."""

    def __init__(self, style_dim: int, queue_size: int = 128) -> None:
        super().__init__()
        if int(queue_size) < 1:
            raise ValueError("queue_size must be positive")
        self.queue_size = int(queue_size)
        self.style_dim = int(style_dim)
        self.register_buffer("queue", torch.zeros(self.queue_size, self.style_dim))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, style: torch.Tensor) -> None:
        style = F.normalize(style.detach(), dim=1)
        for row in style:
            pointer = int(self.queue_pointer.item())
            self.queue[pointer].copy_(row)
            self.queue_pointer.fill_((pointer + 1) % self.queue_size)
            self.queue_count.fill_(min(int(self.queue_count.item()) + 1, self.queue_size))

    def active(self) -> torch.Tensor:
        count = int(self.queue_count.item())
        return self.queue[:count]


class DiagnosticOrthogonalConditioner(nn.Module):
    """Build target-reference style codes with diagnosis leakage suppression."""

    def __init__(
        self,
        input_channels: int,
        style_dim: int,
        generator_style_dim: int,
        num_classes: int = 2,
        encoder_widths: Sequence[int] = (32, 64, 128, 256),
        projector_momentum: float = 0.95,
        grl_strength: float = 1.0,
        queue_size: int = 128,
        contrastive_temperature: float = 0.07,
        enable_projection: bool = True,
    ) -> None:
        super().__init__()
        self.style_dim = int(style_dim)
        self.generator_style_dim = int(generator_style_dim)
        self.num_classes = int(num_classes)
        self.grl_strength = float(grl_strength)
        self.contrastive_temperature = float(contrastive_temperature)
        self.enable_projection = bool(enable_projection)

        self.encoder = MultiScaleMomentStyleEncoder(
            input_channels=int(input_channels),
            style_dim=self.style_dim,
            widths=encoder_widths,
        )
        self.style_norm = nn.LayerNorm(self.style_dim)
        self.projector = DiagnosticSubspaceProjector(
            style_dim=self.style_dim,
            num_classes=self.num_classes,
            momentum=float(projector_momentum),
        )
        self.diagnostic_head = _MLPHead(self.style_dim, self.num_classes)
        self.domain_head = _MLPHead(self.style_dim, 2)
        self.to_generator_style = nn.Sequential(
            nn.LayerNorm(self.style_dim),
            nn.Linear(self.style_dim, self.generator_style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.generator_style_dim, self.generator_style_dim),
        )
        self.reference_queue = ReferenceStyleQueue(self.style_dim, queue_size=queue_size)
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))

    def encode_raw(self, image: torch.Tensor) -> torch.Tensor:
        return self.style_norm(self.encoder(image))

    def encode_projected(self, image: torch.Tensor) -> torch.Tensor:
        style = self.encode_raw(image)
        return self.projector(style) if self.enable_projection else style

    def encode_condition(self, target_reference: torch.Tensor) -> torch.Tensor:
        target_style = self.encode_projected(target_reference)
        return self.to_generator_style(target_style)

    def _instance_contrastive_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(query, dim=1)
        positive = F.normalize(positive, dim=1)
        positive_logit = (query * positive).sum(dim=1, keepdim=True)
        # The queue is updated later in the same forward pass. Clone the active
        # slice so autograd does not retain storage whose version will change.
        negatives = self.reference_queue.active().detach().clone()
        if negatives.numel() == 0:
            return query.sum() * 0.0
        negative_logits = query @ negatives.t()
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        logits = logits / self.contrastive_temperature
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, labels)

    def build_context(
        self,
        source_image: torch.Tensor,
        target_reference: torch.Tensor,
        source_labels: torch.Tensor,
        update_projector: bool = True,
        update_queue: bool = True,
    ) -> dict[str, torch.Tensor]:
        source_labels = source_labels.reshape(-1).long()
        source_raw = self.encode_raw(source_image)
        target_raw = self.encode_raw(target_reference)

        if update_projector and self.enable_projection:
            self.projector.update(source_raw.detach(), source_labels.detach())

        source_style = self.projector(source_raw) if self.enable_projection else source_raw
        target_style = self.projector(target_raw) if self.enable_projection else target_raw
        valid = (source_labels >= 0) & (source_labels < self.num_classes)
        if torch.any(valid):
            diagnostic_logits = self.diagnostic_head(
                gradient_reverse(source_style[valid], self.grl_strength)
            )
            diagnostic_loss = F.cross_entropy(diagnostic_logits, source_labels[valid])
            diagnostic_accuracy = (
                diagnostic_logits.argmax(dim=1) == source_labels[valid]
            ).float().mean()
        else:
            diagnostic_loss = source_style.sum() * 0.0
            diagnostic_accuracy = source_style.new_tensor(float("nan"))

        domain_style = torch.cat([source_style, target_style], dim=0)
        domain_labels = torch.cat(
            [
                torch.zeros(source_style.shape[0], dtype=torch.long, device=source_style.device),
                torch.ones(target_style.shape[0], dtype=torch.long, device=target_style.device),
            ],
            dim=0,
        )
        domain_logits = self.domain_head(domain_style)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        domain_accuracy = (domain_logits.argmax(dim=1) == domain_labels).float().mean()

        flipped_reference = torch.flip(target_reference, dims=(3,))
        positive_style = self.encode_projected(flipped_reference)
        instance_loss = self._instance_contrastive_loss(target_style, positive_style)
        if update_queue:
            self.reference_queue.enqueue(positive_style)

        condition = self.to_generator_style(target_style)
        return {
            "condition": condition,
            "source_style": source_style,
            "target_style": target_style,
            "diagnostic_loss": diagnostic_loss,
            "diagnostic_accuracy": diagnostic_accuracy,
            "domain_loss": domain_loss,
            "domain_accuracy": domain_accuracy,
            "instance_loss": instance_loss,
            "removed_energy": (
                self.projector.removed_energy_ratio(target_raw)
                if self.enable_projection
                else target_raw.sum() * 0.0
            ),
        }

    def style_reconstruction_loss(
        self,
        translated_image: torch.Tensor,
        target_reference: torch.Tensor,
    ) -> torch.Tensor:
        translated_style = F.normalize(self.encode_projected(translated_image), dim=1)
        with torch.no_grad():
            reference_style = F.normalize(self.encode_projected(target_reference), dim=1)
        return (1.0 - (translated_style * reference_style).sum(dim=1)).mean()

    def mix_with_noise(
        self,
        condition: torch.Tensor,
        noise_ratio: float,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ratio = float(noise_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("noise_ratio must be in [0, 1]")
        if ratio == 0.0:
            return condition
        if noise is None:
            noise = torch.randn_like(condition)
        style_scale = (1.0 - ratio**2) ** 0.5
        return style_scale * condition + ratio * noise

    @torch.no_grad()
    def advance_step(self) -> None:
        self.training_step.add_(1)

    def warmup_scale(self, warmup_steps: int) -> float:
        warmup_steps = int(warmup_steps)
        if warmup_steps <= 0:
            return 1.0
        return min(float(self.training_step.item()) / float(warmup_steps), 1.0)


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return true-class logit minus log-sum-exp of all competing classes."""

    if logits.ndim != 2:
        raise ValueError(f"Expected logits [B,C], got shape={tuple(logits.shape)}")
    labels = labels.reshape(-1).long()
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("logits and labels must have the same batch size")
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    class_mask = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    competing_logits = logits.masked_fill(class_mask, float("-inf"))
    competing = torch.logsumexp(competing_logits, dim=1)
    return true_logits - competing


def diagnostic_non_degradation_loss(
    teacher: nn.Module,
    source_image: torch.Tensor,
    translated_image: torch.Tensor,
    source_labels: torch.Tensor,
    tolerance: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize translated images whose true-label margin falls too far."""

    labels = source_labels.reshape(-1).long()
    valid = (labels >= 0)
    if not torch.any(valid):
        zero = translated_image.sum() * 0.0
        return zero, zero.detach()
    labels = labels[valid]
    with torch.no_grad():
        source_margin = true_class_margin(teacher(source_image[valid]), labels)
    translated_margin = true_class_margin(teacher(translated_image[valid]), labels)
    drop = source_margin.detach() - translated_margin
    loss = F.relu(drop - float(tolerance)).mean()
    return loss, drop.mean().detach()


def parse_widths(value: str) -> tuple[int, ...]:
    widths = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(f"Invalid encoder widths: {value!r}")
    return widths
