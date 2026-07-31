"""Target-reference style conditioning and output-safety modules for UNSB.

This file is intentionally self-contained so it can be copied into the upstream
UNSB ``models`` package by ``scripts/install_dosc_unsb_overlay.py``.

The historical ``dosc`` module and option prefixes are retained for checkpoint
and command compatibility. The experiments do not establish diagnostic
orthogonality: projection and gradient reversal are optional ablations, while
CIDP audits output-level diagnostic preservation.
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
            raise ValueError(
                f"Expected BCHW image tensor, got shape={tuple(image.shape)}"
            )
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
        self.register_buffer(
            "class_means", torch.zeros(self.num_classes, self.style_dim)
        )
        self.register_buffer(
            "class_initialized", torch.zeros(self.num_classes, dtype=torch.bool)
        )
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
            self.queue_count.fill_(
                min(int(self.queue_count.item()) + 1, self.queue_size)
            )

    def active(self) -> torch.Tensor:
        count = int(self.queue_count.item())
        return self.queue[:count]


class TargetReferenceStyleConditioner(nn.Module):
    """Build target-reference style codes with optional leakage-control ablations."""

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
        enable_projection: bool = False,
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
        self.reference_queue = ReferenceStyleQueue(
            self.style_dim, queue_size=queue_size
        )
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

        source_style = (
            self.projector(source_raw) if self.enable_projection else source_raw
        )
        target_style = (
            self.projector(target_raw) if self.enable_projection else target_raw
        )
        valid = (source_labels >= 0) & (source_labels < self.num_classes)
        if torch.any(valid):
            diagnostic_logits = self.diagnostic_head(
                gradient_reverse(source_style[valid], self.grl_strength)
            )
            diagnostic_loss = F.cross_entropy(diagnostic_logits, source_labels[valid])
            diagnostic_accuracy = (
                (diagnostic_logits.argmax(dim=1) == source_labels[valid]).float().mean()
            )
        else:
            diagnostic_loss = source_style.sum() * 0.0
            diagnostic_accuracy = source_style.new_tensor(float("nan"))

        domain_style = torch.cat([source_style, target_style], dim=0)
        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_style.shape[0], dtype=torch.long, device=source_style.device
                ),
                torch.ones(
                    target_style.shape[0], dtype=torch.long, device=target_style.device
                ),
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
            reference_style = F.normalize(
                self.encode_projected(target_reference), dim=1
            )
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


# Backward-compatible import name for existing overlays and external scripts.
# It must not be used as a claim that the representation is diagnostic-orthogonal.
DiagnosticOrthogonalConditioner = TargetReferenceStyleConditioner


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


def binary_diagnostic_score(logits: torch.Tensor) -> torch.Tensor:
    """Return the positive-versus-negative score ``z1 - z0``."""

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "CIDP requires binary teacher logits with shape [B,2], "
            f"got {tuple(logits.shape)}"
        )
    return logits[:, 1] - logits[:, 0]


class ClassBalancedScoreQueue(nn.Module):
    """Keep an equal-capacity FIFO of detached score pairs for each class."""

    def __init__(self, queue_size: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        if int(queue_size) < int(num_classes):
            raise ValueError("queue_size must be at least num_classes")
        self.num_classes = int(num_classes)
        self.capacity_per_class = int(queue_size) // self.num_classes
        self.queue_size = self.capacity_per_class * self.num_classes
        shape = (self.num_classes, self.capacity_per_class)
        self.register_buffer("source_scores", torch.zeros(shape))
        self.register_buffer("translated_scores", torch.zeros(shape))
        self.register_buffer(
            "pointers", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer("counts", torch.zeros(self.num_classes, dtype=torch.long))

    @torch.no_grad()
    def enqueue(
        self,
        source_score: torch.Tensor,
        translated_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        source_score = source_score.detach().reshape(-1)
        translated_score = translated_score.detach().reshape(-1)
        labels = labels.detach().reshape(-1).long()
        if not (source_score.shape == translated_score.shape == labels.shape):
            raise ValueError("score pairs and labels must have the same shape")
        valid = (labels >= 0) & (labels < self.num_classes)
        for source_value, translated_value, label in zip(
            source_score[valid],
            translated_score[valid],
            labels[valid],
        ):
            class_index = int(label.item())
            pointer = int(self.pointers[class_index].item())
            self.source_scores[class_index, pointer].copy_(source_value)
            self.translated_scores[class_index, pointer].copy_(translated_value)
            self.pointers[class_index] = (pointer + 1) % self.capacity_per_class
            self.counts[class_index] = min(
                int(self.counts[class_index].item()) + 1,
                self.capacity_per_class,
            )

    def active(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_rows = []
        translated_rows = []
        label_rows = []
        for class_index in range(self.num_classes):
            count = int(self.counts[class_index].item())
            if count == 0:
                continue
            source_rows.append(self.source_scores[class_index, :count])
            translated_rows.append(self.translated_scores[class_index, :count])
            label_rows.append(
                torch.full(
                    (count,),
                    class_index,
                    dtype=torch.long,
                    device=self.source_scores.device,
                )
            )
        if not source_rows:
            empty_score = self.source_scores.new_empty((0,))
            empty_label = self.counts.new_empty((0,))
            return empty_score, empty_score.clone(), empty_label
        return (
            torch.cat(source_rows),
            torch.cat(translated_rows),
            torch.cat(label_rows),
        )


class CalibrationInvariantDiagnosticPreservation(nn.Module):
    """Calibration-decoupled diagnostic non-degradation for binary diagnosis.

    A positive affine map is fitted on detached source/translated teacher
    scores from a class-balanced FIFO. The fitted map absorbs global offset and
    temperature drift. A calibrated true-class margin term handles local
    confidence loss, while a pairwise rank term preserves separability that no
    threshold or positive affine map can recover.
    """

    def __init__(
        self,
        queue_size: int = 128,
        min_per_class: int = 8,
        affine_ridge: float = 1e-4,
        min_scale: float = 0.05,
        max_scale: float = 20.0,
    ) -> None:
        super().__init__()
        if int(min_per_class) < 1:
            raise ValueError("min_per_class must be positive")
        if float(affine_ridge) < 0.0:
            raise ValueError("affine_ridge must be non-negative")
        if not 0.0 < float(min_scale) <= float(max_scale):
            raise ValueError("CIDP scales must satisfy 0 < min_scale <= max_scale")
        self.min_per_class = int(min_per_class)
        self.affine_ridge = float(affine_ridge)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.score_queue = ClassBalancedScoreQueue(
            queue_size=int(queue_size),
            num_classes=2,
        )
        if self.min_per_class > self.score_queue.capacity_per_class:
            raise ValueError("min_per_class cannot exceed the per-class queue capacity")

    @torch.no_grad()
    def _estimate_affine(
        self,
        translated_score: torch.Tensor,
        source_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit ``source ~= a * translated + b`` with equal class weight."""

        translated_score = translated_score.detach().reshape(-1)
        source_score = source_score.detach().reshape(-1)
        labels = labels.detach().reshape(-1).long()
        counts = torch.bincount(labels, minlength=2)
        ready = torch.all(counts[:2] >= self.min_per_class)
        if not bool(ready):
            one = translated_score.new_tensor(1.0)
            zero = translated_score.new_tensor(0.0)
            return one, zero, zero

        class_counts = counts[labels].to(dtype=translated_score.dtype)
        weights = class_counts.reciprocal()
        weights = weights / weights.sum().clamp_min(1e-12)
        translated_mean = (weights * translated_score).sum()
        source_mean = (weights * source_score).sum()
        centered_translated = translated_score - translated_mean
        centered_source = source_score - source_mean
        variance = (weights * centered_translated.square()).sum()
        covariance = (weights * centered_translated * centered_source).sum()
        scale = covariance / (variance + self.affine_ridge)
        scale = scale.clamp(self.min_scale, self.max_scale)
        bias = source_mean - scale * translated_mean
        return scale, bias, translated_score.new_tensor(1.0)

    @staticmethod
    def _binary_margin(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sign = labels.to(dtype=score.dtype).mul(2.0).sub(1.0)
        return sign * score

    @staticmethod
    def _rank_drop(
        source_positive: torch.Tensor,
        source_negative: torch.Tensor,
        translated_positive: torch.Tensor,
        translated_negative: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        if source_positive.numel() == 0 or source_negative.numel() == 0:
            return translated_positive.new_empty((0,))
        source_gap = source_positive[:, None] - source_negative[None, :]
        translated_gap = translated_positive[:, None] - translated_negative[None, :]
        # Do not preserve mistakes made by the frozen source teacher.
        source_correct = source_gap > 0.0
        if not torch.any(source_correct):
            return translated_gap.new_empty((0,))
        return F.relu(
            source_gap.detach()[source_correct]
            - translated_gap[source_correct]
            - float(tolerance)
        )

    def _pairwise_rank_loss(
        self,
        source_score: torch.Tensor,
        calibrated_score: torch.Tensor,
        labels: torch.Tensor,
        bank_source: torch.Tensor,
        bank_calibrated: torch.Tensor,
        bank_labels: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        current_positive = labels == 1
        current_negative = labels == 0
        bank_positive = bank_labels == 1
        bank_negative = bank_labels == 0
        drops = [
            self._rank_drop(
                source_score[current_positive],
                source_score[current_negative],
                calibrated_score[current_positive],
                calibrated_score[current_negative],
                tolerance,
            ),
            self._rank_drop(
                source_score[current_positive],
                bank_source[bank_negative],
                calibrated_score[current_positive],
                bank_calibrated[bank_negative],
                tolerance,
            ),
            self._rank_drop(
                bank_source[bank_positive],
                source_score[current_negative],
                bank_calibrated[bank_positive],
                calibrated_score[current_negative],
                tolerance,
            ),
        ]
        non_empty = [drop for drop in drops if drop.numel() > 0]
        if not non_empty:
            return calibrated_score.sum() * 0.0
        return torch.cat(non_empty).mean()

    def forward_scores(
        self,
        source_score: torch.Tensor,
        translated_score: torch.Tensor,
        labels: torch.Tensor,
        margin_tolerance: float = 0.1,
        rank_tolerance: float = 0.1,
        rank_weight: float = 1.0,
        update_queue: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute CIDP from paired teacher scores."""

        source_score = source_score.reshape(-1)
        translated_score = translated_score.reshape(-1)
        labels = labels.reshape(-1).long()
        if not (source_score.shape == translated_score.shape == labels.shape):
            raise ValueError("score pairs and labels must have the same shape")
        if torch.any((labels < 0) | (labels > 1)):
            raise ValueError("CIDP accepts binary labels 0/1 only")
        if source_score.numel() == 0:
            zero = translated_score.sum() * 0.0
            return {
                "loss": zero,
                "margin_loss": zero,
                "rank_loss": zero,
                "margin_drop": zero.detach(),
                "raw_margin_drop": zero.detach(),
                "affine_scale": zero.detach() + 1.0,
                "affine_bias": zero.detach(),
                "ready": zero.detach(),
                "class0_charge": zero.detach(),
                "class1_charge": zero.detach(),
            }

        bank_source, bank_translated, bank_labels = self.score_queue.active()
        fit_source = torch.cat([bank_source, source_score.detach()])
        fit_translated = torch.cat([bank_translated, translated_score.detach()])
        fit_labels = torch.cat([bank_labels, labels.detach()])
        scale, bias, ready = self._estimate_affine(
            fit_translated,
            fit_source,
            fit_labels,
        )
        scale = scale.detach()
        bias = bias.detach()
        calibrated_score = scale * translated_score + bias
        bank_calibrated = scale * bank_translated + bias

        source_margin = self._binary_margin(source_score.detach(), labels)
        raw_margin = self._binary_margin(translated_score, labels)
        calibrated_margin = self._binary_margin(calibrated_score, labels)
        margin_drop = source_margin - calibrated_margin
        margin_charge = F.relu(margin_drop - float(margin_tolerance))
        margin_loss = margin_charge.mean()
        rank_loss = self._pairwise_rank_loss(
            source_score.detach(),
            calibrated_score,
            labels,
            bank_source,
            bank_calibrated,
            bank_labels,
            tolerance=float(rank_tolerance),
        )
        total = ready * (margin_loss + float(rank_weight) * rank_loss)

        class_charges = []
        for class_index in range(2):
            class_mask = labels == class_index
            if torch.any(class_mask):
                class_charges.append(margin_charge[class_mask].mean().detach())
            else:
                class_charges.append(margin_charge.new_tensor(0.0))

        if update_queue:
            self.score_queue.enqueue(
                source_score,
                translated_score,
                labels,
            )
        raw_drop = source_margin - raw_margin
        return {
            "loss": total,
            "margin_loss": ready * margin_loss,
            "rank_loss": ready * rank_loss,
            "margin_drop": margin_drop.mean().detach(),
            "raw_margin_drop": raw_drop.mean().detach(),
            "affine_scale": scale,
            "affine_bias": bias,
            "ready": ready,
            "class0_charge": ready * class_charges[0],
            "class1_charge": ready * class_charges[1],
        }

    def forward(
        self,
        teacher: nn.Module,
        source_image: torch.Tensor,
        translated_image: torch.Tensor,
        source_labels: torch.Tensor,
        margin_tolerance: float = 0.1,
        rank_tolerance: float = 0.1,
        rank_weight: float = 1.0,
        update_queue: bool = True,
    ) -> dict[str, torch.Tensor]:
        labels = source_labels.reshape(-1).long()
        valid = (labels >= 0) & (labels <= 1)
        if not torch.any(valid):
            zero = translated_image.sum() * 0.0
            return self.forward_scores(
                zero.new_empty((0,)),
                zero.new_empty((0,)),
                labels.new_empty((0,)),
                margin_tolerance=margin_tolerance,
                rank_tolerance=rank_tolerance,
                rank_weight=rank_weight,
                update_queue=False,
            )
        labels = labels[valid]
        with torch.no_grad():
            source_score = binary_diagnostic_score(teacher(source_image[valid]))
        translated_score = binary_diagnostic_score(teacher(translated_image[valid]))
        return self.forward_scores(
            source_score,
            translated_score,
            labels,
            margin_tolerance=margin_tolerance,
            rank_tolerance=rank_tolerance,
            rank_weight=rank_weight,
            update_queue=update_queue,
        )


def diagnostic_non_degradation_loss(
    teacher: nn.Module,
    source_image: torch.Tensor,
    translated_image: torch.Tensor,
    source_labels: torch.Tensor,
    tolerance: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize translated images whose true-label margin falls too far."""

    labels = source_labels.reshape(-1).long()
    valid = labels >= 0
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
