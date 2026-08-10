"""Core modules for SafeBridge-UDA state selection and feature fusion.

The implementation is intentionally independent of the upstream UNSB package
so the geometry and protocol can be unit-tested in this repository.  No target
diagnosis label is accepted by any public API in this module.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_score(logits: torch.Tensor) -> torch.Tensor:
    """Return the positive-versus-negative logit for a binary classifier."""

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"Expected binary logits [B,2], got {tuple(logits.shape)}"
        )
    return logits[:, 1] - logits[:, 0]


def parse_candidate_states(value: str, num_timesteps: int) -> tuple[int, ...]:
    """Parse one-based UNSB output indices and enforce increasing depth."""

    try:
        states = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"Invalid candidate-state list: {value!r}") from error
    if not states:
        raise ValueError("At least one candidate state is required")
    if any(state < 1 or state > int(num_timesteps) for state in states):
        raise ValueError(
            f"Candidate states must lie in [1,{int(num_timesteps)}], got {states}"
        )
    if tuple(sorted(set(states))) != states:
        raise ValueError("Candidate states must be unique and strictly increasing")
    return states


class TargetDomainProbe(nn.Module):
    """Small probe that distinguishes source from real unlabeled target features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if int(feature_dim) < 1 or int(hidden_dim) < 2:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected pooled features [B,C], got {features.shape}")
        return self.network(features)


class TargetGradientSubspace(nn.Module):
    """Track target-domain gradient directions and their low-rank span.

    Gradients are normalized before entering the FIFO.  The right singular
    vectors of the uncentered gradient matrix define directions that the real
    target-domain probe currently uses.  The basis and target-score threshold
    are buffers, not trainable parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        rank: int = 8,
        queue_size: int = 128,
        min_samples: int = 16,
        target_quantile: float = 0.25,
        threshold_momentum: float = 0.90,
        minimum_domain_accuracy: float = 0.60,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        rank = int(rank)
        queue_size = int(queue_size)
        min_samples = int(min_samples)
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if rank < 1 or rank > feature_dim:
            raise ValueError("rank must lie in [1, feature_dim]")
        if queue_size < rank or min_samples < rank or min_samples > queue_size:
            raise ValueError("Require rank <= min_samples <= queue_size")
        if not 0.0 <= float(target_quantile) <= 1.0:
            raise ValueError("target_quantile must be in [0, 1]")
        if not 0.0 <= float(threshold_momentum) < 1.0:
            raise ValueError("threshold_momentum must be in [0, 1)")
        if not 0.0 <= float(minimum_domain_accuracy) <= 1.0:
            raise ValueError("minimum_domain_accuracy must be in [0, 1]")

        self.feature_dim = feature_dim
        self.maximum_rank = rank
        self.queue_size = queue_size
        self.min_samples = min_samples
        self.target_quantile = float(target_quantile)
        self.threshold_momentum = float(threshold_momentum)
        self.minimum_domain_accuracy = float(minimum_domain_accuracy)
        self.eps = float(eps)

        self.register_buffer(
            "gradient_queue", torch.zeros(queue_size, feature_dim)
        )
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("basis", torch.zeros(feature_dim, rank))
        self.register_buffer("basis_rank", torch.zeros((), dtype=torch.long))
        self.register_buffer("target_threshold", torch.zeros(()))
        self.register_buffer(
            "threshold_initialized", torch.zeros((), dtype=torch.bool)
        )
        self.register_buffer("domain_accuracy_ema", torch.zeros(()))
        self.register_buffer(
            "accuracy_initialized", torch.zeros((), dtype=torch.bool)
        )

    @torch.no_grad()
    def _enqueue_gradients(self, gradients: torch.Tensor) -> None:
        if gradients.ndim != 2 or gradients.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected gradients [B,{self.feature_dim}], got {gradients.shape}"
            )
        gradients = gradients.detach()
        finite = torch.isfinite(gradients).all(dim=1)
        norms = gradients.norm(dim=1)
        valid = finite & (norms > self.eps)
        normalized = gradients[valid] / norms[valid, None]
        for row in normalized:
            pointer = int(self.queue_pointer.item())
            self.gradient_queue[pointer].copy_(row)
            self.queue_pointer.fill_((pointer + 1) % self.queue_size)
            self.queue_count.fill_(min(int(self.queue_count.item()) + 1, self.queue_size))

    @torch.no_grad()
    def _refresh_basis(self) -> None:
        count = int(self.queue_count.item())
        self.basis.zero_()
        self.basis_rank.zero_()
        if count < self.min_samples:
            return
        active = self.gradient_queue[:count]
        # The FIFO is much shorter than the feature dimension.  Decomposing
        # its Gram matrix is equivalent to a full SVD but substantially cheaper.
        gram = active @ active.t()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        numerical_rank = int((eigenvalues > self.eps**2).sum().item())
        rank = min(numerical_rank, self.maximum_rank)
        if rank > 0:
            singular_values = eigenvalues[:rank].clamp_min(self.eps**2).sqrt()
            right_vectors = (
                active.t() @ eigenvectors[:, :rank]
            ) / singular_values[None, :]
            right_vectors, _ = torch.linalg.qr(right_vectors, mode="reduced")
            self.basis[:, :rank].copy_(right_vectors[:, :rank])
            self.basis_rank.fill_(rank)

    @torch.no_grad()
    def update(
        self,
        gradients: torch.Tensor,
        real_target_scores: torch.Tensor,
        domain_accuracy: torch.Tensor | float,
    ) -> None:
        """Update the basis and an unlabeled real-target adequacy threshold."""

        scores = real_target_scores.detach().reshape(-1)
        scores = scores[torch.isfinite(scores)]
        if scores.numel() == 0:
            raise ValueError("real_target_scores contains no finite value")
        batch_threshold = torch.quantile(scores, self.target_quantile)
        if bool(self.threshold_initialized):
            self.target_threshold.mul_(self.threshold_momentum).add_(
                batch_threshold, alpha=1.0 - self.threshold_momentum
            )
        else:
            self.target_threshold.copy_(batch_threshold)
            self.threshold_initialized.fill_(True)

        accuracy = torch.as_tensor(
            domain_accuracy,
            device=self.domain_accuracy_ema.device,
            dtype=self.domain_accuracy_ema.dtype,
        ).detach()
        if not torch.isfinite(accuracy):
            raise ValueError("domain_accuracy must be finite")
        if bool(self.accuracy_initialized):
            self.domain_accuracy_ema.mul_(0.90).add_(accuracy, alpha=0.10)
        else:
            self.domain_accuracy_ema.copy_(accuracy)
            self.accuracy_initialized.fill_(True)

        self._enqueue_gradients(gradients)
        self._refresh_basis()

    def is_ready(self) -> bool:
        return (
            int(self.queue_count.item()) >= self.min_samples
            and int(self.basis_rank.item()) > 0
            and bool(self.threshold_initialized)
            and bool(self.accuracy_initialized)
            and float(self.domain_accuracy_ema.item()) >= self.minimum_domain_accuracy
        )

    def project(self, residual: torch.Tensor) -> torch.Tensor:
        """Project BCHW or BKCHW residuals onto the active channel basis."""

        if residual.ndim not in (4, 5):
            raise ValueError("residual must be BCHW or BKCHW")
        channel_dim = 1 if residual.ndim == 4 else 2
        if residual.shape[channel_dim] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} channels, got {residual.shape[channel_dim]}"
            )
        rank = int(self.basis_rank.item())
        if rank == 0:
            return torch.zeros_like(residual)
        active_basis = self.basis[:, :rank].detach()
        if residual.ndim == 4:
            vectors = residual.permute(0, 2, 3, 1)
            projected = (vectors @ active_basis) @ active_basis.t()
            return projected.permute(0, 3, 1, 2)
        vectors = residual.permute(0, 1, 3, 4, 2)
        projected = (vectors @ active_basis) @ active_basis.t()
        return projected.permute(0, 1, 4, 2, 3)


class StateWiseDiagnosticCalibrator(nn.Module):
    """Positive-affine calibration and case-order checks for each bridge state."""

    def __init__(
        self,
        state_count: int,
        queue_size: int = 128,
        min_per_class: int = 8,
        affine_ridge: float = 1e-4,
        min_scale: float = 0.05,
        max_scale: float = 20.0,
    ) -> None:
        super().__init__()
        state_count = int(state_count)
        capacity = int(queue_size) // 2
        if state_count < 1:
            raise ValueError("state_count must be positive")
        if capacity < int(min_per_class):
            raise ValueError("queue_size must hold min_per_class items per class")
        if float(affine_ridge) < 0.0:
            raise ValueError("affine_ridge must be non-negative")
        if not 0.0 < float(min_scale) <= float(max_scale):
            raise ValueError("Require 0 < min_scale <= max_scale")
        self.state_count = state_count
        self.capacity = capacity
        self.min_per_class = int(min_per_class)
        self.affine_ridge = float(affine_ridge)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        shape = (state_count, 2, capacity)
        self.register_buffer("source_scores", torch.zeros(shape))
        self.register_buffer("candidate_scores", torch.zeros(shape))
        self.register_buffer(
            "pointers", torch.zeros(state_count, 2, dtype=torch.long)
        )
        self.register_buffer(
            "counts", torch.zeros(state_count, 2, dtype=torch.long)
        )

    @torch.no_grad()
    def _enqueue(
        self,
        source_score: torch.Tensor,
        candidate_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        for state in range(self.state_count):
            for source_value, candidate_value, label in zip(
                source_score.detach(), candidate_score[:, state].detach(), labels.detach()
            ):
                class_index = int(label.item())
                if class_index not in (0, 1):
                    continue
                pointer = int(self.pointers[state, class_index].item())
                self.source_scores[state, class_index, pointer].copy_(source_value)
                self.candidate_scores[state, class_index, pointer].copy_(candidate_value)
                self.pointers[state, class_index] = (pointer + 1) % self.capacity
                self.counts[state, class_index] = min(
                    int(self.counts[state, class_index].item()) + 1,
                    self.capacity,
                )

    def _active(self, state: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_rows = []
        candidate_rows = []
        label_rows = []
        for class_index in range(2):
            count = int(self.counts[state, class_index].item())
            source_rows.append(self.source_scores[state, class_index, :count])
            candidate_rows.append(self.candidate_scores[state, class_index, :count])
            label_rows.append(
                torch.full(
                    (count,),
                    class_index,
                    dtype=torch.long,
                    device=self.source_scores.device,
                )
            )
        return (
            torch.cat(source_rows),
            torch.cat(candidate_rows),
            torch.cat(label_rows),
        )

    @torch.no_grad()
    def _estimate_affine(self, state: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
        ready = bool(torch.all(self.counts[state] >= self.min_per_class))
        one = self.source_scores.new_tensor(1.0)
        zero = self.source_scores.new_tensor(0.0)
        if not ready:
            return one, zero, False
        source, candidate, labels = self._active(state)
        class_counts = torch.bincount(labels, minlength=2)[labels].to(source.dtype)
        weights = class_counts.reciprocal()
        weights = weights / weights.sum().clamp_min(1e-12)
        candidate_mean = (weights * candidate).sum()
        source_mean = (weights * source).sum()
        centered_candidate = candidate - candidate_mean
        centered_source = source - source_mean
        variance = (weights * centered_candidate.square()).sum()
        covariance = (weights * centered_candidate * centered_source).sum()
        scale = covariance / (variance + self.affine_ridge)
        scale = scale.clamp(self.min_scale, self.max_scale)
        bias = source_mean - scale * candidate_mean
        return scale, bias, True

    def _rank_violation(
        self,
        state: int,
        source_score: torch.Tensor,
        calibrated_score: torch.Tensor,
        labels: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        bank_source, bank_candidate, bank_labels = self._active(state)
        bank_calibrated = scale * bank_candidate + bias
        batch_size, reference_count = calibrated_score.shape
        result = calibrated_score.new_zeros((batch_size, reference_count))
        for batch_index in range(batch_size):
            label = int(labels[batch_index].item())
            opposite = bank_labels == (1 - label)
            if not torch.any(opposite):
                continue
            if label == 1:
                source_gap = source_score[batch_index] - bank_source[opposite]
                candidate_gap = (
                    calibrated_score[batch_index, :, None]
                    - bank_calibrated[opposite][None, :]
                )
            else:
                source_gap = bank_source[opposite] - source_score[batch_index]
                candidate_gap = (
                    bank_calibrated[opposite][None, :]
                    - calibrated_score[batch_index, :, None]
                )
            source_correct = source_gap > 0.0
            if torch.any(source_correct):
                drops = F.relu(
                    source_gap.detach()[None, source_correct]
                    - candidate_gap[:, source_correct]
                    - float(tolerance)
                )
                result[batch_index] = drops.mean(dim=1)
        return result

    @torch.no_grad()
    def evaluate_and_update(
        self,
        source_score: torch.Tensor,
        candidate_score: torch.Tensor,
        labels: torch.Tensor,
        margin_tolerance: float = 0.10,
        rank_tolerance: float = 0.10,
        maximum_rank_violation: float = 0.10,
    ) -> dict[str, torch.Tensor]:
        """Evaluate B-by-K-by-S candidate scores after state-wise calibration."""

        if candidate_score.ndim != 3:
            raise ValueError("candidate_score must be [B,K,S]")
        batch_size, _, state_count = candidate_score.shape
        source_score = source_score.reshape(-1)
        labels = labels.reshape(-1).long()
        if source_score.shape != (batch_size,) or labels.shape != (batch_size,):
            raise ValueError("source_score, labels, and candidates must share B")
        if state_count != self.state_count:
            raise ValueError(
                f"Expected {self.state_count} states, got {state_count}"
            )
        if torch.any((labels < 0) | (labels > 1)):
            raise ValueError("Only binary source labels 0/1 are supported")
        if min(float(margin_tolerance), float(rank_tolerance), float(maximum_rank_violation)) < 0.0:
            raise ValueError("Diagnostic tolerances must be non-negative")

        self._enqueue(source_score, candidate_score.mean(dim=1), labels)
        calibrated = torch.empty_like(candidate_score)
        margin_drop = torch.empty_like(candidate_score)
        rank_violation = torch.empty_like(candidate_score)
        scales = source_score.new_empty((state_count,))
        biases = source_score.new_empty((state_count,))
        ready = torch.zeros(state_count, dtype=torch.bool, device=source_score.device)
        sign = labels.to(source_score.dtype).mul(2.0).sub(1.0)
        source_margin = sign * source_score
        for state in range(state_count):
            scale, bias, state_ready = self._estimate_affine(state)
            current = scale * candidate_score[:, :, state] + bias
            current_margin = sign[:, None] * current
            current_drop = source_margin[:, None] - current_margin
            current_rank = self._rank_violation(
                state,
                source_score,
                current,
                labels,
                scale,
                bias,
                tolerance=rank_tolerance,
            )
            calibrated[:, :, state] = current
            margin_drop[:, :, state] = current_drop
            rank_violation[:, :, state] = current_rank
            scales[state] = scale
            biases[state] = bias
            ready[state] = state_ready

        safe = (
            ready[None, None, :]
            & (margin_drop <= float(margin_tolerance))
            & (rank_violation <= float(maximum_rank_violation))
        )
        return {
            "safe": safe,
            "calibrated_score": calibrated,
            "margin_drop": margin_drop,
            "rank_violation": rank_violation,
            "affine_scale": scales,
            "affine_bias": biases,
            "ready": ready,
        }


def state_residual_agreement(
    source_features: torch.Tensor,
    candidate_features: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Measure agreement of two stochastic trajectories for every state."""

    if source_features.ndim != 4 or candidate_features.ndim != 7:
        raise ValueError("Expected source BCHW and candidates BKRSC... tensors")
    if candidate_features.shape[2] != 2:
        raise ValueError("Exactly two stochastic trajectories are required")
    if candidate_features.shape[0] != source_features.shape[0]:
        raise ValueError("Source and candidate batches differ")
    source = source_features[:, None, None]
    delta_first = candidate_features[:, :, 0] - source
    delta_second = candidate_features[:, :, 1] - source
    difference = (delta_first - delta_second).flatten(3).norm(dim=3)
    magnitude = (
        delta_first.flatten(3).norm(dim=3)
        + delta_second.flatten(3).norm(dim=3)
    )
    return (1.0 - difference / magnitude.clamp_min(float(eps))).clamp(0.0, 1.0)


def select_minimal_adequate_state(
    diagnostic_safe: torch.Tensor,
    domain_scores: torch.Tensor,
    source_domain_score: torch.Tensor,
    target_threshold: torch.Tensor | float,
    stability: torch.Tensor,
    state_ids: Sequence[int],
    minimum_domain_gain: float = 0.0,
    minimum_stability: float = 0.60,
    selector_ready: bool = True,
) -> dict[str, torch.Tensor]:
    """Choose the shallowest safe state that reaches real-target support."""

    if diagnostic_safe.ndim != 3 or domain_scores.shape != diagnostic_safe.shape:
        raise ValueError("diagnostic_safe and domain_scores must be [B,K,S]")
    if stability.shape != diagnostic_safe.shape:
        raise ValueError("stability must match candidate state tensors")
    batch_size, reference_count, state_count = diagnostic_safe.shape
    if len(tuple(state_ids)) != state_count:
        raise ValueError("state_ids does not match the state dimension")
    source_domain_score = source_domain_score.reshape(-1)
    if source_domain_score.shape != (batch_size,):
        raise ValueError("source_domain_score must have shape [B]")

    threshold = torch.as_tensor(
        target_threshold,
        device=domain_scores.device,
        dtype=domain_scores.dtype,
    )
    domain_gain = domain_scores - source_domain_score[:, None, None]
    eligible = (
        diagnostic_safe
        & (stability >= float(minimum_stability))
        & (domain_gain > float(minimum_domain_gain))
        & (domain_scores >= threshold)
    )
    if not selector_ready:
        eligible = torch.zeros_like(eligible)
    accepted = eligible.any(dim=2)
    selected_index = eligible.to(torch.int64).argmax(dim=2)
    state_tensor = torch.as_tensor(
        tuple(state_ids), device=selected_index.device, dtype=torch.long
    )
    selected_state = state_tensor[selected_index]
    selected_state = torch.where(
        accepted, selected_state, torch.full_like(selected_state, -1)
    )
    selected_index = torch.where(
        accepted, selected_index, torch.zeros_like(selected_index)
    )
    return {
        "accepted": accepted,
        "selected_index": selected_index,
        "selected_state": selected_state,
        "eligible": eligible,
        "domain_gain": domain_gain,
    }


def gather_state_features(
    candidate_features: torch.Tensor,
    selected_index: torch.Tensor,
) -> torch.Tensor:
    """Gather B-by-K state indices from B-by-K-by-R-by-S feature maps."""

    if candidate_features.ndim != 7:
        raise ValueError("candidate_features must be [B,K,R,S,C,H,W]")
    batch_size, reference_count, rollout_count, _, channels, height, width = (
        candidate_features.shape
    )
    if selected_index.shape != (batch_size, reference_count):
        raise ValueError("selected_index must be [B,K]")
    index = selected_index[:, :, None, None, None, None, None].expand(
        -1, -1, rollout_count, 1, channels, height, width
    )
    return torch.gather(candidate_features, dim=3, index=index).squeeze(3)


def build_stable_residual(
    source_features: torch.Tensor,
    selected_features: torch.Tensor,
    noise_floor: float = 0.01,
    spatial_threshold: float = 1.0,
    channel_threshold: float = 1.0,
    spatial_temperature: float = 0.20,
    channel_temperature: float = 0.20,
) -> dict[str, torch.Tensor]:
    """Filter the mean two-rollout residual with spatial and channel SNR gates."""

    if source_features.ndim != 4 or selected_features.ndim != 6:
        raise ValueError("Expected source BCHW and selected BKRCHW")
    if selected_features.shape[2] != 2:
        raise ValueError("Exactly two selected rollouts are required")
    if min(
        float(noise_floor),
        float(spatial_temperature),
        float(channel_temperature),
    ) <= 0.0:
        raise ValueError("Noise floor and temperatures must be positive")
    source = source_features[:, None]
    delta_first = selected_features[:, :, 0] - source
    delta_second = selected_features[:, :, 1] - source
    mean_delta = 0.5 * (delta_first + delta_second)
    variance = 0.5 * (
        (delta_first - mean_delta).square()
        + (delta_second - mean_delta).square()
    )
    reliability = mean_delta.abs() / torch.sqrt(
        variance + float(noise_floor) ** 2
    )
    spatial_reliability = reliability.mean(dim=2, keepdim=True)
    channel_reliability = reliability.mean(dim=(3, 4), keepdim=True)
    spatial_mask = torch.sigmoid(
        (spatial_reliability - float(spatial_threshold))
        / float(spatial_temperature)
    )
    channel_mask = torch.sigmoid(
        (channel_reliability - float(channel_threshold))
        / float(channel_temperature)
    )
    mask = (spatial_mask * channel_mask).detach()
    return {
        "residual": mask * mean_delta,
        "mean_delta": mean_delta,
        "variance": variance,
        "mask": mask,
        "mask_mean": mask.mean().detach(),
    }


def project_diagnostic_nullspace(
    residual: torch.Tensor,
    diagnostic_gradient: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Remove each residual's component along the local diagnosis Jacobian."""

    if residual.ndim != 5 or diagnostic_gradient.ndim != 4:
        raise ValueError("Expected residual BKCHW and gradient BCHW")
    if residual.shape[0] != diagnostic_gradient.shape[0] or residual.shape[2:] != diagnostic_gradient.shape[1:]:
        raise ValueError("Residual and diagnostic-gradient shapes are incompatible")
    gradient = diagnostic_gradient.detach()[:, None]
    inner = (residual * gradient).sum(dim=(2, 3, 4), keepdim=True)
    denominator = gradient.square().sum(dim=(2, 3, 4), keepdim=True)
    projected = residual - gradient * inner / denominator.clamp_min(float(eps))
    post_inner = (projected * gradient).sum(dim=(2, 3, 4))
    return {
        "residual": projected,
        "pre_inner": inner.squeeze(-1).squeeze(-1).squeeze(-1).detach(),
        "post_inner": post_inner.detach(),
    }


def choose_backtracking_alpha(
    domain_scores: torch.Tensor,
    true_class_margins: torch.Tensor,
    source_domain_score: torch.Tensor,
    source_true_class_margin: torch.Tensor,
    accepted: torch.Tensor,
    alphas: Sequence[float],
    minimum_domain_gain: float = 0.0,
    margin_tolerance: float = 0.10,
    required_domain_gain: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Choose the largest tested strength satisfying nonlinear checks."""

    if domain_scores.ndim != 3 or true_class_margins.shape != domain_scores.shape:
        raise ValueError("Backtracking score tensors must be [B,K,A]")
    batch_size, reference_count, alpha_count = domain_scores.shape
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if len(alpha_values) != alpha_count or not alpha_values:
        raise ValueError("alphas does not match the score tensor")
    if any(alpha <= 0.0 for alpha in alpha_values):
        raise ValueError("Backtracking alphas must be positive")
    if any(left < right for left, right in zip(alpha_values, alpha_values[1:])):
        raise ValueError("Backtracking alphas must be ordered from largest to smallest")
    if accepted.shape != (batch_size, reference_count):
        raise ValueError("accepted must be [B,K]")

    source_domain_score = source_domain_score.reshape(batch_size)
    source_true_class_margin = source_true_class_margin.reshape(batch_size)
    required_gain = domain_scores.new_full(
        (batch_size, reference_count),
        float(minimum_domain_gain),
    )
    if required_domain_gain is not None:
        if required_domain_gain.shape != (batch_size, reference_count):
            raise ValueError("required_domain_gain must be [B,K]")
        required_gain = torch.maximum(
            required_gain,
            required_domain_gain.to(domain_scores.dtype),
        )
    valid = (
        accepted[:, :, None]
        & (
            domain_scores
            >= source_domain_score[:, None, None] + required_gain[:, :, None]
        )
        & (
            true_class_margins
            >= source_true_class_margin[:, None, None] - float(margin_tolerance)
        )
    )
    final_accepted = valid.any(dim=2)
    selected_index = valid.to(torch.int64).argmax(dim=2)
    alpha_tensor = domain_scores.new_tensor(alpha_values)
    selected_alpha = alpha_tensor[selected_index]
    selected_alpha = torch.where(
        final_accepted, selected_alpha, torch.zeros_like(selected_alpha)
    )
    return {
        "accepted": final_accepted,
        "alpha": selected_alpha,
        "valid": valid,
        "selected_index": selected_index,
    }


def masked_multi_reference_cross_entropy(
    raw_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    accepted: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Use equal raw/candidate group weight only when a case has a candidate."""

    if raw_logits.ndim != 2 or candidate_logits.ndim != 3:
        raise ValueError("Expected raw [B,C] and candidate [B,K,C] logits")
    batch_size, reference_count, class_count = candidate_logits.shape
    if raw_logits.shape != (batch_size, class_count):
        raise ValueError("Raw and candidate logits are incompatible")
    if labels.shape != (batch_size,) or accepted.shape != (batch_size, reference_count):
        raise ValueError("labels or accepted has an incompatible shape")

    raw_per_case = F.cross_entropy(raw_logits, labels, reduction="none")
    repeated_labels = labels[:, None].expand(-1, reference_count).reshape(-1)
    candidate_per_view = F.cross_entropy(
        candidate_logits.reshape(-1, class_count),
        repeated_labels,
        reduction="none",
    ).reshape(batch_size, reference_count)
    weights = accepted.to(candidate_per_view.dtype)
    counts = weights.sum(dim=1)
    candidate_per_case = (candidate_per_view * weights).sum(dim=1) / counts.clamp_min(1.0)
    has_candidate = counts > 0
    total_per_case = torch.where(
        has_candidate,
        0.5 * (raw_per_case + candidate_per_case),
        raw_per_case,
    )
    accepted_count = weights.sum()
    candidate_mean = (candidate_per_view * weights).sum() / accepted_count.clamp_min(1.0)
    candidate_mean = torch.where(
        accepted_count > 0,
        candidate_mean,
        candidate_per_view.sum() * 0.0,
    )
    return {
        "total": total_per_case.mean(),
        "raw": raw_per_case.mean(),
        "candidate": candidate_mean,
        "accepted_fraction": weights.mean().detach(),
        "accepted_case_fraction": has_candidate.float().mean().detach(),
    }
