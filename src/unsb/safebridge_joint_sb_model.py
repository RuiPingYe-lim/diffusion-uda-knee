"""SafeBridge-UDA: state rejection and diagnosis-prioritized feature fusion."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import util

from .dosc_modules import true_class_margin
from .dosc_sb_model import DoscSBModel
from .safebridge_modules import (
    StateWiseDiagnosticCalibrator,
    TargetDomainProbe,
    TargetGradientSubspace,
    binary_score,
    build_stable_residual,
    choose_backtracking_alpha,
    gather_state_features,
    masked_multi_reference_cross_entropy,
    parse_candidate_states,
    project_diagnostic_nullspace,
    select_minimal_adequate_state,
    state_residual_agreement,
)
from .trsc_joint_sb_model import TrscJointSBModel


class SafebridgeJointSBModel(TrscJointSBModel):
    """Train UNSB normally while adapting C with selected safe feature changes.

    The classifier loss never enters the translator.  Candidate trajectories
    are rendered under ``torch.no_grad``; G/S still receive their original
    GAN, bridge, PatchNCE, and target-reference style objectives.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = TrscJointSBModel.modify_commandline_options(
            parser,
            is_train=is_train,
        )
        parser.add_argument(
            "--safebridge_candidate_states",
            type=str,
            default="1,3,5",
            help="One-based UNSB outputs considered by DSSR",
        )
        parser.add_argument(
            "--safebridge_fusion_mode",
            choices=(
                "raw_only",
                "selected_feature",
                "stable_residual",
                "target_projected",
                "full",
            ),
            default="full",
        )
        parser.add_argument(
            "--safebridge_teacher_path",
            type=str,
            default="",
            help="Frozen render-robust binary diagnostic anchor",
        )
        parser.add_argument("--safebridge_diag_queue_size", type=int, default=128)
        parser.add_argument("--safebridge_diag_min_per_class", type=int, default=8)
        parser.add_argument("--safebridge_diag_affine_ridge", type=float, default=1e-4)
        parser.add_argument("--safebridge_diag_min_scale", type=float, default=0.05)
        parser.add_argument("--safebridge_diag_max_scale", type=float, default=20.0)
        parser.add_argument("--safebridge_diag_margin_tolerance", type=float, default=0.10)
        parser.add_argument("--safebridge_diag_rank_tolerance", type=float, default=0.10)
        parser.add_argument("--safebridge_diag_max_rank_violation", type=float, default=0.10)
        parser.add_argument("--safebridge_minimum_stability", type=float, default=0.60)
        parser.add_argument("--safebridge_state_minimum_domain_gain", type=float, default=0.0)

        parser.add_argument("--safebridge_domain_hidden_dim", type=int, default=256)
        parser.add_argument("--safebridge_domain_lr", type=float, default=1e-4)
        parser.add_argument("--safebridge_domain_weight_decay", type=float, default=1e-4)
        parser.add_argument("--safebridge_target_rank", type=int, default=8)
        parser.add_argument("--safebridge_target_queue_size", type=int, default=128)
        parser.add_argument("--safebridge_target_min_samples", type=int, default=16)
        parser.add_argument("--safebridge_target_quantile", type=float, default=0.25)
        parser.add_argument("--safebridge_target_threshold_momentum", type=float, default=0.90)
        parser.add_argument("--safebridge_domain_min_accuracy", type=float, default=0.60)

        parser.add_argument("--safebridge_noise_floor", type=float, default=0.01)
        parser.add_argument("--safebridge_spatial_threshold", type=float, default=1.0)
        parser.add_argument("--safebridge_channel_threshold", type=float, default=1.0)
        parser.add_argument("--safebridge_spatial_temperature", type=float, default=0.20)
        parser.add_argument("--safebridge_channel_temperature", type=float, default=0.20)
        parser.add_argument(
            "--safebridge_backtracking_alphas",
            type=str,
            default="1.0,0.5,0.25,0.125",
        )
        parser.add_argument("--safebridge_backtracking_domain_gain", type=float, default=0.0)
        parser.add_argument(
            "--safebridge_backtracking_gain_retention",
            type=float,
            default=0.50,
            help="Minimum fraction of the selected state's domain gain to retain",
        )
        parser.add_argument("--safebridge_backtracking_margin_tolerance", type=float, default=0.10)
        parser.add_argument(
            "--safebridge_freeze_classifier_bn",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Freeze BN statistics so per-case diagnosis Jacobians do not mix cases",
        )
        parser.add_argument(
            "--safebridge_audit_jsonl",
            type=str,
            default="",
            help="Optional per-source/reference DSSR and backtracking audit log",
        )
        parser.set_defaults(
            dataset_mode="trsc_unaligned",
            trsc_num_references=1,
            lambda_TRSC_task=0.0,
            dosc_noise_ratio=0.0,
            lambda_DOSC_diag=0.0,
            lambda_DOSC_safe=0.0,
        )
        return parser

    def __init__(self, opt):
        self._candidate_states = parse_candidate_states(
            opt.safebridge_candidate_states,
            opt.num_timesteps,
        )
        self._backtracking_alphas = self._parse_backtracking_alphas(
            opt.safebridge_backtracking_alphas
        )
        self._validate_safebridge_options(opt)
        super().__init__(opt)

        feature_dim = 1024
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            self.netH = TargetDomainProbe(
                feature_dim=feature_dim,
                hidden_dim=opt.safebridge_domain_hidden_dim,
            ).to(self.device)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        self.netT = TargetGradientSubspace(
            feature_dim=feature_dim,
            rank=opt.safebridge_target_rank,
            queue_size=opt.safebridge_target_queue_size,
            min_samples=opt.safebridge_target_min_samples,
            target_quantile=opt.safebridge_target_quantile,
            threshold_momentum=opt.safebridge_target_threshold_momentum,
            minimum_domain_accuracy=opt.safebridge_domain_min_accuracy,
        ).to(self.device)
        self.netJ = StateWiseDiagnosticCalibrator(
            state_count=len(self._candidate_states),
            queue_size=opt.safebridge_diag_queue_size,
            min_per_class=opt.safebridge_diag_min_per_class,
            affine_ridge=opt.safebridge_diag_affine_ridge,
            min_scale=opt.safebridge_diag_min_scale,
            max_scale=opt.safebridge_diag_max_scale,
        ).to(self.device)
        self.model_names += ["H", "T", "J"]

        self.netSafeBridgeTeacher = None
        if self.isTrain:
            self.netSafeBridgeTeacher = self._load_safebridge_teacher(
                opt.safebridge_teacher_path
            )
            self.optimizer_H = torch.optim.AdamW(
                self.netH.parameters(),
                lr=float(opt.safebridge_domain_lr),
                weight_decay=float(opt.safebridge_domain_weight_decay),
            )
            self.optimizers.append(self.optimizer_H)
            self.loss_names += [
                "SB_domain_probe",
                "SB_domain_accuracy",
                "SB_domain_accuracy_ema",
                "SB_target_threshold",
                "SB_basis_rank",
                "SB_selector_ready",
                "SB_accept",
                "SB_accept_c0",
                "SB_accept_c1",
                "SB_selected_depth",
                "SB_diag_margin_drop",
                "SB_diag_rank_violation",
                "SB_stability",
                "SB_reliability_mask",
                "SB_target_energy_retention",
                "SB_domain_gain_retention",
                "SB_null_inner",
                "SB_alpha",
            ]

        self.state_candidates: torch.Tensor | None = None
        self._raw_logits: torch.Tensor | None = None
        self._safe_logits: torch.Tensor | None = None
        self._safe_acceptance: torch.Tensor | None = None
        self._probe_source_pool: torch.Tensor | None = None
        self._probe_target_pool: torch.Tensor | None = None
        self._source_domain_score: torch.Tensor | None = None
        self._source_true_margin: torch.Tensor | None = None
        self._audit_source_paths: list[str] = []
        self._audit_reference_paths: list[list[str]] = []

    @staticmethod
    def _parse_backtracking_alphas(value: str) -> tuple[float, ...]:
        try:
            alphas = tuple(
                float(item.strip()) for item in str(value).split(",") if item.strip()
            )
        except ValueError as error:
            raise ValueError(f"Invalid backtracking alphas: {value!r}") from error
        if not alphas or any(alpha <= 0.0 for alpha in alphas):
            raise ValueError("Backtracking alphas must be positive")
        if any(left < right for left, right in zip(alphas, alphas[1:])):
            raise ValueError("Backtracking alphas must be descending")
        return alphas

    @staticmethod
    def _validate_safebridge_options(opt) -> None:
        if int(opt.dosc_num_classes) != 2:
            raise ValueError("SafeBridge-UDA currently supports binary diagnosis only")
        if float(opt.lambda_TRSC_task) != 0.0:
            raise ValueError(
                "SafeBridge-UDA blocks classifier gradients at translated images; "
                "set --lambda_TRSC_task 0.0"
            )
        non_negative = (
            "safebridge_diag_margin_tolerance",
            "safebridge_diag_rank_tolerance",
            "safebridge_diag_max_rank_violation",
            "safebridge_state_minimum_domain_gain",
            "safebridge_backtracking_domain_gain",
            "safebridge_backtracking_margin_tolerance",
        )
        for name in non_negative:
            if float(getattr(opt, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(opt.safebridge_minimum_stability) <= 1.0:
            raise ValueError("safebridge_minimum_stability must be in [0, 1]")
        if not 0.0 <= float(opt.safebridge_backtracking_gain_retention) <= 1.0:
            raise ValueError(
                "safebridge_backtracking_gain_retention must be in [0, 1]"
            )

    def _load_safebridge_teacher(self, value: str):
        path_value = str(value).strip()
        if not path_value:
            raise ValueError(
                "SafeBridge training requires --safebridge_teacher_path"
            )
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SafeBridge diagnostic anchor not found: {path}")
        teacher = torch.jit.load(str(path), map_location=self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f"[SafeBridge] loaded frozen diagnostic anchor: {path}")
        return teacher

    def _freeze_classifier_bn(self) -> None:
        if not bool(self.opt.safebridge_freeze_classifier_bn):
            return
        for module in self.netC.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    @staticmethod
    def _unwrap(network: nn.Module) -> nn.Module:
        return network.module if hasattr(network, "module") else network

    def _classifier_module(self):
        return self._unwrap(self.netC)

    def _domain_probe_module(self):
        return self._unwrap(self.netH)

    def _target_subspace_module(self):
        return self._unwrap(self.netT)

    def _diagnostic_calibrator_module(self):
        return self._unwrap(self.netJ)

    def set_input(self, input, input2=None):
        super().set_input(input, input2)
        if not self.isTrain:
            return
        source_paths = input.get("A_paths", [])
        if isinstance(source_paths, str):
            source_paths = [source_paths]
        self._audit_source_paths = [str(path) for path in source_paths]
        reference_paths = input.get("B_ref_paths", [])
        if reference_paths:
            if isinstance(reference_paths[0], str):
                reference_paths = [[str(path)] for path in reference_paths]
            elif len(reference_paths) == int(self.opt.trsc_num_references):
                # Default collation transposes a list-valued field into K rows.
                reference_paths = list(map(list, zip(*reference_paths)))
        self._audit_reference_paths = [
            [str(path) for path in row] for row in reference_paths
        ]

    def _classifier_resize(self, images: torch.Tensor) -> torch.Tensor:
        size = int(self.opt.trsc_classifier_input_size)
        return F.interpolate(
            images,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def generate_state_candidates(self) -> torch.Tensor:
        """Render two stochastic trajectories for each source/reference pair."""

        if self.real_B_references is None:
            raise RuntimeError("Target references are not initialized")
        references = self.real_B_references
        batch_size, reference_count = references.shape[:2]
        source = (
            self.real_A[:, None]
            .expand(-1, reference_count, -1, -1, -1)
            .reshape(batch_size * reference_count, *self.real_A.shape[1:])
        )
        flat_references = references.reshape(
            batch_size * reference_count,
            *references.shape[2:],
        )
        condition = self._style_module().encode_condition(flat_references)
        rollouts = []
        for _ in range(2):
            path = self.translate_with_condition(source, condition)
            selected = torch.stack(
                [path[state - 1] for state in self._candidate_states],
                dim=1,
            )
            rollouts.append(
                selected.reshape(
                    batch_size,
                    reference_count,
                    len(self._candidate_states),
                    *selected.shape[2:],
                )
            )
        return torch.stack(rollouts, dim=2)

    @staticmethod
    def _mean_over_accepted(
        values: torch.Tensor,
        accepted: torch.Tensor,
    ) -> torch.Tensor:
        weights = accepted.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def _diagnostic_scores(
        self,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.netSafeBridgeTeacher is None:
            raise RuntimeError("Diagnostic anchor is not initialized")
        batch_size, reference_count, rollout_count, state_count = candidates.shape[:4]
        flat = candidates.reshape(-1, *candidates.shape[4:])
        with torch.no_grad():
            source_score = binary_score(self.netSafeBridgeTeacher(self.real_A))
            candidate_score = binary_score(self.netSafeBridgeTeacher(flat)).reshape(
                batch_size,
                reference_count,
                rollout_count,
                state_count,
            )
        return source_score, candidate_score.mean(dim=2)

    def _extract_candidate_features(
        self,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, reference_count, rollout_count, state_count = candidates.shape[:4]
        flat = candidates.detach().reshape(-1, *candidates.shape[4:])
        features = self._classifier_module().forward_to_layer3(
            self._classifier_resize(flat)
        )
        return features.reshape(
            batch_size,
            reference_count,
            rollout_count,
            state_count,
            *features.shape[1:],
        )

    def _backtrack(
        self,
        source_features: torch.Tensor,
        residual: torch.Tensor,
        accepted: torch.Tensor,
        source_domain_score: torch.Tensor,
        source_margin: torch.Tensor,
        required_domain_gain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, reference_count = residual.shape[:2]
        domain_rows = []
        margin_rows = []
        classifier = self._classifier_module()
        domain_probe = self._domain_probe_module()
        with torch.no_grad():
            for alpha in self._backtracking_alphas:
                fused = (
                    source_features.detach()[:, None]
                    + float(alpha) * residual.detach()
                ).reshape(-1, *source_features.shape[1:])
                logits = classifier.forward_from_layer3(fused)
                labels = self.real_A_label[:, None].expand(
                    -1, reference_count
                ).reshape(-1)
                margin_rows.append(
                    true_class_margin(logits, labels).reshape(
                        batch_size, reference_count
                    )
                )
                pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
                domain_rows.append(
                    binary_score(domain_probe(pooled)).reshape(
                        batch_size, reference_count
                    )
                )
        domain_scores = torch.stack(domain_rows, dim=2)
        margins = torch.stack(margin_rows, dim=2)
        decision = choose_backtracking_alpha(
            domain_scores,
            margins,
            source_domain_score,
            source_margin,
            accepted,
            self._backtracking_alphas,
            minimum_domain_gain=self.opt.safebridge_backtracking_domain_gain,
            margin_tolerance=self.opt.safebridge_backtracking_margin_tolerance,
            required_domain_gain=required_domain_gain,
        )
        decision["selected_domain_score"] = torch.gather(
            domain_scores,
            dim=2,
            index=decision["selected_index"][:, :, None],
        ).squeeze(2)
        return decision

    def _prepare_safebridge_task(self) -> None:
        if self.state_candidates is None:
            raise RuntimeError("SafeBridge candidates are not initialized")
        candidates = self.state_candidates
        batch_size, reference_count, rollout_count, state_count = candidates.shape[:4]
        source_input = self._classifier_resize(self.real_A)
        target_input = self._classifier_resize(self.real_B)
        classifier = self._classifier_module()
        domain_probe = self._domain_probe_module()
        target_subspace = self._target_subspace_module()
        diagnostic_calibrator = self._diagnostic_calibrator_module()
        source_features = classifier.forward_to_layer3(source_input)
        with torch.no_grad():
            target_features = classifier.forward_to_layer3(target_input)
        candidate_features = self._extract_candidate_features(candidates)
        source_pool = F.adaptive_avg_pool2d(source_features, 1).flatten(1)
        target_pool = F.adaptive_avg_pool2d(target_features, 1).flatten(1)
        candidate_pool = F.adaptive_avg_pool2d(
            candidate_features.reshape(-1, *candidate_features.shape[4:]), 1
        ).flatten(1)

        source_logits = classifier.forward_from_layer3(source_features)
        source_margin = true_class_margin(source_logits, self.real_A_label)
        with torch.no_grad():
            source_domain_score = binary_score(domain_probe(source_pool.detach()))
            candidate_domain_score = binary_score(
                domain_probe(candidate_pool.detach())
            ).reshape(
                batch_size,
                reference_count,
                rollout_count,
                state_count,
            ).mean(dim=2)

        source_teacher_score, candidate_teacher_score = self._diagnostic_scores(
            candidates
        )
        diagnostic = diagnostic_calibrator.evaluate_and_update(
            source_teacher_score,
            candidate_teacher_score,
            self.real_A_label,
            margin_tolerance=self.opt.safebridge_diag_margin_tolerance,
            rank_tolerance=self.opt.safebridge_diag_rank_tolerance,
            maximum_rank_violation=self.opt.safebridge_diag_max_rank_violation,
        )
        stability = state_residual_agreement(
            source_features.detach(),
            candidate_features.detach(),
        )
        selector_ready = target_subspace.is_ready()
        selection = select_minimal_adequate_state(
            diagnostic["safe"],
            candidate_domain_score,
            source_domain_score,
            target_subspace.target_threshold,
            stability,
            self._candidate_states,
            minimum_domain_gain=self.opt.safebridge_state_minimum_domain_gain,
            minimum_stability=self.opt.safebridge_minimum_stability,
            selector_ready=selector_ready,
        )
        selected_candidate_domain_score = torch.gather(
            candidate_domain_score,
            dim=2,
            index=selection["selected_index"][:, :, None],
        ).squeeze(2)
        selected_domain_gain = F.relu(
            selected_candidate_domain_score - source_domain_score[:, None]
        )
        required_domain_gain = (
            float(self.opt.safebridge_backtracking_gain_retention)
            * selected_domain_gain
        ).detach()
        selected_features = gather_state_features(
            candidate_features,
            selection["selected_index"],
        )
        selected_mean = selected_features.mean(dim=2)
        selected_residual = selected_mean - source_features[:, None]
        stable = build_stable_residual(
            source_features,
            selected_features,
            noise_floor=self.opt.safebridge_noise_floor,
            spatial_threshold=self.opt.safebridge_spatial_threshold,
            channel_threshold=self.opt.safebridge_channel_threshold,
            spatial_temperature=self.opt.safebridge_spatial_temperature,
            channel_temperature=self.opt.safebridge_channel_temperature,
        )

        fusion_mode = str(self.opt.safebridge_fusion_mode)
        if fusion_mode == "raw_only":
            residual = torch.zeros_like(stable["residual"])
            accepted = torch.zeros_like(selection["accepted"])
        elif fusion_mode == "selected_feature":
            residual = selected_residual
            accepted = selection["accepted"]
        elif fusion_mode == "stable_residual":
            residual = stable["residual"]
            accepted = selection["accepted"]
        else:
            residual = target_subspace.project(stable["residual"])
            accepted = selection["accepted"]

        pre_target_norm = stable["residual"].flatten(2).norm(dim=2)
        post_target_norm = residual.flatten(2).norm(dim=2)
        target_retention = post_target_norm / pre_target_norm.clamp_min(1e-8)
        null_inner = residual.new_zeros((batch_size, reference_count))
        if fusion_mode == "full":
            diagnostic_gradient = torch.autograd.grad(
                source_margin.sum(),
                source_features,
                retain_graph=True,
                create_graph=False,
            )[0].detach()
            null_projection = project_diagnostic_nullspace(
                residual,
                diagnostic_gradient,
            )
            residual = null_projection["residual"]
            null_inner = null_projection["post_inner"].abs()

        backtracking = self._backtrack(
            source_features,
            residual,
            accepted,
            source_domain_score,
            source_margin.detach(),
            required_domain_gain,
        )
        final_accepted = backtracking["accepted"]
        alpha = backtracking["alpha"].detach()
        final_domain_gain = F.relu(
            backtracking["selected_domain_score"]
            - source_domain_score[:, None]
        )
        domain_gain_retention = final_domain_gain / selected_domain_gain.clamp_min(
            1e-8
        )
        safe_features = (
            source_features[:, None]
            + alpha[:, :, None, None, None] * residual
        )
        safe_logits = classifier.forward_from_layer3(
            safe_features.reshape(-1, *safe_features.shape[2:])
        ).reshape(batch_size, reference_count, -1)

        self._raw_logits = source_logits
        self._safe_logits = safe_logits
        self._safe_acceptance = final_accepted
        self._probe_source_pool = source_pool.detach()
        self._probe_target_pool = target_pool.detach()
        self._source_domain_score = source_domain_score.detach()
        self._source_true_margin = source_margin.detach()

        self.loss_SB_selector_ready = source_features.new_tensor(
            float(selector_ready)
        )
        self.loss_SB_accept = final_accepted.float().mean()
        for class_index in (0, 1):
            class_mask = self.real_A_label == class_index
            value = (
                final_accepted[class_mask].float().mean()
                if torch.any(class_mask)
                else source_features.new_tensor(0.0)
            )
            setattr(self, f"loss_SB_accept_c{class_index}", value)
        selected_depth = selection["selected_state"].clamp_min(0).to(source_features.dtype)
        self.loss_SB_selected_depth = self._mean_over_accepted(
            selected_depth,
            final_accepted,
        )
        self.loss_SB_diag_margin_drop = diagnostic["margin_drop"].mean()
        self.loss_SB_diag_rank_violation = diagnostic["rank_violation"].mean()
        self.loss_SB_stability = stability.mean()
        self.loss_SB_reliability_mask = stable["mask_mean"]
        self.loss_SB_target_energy_retention = self._mean_over_accepted(
            target_retention,
            selection["accepted"],
        )
        self.loss_SB_domain_gain_retention = self._mean_over_accepted(
            domain_gain_retention,
            final_accepted,
        )
        self.loss_SB_null_inner = self._mean_over_accepted(
            null_inner,
            final_accepted,
        )
        self.loss_SB_alpha = self._mean_over_accepted(alpha, final_accepted)
        self.loss_SB_target_threshold = target_subspace.target_threshold.detach()
        self.loss_SB_basis_rank = source_features.new_tensor(
            float(target_subspace.basis_rank.item())
        )
        self.loss_SB_domain_accuracy_ema = (
            target_subspace.domain_accuracy_ema.detach()
        )
        self._write_safebridge_audit(
            selection=selection,
            diagnostic=diagnostic,
            stability=stability,
            candidate_domain_score=candidate_domain_score,
            source_domain_score=source_domain_score,
            final_accepted=final_accepted,
            alpha=alpha,
            domain_gain_retention=domain_gain_retention,
        )

    @torch.no_grad()
    def _write_safebridge_audit(
        self,
        selection: dict[str, torch.Tensor],
        diagnostic: dict[str, torch.Tensor],
        stability: torch.Tensor,
        candidate_domain_score: torch.Tensor,
        source_domain_score: torch.Tensor,
        final_accepted: torch.Tensor,
        alpha: torch.Tensor,
        domain_gain_retention: torch.Tensor,
    ) -> None:
        path_value = str(self.opt.safebridge_audit_jsonl).strip()
        if not path_value:
            return
        selected_index = selection["selected_index"]
        target_subspace = self._target_subspace_module()

        def gather(values: torch.Tensor) -> torch.Tensor:
            return torch.gather(
                values,
                dim=2,
                index=selected_index[:, :, None],
            ).squeeze(2)

        selected_margin = gather(diagnostic["margin_drop"])
        selected_rank = gather(diagnostic["rank_violation"])
        selected_stability = gather(stability)
        selected_domain = gather(candidate_domain_score)
        rows = []
        batch_size, reference_count = final_accepted.shape
        for batch_index in range(batch_size):
            source_path = (
                self._audit_source_paths[batch_index]
                if batch_index < len(self._audit_source_paths)
                else ""
            )
            for reference_index in range(reference_count):
                reference_path = ""
                if batch_index < len(self._audit_reference_paths):
                    paths = self._audit_reference_paths[batch_index]
                    if reference_index < len(paths):
                        reference_path = paths[reference_index]
                rows.append(
                    {
                        "source_path": source_path,
                        "source_label": int(self.real_A_label[batch_index].item()),
                        "target_reference_path": reference_path,
                        "selected_state": int(
                            selection["selected_state"][
                                batch_index, reference_index
                            ].item()
                        ),
                        "dssr_accepted": bool(
                            selection["accepted"][
                                batch_index, reference_index
                            ].item()
                        ),
                        "final_accepted": bool(
                            final_accepted[batch_index, reference_index].item()
                        ),
                        "alpha": float(alpha[batch_index, reference_index].item()),
                        "domain_gain_retention": float(
                            domain_gain_retention[
                                batch_index, reference_index
                            ].item()
                        ),
                        "diagnostic_margin_drop": float(
                            selected_margin[batch_index, reference_index].item()
                        ),
                        "diagnostic_rank_violation": float(
                            selected_rank[batch_index, reference_index].item()
                        ),
                        "trajectory_stability": float(
                            selected_stability[batch_index, reference_index].item()
                        ),
                        "source_domain_score": float(
                            source_domain_score[batch_index].item()
                        ),
                        "candidate_domain_score": float(
                            selected_domain[batch_index, reference_index].item()
                        ),
                        "target_threshold": float(
                            target_subspace.target_threshold.item()
                        ),
                        "selector_ready": bool(target_subspace.is_ready()),
                    }
                )
        audit_path = Path(path_value).expanduser()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def forward(self):
        DoscSBModel.forward(self)
        if not self.isTrain or self._data_dependent_initializing:
            return
        self._freeze_classifier_bn()
        self.state_candidates = self.generate_state_candidates()
        self._prepare_safebridge_task()

    def compute_G_loss(self):
        if self._data_dependent_initializing:
            return DoscSBModel.compute_G_loss(self)
        base_loss = DoscSBModel.compute_G_loss(self)
        if (
            self._raw_logits is None
            or self._safe_logits is None
            or self._safe_acceptance is None
        ):
            raise RuntimeError("SafeBridge task logits are not initialized")
        task = masked_multi_reference_cross_entropy(
            self._raw_logits,
            self._safe_logits,
            self.real_A_label,
            self._safe_acceptance,
        )
        self.loss_TRSC_task = task["total"]
        self.loss_TRSC_raw_CE = task["raw"]
        self.loss_TRSC_candidate_CE = task["candidate"]
        self.loss_G = base_loss + self.loss_TRSC_task
        return self.loss_G

    def _update_domain_probe(self) -> None:
        if self._probe_source_pool is None or self._probe_target_pool is None:
            raise RuntimeError("Domain-probe features are not initialized")
        source_pool = self._probe_source_pool.detach()
        target_pool = self._probe_target_pool.detach()
        features = torch.cat([source_pool, target_pool], dim=0)
        domain_probe = self._domain_probe_module()
        target_subspace = self._target_subspace_module()
        labels = torch.cat(
            [
                torch.zeros(
                    source_pool.shape[0], dtype=torch.long, device=self.device
                ),
                torch.ones(
                    target_pool.shape[0], dtype=torch.long, device=self.device
                ),
            ],
            dim=0,
        )
        self.optimizer_H.zero_grad()
        logits = domain_probe(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer_H.step()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().detach()

        direction_input = features.detach().requires_grad_(True)
        direction_score = binary_score(domain_probe(direction_input))
        gradients = torch.autograd.grad(
            direction_score.sum(),
            direction_input,
            create_graph=False,
        )[0].detach()
        with torch.no_grad():
            target_scores = binary_score(domain_probe(target_pool))
            target_subspace.update(gradients, target_scores, accuracy)
        self.loss_SB_domain_probe = loss.detach()
        self.loss_SB_domain_accuracy = accuracy
        self.loss_SB_domain_accuracy_ema = (
            target_subspace.domain_accuracy_ema.detach()
        )
        self.loss_SB_target_threshold = target_subspace.target_threshold.detach()
        self.loss_SB_basis_rank = features.new_tensor(
            float(target_subspace.basis_rank.item())
        )

    def optimize_parameters(self):
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self.netS.train()
        self.netC.train()
        self.netH.train()
        self._freeze_classifier_bn()
        if self.netP is not None:
            self.netP.train()
        if self.netSafeBridgeTeacher is not None:
            self.netSafeBridgeTeacher.eval()
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        self.optimizer_C.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()
        self.optimizer_C.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()

        self._update_domain_probe()
        self._style_module().advance_step()
