"""TRSC joint training with diagnosis-aware U1 residual repair."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .dabrf_modules import (
    DiagnosisAwareBridgeResidualRepair,
    residual_radius_loss,
    target_style_progress_loss,
)
from .dosc_modules import (
    CalibrationInvariantDiagnosticPreservation,
    binary_diagnostic_score,
)
from .trsc_joint_modules import route_task_gradient
from .trsc_joint_sb_model import TrscJointSBModel


class TrscDabrfJointSBModel(TrscJointSBModel):
    """Repair every unfiltered U1 view before joint classifier training."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = TrscJointSBModel.modify_commandline_options(
            parser,
            is_train=is_train,
        )
        parser.add_argument(
            "--dabrf_mode",
            choices=("identity", "fixed_scale", "norm_clip", "learned"),
            default="learned",
            help="DA-BRF arm; identity reproduces the unmodified K-view baseline",
        )
        parser.add_argument("--dabrf_hidden_channels", type=int, default=32)
        parser.add_argument("--dabrf_fixed_scale", type=float, default=0.8)
        parser.add_argument("--dabrf_max_radius_ratio", type=float, default=1.0)
        parser.add_argument("--dabrf_gate_floor", type=float, default=0.0)
        parser.add_argument("--dabrf_gate_init", type=float, default=0.95)
        parser.add_argument("--dabrf_lr", type=float, default=1e-4)
        parser.add_argument("--dabrf_weight_decay", type=float, default=1e-4)
        parser.add_argument(
            "--dabrf_teacher_path",
            type=str,
            default="",
            help=(
                "Frozen render-robust source teacher exported by "
                "scripts/export_diagnostic_teacher.py"
            ),
        )
        parser.add_argument("--dabrf_queue_size", type=int, default=128)
        parser.add_argument("--dabrf_min_per_class", type=int, default=8)
        parser.add_argument("--dabrf_affine_ridge", type=float, default=1e-4)
        parser.add_argument("--dabrf_min_scale", type=float, default=0.05)
        parser.add_argument("--dabrf_max_scale", type=float, default=20.0)
        parser.add_argument("--dabrf_margin_tolerance", type=float, default=0.10)
        parser.add_argument("--dabrf_rank_tolerance", type=float, default=0.10)
        parser.add_argument("--dabrf_rank_weight", type=float, default=1.0)
        parser.add_argument("--dabrf_progress_retention", type=float, default=0.80)
        parser.add_argument("--dabrf_progress_min_gain", type=float, default=1e-4)
        parser.add_argument("--lambda_DABRF_diag", type=float, default=1.0)
        parser.add_argument("--lambda_DABRF_progress", type=float, default=1.0)
        parser.add_argument("--lambda_DABRF_radius", type=float, default=1.0)
        parser.set_defaults(
            dataset_mode="trsc_unaligned",
            dosc_noise_ratio=0.0,
            lambda_DOSC_diag=0.0,
            lambda_DOSC_safe=0.0,
        )
        return parser

    def __init__(self, opt):
        self._validate_dabrf_options(opt)
        super().__init__(opt)

        # Keep the identity arm on the historical K3 random stream. The new
        # repair weights must not shift target-reference or bridge-noise draws.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            self.netR = DiagnosisAwareBridgeResidualRepair(
                input_channels=opt.output_nc,
                hidden_channels=opt.dabrf_hidden_channels,
                mode=opt.dabrf_mode,
                fixed_scale=opt.dabrf_fixed_scale,
                max_radius_ratio=opt.dabrf_max_radius_ratio,
                gate_floor=opt.dabrf_gate_floor,
                gate_init=opt.dabrf_gate_init,
            ).to(self.device)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.netQ = CalibrationInvariantDiagnosticPreservation(
            queue_size=opt.dabrf_queue_size,
            min_per_class=opt.dabrf_min_per_class,
            affine_ridge=opt.dabrf_affine_ridge,
            min_scale=opt.dabrf_min_scale,
            max_scale=opt.dabrf_max_scale,
        ).to(self.device)
        self.model_names.extend(["R", "Q"])

        self.netDABRFTeacher = None
        if self.isTrain:
            self.netDABRFTeacher = self._load_dabrf_teacher(opt.dabrf_teacher_path)
            self.loss_names += [
                "DABRF_total",
                "DABRF_diag",
                "DABRF_diag_margin",
                "DABRF_diag_rank",
                "DABRF_progress",
                "DABRF_radius",
                "DABRF_affine_scale",
                "DABRF_affine_bias",
                "DABRF_calibration_ready",
                "DABRF_progress_retention",
                "DABRF_progress_valid",
                "DABRF_candidate_style_gain",
                "DABRF_repaired_style_gain",
                "DABRF_radius_ratio",
                "DABRF_radius_max",
                "DABRF_gate_mean",
            ]
            self.optimizer_R = torch.optim.AdamW(
                self.netR.parameters(),
                lr=float(opt.dabrf_lr),
                weight_decay=float(opt.dabrf_weight_decay),
            )
            self.optimizers.append(self.optimizer_R)

        self.task_repaired_candidates: torch.Tensor | None = None
        self.constraint_repaired_candidates: torch.Tensor | None = None
        self._dabrf_constraint_context: dict[str, torch.Tensor] = {}

    @staticmethod
    def _validate_dabrf_options(opt) -> None:
        non_negative = (
            "lambda_DABRF_diag",
            "lambda_DABRF_progress",
            "lambda_DABRF_radius",
            "dabrf_rank_weight",
            "dabrf_margin_tolerance",
            "dabrf_rank_tolerance",
            "dabrf_progress_min_gain",
        )
        for name in non_negative:
            if float(getattr(opt, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(opt.dabrf_progress_retention) <= 1.0:
            raise ValueError("dabrf_progress_retention must be in [0, 1]")
        if int(opt.dosc_num_classes) != 2:
            raise ValueError("DA-BRF currently supports binary diagnosis only")

    def _load_dabrf_teacher(self, value: str):
        path_value = str(value).strip()
        if not path_value:
            raise ValueError(
                "DA-BRF training requires --dabrf_teacher_path; export a frozen "
                "render-robust source teacher first"
            )
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DA-BRF teacher not found: {path}")
        teacher = torch.jit.load(str(path), map_location=self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f"[DA-BRF] loaded frozen diagnostic teacher: {path}")
        return teacher

    def _calibration_module(self) -> CalibrationInvariantDiagnosticPreservation:
        return self.netQ.module if hasattr(self.netQ, "module") else self.netQ

    def _flat_multi_reference_inputs(
        self,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.real_B_references is None:
            raise RuntimeError("Target references are not initialized")
        if candidates.ndim != 5:
            raise ValueError(
                f"candidates must be [B,K,C,H,W], got {tuple(candidates.shape)}"
            )
        batch_size, reference_count = candidates.shape[:2]
        source = (
            self.real_A[:, None]
            .expand(-1, reference_count, -1, -1, -1)
            .reshape(batch_size * reference_count, *self.real_A.shape[1:])
        )
        references = self.real_B_references.reshape(
            batch_size * reference_count,
            *self.real_B_references.shape[2:],
        )
        return source, references, batch_size, reference_count

    def forward(self):
        super().forward()
        if not self.isTrain or self._data_dependent_initializing:
            return
        if self.task_candidates is None:
            raise RuntimeError("TRSC U1 candidates are not initialized")

        source, _, batch_size, reference_count = self._flat_multi_reference_inputs(
            self.task_candidates
        )
        task_input = route_task_gradient(
            self.task_candidates,
            self.opt.lambda_TRSC_task,
        ).reshape(batch_size * reference_count, *self.task_candidates.shape[2:])
        task_result = self.netR(source, task_input)
        self.task_repaired_candidates = task_result["repaired"].reshape_as(
            self.task_candidates
        )

        # The constraint branch trains R but cannot move G to game its teacher.
        constraint_result = self.netR(
            source,
            self.task_candidates.detach().reshape(
                batch_size * reference_count,
                *self.task_candidates.shape[2:],
            ),
        )
        self.constraint_repaired_candidates = constraint_result["repaired"].reshape_as(
            self.task_candidates
        )
        self._dabrf_constraint_context = constraint_result

    def _task_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task_repaired_candidates is None:
            raise RuntimeError("DA-BRF task candidates are not initialized")
        batch_size, reference_count = self.task_repaired_candidates.shape[:2]
        all_views = torch.cat(
            [self.real_A[:, None], self.task_repaired_candidates],
            dim=1,
        ).reshape(
            batch_size * (reference_count + 1),
            *self.real_A.shape[1:],
        )
        classifier_input = F.interpolate(
            all_views,
            size=(
                int(self.opt.trsc_classifier_input_size),
                int(self.opt.trsc_classifier_input_size),
            ),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.netC(classifier_input).reshape(
            batch_size,
            reference_count + 1,
            int(self.opt.dosc_num_classes),
        )
        return logits[:, 0], logits[:, 1:]

    def _diagnostic_repair_loss(
        self,
        repaired_flat: torch.Tensor,
        batch_size: int,
        reference_count: int,
    ) -> dict[str, torch.Tensor]:
        if self.netDABRFTeacher is None:
            raise RuntimeError("DA-BRF diagnostic teacher is not initialized")
        with torch.no_grad():
            source_score = binary_diagnostic_score(
                self.netDABRFTeacher(self.real_A)
            )
        repaired_score = binary_diagnostic_score(
            self.netDABRFTeacher(repaired_flat)
        )
        repeated_source_score = source_score[:, None].expand(
            -1, reference_count
        ).reshape(-1)
        repeated_labels = self.real_A_label[:, None].expand(
            -1, reference_count
        ).reshape(-1)
        calibration = self._calibration_module()
        result = calibration.forward_scores(
            repeated_source_score,
            repaired_score,
            repeated_labels,
            margin_tolerance=self.opt.dabrf_margin_tolerance,
            rank_tolerance=self.opt.dabrf_rank_tolerance,
            rank_weight=self.opt.dabrf_rank_weight,
            update_queue=False,
        )
        calibration.score_queue.enqueue(
            source_score,
            repaired_score.reshape(batch_size, reference_count).mean(dim=1),
            self.real_A_label,
        )
        return result

    def compute_G_loss(self):
        if self._data_dependent_initializing:
            return super().compute_G_loss()
        base_loss = super().compute_G_loss()
        if self.task_candidates is None or self.constraint_repaired_candidates is None:
            raise RuntimeError("DA-BRF candidates are not initialized")

        source, references, batch_size, reference_count = (
            self._flat_multi_reference_inputs(self.task_candidates)
        )
        candidate_flat = self.task_candidates.detach().reshape(
            batch_size * reference_count,
            *self.task_candidates.shape[2:],
        )
        repaired_flat = self.constraint_repaired_candidates.reshape(
            batch_size * reference_count,
            *self.constraint_repaired_candidates.shape[2:],
        )
        diagnostic = self._diagnostic_repair_loss(
            repaired_flat,
            batch_size,
            reference_count,
        )
        progress = target_style_progress_loss(
            source.detach(),
            candidate_flat,
            repaired_flat,
            references.detach(),
            minimum_retention=self.opt.dabrf_progress_retention,
            minimum_gain=self.opt.dabrf_progress_min_gain,
        )
        radius = residual_radius_loss(
            source.detach(),
            candidate_flat,
            repaired_flat,
            max_ratio=self.opt.dabrf_max_radius_ratio,
        )

        self.loss_DABRF_diag = diagnostic["loss"]
        self.loss_DABRF_diag_margin = diagnostic["margin_loss"]
        self.loss_DABRF_diag_rank = diagnostic["rank_loss"]
        self.loss_DABRF_progress = progress["loss"]
        self.loss_DABRF_radius = radius["loss"]
        self.loss_DABRF_affine_scale = diagnostic["affine_scale"]
        self.loss_DABRF_affine_bias = diagnostic["affine_bias"]
        self.loss_DABRF_calibration_ready = diagnostic["ready"]
        self.loss_DABRF_progress_retention = progress["retention"]
        self.loss_DABRF_progress_valid = progress["valid_fraction"]
        self.loss_DABRF_candidate_style_gain = progress["candidate_gain"]
        self.loss_DABRF_repaired_style_gain = progress["repaired_gain"]
        self.loss_DABRF_radius_ratio = radius["ratio"]
        self.loss_DABRF_radius_max = radius["maximum_ratio"]
        self.loss_DABRF_gate_mean = self._dabrf_constraint_context[
            "gate_mean"
        ].detach()
        self.loss_DABRF_total = (
            float(self.opt.lambda_DABRF_diag) * self.loss_DABRF_diag
            + float(self.opt.lambda_DABRF_progress) * self.loss_DABRF_progress
            + float(self.opt.lambda_DABRF_radius) * self.loss_DABRF_radius
        )
        self.loss_G = base_loss + self.loss_DABRF_total
        return self.loss_G

    def optimize_parameters(self):
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self.netS.train()
        self.netC.train()
        self.netR.train()
        self.netQ.train()
        if self.netP is not None:
            self.netP.train()
        if self.netDABRFTeacher is not None:
            self.netDABRFTeacher.eval()
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
        self.optimizer_R.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()
        self.optimizer_C.step()
        self.optimizer_R.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()
        self._style_module().advance_step()
