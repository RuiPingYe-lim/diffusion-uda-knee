"""UNSB model overlay with diagnostic-orthogonal style conditioning.

Copy this file and ``dosc_modules.py`` into the upstream UNSB ``models``
directory, then train with ``--model dosc_sb``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .dosc_modules import (
    DiagnosticOrthogonalConditioner,
    diagnostic_non_degradation_loss,
    parse_widths,
)
from .sb_model import SBModel


class DoscSBModel(SBModel):
    """Replace UNSB random style noise with diagnosis-safe target exemplars."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train=is_train)
        parser.add_argument("--dosc_style_dim", type=int, default=128)
        parser.add_argument("--dosc_num_classes", type=int, default=2)
        parser.add_argument("--dosc_encoder_widths", type=str, default="32,64,128,256")
        parser.add_argument("--dosc_projector_momentum", type=float, default=0.95)
        parser.add_argument("--dosc_disable_projection", action="store_true")
        parser.add_argument("--dosc_grl_strength", type=float, default=1.0)
        parser.add_argument("--dosc_queue_size", type=int, default=128)
        parser.add_argument("--dosc_contrastive_temperature", type=float, default=0.07)
        parser.add_argument(
            "--dosc_noise_ratio",
            type=float,
            default=0.10,
            help="Residual Gaussian fraction mixed into the target-reference condition",
        )
        parser.add_argument("--dosc_style_lr", type=float, default=-1.0)
        parser.add_argument("--lambda_DOSC_diag", type=float, default=0.10)
        parser.add_argument("--lambda_DOSC_domain", type=float, default=0.10)
        parser.add_argument("--lambda_DOSC_instance", type=float, default=0.10)
        parser.add_argument("--lambda_DOSC_recon", type=float, default=1.00)
        parser.add_argument("--lambda_DOSC_safe", type=float, default=0.50)
        parser.add_argument("--dosc_safe_tolerance", type=float, default=0.10)
        parser.add_argument("--dosc_safe_warmup_steps", type=int, default=1000)
        parser.add_argument(
            "--dosc_teacher_path",
            type=str,
            default="",
            help="TorchScript source diagnostic teacher; required when lambda_DOSC_safe > 0",
        )
        parser.set_defaults(direction="AtoB", dataset_mode="dosc_unaligned")
        return parser

    def __init__(self, opt):
        if opt.direction != "AtoB":
            raise ValueError(
                "DoscSBModel requires AtoB: domain A is labeled source and domain B is unlabeled target"
            )
        if len(opt.gpu_ids) > 1:
            raise ValueError(
                "DoscSBModel currently supports one GPU because the EMA projector and style queue "
                "must have a single authoritative state"
            )
        if int(opt.num_timesteps) < 2:
            raise ValueError("DoscSBModel requires num_timesteps >= 2")
        if not 0.0 <= float(opt.dosc_noise_ratio) <= 1.0:
            raise ValueError("dosc_noise_ratio must be in [0, 1]")

        super().__init__(opt)
        self.netS = DiagnosticOrthogonalConditioner(
            input_channels=opt.output_nc,
            style_dim=opt.dosc_style_dim,
            generator_style_dim=4 * opt.ngf,
            num_classes=opt.dosc_num_classes,
            encoder_widths=parse_widths(opt.dosc_encoder_widths),
            projector_momentum=opt.dosc_projector_momentum,
            grl_strength=opt.dosc_grl_strength,
            queue_size=opt.dosc_queue_size,
            contrastive_temperature=opt.dosc_contrastive_temperature,
            enable_projection=not opt.dosc_disable_projection,
        ).to(self.device)
        self.model_names.append("S")
        self._dosc_context: dict[str, torch.Tensor] = {}
        self._nce_conditions: list[torch.Tensor] = []
        self._nce_condition_index = 0

        self.netTeacher = None
        if self.isTrain:
            self.loss_names += [
                "DOSC_diag",
                "DOSC_domain",
                "DOSC_instance",
                "DOSC_recon",
                "DOSC_safe",
                "DOSC_margin_drop",
                "DOSC_removed",
                "DOSC_diag_acc",
                "DOSC_domain_acc",
            ]
            style_lr = float(opt.dosc_style_lr)
            if style_lr <= 0.0:
                style_lr = float(opt.lr)
            self.optimizer_S = torch.optim.Adam(
                self.netS.parameters(),
                lr=style_lr,
                betas=(opt.beta1, opt.beta2),
            )
            self.optimizers.append(self.optimizer_S)
            self.netTeacher = self._load_teacher_if_required(opt)

    def _load_teacher_if_required(self, opt):
        teacher_path = str(opt.dosc_teacher_path).strip()
        if float(opt.lambda_DOSC_safe) <= 0.0:
            return None
        if not teacher_path:
            raise ValueError(
                "lambda_DOSC_safe > 0 requires --dosc_teacher_path. "
                "Use scripts/export_diagnostic_teacher.py to create it."
            )
        path = Path(teacher_path)
        if not path.is_file():
            raise FileNotFoundError(f"Diagnostic teacher not found: {path}")
        teacher = torch.jit.load(str(path), map_location=self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    def _style_module(self) -> DiagnosticOrthogonalConditioner:
        return self.netS.module if hasattr(self.netS, "module") else self.netS

    def set_input(self, input, input2=None):
        super().set_input(input, input2)
        if self.isTrain:
            if "A_label" not in input:
                raise KeyError(
                    "DoscSBModel requires A_label from --dataset_mode dosc_unaligned"
                )
            self.real_A_label = input["A_label"].to(self.device).long()
            if input2 is None or "A_label" not in input2:
                raise KeyError("The second UNSB data stream must also provide A_label")
            self.real_A_label2 = input2["A_label"].to(self.device).long()

    def _bridge_times(self) -> torch.Tensor:
        count = int(self.opt.num_timesteps)
        increments = torch.tensor(
            [0.0] + [1.0 / float(index + 1) for index in range(count - 1)],
            dtype=self.real_A.dtype,
            device=self.real_A.device,
        )
        times = torch.cumsum(increments, dim=0)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        return torch.cat([times.new_zeros(1), times], dim=0)

    def _condition_with_noise(
        self,
        condition: torch.Tensor,
        detach: bool = False,
    ) -> torch.Tensor:
        if detach:
            condition = condition.detach()
        return self._style_module().mix_with_noise(
            condition,
            noise_ratio=self.opt.dosc_noise_ratio,
        )

    def _rollout_training_states(
        self,
        times: torch.Tensor,
        condition: torch.Tensor,
        condition2: torch.Tensor,
        identity_condition: torch.Tensor,
    ) -> None:
        tau = float(self.opt.tau)
        with torch.no_grad():
            self.netG.eval()
            state = self.real_A
            state2 = self.real_A2
            identity_state = self.real_B
            next_state = state
            next_state2 = state2
            next_identity = identity_state
            for step in range(int(self.time_idx.item()) + 1):
                if step > 0:
                    delta = times[step] - times[step - 1]
                    denominator = times[-1] - times[step - 1]
                    interpolation = (delta / denominator).reshape(1, 1, 1, 1)
                    scale = (delta * (1.0 - delta / denominator)).reshape(1, 1, 1, 1)
                    state = (
                        (1.0 - interpolation) * state
                        + interpolation * next_state.detach()
                        + (scale * tau).sqrt() * torch.randn_like(state)
                    )
                    state2 = (
                        (1.0 - interpolation) * state2
                        + interpolation * next_state2.detach()
                        + (scale * tau).sqrt() * torch.randn_like(state2)
                    )
                    identity_state = (
                        (1.0 - interpolation) * identity_state
                        + interpolation * next_identity.detach()
                        + (scale * tau).sqrt() * torch.randn_like(identity_state)
                    )

                batch_time = torch.full(
                    (self.real_A.shape[0],),
                    step,
                    dtype=torch.long,
                    device=self.real_A.device,
                )
                next_state = self.netG(
                    state,
                    batch_time,
                    self._condition_with_noise(condition, detach=True),
                )
                next_state2 = self.netG(
                    state2,
                    batch_time,
                    self._condition_with_noise(condition2, detach=True),
                )
                if self.opt.nce_idt:
                    next_identity = self.netG(
                        identity_state,
                        batch_time,
                        self._condition_with_noise(identity_condition, detach=True),
                    )

            self.real_A_noisy = state.detach()
            self.real_A_noisy2 = state2.detach()
            if self.opt.nce_idt:
                self.XtB = identity_state.detach()

    def _forward_train(self) -> None:
        style_module = self._style_module()
        context = style_module.build_context(
            self.real_A,
            self.real_B,
            self.real_A_label,
            update_projector=True,
            update_queue=True,
        )
        context2 = style_module.build_context(
            self.real_A2,
            self.real_B2,
            self.real_A_label2,
            update_projector=False,
            update_queue=False,
        )
        self._dosc_context = context
        source_condition = context["condition"]
        source_condition2 = context2["condition"]
        identity_condition = style_module.encode_condition(self.real_B)

        times = self._bridge_times()
        self.times = times
        self.time_idx = torch.randint(
            int(self.opt.num_timesteps),
            size=(1,),
            device=self.real_A.device,
        ).long()
        self.timestep = times[self.time_idx]
        self._rollout_training_states(
            times,
            source_condition,
            source_condition2,
            identity_condition,
        )
        self.netG.train()

        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if self.opt.nce_idt
            else self.real_A
        )
        self.realt = (
            torch.cat((self.real_A_noisy, self.XtB), dim=0)
            if self.opt.nce_idt
            else self.real_A_noisy
        )
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (
                torch.rand((), device=self.real_A.device).item() < 0.5
            )
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, dims=(3,))
                self.realt = torch.flip(self.realt, dims=(3,))

        main_condition = self._condition_with_noise(source_condition)
        if self.opt.nce_idt:
            main_condition = torch.cat(
                [main_condition, self._condition_with_noise(identity_condition)],
                dim=0,
            )
        self.fake = self.netG(self.realt, self.time_idx, main_condition)
        self.fake_B2 = self.netG(
            self.real_A_noisy2,
            self.time_idx,
            self._condition_with_noise(source_condition2),
        )
        self.fake_B = self.fake[: self.real_A.shape[0]]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.shape[0] :]

        self._nce_conditions = [source_condition, identity_condition]

    def _forward_test(self) -> None:
        self.real = self.real_A
        times = self._bridge_times()
        self.times = times
        condition = self._style_module().encode_condition(self.real_B)
        state = self.real_A
        next_state = state
        tau = float(self.opt.tau)
        for step in range(int(self.opt.num_timesteps)):
            if step > 0:
                delta = times[step] - times[step - 1]
                denominator = times[-1] - times[step - 1]
                interpolation = (delta / denominator).reshape(1, 1, 1, 1)
                scale = (delta * (1.0 - delta / denominator)).reshape(1, 1, 1, 1)
                state = (
                    (1.0 - interpolation) * state
                    + interpolation * next_state.detach()
                    + (scale * tau).sqrt() * torch.randn_like(state)
                )
            batch_time = torch.full(
                (self.real_A.shape[0],),
                step,
                dtype=torch.long,
                device=self.real_A.device,
            )
            next_state = self.netG(
                state,
                batch_time,
                self._condition_with_noise(condition),
            )
            setattr(self, f"fake_{step + 1}", next_state)

    def forward(self):
        if self.isTrain:
            self._forward_train()
        else:
            self._forward_test()

    def calculate_NCE_loss(self, src, tgt):
        if not self._nce_conditions:
            return super().calculate_NCE_loss(src, tgt)
        condition_index = min(
            self._nce_condition_index,
            len(self._nce_conditions) - 1,
        )
        condition = self._nce_conditions[condition_index]
        self._nce_condition_index += 1
        condition = self._condition_with_noise(condition)

        feature_query = self.netG(
            tgt,
            self.time_idx * 0,
            condition,
            self.nce_layers,
            encode_only=True,
        )
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feature_query = [torch.flip(feature, dims=(3,)) for feature in feature_query]
        feature_key = self.netG(
            src,
            self.time_idx * 0,
            condition,
            self.nce_layers,
            encode_only=True,
        )
        pooled_key, sample_ids = self.netF(feature_key, self.opt.num_patches, None)
        pooled_query, _ = self.netF(feature_query, self.opt.num_patches, sample_ids)

        total = src.sum() * 0.0
        for query, key, criterion in zip(
            pooled_query,
            pooled_key,
            self.criterionNCE,
        ):
            total = total + (criterion(query, key) * self.opt.lambda_NCE).mean()
        return total / max(len(self.nce_layers), 1)

    def compute_G_loss(self):
        self._nce_condition_index = 0
        base_loss = super().compute_G_loss()
        context = self._dosc_context
        style_module = self._style_module()

        self.loss_DOSC_diag = context["diagnostic_loss"]
        self.loss_DOSC_domain = context["domain_loss"]
        self.loss_DOSC_instance = context["instance_loss"]
        self.loss_DOSC_removed = context["removed_energy"].detach()
        self.loss_DOSC_diag_acc = context["diagnostic_accuracy"].detach()
        self.loss_DOSC_domain_acc = context["domain_accuracy"].detach()
        self.loss_DOSC_recon = style_module.style_reconstruction_loss(
            self.fake_B,
            self.real_B,
        )

        if self.netTeacher is not None:
            self.loss_DOSC_safe, self.loss_DOSC_margin_drop = (
                diagnostic_non_degradation_loss(
                    self.netTeacher,
                    self.real_A,
                    self.fake_B,
                    self.real_A_label,
                    tolerance=self.opt.dosc_safe_tolerance,
                )
            )
        else:
            self.loss_DOSC_safe = self.fake_B.sum() * 0.0
            self.loss_DOSC_margin_drop = self.loss_DOSC_safe.detach()

        safe_scale = style_module.warmup_scale(self.opt.dosc_safe_warmup_steps)
        dosc_loss = (
            float(self.opt.lambda_DOSC_diag) * self.loss_DOSC_diag
            + float(self.opt.lambda_DOSC_domain) * self.loss_DOSC_domain
            + float(self.opt.lambda_DOSC_instance) * self.loss_DOSC_instance
            + float(self.opt.lambda_DOSC_recon) * self.loss_DOSC_recon
            + safe_scale
            * float(self.opt.lambda_DOSC_safe)
            * self.loss_DOSC_safe
        )
        self.loss_G = base_loss + dosc_loss
        return self.loss_G

    def optimize_parameters(self):
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self.netS.train()
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
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()
        self._style_module().advance_step()
