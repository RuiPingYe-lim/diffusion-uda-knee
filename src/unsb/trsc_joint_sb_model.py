"""UNSB + TRSC + classifier joint training with multiple target references."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from util import util

from .dosc_sb_model import DoscSBModel
from .trsc_joint_modules import (
    SourceWarmStartResNet50,
    multi_reference_cross_entropy,
    route_task_gradient,
)


class TrscJointSBModel(DoscSBModel):
    """Train UNSB and a classifier on raw plus unfiltered U1 candidates."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = DoscSBModel.modify_commandline_options(
            parser,
            is_train=is_train,
        )
        parser.add_argument("--trsc_num_references", type=int, default=3)
        parser.add_argument(
            "--lambda_TRSC_task",
            type=float,
            default=1.0,
            help=(
                "Gradient scale from candidate classification CE into netG/netS. "
                "The classifier always receives the full task gradient."
            ),
        )
        parser.add_argument(
            "--trsc_view_weighting",
            choices=("equal_groups", "equal_views"),
            default="equal_groups",
            help=(
                "equal_groups gives raw and the candidate set equal total weight; "
                "equal_views gives every raw/candidate image equal weight"
            ),
        )
        parser.add_argument(
            "--trsc_source_classifier_path",
            type=str,
            default="",
            help=(
                "Source-only custom_resnet50_space checkpoint used to warm-start "
                "the fully trainable joint classifier"
            ),
        )
        parser.add_argument(
            "--trsc_translator_init_dir",
            type=str,
            default="",
            help=(
                "Pretrained TRSC checkpoint directory containing G/F/D/E/S; "
                "all components remain unfrozen during joint training"
            ),
        )
        parser.add_argument(
            "--trsc_translator_init_epoch",
            type=str,
            default="latest",
        )
        parser.add_argument(
            "--trsc_allow_random_init",
            action="store_true",
            help="Debug-only escape hatch; canonical experiments require both warm starts",
        )
        parser.add_argument("--trsc_classifier_input_size", type=int, default=224)
        parser.add_argument("--trsc_classifier_lr", type=float, default=1e-4)
        parser.add_argument(
            "--trsc_classifier_weight_decay",
            type=float,
            default=1e-4,
        )
        parser.add_argument(
            "--trsc_deterministic",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=True,
        )
        parser.set_defaults(
            dataset_mode="trsc_unaligned",
            dosc_noise_ratio=0.0,
            lambda_DOSC_diag=0.0,
            lambda_DOSC_safe=0.0,
        )
        return parser

    def __init__(self, opt):
        if int(opt.trsc_num_references) < 1:
            raise ValueError("trsc_num_references must be at least one")
        if float(opt.lambda_TRSC_task) < 0.0:
            raise ValueError("lambda_TRSC_task must be non-negative")
        if int(opt.trsc_classifier_input_size) < 32:
            raise ValueError("trsc_classifier_input_size must be at least 32")

        super().__init__(opt)

        # Upstream UNSB's BaseModel enables cuDNN benchmarking in its
        # constructor. Re-apply the experiment protocol after every upstream
        # constructor has returned so the requested setting cannot be
        # silently overwritten.
        if bool(opt.trsc_deterministic):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.netC = SourceWarmStartResNet50(
            num_classes=opt.dosc_num_classes,
        ).to(self.device)
        self.model_names.append("C")

        self._fresh_training = (
            self.isTrain
            and not bool(getattr(opt, "continue_train", False))
            and not bool(getattr(opt, "pretrained_name", None))
        )
        self._translator_init_dir = Path(
            str(opt.trsc_translator_init_dir).strip()
        ).expanduser()
        self._translator_init_epoch = str(opt.trsc_translator_init_epoch)
        self._pending_netF_warm_start = False
        if self._fresh_training:
            self._initialize_warm_starts()

        self.real_B_references: torch.Tensor | None = None
        self.task_candidates: torch.Tensor | None = None
        self._data_dependent_initializing = False
        if self.isTrain:
            self.loss_names += [
                "TRSC_task",
                "TRSC_raw_CE",
                "TRSC_candidate_CE",
            ]
            self.optimizer_C = torch.optim.AdamW(
                self.netC.parameters(),
                lr=float(opt.trsc_classifier_lr),
                weight_decay=float(opt.trsc_classifier_weight_decay),
            )
            self.optimizers.append(self.optimizer_C)

    @staticmethod
    def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        if hasattr(blob, "state_dict"):
            blob = blob.state_dict()
        if isinstance(blob, dict):
            for key in ("state_dict", "model"):
                if key in blob and isinstance(blob[key], dict):
                    blob = blob[key]
                    break
        if not isinstance(blob, dict):
            raise TypeError(f"Unsupported checkpoint payload in {path}")
        return {
            str(key).removeprefix("module."): value
            for key, value in blob.items()
        }

    def _load_network_strict(
        self,
        network: torch.nn.Module,
        checkpoint: Path,
        name: str,
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name} warm-start checkpoint not found: {checkpoint}")
        state = self._checkpoint_state(checkpoint)
        result = network.load_state_dict(state, strict=False)
        missing = [
            key for key in result.missing_keys if "num_batches_tracked" not in key
        ]
        unexpected = [
            key for key in result.unexpected_keys if "num_batches_tracked" not in key
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"{name} warm-start mismatch: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        print(f"[TRSC warm start] loaded {name}: {checkpoint}")

    def _translator_checkpoint(self, network_name: str) -> Path:
        return (
            self._translator_init_dir
            / f"{self._translator_init_epoch}_net_{network_name}.pth"
        )

    def _initialize_warm_starts(self) -> None:
        source_path = Path(
            str(self.opt.trsc_source_classifier_path).strip()
        ).expanduser()
        translator_value = str(self.opt.trsc_translator_init_dir).strip()
        missing = []
        if not str(self.opt.trsc_source_classifier_path).strip():
            missing.append("--trsc_source_classifier_path")
        if not translator_value:
            missing.append("--trsc_translator_init_dir")
        if missing and not bool(self.opt.trsc_allow_random_init):
            raise ValueError(
                "Canonical TRSC joint training requires warm starts: "
                + ", ".join(missing)
            )

        if str(self.opt.trsc_source_classifier_path).strip():
            self._load_network_strict(
                self.netC,
                source_path,
                "source classifier C",
            )
        if translator_value:
            for name in ("G", "D", "E", "S"):
                self._load_network_strict(
                    getattr(self, f"net{name}"),
                    self._translator_checkpoint(name),
                    f"translator {name}",
                )
            self._pending_netF_warm_start = True

    def data_dependent_initialize(self, data, data2):
        self._data_dependent_initializing = True
        try:
            # Upstream uses this pass only to instantiate netF's lazy MLPs.
            # Exclude task CE so the source classifier warm start does not
            # receive a hidden BatchNorm update before the first real step.
            super().data_dependent_initialize(data, data2)
        finally:
            self._data_dependent_initializing = False
        if self._fresh_training and self._pending_netF_warm_start:
            self._load_network_strict(
                self.netF,
                self._translator_checkpoint("F"),
                "translator F",
            )
            self._pending_netF_warm_start = False

    def set_input(self, input, input2=None):
        super().set_input(input, input2)
        if not self.isTrain:
            return
        if "B_refs" not in input:
            raise KeyError(
                "TrscJointSBModel requires B_refs from "
                "--dataset_mode trsc_unaligned"
            )
        references = input["B_refs"].to(self.device)
        if references.ndim != 5:
            raise ValueError(
                "B_refs must be [B,K,C,H,W], "
                f"got {tuple(references.shape)}"
            )
        expected = int(self.opt.trsc_num_references)
        if references.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} target references, got {references.shape[1]}"
            )
        if references.shape[0] != self.real_A.shape[0]:
            raise ValueError("B_refs and source batches have different sizes")
        if not torch.equal(references[:, 0], self.real_B):
            raise ValueError(
                "The first target reference must equal real_B so the UNSB and "
                "multi-candidate objectives use the same target stream"
            )
        self.real_B_references = references

    def generate_u1_candidates(
        self,
        source_image: torch.Tensor | None = None,
        target_references: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate one differentiable U1 for every source/reference pair."""

        source = self.real_A if source_image is None else source_image
        references = (
            self.real_B_references
            if target_references is None
            else target_references
        )
        if references is None:
            raise RuntimeError("Target references are not initialized")
        if source.ndim != 4 or references.ndim != 5:
            raise ValueError("Expected source [B,C,H,W] and references [B,K,C,H,W]")
        batch_size, reference_count = references.shape[:2]
        if source.shape[0] != batch_size:
            raise ValueError("Source and target-reference batch sizes differ")
        if references.shape[2:] != source.shape[1:]:
            raise ValueError("Source and target-reference image shapes differ")

        expanded_source = (
            source[:, None]
            .expand(-1, reference_count, -1, -1, -1)
            .reshape(batch_size * reference_count, *source.shape[1:])
        )
        flat_references = references.reshape(
            batch_size * reference_count,
            *references.shape[2:],
        )
        style = self._style_module().encode_condition(flat_references)
        condition = self._condition_with_noise(style)
        timestep = torch.zeros(
            batch_size * reference_count,
            dtype=torch.long,
            device=source.device,
        )
        translated = self.netG(expanded_source, timestep, condition)
        return translated.reshape(
            batch_size,
            reference_count,
            *translated.shape[1:],
        )

    def forward(self):
        super().forward()
        if self.isTrain and not self._data_dependent_initializing:
            self.task_candidates = self.generate_u1_candidates()

    def _task_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.task_candidates is None:
            raise RuntimeError("Task candidates are not initialized")
        batch_size, reference_count = self.task_candidates.shape[:2]
        routed_candidates = route_task_gradient(
            self.task_candidates,
            self.opt.lambda_TRSC_task,
        )
        all_views = torch.cat(
            [self.real_A[:, None], routed_candidates],
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

    def compute_G_loss(self):
        if self._data_dependent_initializing:
            return super().compute_G_loss()
        base_loss = super().compute_G_loss()
        raw_logits, candidate_logits = self._task_logits()
        task_losses = multi_reference_cross_entropy(
            raw_logits,
            candidate_logits,
            self.real_A_label,
            weighting=self.opt.trsc_view_weighting,
        )
        self.loss_TRSC_task = task_losses["total"]
        self.loss_TRSC_raw_CE = task_losses["raw"]
        self.loss_TRSC_candidate_CE = task_losses["candidate"]
        self.loss_G = base_loss + self.loss_TRSC_task
        return self.loss_G

    def optimize_parameters(self):
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self.netS.train()
        self.netC.train()
        if self.netP is not None:
            self.netP.train()
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
        self._style_module().advance_step()
