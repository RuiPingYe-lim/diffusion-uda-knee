"""Train a classifier on task-preserving source-to-target style paths.

Protocol
--------
* source train labels: classification supervision;
* target train images: unlabeled mean/std style bank only;
* source validation labels: checkpoint selection;
* target test labels: final evaluation only, after checkpoint selection;
* inference: raw target image -> classifier (no VAE, bridge, or translation).

This is a C2 gate experiment, not evidence of efficacy by itself.  Compare
``none``, ``endpoint``, ``linear`` and ``brownian`` under identical seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_existing_classifier_on_csv import (  # noqa: E402
    IMAGE_COL_CANDIDATES,
    LABEL_COL_CANDIDATES,
    build_model,
    detect_column,
    resolve_path,
    set_seed,
)
from train_source_classifier import (  # noqa: E402
    CASE_ID_COL_CANDIDATES,
    CSVBinaryDataset,
    binary_metrics_full,
    build_optimizer,
    build_scheduler,
)

try:  # Support both ``python file.py`` and package import.
    from .path_augmentation import (
        build_target_style_bank,
        classifier_normalize,
        load_gray01,
        make_style_path_view,
    )
except ImportError:  # pragma: no cover - exercised by the command-line entrypoint
    from path_augmentation import (
        build_target_style_bank,
        classifier_normalize,
        load_gray01,
        make_style_path_view,
    )


PATH_MODES = ("none", "endpoint", "linear", "brownian")


def _jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def save_json(data: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


class StylePathSourceDataset(Dataset):
    """Labeled source image plus K content-identical style-path views."""

    def __init__(
        self,
        source_csv: Path,
        image_size: int,
        path_mode: str,
        style_bank: Optional[Dict[str, object]],
        root_dir: Optional[Path] = None,
        image_col: Optional[str] = None,
        label_col: Optional[str] = None,
        case_id_col: Optional[str] = None,
        num_path_views: int = 2,
        alpha_min: float = 0.1,
        alpha_max: float = 1.0,
        endpoint_frac: float = 0.25,
        style_sampling: str = "per_image",
        path_noise_std: float = 0.06,
        noise_blur_kernel: int = 21,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.source_csv = Path(source_csv)
        if not self.source_csv.exists():
            raise FileNotFoundError(f"Source training CSV not found: {self.source_csv}")
        self.frame = pd.read_csv(self.source_csv)
        if self.frame.empty:
            raise ValueError(f"Source training CSV has no rows: {self.source_csv}")
        self.image_col = detect_column(self.frame, image_col, IMAGE_COL_CANDIDATES, required=True)
        self.label_col = detect_column(self.frame, label_col, LABEL_COL_CANDIDATES, required=True)
        self.case_id_col = detect_column(self.frame, case_id_col, CASE_ID_COL_CANDIDATES, required=False)
        labels = pd.to_numeric(self.frame[self.label_col], errors="coerce")
        if not labels.isin([0, 1]).all():
            bad = self.frame.loc[~labels.isin([0, 1]), self.label_col].head(5).tolist()
            raise ValueError(f"Source labels must all be binary 0/1; examples={bad}")

        self.image_size = int(image_size)
        self.path_mode = str(path_mode).lower()
        if self.path_mode not in PATH_MODES:
            raise ValueError(f"path_mode must be one of {PATH_MODES}")
        self.style_bank = style_bank
        if self.path_mode != "none" and not style_bank:
            raise ValueError(f"path_mode={self.path_mode} requires an unlabeled target style bank")
        self.style_means = list(style_bank.get("means", [])) if style_bank else []
        self.style_stds = list(style_bank.get("stds", [])) if style_bank else []
        if self.path_mode != "none" and (not self.style_means or len(self.style_means) != len(self.style_stds)):
            raise ValueError("Target style bank is empty or malformed")

        self.root_dir = Path(root_dir) if root_dir else None
        self.csv_parent = self.source_csv.parent
        self.num_path_views = max(0, int(num_path_views))
        if self.path_mode != "none" and self.num_path_views < 1:
            raise ValueError("Path training requires num_path_views >= 1")
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        if not 0.0 <= self.alpha_min <= self.alpha_max <= 1.0:
            raise ValueError("Expected 0 <= alpha_min <= alpha_max <= 1")
        self.endpoint_frac = float(endpoint_frac)
        if not 0.0 <= self.endpoint_frac <= 1.0:
            raise ValueError("endpoint_frac must be in [0,1]")
        self.style_sampling = str(style_sampling).lower()
        if self.style_sampling not in {"global", "per_image"}:
            raise ValueError("style_sampling must be 'global' or 'per_image'")
        self.path_noise_std = float(path_noise_std)
        self.noise_blur_kernel = int(noise_blur_kernel)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _sample_style(self, rng: np.random.RandomState) -> Tuple[float, float, int]:
        if self.style_sampling == "global":
            assert self.style_bank is not None
            return (
                float(self.style_bank["bank_mean"]),
                float(self.style_bank["bank_std_mean"]),
                -1,
            )
        style_index = int(rng.randint(0, len(self.style_means)))
        return float(self.style_means[style_index]), float(self.style_stds[style_index]), style_index

    def _sample_alpha(self, rng: np.random.RandomState) -> float:
        if self.path_mode == "endpoint":
            return 1.0
        if float(rng.rand()) < self.endpoint_frac:
            return 1.0
        return float(rng.uniform(self.alpha_min, self.alpha_max))

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[int(index)]
        path = resolve_path(str(row[self.image_col]), self.root_dir, self.csv_parent)
        source01 = load_gray01(path, image_size=self.image_size)

        # Index/epoch-derived RNG makes views reproducible with any worker count.
        item_seed = self.seed + 1_000_003 * self.epoch + 97_409 * int(index)
        rng = np.random.RandomState(item_seed % (2**32 - 1))
        if float(rng.rand()) < self.hflip_prob:
            source01 = torch.flip(source01, dims=(-1,))
        if float(rng.rand()) < self.vflip_prob:
            source01 = torch.flip(source01, dims=(-2,))

        clean = classifier_normalize(source01)
        views: List[torch.Tensor] = []
        alphas: List[float] = []
        style_indices: List[int] = []
        if self.path_mode != "none":
            for view_index in range(self.num_path_views):
                target_mean, target_std, style_index = self._sample_style(rng)
                alpha = self._sample_alpha(rng)
                generator = torch.Generator()
                generator.manual_seed(item_seed + 15_485_863 * (view_index + 1))
                view01, _ = make_style_path_view(
                    source01=source01,
                    target_mean=target_mean,
                    target_std=target_std,
                    alpha=alpha,
                    noise_std=self.path_noise_std if self.path_mode == "brownian" else 0.0,
                    blur_kernel=self.noise_blur_kernel,
                    generator=generator,
                )
                views.append(classifier_normalize(view01))
                alphas.append(alpha)
                style_indices.append(style_index)

        if views:
            path_views = torch.stack(views, dim=0)
        else:
            path_views = torch.empty((0, 3, self.image_size, self.image_size), dtype=clean.dtype)
        case_id = str(row[self.case_id_col]) if self.case_id_col else str(index)
        return {
            "clean": clean,
            "path_views": path_views,
            "alphas": torch.tensor(alphas, dtype=torch.float32),
            "style_indices": torch.tensor(style_indices, dtype=torch.long),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "case_id": case_id,
            "path": str(path),
        }


def _model_features_and_logits(
    model: nn.Module,
    inputs: torch.Tensor,
    need_features: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if not need_features:
        return None, model(inputs)
    if not hasattr(model, "extract_features") or not hasattr(model, "classifier"):
        raise ValueError(
            "Feature consistency requires a model with extract_features() and classifier; "
            "use custom_resnet18/50_* or set --feature_weight 0."
        )
    features = model.extract_features(inputs)  # type: ignore[attr-defined]
    logits = model.classifier(features)  # type: ignore[attr-defined]
    return features, logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    label_smoothing: float,
    clean_ce_weight: float,
    path_ce_weight: float,
    consistency_weight: float,
    feature_weight: float,
    consistency_temperature: float,
    consistency_ramp: float,
) -> Dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("loss", "clean_ce", "path_ce", "consistency", "feature")}
    n_batches = 0
    need_features = float(feature_weight) > 0.0
    temperature = max(float(consistency_temperature), 1e-6)

    for batch in loader:
        clean = batch["clean"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        path_views = batch["path_views"].to(device, non_blocking=True)
        batch_size, num_views = int(clean.shape[0]), int(path_views.shape[1])
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
            clean_features, clean_logits = _model_features_and_logits(model, clean, need_features=need_features)
            clean_ce = F.cross_entropy(clean_logits, labels, label_smoothing=float(label_smoothing))
            zero = clean_ce.new_zeros(())
            path_ce = zero
            consistency = zero
            feature_loss = zero

            if num_views > 0:
                flat_views = path_views.reshape(batch_size * num_views, *path_views.shape[2:])
                repeated_labels = labels.repeat_interleave(num_views)
                path_features, path_logits = _model_features_and_logits(model, flat_views, need_features=need_features)
                path_ce = F.cross_entropy(path_logits, repeated_labels, label_smoothing=float(label_smoothing))

                clean_teacher = torch.softmax(clean_logits.detach() / temperature, dim=1)
                clean_teacher = clean_teacher.repeat_interleave(num_views, dim=0)
                consistency = F.kl_div(
                    F.log_softmax(path_logits / temperature, dim=1),
                    clean_teacher,
                    reduction="batchmean",
                ) * (temperature**2)

                if need_features:
                    assert clean_features is not None and path_features is not None
                    teacher_features = clean_features.detach().repeat_interleave(num_views, dim=0)
                    feature_loss = 1.0 - F.cosine_similarity(
                        F.normalize(path_features, dim=1),
                        F.normalize(teacher_features, dim=1),
                        dim=1,
                    ).mean()

            loss = (
                float(clean_ce_weight) * clean_ce
                + float(path_ce_weight) * path_ce
                + float(consistency_weight) * float(consistency_ramp) * consistency
                + float(feature_weight) * float(consistency_ramp) * feature_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(loss.item())
        totals["clean_ce"] += float(clean_ce.item())
        totals["path_ce"] += float(path_ce.item())
        totals["consistency"] += float(consistency.item())
        totals["feature"] += float(feature_loss.item())
        n_batches += 1

    return {name: value / max(n_batches, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()
    probabilities: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    has_labels: List[torch.Tensor] = []
    case_ids: List[str] = []
    paths: List[str] = []

    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        prob = torch.softmax(model(inputs), dim=1)[:, 1].cpu()
        probabilities.append(prob)
        labels.append(batch["label"].cpu())
        raw_has_label = batch["has_label"]
        has_labels.append(raw_has_label.cpu().bool() if isinstance(raw_has_label, torch.Tensor) else torch.tensor(raw_has_label).bool())
        case_ids.extend([str(value) for value in batch["case_id"]])
        paths.extend([str(value) for value in batch["path"]])

    probability_np = torch.cat(probabilities).numpy()
    label_np = torch.cat(labels).numpy()
    has_label_np = torch.cat(has_labels).numpy()
    sample_frame = pd.DataFrame(
        {
            "case_id": case_ids,
            "image_path": paths,
            "label": label_np,
            "prob_positive": probability_np,
        }
    )
    sample_frame.loc[~has_label_np, "label"] = np.nan
    valid = has_label_np & np.isin(label_np, [0, 1])

    metric_names = ["auc", "ap", "acc", "sens", "recall", "spec", "precision", "f1", "mcc"]
    if np.any(valid):
        sample_pred = (probability_np[valid] >= 0.5).astype(int)
        sample_metrics = binary_metrics_full(label_np[valid], probability_np[valid], sample_pred)
        valid_frame = sample_frame.loc[valid].copy()
        case_frame = (
            valid_frame.groupby("case_id", sort=False)
            .agg(label=("label", "first"), prob_positive=("prob_positive", "mean"), n_images=("image_path", "size"))
            .reset_index()
        )
        case_prob = case_frame["prob_positive"].to_numpy(dtype=float)
        case_label = case_frame["label"].to_numpy(dtype=int)
        case_pred = (case_prob >= 0.5).astype(int)
        case_metrics = binary_metrics_full(case_label, case_prob, case_pred)
    else:
        sample_metrics = {name: float("nan") for name in metric_names}
        case_metrics = {name: float("nan") for name in metric_names}
        case_frame = pd.DataFrame(columns=["case_id", "label", "prob_positive", "n_images"])

    metrics: Dict[str, float] = {
        **{f"sample_{key}": float(value) for key, value in sample_metrics.items()},
        **{f"case_{key}": float(value) for key, value in case_metrics.items()},
        "n_samples": float(len(sample_frame)),
        "n_labeled_samples": float(int(valid.sum())),
        "n_cases": float(len(case_frame)),
    }
    return metrics, sample_frame, case_frame


def build_eval_loader(
    csv_path: Optional[Path],
    root_dir: Optional[Path],
    image_col: Optional[str],
    label_col: Optional[str],
    case_id_col: Optional[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Optional[DataLoader]:
    if csv_path is None:
        return None
    dataset = CSVBinaryDataset(
        csv_path=csv_path,
        image_col=image_col,
        label_col=label_col,
        case_id_col=case_id_col,
        image_size=image_size,
        root_dir=root_dir,
        is_train=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_and_save(
    split: str,
    loader: Optional[DataLoader],
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    metrics, sample_frame, case_frame = evaluate_classifier(model, loader, device)
    sample_frame.to_csv(output_dir / f"pred_{split}_per_image.csv", index=False)
    case_frame.to_csv(output_dir / f"pred_{split}_per_case.csv", index=False)
    save_json(metrics, output_dir / f"metrics_{split}.json")
    print(
        f"[{split}] case_AUC={metrics.get('case_auc', float('nan')):.4f} "
        f"sample_AUC={metrics.get('sample_auc', float('nan')):.4f} n_case={int(metrics['n_cases'])}"
    )
    return metrics


def load_initial_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    cleaned = {str(key).replace("module.", ""): value for key, value in state.items()}
    result = model.load_state_dict(cleaned, strict=False)
    bad_missing = [key for key in result.missing_keys if "rsa" not in key.lower()]
    bad_unexpected = [key for key in result.unexpected_keys if "rsa" not in key.lower()]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Classifier checkpoint mismatch: "
            f"missing(non-rsa)={bad_missing[:8]}, unexpected(non-rsa)={bad_unexpected[:8]}"
        )
    print(
        f"[Init] loaded {checkpoint_path} "
        f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}; rsa ignored)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("C2: task-preserving source-to-target path classifier")
    parser.add_argument("--src_train_csv", type=Path, required=True)
    parser.add_argument("--src_val_csv", type=Path, required=True)
    parser.add_argument("--src_test_csv", type=Path, default=None)
    parser.add_argument("--tgt_train_csv", type=Path, default=None, help="Unlabeled target train split; style statistics only")
    parser.add_argument("--tgt_test_csv", type=Path, default=None, help="Final evaluation only; never used for selection")
    parser.add_argument("--src_root_dir", type=Path, default=None)
    parser.add_argument("--tgt_root_dir", type=Path, default=None)
    parser.add_argument("--source_image_col", type=str, default=None)
    parser.add_argument("--target_image_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--case_id_col", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--path_mode", choices=PATH_MODES, default="linear")
    parser.add_argument("--style_sampling", choices=["global", "per_image"], default="per_image")
    parser.add_argument("--style_bank_size", type=int, default=1000)
    parser.add_argument("--style_bank_cache", type=Path, default=None, help="Optional cache shared across ablation runs")
    parser.add_argument("--num_path_views", type=int, default=2)
    parser.add_argument("--alpha_min", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--endpoint_frac", type=float, default=0.25)
    parser.add_argument("--path_noise_std", type=float, default=0.06)
    parser.add_argument("--noise_blur_kernel", type=int, default=21)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.5)

    parser.add_argument("--clean_ce_weight", type=float, default=1.0)
    parser.add_argument("--path_ce_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--feature_weight", type=float, default=0.1)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    parser.add_argument("--consistency_ramp_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--backbone", type=str, default="custom_resnet50_space")
    parser.add_argument("--pretrained", type=str, default="imagenet")
    parser.add_argument("--init_weights", type=Path, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default="cosine")
    parser.add_argument("--metric_for_best", choices=["case_auc", "case_ap", "sample_auc"], default="case_auc")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device if ("cuda" not in args.device.lower() or torch.cuda.is_available()) else "cpu"
    device = torch.device(device_name)

    if args.path_mode != "none" and args.tgt_train_csv is None:
        raise ValueError(f"--path_mode {args.path_mode} requires --tgt_train_csv")
    if args.feature_weight < 0 or args.consistency_weight < 0:
        raise ValueError("Consistency weights must be non-negative")

    style_bank: Optional[Dict[str, object]] = None
    if args.path_mode != "none":
        style_cache = args.style_bank_cache or (args.output_dir / "target_style_bank.json")
        style_bank = build_target_style_bank(
            target_csv=args.tgt_train_csv,
            image_size=args.image_size,
            max_samples=args.style_bank_size,
            seed=args.seed,
            root_dir=args.tgt_root_dir,
            image_col=args.target_image_col,
            cache_path=style_cache,
        )
        print(
            "[UDA audit] target_train labels ignored; "
            f"style_bank n={style_bank['n_styles']} mean={style_bank['bank_mean']:.4f} "
            f"mean_std={style_bank['bank_std_mean']:.4f}"
        )
    else:
        print("[UDA audit] path_mode=none: target_train CSV is not read")

    protocol = {
        "config": _jsonable_args(args),
        "target_train_usage": "unlabeled image mean/std only" if style_bank else "not accessed",
        "target_train_labels_used": False,
        "target_test_used_for_training": False,
        "target_test_used_for_checkpoint_selection": False,
        "checkpoint_selection_split": "source validation",
        "inference_requires_generator_or_vae": False,
    }
    save_json(protocol, args.output_dir / "config_used.json")

    train_dataset = StylePathSourceDataset(
        source_csv=args.src_train_csv,
        image_size=args.image_size,
        path_mode=args.path_mode,
        style_bank=style_bank,
        root_dir=args.src_root_dir,
        image_col=args.source_image_col,
        label_col=args.label_col,
        case_id_col=args.case_id_col,
        num_path_views=args.num_path_views,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        endpoint_frac=args.endpoint_frac,
        style_sampling=args.style_sampling,
        path_noise_std=args.path_noise_std,
        noise_blur_kernel=args.noise_blur_kernel,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        seed=args.seed,
    )
    loader_generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    src_val_loader = build_eval_loader(
        args.src_val_csv,
        args.src_root_dir,
        args.source_image_col,
        args.label_col,
        args.case_id_col,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    assert src_val_loader is not None

    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device).to(device)
    if args.init_weights is not None:
        load_initial_weights(model, args.init_weights, device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    best_score = -float("inf")
    best_epoch = 0
    best_path = args.output_dir / "best_checkpoint.pt"
    history: List[Dict[str, object]] = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        ramp_epochs = max(1, int(args.consistency_ramp_epochs))
        ramp = min(1.0, float(epoch) / float(ramp_epochs))
        losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=args.amp,
            label_smoothing=args.label_smoothing,
            clean_ce_weight=args.clean_ce_weight,
            path_ce_weight=args.path_ce_weight,
            consistency_weight=args.consistency_weight,
            feature_weight=args.feature_weight,
            consistency_temperature=args.consistency_temperature,
            consistency_ramp=ramp,
        )
        val_metrics, _, _ = evaluate_classifier(model, src_val_loader, device)
        score = float(val_metrics.get(args.metric_for_best, float("nan")))
        if math.isnan(score):
            raise RuntimeError(
                f"Source validation metric {args.metric_for_best} is NaN; "
                "check labels and case split before using target test."
            )
        row: Dict[str, object] = {
            "epoch": epoch,
            "consistency_ramp": ramp,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in losses.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} loss={losses['loss']:.5f} "
            f"src_val_case_auc={val_metrics['case_auc']:.4f} "
            f"path_ce={losses['path_ce']:.5f} cons={losses['consistency']:.5f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "args": _jsonable_args(args),
                    "source_val_metrics": val_metrics,
                    "best_metric_name": args.metric_for_best,
                    "best_metric_value": score,
                    "uda_audit": protocol,
                },
                best_path,
            )
        if epoch % max(1, int(args.save_every)) == 0 or epoch == args.epochs:
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "args": _jsonable_args(args)},
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )
        if scheduler is not None:
            scheduler.step()

    pd.DataFrame(history).to_csv(args.output_dir / "train_log.csv", index=False)
    print(f"[Train] best_epoch={best_epoch} best_{args.metric_for_best}={best_score:.6f}")

    # Only now load target test and use its labels: checkpoint selection is finished.
    selected = torch.load(best_path, map_location=device, weights_only=False)
    result = model.load_state_dict(selected["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Selected checkpoint failed strict reload: {result}")
    final_metrics: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_source_val_metric": best_score,
        "metric_for_best": args.metric_for_best,
    }
    final_metrics["src_val"] = evaluate_and_save("src_val", src_val_loader, model, device, args.output_dir)

    src_test_loader = build_eval_loader(
        args.src_test_csv,
        args.src_root_dir,
        args.source_image_col,
        args.label_col,
        args.case_id_col,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    final_metrics["src_test"] = evaluate_and_save("src_test", src_test_loader, model, device, args.output_dir)
    tgt_test_loader = build_eval_loader(
        args.tgt_test_csv,
        args.tgt_root_dir,
        args.target_image_col,
        args.label_col,
        args.case_id_col,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    final_metrics["tgt_test"] = evaluate_and_save("tgt_test", tgt_test_loader, model, device, args.output_dir)
    save_json(final_metrics, args.output_dir / "final_metrics.json")


if __name__ == "__main__":
    main()
