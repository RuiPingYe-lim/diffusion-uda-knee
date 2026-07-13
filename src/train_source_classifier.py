import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from eval_existing_classifier_on_csv import (
    IMAGE_COL_CANDIDATES,
    LABEL_COL_CANDIDATES,
    array_to_pil,
    build_model,
    detect_column,
    load_weights,
    resolve_path,
    set_seed,
)


CASE_ID_COL_CANDIDATES = ["case_id", "id", "case", "study_id", "patient_id"]


def build_eval_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_train_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class CSVBinaryDataset(Dataset):
    """CSV dataset for binary classification with optional case_id passthrough."""

    def __init__(
        self,
        csv_path: Path,
        image_col: Optional[str],
        label_col: Optional[str],
        case_id_col: Optional[str],
        image_size: int,
        root_dir: Optional[Path],
        is_train: bool,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        if len(self.df) == 0:
            raise ValueError(f"CSV has no rows: {self.csv_path}")

        self.image_col = detect_column(self.df, image_col, IMAGE_COL_CANDIDATES, required=True)
        self.label_col = detect_column(self.df, label_col, LABEL_COL_CANDIDATES, required=False)
        self.case_id_col = detect_column(self.df, case_id_col, CASE_ID_COL_CANDIDATES, required=False)
        self.root_dir = Path(root_dir) if root_dir else None
        self.csv_parent = self.csv_path.parent
        self.transform = build_train_transform(image_size) if is_train else build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        resolved_path = resolve_path(str(row[self.image_col]), self.root_dir, self.csv_parent)
        suffix = resolved_path.suffix.lower()

        if suffix == ".npy":
            arr = np.load(resolved_path)
            img = array_to_pil(arr)
        else:
            img = Image.open(resolved_path).convert("L")

        x = self.transform(img)

        has_label = self.label_col is not None and self.label_col in row and pd.notna(row[self.label_col])
        label = int(row[self.label_col]) if has_label else -1
        if has_label and label not in (0, 1):
            raise ValueError(f"Expected binary label 0/1, got {label} at row {idx} in {self.csv_path}")

        case_id = str(row[self.case_id_col]) if self.case_id_col is not None and self.case_id_col in row else str(idx)
        return {
            "image": x,
            "label": torch.tensor(label, dtype=torch.long),
            "has_label": bool(has_label),
            "case_id": case_id,
            "path": str(resolved_path),
        }


def binary_metrics_full(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    specificity = tn / max(tn + fp, 1)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    auc = float("nan")
    ap = float("nan")
    if pos > 0 and neg > 0:
        order = np.argsort(-y_prob)
        ys = y_true[order]
        tp_cum = np.cumsum(ys == 1)
        fp_cum = np.cumsum(ys == 0)

        tpr = np.concatenate([[0.0], tp_cum / max(pos, 1)])
        fpr = np.concatenate([[0.0], fp_cum / max(neg, 1)])
        auc = float(np.trapz(tpr, fpr))

        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        recall_curve = tp_cum / max(pos, 1)
        precision_curve = np.concatenate([[1.0], precision_curve])
        recall_curve = np.concatenate([[0.0], recall_curve])
        ap = float(np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:]))

    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = float((tp * tn - fp * fn) / denom)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "acc": float(acc),
        "sens": float(recall),
        "recall": float(recall),
        "spec": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    probs_list: List[torch.Tensor] = []
    preds_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    has_label_list: List[torch.Tensor] = []
    paths: List[str] = []
    case_ids: List[str] = []

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True)
        logits = model(xb)
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu()
        pred = (p1 >= float(threshold)).long().cpu()

        probs_list.append(p1)
        preds_list.append(pred)
        labels_list.append(batch["label"].cpu())
        if isinstance(batch["has_label"], torch.Tensor):
            has_label_list.append(batch["has_label"].detach().cpu().bool())
        else:
            has_label_list.append(torch.tensor(batch["has_label"], dtype=torch.bool))
        paths.extend(batch["path"])
        case_ids.extend(batch["case_id"])

    probs = torch.cat(probs_list).numpy()
    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    has_label = torch.cat(has_label_list).numpy()

    pred_df = pd.DataFrame(
        {
            "case_id": case_ids,
            "image_path": paths,
            "label": labels,
            "prob_positive": probs,
            "predicted_class": preds,
        }
    )
    if not np.all(has_label):
        pred_df.loc[~has_label, "label"] = np.nan

    valid = has_label & np.isin(labels, [0, 1])
    if np.any(valid):
        metrics = binary_metrics_full(labels[valid], probs[valid], preds[valid])
    else:
        metrics = {k: float("nan") for k in ["auc", "ap", "acc", "sens", "recall", "spec", "precision", "f1", "mcc"]}
    metrics["n_samples"] = float(len(labels))
    metrics["n_labeled"] = float(int(np.sum(valid)))
    return metrics, pred_df


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True)
        yb = batch["label"].to(device, non_blocking=True).long()
        valid = (yb >= 0) & (yb <= 1)
        if not torch.any(valid):
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
            logits = model(xb[valid])
            loss = criterion(logits, yb[valid])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(n_batches, 1)


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    n = name.lower()
    if n == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, name: str, epochs: int):
    n = name.lower()
    if n == "none":
        return None
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if n == "step":
        step_size = max(1, epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {name}")


def score_for_best(metrics: Dict[str, float], metric_name: str) -> float:
    v = float(metrics.get(metric_name, float("nan")))
    if np.isnan(v):
        return -float("inf")
    return v


def save_json(obj: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def maybe_build_loader(
    csv_path: Optional[Path],
    root_dir: Optional[Path],
    image_col: Optional[str],
    label_col: Optional[str],
    case_id_col: Optional[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    is_train: bool,
) -> Optional[DataLoader]:
    if csv_path is None:
        return None
    ds = CSVBinaryDataset(
        csv_path=csv_path,
        image_col=image_col,
        label_col=label_col,
        case_id_col=case_id_col,
        image_size=image_size,
        root_dir=root_dir,
        is_train=is_train,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_and_save_split(
    split_name: str,
    loader: Optional[DataLoader],
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> None:
    if loader is None:
        return
    metrics, pred_df = evaluate(model=model, loader=loader, device=device)
    pred_path = output_dir / f"pred_{split_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    metrics_path = output_dir / f"metrics_{split_name}.json"
    save_json(metrics, metrics_path)
    print(f"[{split_name}] AUC={metrics.get('auc', float('nan')):.4f} AP={metrics.get('ap', float('nan')):.4f}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Train source-domain binary classifier and evaluate on source/target test CSVs")
    ap.add_argument("--src_train_csv", type=Path, default=None)
    ap.add_argument("--src_val_csv", type=Path, required=True)
    ap.add_argument("--src_test_csv", type=Path, default=None)
    ap.add_argument("--src_root_dir", type=Path, default=None)
    ap.add_argument("--tgt_test_csv", type=Path, default=None)
    ap.add_argument("--tgt_recon_test_csv", type=Path, default=None)
    ap.add_argument("--tgt_root_dir", type=Path, default=None)
    ap.add_argument("--image_col", type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--case_id_col", type=str, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="none")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric_for_best", type=str, default="auc", choices=["auc", "ap", "acc", "f1", "mcc"])
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="Skip training and evaluate with --weights")
    ap.add_argument("--weights", type=Path, default=None, help="Checkpoint for eval_only")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    if args.num_classes != 2:
        raise ValueError("This script is for binary classification only. Please set --num_classes 2.")

    if args.eval_only and args.weights is None:
        raise ValueError("--eval_only requires --weights.")
    if not args.eval_only and args.src_train_csv is None:
        raise ValueError("Training mode requires --src_train_csv.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = None
    if not args.eval_only:
        train_loader = maybe_build_loader(
            csv_path=args.src_train_csv,
            root_dir=args.src_root_dir,
            image_col=args.image_col,
            label_col=args.label_col,
            case_id_col=args.case_id_col,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_train=True,
        )
    val_loader = maybe_build_loader(
        csv_path=args.src_val_csv,
        root_dir=args.src_root_dir,
        image_col=args.image_col,
        label_col=args.label_col,
        case_id_col=args.case_id_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    src_test_loader = maybe_build_loader(
        csv_path=args.src_test_csv,
        root_dir=args.src_root_dir,
        image_col=args.image_col,
        label_col=args.label_col,
        case_id_col=args.case_id_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    tgt_test_loader = maybe_build_loader(
        csv_path=args.tgt_test_csv,
        root_dir=args.tgt_root_dir,
        image_col=args.image_col,
        label_col=args.label_col,
        case_id_col=args.case_id_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    tgt_recon_test_loader = maybe_build_loader(
        csv_path=args.tgt_recon_test_csv,
        root_dir=args.tgt_root_dir,
        image_col=args.image_col,
        label_col=args.label_col,
        case_id_col=args.case_id_col,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    model = build_model(backbone=args.backbone, num_classes=args.num_classes, pretrained=args.pretrained, device=device)
    model.to(device)

    best_ckpt_path = args.output_dir / "best_checkpoint.pt"
    last_ckpt_path = args.output_dir / "last_checkpoint.pt"
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model=model, name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_scheduler(optimizer=optimizer, name=args.scheduler, epochs=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

        best_score = -float("inf")
        best_epoch = 0
        train_log: List[Dict] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                scaler=scaler,
                use_amp=args.amp,
            )
            val_metrics, val_pred_df = evaluate(model=model, loader=val_loader, device=device)
            val_score = score_for_best(val_metrics, args.metric_for_best)

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                "best_metric_name": args.metric_for_best,
                "best_metric_value": float(val_score),
            }
            train_log.append(row)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.6f} val_auc={val_metrics.get('auc', float('nan')):.4f} "
                f"val_ap={val_metrics.get('ap', float('nan')):.4f} val_acc={val_metrics.get('acc', float('nan')):.4f}"
            )

            if scheduler is not None:
                scheduler.step()

            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "val_metrics": val_metrics,
                        "best_metric_name": args.metric_for_best,
                        "best_metric_value": float(best_score),
                    },
                    best_ckpt_path,
                )
                val_pred_df.to_csv(args.output_dir / "val_pred.csv", index=False)

            if epoch % max(1, args.save_every) == 0 or epoch == args.epochs:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "val_metrics": val_metrics,
                    },
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                )

        torch.save(
            {
                "epoch": args.epochs,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_metric_name": args.metric_for_best,
                "best_metric_value": float(best_score),
            },
            last_ckpt_path,
        )
        pd.DataFrame(train_log).to_csv(args.output_dir / "train_log.csv", index=False)
        print(f"[Train] Done. best_epoch={best_epoch} best_{args.metric_for_best}={best_score:.6f}")

    # Load checkpoint for final evaluation (best for training mode, user-provided for eval_only mode)
    eval_ckpt = args.weights if args.eval_only else best_ckpt_path
    ckpt = torch.load(eval_ckpt, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {eval_ckpt}")
    res = model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    print(f"[Eval] Loaded checkpoint {eval_ckpt} (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")

    # Re-run source val for reproducibility record with selected checkpoint
    val_metrics, val_pred_df = evaluate(model=model, loader=val_loader, device=device)
    val_pred_df.to_csv(args.output_dir / "val_pred.csv", index=False)
    save_json(val_metrics, args.output_dir / "metrics_src_val.json")

    evaluate_and_save_split("src_test", src_test_loader, model, device, args.output_dir)
    evaluate_and_save_split("tgt_test", tgt_test_loader, model, device, args.output_dir)
    evaluate_and_save_split("tgt_recon_test", tgt_recon_test_loader, model, device, args.output_dir)


if __name__ == "__main__":
    main()
