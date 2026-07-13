import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import models


IMAGE_COL_CANDIDATES = ["image_path", "recon_image_path", "path", "img_path", "filepath", "file"]
LABEL_COL_CANDIDATES = ["label", "target", "y", "class", "cls"]


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_column(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], required: bool = True) -> Optional[str]:
    if preferred:
        if preferred in df.columns:
            return preferred
        raise ValueError(f"Column '{preferred}' not found. Available columns: {list(df.columns)}")

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise ValueError(f"Could not auto-detect required column. Available columns: {list(df.columns)}")
    return None


def resolve_path(path_value: str, root_dir: Optional[Path], csv_parent: Path) -> Path:
    p = Path(str(path_value).strip()).expanduser()
    if p.exists():
        return p

    if root_dir is not None:
        rp = (root_dir / p).resolve()
        if rp.exists():
            return rp

    cp = (csv_parent / p).resolve()
    if cp.exists():
        return cp

    if p.is_absolute():
        return p
    if root_dir is not None:
        return (root_dir / p).resolve()
    return cp


def build_transform(resize: int) -> T.Compose:
    # Mirrors old pipeline in data.py: ToTensor -> Resize -> repeat(3) for grayscale -> Normalize(0.5)
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((resize, resize), antialias=True),
            T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image-like array, got shape {arr.shape}")

    arr = arr.astype(np.float32)
    if arr.max() > 1.0 or arr.min() < 0.0:
        lo = float(np.percentile(arr, 1.0))
        hi = float(np.percentile(arr, 99.0))
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L")


class CSVImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_col: Optional[str] = None,
        label_col: Optional[str] = None,
        resize: int = 224,
        root_dir: Optional[Path] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"test_csv not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if len(self.df) == 0:
            raise ValueError(f"test_csv has no rows: {self.csv_path}")

        self.image_col = detect_column(self.df, image_col, IMAGE_COL_CANDIDATES, required=True)
        self.label_col = detect_column(self.df, label_col, LABEL_COL_CANDIDATES, required=False)
        self.transform = build_transform(resize)
        self.root_dir = Path(root_dir) if root_dir else None
        self.csv_parent = self.csv_path.parent

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        resolved_path = resolve_path(str(row[self.image_col]), self.root_dir, self.csv_parent)
        suffix = resolved_path.suffix.lower()

        if suffix == ".npy":
            arr = np.load(resolved_path)
            img = array_to_pil(arr)
        else:
            img = Image.open(resolved_path).convert("L")

        x = self.transform(img)

        label = -1
        has_label = self.label_col is not None and self.label_col in row and pd.notna(row[self.label_col])
        if has_label:
            label = int(row[self.label_col])

        return {
            "image": x,
            "label": torch.tensor(label, dtype=torch.long),
            "has_label": bool(has_label),
            "path": str(resolved_path),
        }


class SpaceAttention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.conv(x)
        b, c, h, w = att.shape
        att = att.view(b, c, -1)
        att = self.soft(att)
        att = att.view(b, c, h, w)
        m = att.view(b, c, -1).amax(dim=2, keepdim=True).clamp_min(1e-6)
        att = (att.view(b, c, -1) / m).view(b, c, h, w)
        return x * att


class RSABlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1)
        return x + self.net(x)


class HybridResNet(nn.Module):
    def __init__(self, method: str = "resnet50_space", num_classes: int = 2, pretrained: str = "none"):
        super().__init__()
        self.use_space = "space" in method
        self.use_rsa = "rsa" in method
        use_imagenet = isinstance(pretrained, str) and pretrained.lower() == "imagenet"

        if "resnet50" in method:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_imagenet else None
            base = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet else None
            base = models.resnet18(weights=weights)
            feat_dim = 512

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.use_space:
            self.space_attn = SpaceAttention(feat_dim)
        if self.use_rsa:
            self.rsa = RSABlock(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.use_space:
            x = self.space_attn(x)
        x = self.avgpool(x).flatten(1)
        if self.use_rsa:
            x = self.rsa(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_model(backbone: str, num_classes: int, pretrained: str, device: torch.device) -> nn.Module:
    # Mirrors old train_test_src.build_model behavior.
    if backbone.startswith("custom_"):
        method = backbone.replace("custom_", "")
        model = HybridResNet(method=method, num_classes=num_classes, pretrained=pretrained)
        if hasattr(model, "rsa"):
            model.rsa = nn.Identity()
            print("[Hotfix] Disable model.rsa -> Identity() during inference")
        return model.to(device)

    use_imagenet = isinstance(pretrained, str) and pretrained.lower() == "imagenet"
    if backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_imagenet else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_imagenet else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone in ("alexnet", "alexnet_mrnet"):
        base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if use_imagenet else None)
        m = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m.to(device)


def load_weights(model: nn.Module, weights: Path, device: torch.device) -> None:
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    sd = torch.load(weights, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict or {'state_dict': ...}")

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    missing = getattr(res, "missing_keys", [])
    unexpected = getattr(res, "unexpected_keys", [])
    print(f"[Init] loaded weights from {weights} (missing={len(missing)}, unexpected={len(unexpected)})")


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    if pos > 0 and neg > 0:
        order = np.argsort(-y_prob)
        ys = y_true[order]
        tp_cum = np.cumsum(ys == 1)
        fp_cum = np.cumsum(ys == 0)
        tpr = np.concatenate([[0.0], tp_cum / max(pos, 1)])
        fpr = np.concatenate([[0.0], fp_cum / max(neg, 1)])
        auc = float(np.trapz(tpr, fpr))
    else:
        auc = float("nan")

    return {
        "auc": float(auc),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    model.eval()

    probs_list: List[torch.Tensor] = []
    preds_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    has_label_list: List[torch.Tensor] = []
    paths: List[str] = []

    t = max(float(temperature), 1e-6)

    for batch in loader:
        xb = batch["image"].to(device, non_blocking=True)
        logits = model(xb) / t
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu()
        pred = (p1 >= float(threshold)).long().cpu()

        probs_list.append(p1)
        preds_list.append(pred)
        labels_list.append(batch["label"].cpu())
        has_label_list.append(torch.tensor(batch["has_label"], dtype=torch.bool))
        paths.extend(batch["path"])

    probs = torch.cat(probs_list).numpy()
    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    has_label = torch.cat(has_label_list).numpy()
    return probs, preds, paths, labels, has_label


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Evaluate an existing classifier checkpoint on CSV image paths")
    ap.add_argument("--test_csv", type=Path, required=True)
    ap.add_argument("--image_col", type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--root_dir", type=Path, default=None, help="Optional root for resolving relative image paths in CSV")

    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="none", help="imagenet or none")

    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device.lower()) else "cpu")

    ds = CSVImageDataset(
        csv_path=args.test_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        resize=args.resize,
        root_dir=args.root_dir,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    load_weights(model, args.weights, device=device)
    model.to(device)

    probs, preds, paths, labels, has_label = run_inference(
        model=model,
        loader=dl,
        device=device,
        temperature=args.temperature,
        threshold=args.threshold,
    )

    out_df = pd.DataFrame(
        {
            "image_path": paths,
            "label": labels,
            "predicted_class": preds,
            "prob_positive": probs,
        }
    )

    if not np.all(has_label):
        out_df.loc[~has_label, "label"] = np.nan

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[Done] saved predictions to {args.out_csv}")

    if np.any(has_label):
        valid = has_label & np.isin(labels, [0, 1])
        if np.any(valid):
            m = binary_metrics(labels[valid], probs[valid], preds[valid])
            print(
                "Metrics | "
                f"AUC={m['auc']:.4f} "
                f"ACC={m['acc']:.4f} "
                f"Precision={m['precision']:.4f} "
                f"Recall={m['recall']:.4f} "
                f"F1={m['f1']:.4f} "
                f"Specificity={m['specificity']:.4f}"
            )
        else:
            print("[Warn] Labels exist but no valid binary labels {0,1}; metrics skipped.")
    else:
        print("[Info] No labels found; only predictions were exported.")


if __name__ == "__main__":
    main()
