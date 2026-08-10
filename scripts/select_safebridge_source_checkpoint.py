#!/usr/bin/env python3
"""Select a SafeBridge classifier checkpoint using source validation AUC only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.unsb.trsc_joint_modules import SourceWarmStartResNet50  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Source validation requires both binary classes")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average tied ranks so AUC is invariant to checkpoint serialization order.
    unique_scores, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    if np.any(counts > 1):
        for score_index, count in enumerate(counts):
            if count > 1:
                mask = inverse == score_index
                ranks[mask] = ranks[mask].mean()
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


class SourceValidationDataset(Dataset):
    """Read labeled source validation images under the classifier contract."""

    def __init__(
        self,
        frame: pd.DataFrame,
        path_column: str,
        label_column: str,
        case_column: str,
        csv_parent: Path,
        root_dir: Path | None,
        input_size: int,
    ) -> None:
        self.rows = []
        selected = frame[[path_column, label_column, case_column]]
        for path_value, label_value, case_value in selected.itertuples(
            index=False,
            name=None,
        ):
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute():
                path = (root_dir if root_dir is not None else csv_parent) / path
            label = int(label_value)
            if label not in (0, 1):
                raise ValueError(f"Invalid source validation label: {label}")
            self.rows.append((path, label, str(case_value)))
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((input_size, input_size), antialias=True),
                T.Lambda(
                    lambda tensor: (
                        tensor.repeat(3, 1, 1)
                        if tensor.shape[0] == 1
                        else tensor
                    )
                ),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, label, case_id = self.rows[index]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, label, case_id


def load_classifier(path: Path, device: torch.device) -> SourceWarmStartResNet50:
    model = SourceWarmStartResNet50().to(device)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    state = {
        str(key).removeprefix("module."): value for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def evaluate(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    model = load_classifier(checkpoint, device)
    rows = []
    for images, labels, case_ids in loader:
        probabilities = F.softmax(model(images.to(device)), dim=1)[:, 1].cpu()
        rows.extend(
            {
                "case_id": str(case_id),
                "label": int(label),
                "prob": float(probability),
            }
            for case_id, label, probability in zip(
                case_ids, labels.tolist(), probabilities.tolist()
            )
        )
    frame = pd.DataFrame(rows)
    label_counts = frame.groupby("case_id")["label"].nunique()
    if bool((label_counts != 1).any()):
        raise ValueError("A source-validation case has inconsistent labels")
    cases = frame.groupby("case_id", as_index=False).agg(
        label=("label", "first"),
        prob=("prob", "mean"),
    )
    return binary_auc(cases["label"].to_numpy(), cases["prob"].to_numpy()), len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--source_val_csv", type=Path, required=True)
    parser.add_argument("--root_dir", type=Path)
    parser.add_argument("--path_col", default="image_path")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--case_col", default="case_id")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_checkpoint", type=Path)
    parser.add_argument("--out_metadata", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.path_col, args.label_col, args.case_col]
    frame = pd.read_csv(args.source_val_csv, usecols=required)
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Source validation CSV is empty or contains missing values")
    dataset = SourceValidationDataset(
        frame,
        args.path_col,
        args.label_col,
        args.case_col,
        args.source_val_csv.resolve().parent,
        args.root_dir.resolve() if args.root_dir is not None else None,
        args.input_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    checkpoints = sorted(args.checkpoint_dir.glob("*_net_C.pth"))
    checkpoints = [
        path
        for path in checkpoints
        if path.name not in {"best_source_val_net_C.pth"}
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No *_net_C.pth checkpoints under {args.checkpoint_dir}"
        )

    results = []
    for checkpoint in checkpoints:
        auc, case_count = evaluate(checkpoint, loader, device)
        results.append(
            {
                "checkpoint": checkpoint.name,
                "source_val_auc": auc,
                "source_val_cases": case_count,
                "sha256": sha256(checkpoint),
            }
        )
        print(f"[source selection] {checkpoint.name}: AUC={auc:.6f}")
    best = max(results, key=lambda item: item["source_val_auc"])
    best_path = args.checkpoint_dir / best["checkpoint"]
    out_checkpoint = (
        args.out_checkpoint
        if args.out_checkpoint is not None
        else args.checkpoint_dir / "best_source_val_net_C.pth"
    )
    out_metadata = (
        args.out_metadata
        if args.out_metadata is not None
        else out_checkpoint.with_suffix(out_checkpoint.suffix + ".json")
    )
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_checkpoint)
    metadata = {
        "selection_domain": "source_validation",
        "target_labels_used": False,
        "source_val_csv": str(args.source_val_csv.resolve()),
        "source_val_csv_sha256": sha256(args.source_val_csv),
        "selected": best,
        "candidates": results,
        "output_checkpoint": str(out_checkpoint.resolve()),
        "output_sha256": sha256(out_checkpoint),
    }
    out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
