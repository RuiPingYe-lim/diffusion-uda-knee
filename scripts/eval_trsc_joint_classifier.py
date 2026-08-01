#!/usr/bin/env python3
"""Write label-free target predictions from a TRSC joint classifier checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
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


class UnlabeledImageDataset(Dataset):
    """Load only image paths and case identifiers; no target label is accepted."""

    def __init__(
        self,
        frame: pd.DataFrame,
        path_column: str,
        case_column: str,
        input_size: int,
        csv_parent: Path,
        root_dir: Path | None,
    ) -> None:
        self.paths = [
            self._resolve_path(value, csv_parent, root_dir)
            for value in frame[path_column].astype(str)
        ]
        self.case_ids = frame[case_column].astype(str).tolist()
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

    @staticmethod
    def _resolve_path(
        value: str,
        csv_parent: Path,
        root_dir: Path | None,
    ) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (root_dir if root_dir is not None else csv_parent) / path

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, self.case_ids[index]


def load_classifier(
    checkpoint: Path,
    device: torch.device,
) -> SourceWarmStartResNet50:
    model = SourceWarmStartResNet50().to(device)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported classifier checkpoint type: {type(state)}")
    state = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="Optional root for relative image paths; defaults to the CSV directory",
    )
    parser.add_argument("--path_col", default="image_path")
    parser.add_argument("--case_col", default="case_id")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    header = pd.read_csv(args.input_csv, nrows=0)
    required = {args.path_col, args.case_col}
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    # Explicit usecols prevents an accidental target-label read.
    frame = pd.read_csv(
        args.input_csv,
        usecols=[args.path_col, args.case_col],
    )
    if frame.empty:
        raise ValueError("Input CSV contains no images")
    if frame[[args.path_col, args.case_col]].isna().any().any():
        raise ValueError("Input CSV contains missing paths or case identifiers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier(args.checkpoint, device)
    dataset = UnlabeledImageDataset(
        frame,
        args.path_col,
        args.case_col,
        args.input_size,
        args.input_csv.resolve().parent,
        args.root_dir.resolve() if args.root_dir is not None else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    with torch.no_grad():
        for images, case_ids in loader:
            probabilities = F.softmax(model(images.to(device)), dim=1)[:, 1]
            rows.extend(
                {"case_id": str(case_id), "prob": float(probability)}
                for case_id, probability in zip(
                    case_ids,
                    probabilities.cpu().numpy(),
                )
            )
    image_predictions = pd.DataFrame(rows)
    case_predictions = (
        image_predictions.groupby("case_id", sort=False, as_index=False)
        .agg(prob=("prob", "mean"), n_images=("prob", "size"))
    )
    if not np.isfinite(case_predictions["prob"]).all():
        raise ValueError("Classifier produced non-finite probabilities")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    case_predictions.to_csv(args.out, index=False)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "input_csv": str(args.input_csv.resolve()),
        "input_csv_sha256": sha256(args.input_csv),
        "root_dir": (
            str(args.root_dir.resolve()) if args.root_dir is not None else None
        ),
        "path_col": args.path_col,
        "case_col": args.case_col,
        "backbone": "custom_resnet50_space",
        "input_size": args.input_size,
        "n_images": len(image_predictions),
        "n_cases": len(case_predictions),
        "target_labels_used": False,
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "predictions": str(args.out),
                "metadata": str(metadata_path),
                "n_cases": len(case_predictions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
