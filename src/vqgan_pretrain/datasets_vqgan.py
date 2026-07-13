from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _import_project_utils():
    import sys

    this_file = Path(__file__).resolve()
    baseline_dir = this_file.parents[1]
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))
    from utils import LABEL_COLUMN_CANDIDATES, detect_column, resolve_path  # type: ignore

    return LABEL_COLUMN_CANDIDATES, detect_column, resolve_path


LABEL_COLUMN_CANDIDATES, detect_column, resolve_path = _import_project_utils()
IMAGE_COLUMN_CANDIDATES = ["image_path", "path", "img_path", "file", "filepath"]


class VQGANSliceDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        root_dir: Optional[Path] = None,
        image_col: Optional[str] = None,
        label_col: Optional[str] = None,
        image_size: int = 128,
        normalize_to_neg_one_one: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if len(self.df) == 0:
            raise ValueError(f"CSV has no rows: {self.csv_path}")

        self.root_dir = Path(root_dir) if root_dir else None
        self.csv_parent = self.csv_path.parent
        self.image_col = detect_column(self.df, image_col, IMAGE_COLUMN_CANDIDATES, kind="image_col", required=True)
        self.label_col = detect_column(
            self.df, label_col, LABEL_COLUMN_CANDIDATES, kind="label_col", required=False
        )
        self.image_size = int(image_size)
        self.normalize = bool(normalize_to_neg_one_one)
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_01(self, image_path: Path) -> np.ndarray:
        suffix = image_path.suffix.lower()
        if suffix == ".npy":
            image = np.load(image_path).astype(np.float32)
            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]
            if image.ndim == 3:
                center = image.shape[0] // 2
                image = image[center]
            if image.ndim != 2:
                raise ValueError(f"Expected 2D grayscale image at {image_path}, got shape {image.shape}")
            image_min, image_max = float(image.min()), float(image.max())
            if image_max > 1.0 or image_min < 0.0:
                denom = max(image_max - image_min, 1e-8)
                image = (image - image_min) / denom
            return np.clip(image, 0.0, 1.0)

        pil = Image.open(image_path).convert("L")
        return np.asarray(pil, dtype=np.float32) / 255.0

    def _resize(self, image_01: np.ndarray) -> np.ndarray:
        if image_01.shape[0] == self.image_size and image_01.shape[1] == self.image_size:
            return image_01.astype(np.float32)
        pil = Image.fromarray((np.clip(image_01, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        pil = pil.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32) / 255.0

    def __getitem__(self, index: int) -> Any:
        row = self.df.iloc[index]
        image_path = resolve_path(str(row[self.image_col]), root_dir=self.root_dir, csv_parent=self.csv_parent)
        image_01 = self._resize(self._load_image_01(image_path))

        tensor = torch.from_numpy(image_01).unsqueeze(0).float()
        if self.normalize:
            tensor = tensor * 2.0 - 1.0

        item: Dict[str, Any] = {"image": tensor}
        item["label"] = int(row[self.label_col]) if self.label_col and pd.notna(row[self.label_col]) else -1
        if self.return_metadata:
            item["metadata"] = {k: row[k] for k in self.df.columns}
            item["resolved_image_path"] = str(image_path)
        return item

