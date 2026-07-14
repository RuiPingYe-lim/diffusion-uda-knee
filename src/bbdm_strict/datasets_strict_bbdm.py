from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_COLUMN_CANDIDATES = ["image_path", "path", "img_path", "file", "filepath"]
LABEL_COLUMN_CANDIDATES = ["label", "target", "y", "class", "cls"]
VOLUME_COLUMN_CANDIDATES = ["original_volume_path", "volume_path", "npy_path"]


def _detect_column(df: pd.DataFrame, user_col: Optional[str], candidates: list[str], required: bool) -> Optional[str]:
    if user_col:
        if user_col in df.columns:
            return user_col
        raise ValueError(f"Column '{user_col}' is not in CSV columns: {list(df.columns)}")
    lower_to_original = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    if required:
        raise ValueError(f"Could not detect required column from {candidates}. Available: {list(df.columns)}")
    return None


def _resolve_path(path_value: str, csv_parent: Path, root_dir: Optional[Path]) -> Path:
    candidate = Path(str(path_value).strip()).expanduser()
    if candidate.exists():
        return candidate
    if root_dir is not None:
        rooted = (root_dir / candidate).resolve()
        if rooted.exists():
            return rooted
    csv_rel = (csv_parent / candidate).resolve()
    if csv_rel.exists():
        return csv_rel
    if root_dir is not None:
        return (root_dir / candidate).resolve()
    return csv_rel


def _load_image(image_path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).float() * 2.0 - 1.0


def _intensity_norm01(arr: np.ndarray) -> np.ndarray:
    # Keep strict branch preprocessing aligned with data.py.
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
    if vmax > vmin:
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr.astype(np.float32)


def _volume_to_2d(arr: np.ndarray, proj: str = "mean") -> np.ndarray:
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        s, h, w = arr.shape[0], arr.shape[-2], arr.shape[-1]
        looks_like_shw = (s >= 4 and h >= 32 and w >= 32)
        looks_like_hwc = (arr.shape[-1] in (1, 3)) and (arr.shape[0] == h) and (arr.shape[1] == w)
        if looks_like_shw and not looks_like_hwc:
            if proj == "max":
                img = arr.max(axis=0)
            elif proj == "median":
                img = np.median(arr, axis=0)
            elif proj == "center":
                img = arr[arr.shape[0] // 2]
            else:
                img = arr.mean(axis=0)
        else:
            c = arr.shape[-1]
            img = arr[..., 0] if c == 1 else arr.mean(axis=-1)
    else:
        img = arr[arr.shape[0] // 2]
    return _intensity_norm01(img.astype(np.float32))


def _load_volume(volume_path: Path, image_size: int) -> torch.Tensor:
    arr = np.load(volume_path, allow_pickle=False)
    img2d = _volume_to_2d(arr, proj="mean")
    img = Image.fromarray((np.clip(img2d, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(out).unsqueeze(0).float() * 2.0 - 1.0


def _resolve_optional_existing_path(
    row: pd.Series,
    col_name: Optional[str],
    csv_parent: Path,
    root_dir: Optional[Path],
) -> Optional[Path]:
    if col_name is None or col_name not in row or pd.isna(row[col_name]):
        return None
    value = str(row[col_name]).strip()
    if not value:
        return None
    path = _resolve_path(value, csv_parent, root_dir)
    if path.exists():
        return path
    return None


class StrictBBDMPairedDataset(Dataset):
    """
    Pair dataset for strict-style pseudo-paired BBDM training.

    Domains in this project:
    - x_A: source domain (MRNet-like style endpoint)
    - x_B: target domain (KneeMRI endpoint)

    pair_mode:
    - paired_csv: source/target rows are index-aligned pairs.
    - label_random: class-consistent pseudo-pairing fallback, target(label=y)
      matched with random source(label=y).
    - moment_self: CONTENT-PRESERVING endpoints. For each target image x_B,
      x_A is the SAME image moment-matched to the source-domain global intensity
      statistics (mean/std). The two bridge endpoints are structurally identical
      and differ only in style, so the bridge learns a pure style map that keeps
      the case's discriminative content. This is the diagnostic that isolates
      whether random (content-mismatched) endpoints were the cause of poor results.

    Important:
    - label_random is pseudo-paired/class-consistent pairing (endpoints share only
      a label, NOT content) -> not official paired BBDM data; the bridge is forced
      to change anatomy, which harms content preservation.
    - Optional pair-cache can freeze target->source mapping within an epoch and
      reshuffle on epoch boundary with deterministic seed control.
    """

    def __init__(
        self,
        source_csv: str,
        target_csv: str,
        image_size: int = 128,
        pair_mode: str = "label_random",
        source_root: Optional[str] = None,
        target_root: Optional[str] = None,
        source_image_col: Optional[str] = None,
        target_image_col: Optional[str] = None,
        source_label_col: Optional[str] = None,
        target_label_col: Optional[str] = None,
        seed: int = 42,
        use_pair_cache: bool = True,
        reshuffle_pairs_each_epoch: bool = True,
    ) -> None:
        self.source_csv = Path(source_csv)
        self.target_csv = Path(target_csv)
        self.source_root = Path(source_root) if source_root else None
        self.target_root = Path(target_root) if target_root else None
        self.image_size = int(image_size)
        self.pair_mode = pair_mode.lower()
        self.seed = int(seed)
        self.use_pair_cache = bool(use_pair_cache)
        self.reshuffle_pairs_each_epoch = bool(reshuffle_pairs_each_epoch)

        self._rng = random.Random(self.seed)
        self._pair_map: Dict[int, int] = {}
        self._pair_epoch = -1

        if self.pair_mode not in {"paired_csv", "label_random", "moment_self"}:
            raise ValueError("pair_mode must be 'paired_csv', 'label_random' or 'moment_self'")

        self.df_source = pd.read_csv(self.source_csv)
        self.df_target = pd.read_csv(self.target_csv)
        if len(self.df_source) == 0 or len(self.df_target) == 0:
            raise ValueError("source_csv and target_csv must both be non-empty")

        self.source_img_col = _detect_column(self.df_source, source_image_col, IMAGE_COLUMN_CANDIDATES, required=True)
        self.target_img_col = _detect_column(self.df_target, target_image_col, IMAGE_COLUMN_CANDIDATES, required=True)
        self.source_vol_col = _detect_column(self.df_source, None, VOLUME_COLUMN_CANDIDATES, required=False)
        self.target_vol_col = _detect_column(self.df_target, None, VOLUME_COLUMN_CANDIDATES, required=False)
        self.source_label_col = _detect_column(self.df_source, source_label_col, LABEL_COLUMN_CANDIDATES, required=False)
        self.target_label_col = _detect_column(self.df_target, target_label_col, LABEL_COLUMN_CANDIDATES, required=False)

        self.source_parent = self.source_csv.parent
        self.target_parent = self.target_csv.parent

        if self.pair_mode == "paired_csv":
            self.length = min(len(self.df_source), len(self.df_target))
            self.source_indices_by_label: Dict[int, list[int]] = {}
        elif self.pair_mode == "moment_self":
            # Content-preserving endpoints derived from the target image itself.
            self.length = len(self.df_target)
            self.source_indices_by_label = {}
            self.src_mean, self.src_std = self._compute_source_stats(n=1000)
        else:
            self.length = len(self.df_target)
            if self.source_label_col is None or self.target_label_col is None:
                raise ValueError("pair_mode='label_random' requires label columns in both CSVs")
            groups: Dict[int, list[int]] = {}
            for i, v in enumerate(self.df_source[self.source_label_col].tolist()):
                if pd.isna(v):
                    continue
                y = int(v)
                groups.setdefault(y, []).append(i)
            if len(groups) == 0:
                raise ValueError("No valid source labels found for label_random pairing")
            self.source_indices_by_label = groups
            if self.use_pair_cache:
                self.set_epoch(0)

    def _compute_source_stats(self, n: int = 1000):
        """Global (mean, std) of source-domain images in [0,1] for moment matching.
        Uses the shared, cached stats module so the moment_self dataset and the
        Oracle evaluation read IDENTICAL statistics (same n/seed/image_size)."""
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from source_style_stats import get_source_stats
        return get_source_stats(self.source_csv, image_col=self.source_img_col,
                                image_size=self.image_size, n=n, seed=self.seed)

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        if self.pair_mode != "label_random" or not self.use_pair_cache:
            return
        epoch_i = int(epoch)
        if (not self.reshuffle_pairs_each_epoch) and self._pair_epoch >= 0:
            return
        if epoch_i == self._pair_epoch:
            return

        rng = random.Random(self.seed + 1000003 * max(epoch_i, 0))
        new_map: Dict[int, int] = {}
        for t_idx in range(len(self.df_target)):
            row = self.df_target.iloc[t_idx]
            tgt_label = int(row[self.target_label_col]) if self.target_label_col and pd.notna(row[self.target_label_col]) else -1
            pool = self.source_indices_by_label.get(int(tgt_label), [])
            if not pool:
                raise ValueError(f"No source sample with label={tgt_label} for label_random pairing")
            new_map[t_idx] = rng.choice(pool)

        self._pair_map = new_map
        self._pair_epoch = epoch_i

    def _pick_source_index(self, idx: int, target_label: int) -> int:
        if self.pair_mode == "paired_csv":
            return idx % len(self.df_source)
        if self.use_pair_cache:
            if not self._pair_map:
                self.set_epoch(0)
            s_idx = self._pair_map.get(int(idx))
            if s_idx is None:
                pool = self.source_indices_by_label.get(int(target_label), [])
                if not pool:
                    raise ValueError(f"No source sample with label={target_label} for label_random pairing")
                s_idx = random.Random(self.seed + int(idx)).choice(pool)
            return int(s_idx)

        pool = self.source_indices_by_label.get(int(target_label), [])
        if not pool:
            raise ValueError(f"No source sample with label={target_label} for label_random pairing")
        return self._rng.choice(pool)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        t_idx = idx % len(self.df_target)
        t_row = self.df_target.iloc[t_idx]
        target_label = int(t_row[self.target_label_col]) if self.target_label_col and pd.notna(t_row[self.target_label_col]) else -1

        if self.pair_mode == "moment_self":
            # x_B = target image; x_A = SAME image moment-matched to source style.
            t_img_path = _resolve_path(str(t_row[self.target_img_col]), self.target_parent, self.target_root)
            g = np.asarray(Image.open(t_img_path).convert("L").resize((self.image_size, self.image_size)),
                           dtype=np.float32) / 255.0
            m, s = float(g.mean()), float(g.std()) + 1e-6
            g_a = np.clip((g - m) / s * self.src_std + self.src_mean, 0.0, 1.0)  # source-styled, same content
            x_b = torch.from_numpy(g).unsqueeze(0).float() * 2.0 - 1.0
            x_a = torch.from_numpy(g_a.astype(np.float32)).unsqueeze(0).float() * 2.0 - 1.0
            # NO target label is used in moment_self (endpoints derived purely from the
            # target image + source-domain global stats). Return -1 so nothing downstream
            # can leak a target training label; SupCon must be disabled in this mode.
            return {
                "x_A": x_a, "x_B": x_b,
                "label": -1, "source_label": -1, "target_label": -1,
                "source_path": str(t_img_path), "target_path": str(t_img_path),
                "pair_mode": self.pair_mode, "pair_epoch": int(self._pair_epoch),
            }

        s_idx = self._pick_source_index(t_idx, target_label)
        s_row = self.df_source.iloc[s_idx]
        source_label = int(s_row[self.source_label_col]) if self.source_label_col and pd.notna(s_row[self.source_label_col]) else -1

        s_vol_path = _resolve_optional_existing_path(s_row, self.source_vol_col, self.source_parent, self.source_root)
        t_vol_path = _resolve_optional_existing_path(t_row, self.target_vol_col, self.target_parent, self.target_root)
        s_img_path = _resolve_path(str(s_row[self.source_img_col]), self.source_parent, self.source_root)
        t_img_path = _resolve_path(str(t_row[self.target_img_col]), self.target_parent, self.target_root)

        x_a = _load_volume(s_vol_path, self.image_size) if s_vol_path is not None else _load_image(s_img_path, self.image_size)
        x_b = _load_volume(t_vol_path, self.image_size) if t_vol_path is not None else _load_image(t_img_path, self.image_size)

        return {
            "x_A": x_a,
            "x_B": x_b,
            "label": target_label,
            "source_label": source_label,
            "target_label": target_label,
            "source_path": str(s_vol_path) if s_vol_path is not None else str(s_img_path),
            "target_path": str(t_vol_path) if t_vol_path is not None else str(t_img_path),
            "pair_mode": self.pair_mode,
            "pair_epoch": int(self._pair_epoch),
        }


class SingleDomainSliceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 128,
        root_dir: Optional[str] = None,
        image_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir else None
        self.image_size = int(image_size)
        self.df = pd.read_csv(self.csv_path)
        if len(self.df) == 0:
            raise ValueError(f"CSV has no rows: {self.csv_path}")
        self.image_col = _detect_column(self.df, image_col, IMAGE_COLUMN_CANDIDATES, required=True)
        self.volume_col = _detect_column(self.df, None, VOLUME_COLUMN_CANDIDATES, required=False)
        self.label_col = _detect_column(self.df, label_col, LABEL_COLUMN_CANDIDATES, required=False)
        self.csv_parent = self.csv_path.parent

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        volume_path = _resolve_optional_existing_path(row, self.volume_col, self.csv_parent, self.root_dir)
        image_path = _resolve_path(str(row[self.image_col]), self.csv_parent, self.root_dir)
        image = _load_volume(volume_path, self.image_size) if volume_path is not None else _load_image(image_path, self.image_size)
        label = int(row[self.label_col]) if self.label_col and pd.notna(row[self.label_col]) else -1
        return {
            "image": image,
            "path": str(volume_path) if volume_path is not None else str(image_path),
            "label": label,
        }
