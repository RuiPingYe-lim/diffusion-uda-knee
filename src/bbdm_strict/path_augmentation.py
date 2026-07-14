"""Task-preserving source-to-target style paths for classifier training.

The target training split is used only to estimate an unlabeled bank of image
mean/std pairs.  A labeled source image is moment-matched to one sampled target
style and becomes the opposite endpoint of a content-preserving path.  The
interior is either a deterministic linear interpolation or a Brownian bridge:

    x_alpha = (1-alpha) * x_source + alpha * x_target_style
              + sigma * sqrt(alpha * (1-alpha)) * low_frequency_noise

Noise is exactly zero at both endpoints.  No target label and no target image
content enters a source training example.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGE_COL_CANDIDATES = ["image_path", "path", "img_path", "file", "filepath"]


def _detect_image_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Column '{preferred}' not found. Available: {list(df.columns)}")
        return preferred
    lower = {str(col).lower(): str(col) for col in df.columns}
    for name in IMAGE_COL_CANDIDATES:
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"Could not detect image column. Available: {list(df.columns)}")


def _resolve_path(value: str, csv_parent: Path, root_dir: Optional[Path]) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.exists():
        return path
    if root_dir is not None:
        rooted = (root_dir / path).resolve()
        if rooted.exists():
            return rooted
    csv_relative = (csv_parent / path).resolve()
    if csv_relative.exists():
        return csv_relative
    if path.is_absolute():
        return path
    return (root_dir / path).resolve() if root_dir is not None else csv_relative


def _array_to_gray01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    elif arr.ndim == 3:
        # Volume input: stay aligned with the mean-projection classifier path.
        arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image or 3-D volume, got shape={arr.shape}")
    arr = arr.astype(np.float32)
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        lo, hi = np.percentile(arr, (1.0, 99.0))
        if hi > lo:
            arr = (arr - float(lo)) / (float(hi) - float(lo) + 1e-6)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_gray01(path: Path, image_size: int) -> torch.Tensor:
    """Load an image/NumPy volume as a [1,H,W] float tensor in [0,1]."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = _array_to_gray01(np.load(path, allow_pickle=False))
        image = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L")
    else:
        image = Image.open(path).convert("L")
    image = image.resize((int(image_size), int(image_size)), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).float()


def _image_column_digest(values: pd.Series) -> str:
    """Hash only the selected target image paths, never the target label column."""
    digest = hashlib.sha256()
    for value in values.astype(str).tolist():
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_target_style_bank(
    target_csv: Path,
    image_size: int,
    max_samples: int = 1000,
    seed: int = 42,
    root_dir: Optional[Path] = None,
    image_col: Optional[str] = None,
    cache_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Return target per-image mean/std values without reading target labels.

    The cache includes a SHA256 digest of the selected image-path column, so
    replacing or editing the split cannot silently reuse stale statistics.
    """
    target_csv = Path(target_csv).expanduser().resolve()
    if not target_csv.exists():
        raise FileNotFoundError(f"Target training CSV not found: {target_csv}")
    root_dir = Path(root_dir).expanduser().resolve() if root_dir else None
    header = pd.read_csv(target_csv, nrows=0)
    resolved_image_col = _detect_image_column(header, image_col)
    # usecols is a protocol guard: target labels are not even loaded in memory.
    frame = pd.read_csv(target_csv, usecols=[resolved_image_col])
    if frame.empty:
        raise ValueError(f"Target training CSV has no rows: {target_csv}")
    digest = _image_column_digest(frame[resolved_image_col])
    expected = {
        "version": 2,
        "target_csv": str(target_csv),
        "target_image_paths_sha256": digest,
        "root_dir": str(root_dir) if root_dir is not None else None,
        "requested_image_col": image_col,
        "image_col": resolved_image_col,
        "image_size": int(image_size),
        "max_samples": int(max_samples),
        "seed": int(seed),
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in expected.items()):
                means = cached.get("means", [])
                stds = cached.get("stds", [])
                if means and len(means) == len(stds):
                    return cached

    indices = list(range(len(frame)))
    random.Random(int(seed)).shuffle(indices)
    indices = indices[: min(max(1, int(max_samples)), len(indices))]

    means = []
    stds = []
    parent = target_csv.parent
    for index in indices:
        value = frame.iloc[index][resolved_image_col]
        path = _resolve_path(str(value), parent, root_dir)
        image = load_gray01(path, image_size=image_size)
        means.append(float(image.mean().item()))
        stds.append(float(image.std(unbiased=False).clamp_min(1e-6).item()))

    result: Dict[str, object] = {
        **expected,
        "n_styles": len(means),
        "means": means,
        "stds": stds,
        "bank_mean": float(np.mean(means)),
        "bank_std_mean": float(np.mean(stds)),
        "target_labels_used": False,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def moment_match_tensor(
    source01: torch.Tensor,
    target_mean: float,
    target_std: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match one source image to target moments while preserving its anatomy."""
    if source01.ndim != 3 or source01.shape[0] != 1:
        raise ValueError(f"Expected source01 shape [1,H,W], got {tuple(source01.shape)}")
    mean = source01.mean()
    std = source01.std(unbiased=False).clamp_min(float(eps))
    endpoint = (source01 - mean) / std * float(target_std) + float(target_mean)
    return endpoint.clamp(0.0, 1.0)


def _normalized_low_frequency_noise(
    reference: torch.Tensor,
    generator: torch.Generator,
    blur_kernel: int,
) -> torch.Tensor:
    noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    kernel = int(blur_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        max_kernel = min(int(reference.shape[-2]), int(reference.shape[-1]))
        if max_kernel % 2 == 0:
            max_kernel -= 1
        kernel = max(1, min(kernel, max_kernel))
        if kernel > 1:
            sigma = max(float(kernel) / 6.0, 0.1)
            noise = TF.gaussian_blur(noise, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    noise = noise - noise.mean()
    return noise / noise.std(unbiased=False).clamp_min(1e-6)


def make_style_path_view(
    source01: torch.Tensor,
    target_mean: float,
    target_std: float,
    alpha: float,
    noise_std: float = 0.0,
    blur_kernel: int = 21,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct one path view and its target-style endpoint.

    ``noise_std`` is the Brownian scale before multiplication by
    ``sqrt(alpha * (1-alpha))``.  It therefore cannot corrupt either endpoint.
    """
    alpha_f = float(min(1.0, max(0.0, alpha)))
    endpoint = moment_match_tensor(source01, target_mean=target_mean, target_std=target_std)
    view = (1.0 - alpha_f) * source01 + alpha_f * endpoint
    if float(noise_std) > 0.0 and 0.0 < alpha_f < 1.0:
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(torch.seed())
        noise = _normalized_low_frequency_noise(source01, generator, blur_kernel=blur_kernel)
        bridge_scale = float(noise_std) * math.sqrt(alpha_f * (1.0 - alpha_f))
        view = view + bridge_scale * noise
    return view.clamp(0.0, 1.0), endpoint


def classifier_normalize(image01: torch.Tensor) -> torch.Tensor:
    """Convert [1,H,W] [0,1] to the repository classifier's [3,H,W] [-1,1]."""
    if image01.ndim != 3 or image01.shape[0] != 1:
        raise ValueError(f"Expected image shape [1,H,W], got {tuple(image01.shape)}")
    return image01.repeat(3, 1, 1) * 2.0 - 1.0
