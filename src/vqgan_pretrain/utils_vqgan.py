from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def import_ssim_global():
    import sys

    this_file = Path(__file__).resolve()
    baseline_dir = this_file.parents[1]
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))
    from utils import ssim_global  # type: ignore

    return ssim_global


ssim_global = import_ssim_global()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def save_comparison_grid(
    x: torch.Tensor,
    recon: torch.Tensor,
    out_path: Path,
    max_items: int = 8,
) -> None:
    x01 = tensor_to_01(x)
    r01 = tensor_to_01(recon)
    n = min(int(x01.shape[0]), int(max_items))
    x01 = x01[:n]
    r01 = r01[:n]
    residual = (x01 - r01).abs()

    panel = torch.cat([x01, r01, residual], dim=0)  # [3N,1,H,W]
    grid = make_grid(panel, nrow=n, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)


def save_single_visualization(
    x: np.ndarray,
    recon: np.ndarray,
    out_path: Path,
) -> None:
    residual = np.abs(x - recon)
    x_u8 = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    r_u8 = (np.clip(recon, 0.0, 1.0) * 255.0).astype(np.uint8)
    res_u8 = (np.clip(residual, 0.0, 1.0) * 255.0).astype(np.uint8)
    canvas = np.concatenate([x_u8, r_u8, res_u8], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(out_path)


def compute_recon_metrics(x01: np.ndarray, recon01: np.ndarray) -> Dict[str, float]:
    x = x01.astype(np.float32)
    y = recon01.astype(np.float32)
    l1 = float(np.mean(np.abs(x - y)))
    mse = float(np.mean((x - y) ** 2))
    psnr = float(20.0 * math.log10(1.0) - 10.0 * math.log10(max(mse, 1e-12)))
    ssim = float(ssim_global(x, y))
    return {"l1": l1, "mse": mse, "psnr": psnr, "ssim": ssim}


def append_csv_row(csv_path: Path, row: Dict[str, float | int | str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_serializable_config(config: Dict) -> Dict:
    out = {}
    for k, v in config.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def maybe_to_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"l1": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    keys = list(rows[0].keys())
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}

