"""
统一的源域强度统计 (mean/std in [0,1]), 供 moment matching 使用。

moment_self 数据集、Oracle 评估、以及任何后续用矩匹配的脚本都从这里取值,
并缓存到 <source_csv>.style_stats_n{n}_s{seed}_sz{size}.json, 保证跨脚本完全一致
(修复"不同脚本各算一次、n/seed/分辨率不同"导致的不可比问题)。
"""
from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

_IMAGE_COLS = ["image_path", "path", "img_path", "file", "filepath"]


def _detect_img_col(df: pd.DataFrame, image_col=None) -> str:
    if image_col and image_col in df.columns:
        return image_col
    low = {c.lower(): c for c in df.columns}
    for k in _IMAGE_COLS:
        if k in low:
            return low[k]
    raise ValueError(f"No image column in {list(df.columns)}")


def _resolve(p: str, csv_parent: Path) -> str:
    pp = Path(str(p).strip())
    if pp.exists():
        return str(pp)
    alt = (csv_parent / pp)
    return str(alt if alt.exists() else pp)


def get_source_stats(source_csv, image_col=None, image_size=128, n=1000, seed=42, cache=True):
    """Return (mean, std) of source images in [0,1]; cached deterministically."""
    source_csv = str(source_csv)
    cache_path = f"{source_csv}.style_stats_n{n}_s{seed}_sz{image_size}.json"
    if cache and os.path.isfile(cache_path):
        d = json.load(open(cache_path))
        return float(d["mean"]), float(d["std"])
    df = pd.read_csv(source_csv)
    col = _detect_img_col(df, image_col)
    parent = Path(source_csv).parent
    rng = random.Random(seed)
    idxs = rng.sample(range(len(df)), min(n, len(df)))
    s = ss = cnt = 0.0
    for i in idxs:
        p = _resolve(str(df.iloc[i][col]), parent)
        g = np.asarray(Image.open(p).convert("L").resize((image_size, image_size)), dtype=np.float32) / 255.0
        s += float(g.sum()); ss += float((g ** 2).sum()); cnt += g.size
    mean = s / cnt
    std = float(np.sqrt(max(ss / cnt - mean ** 2, 1e-8)))
    if cache:
        try:
            json.dump({"mean": mean, "std": std, "num_samples": len(idxs),
                       "image_size": image_size, "seed": seed, "source_csv": source_csv},
                      open(cache_path, "w"), indent=2)
        except Exception:
            pass
    return float(mean), std


def moment_match(g01: np.ndarray, s_mean: float, s_std: float) -> np.ndarray:
    """Per-image affine so its mean/std match source (mean,std); input & output in [0,1]."""
    m, s = float(g01.mean()), float(g01.std()) + 1e-6
    return np.clip((g01 - m) / s * s_std + s_mean, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    m, s = get_source_stats(a.source_csv, image_size=a.image_size, n=a.n, seed=a.seed)
    print("source stats: mean=%.4f std=%.4f (cached)" % (m, s))
