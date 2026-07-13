from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def _series_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }


def _analyze_csv(csv_path: Path, bins: int) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if "prob_positive" not in df.columns:
        raise ValueError(f"prob_positive not found in {csv_path}")

    probs = pd.to_numeric(df["prob_positive"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    hist_counts, hist_edges = np.histogram(probs, bins=bins, range=(0.0, 1.0))

    out: Dict[str, Any] = {
        "file": str(csv_path),
        "n_samples": int(probs.size),
        "histogram": {
            "bins": int(bins),
            "counts": hist_counts.astype(int).tolist(),
            "edges": hist_edges.astype(float).tolist(),
        },
        "overall": _series_stats(probs),
        "by_label": {},
    }

    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
        for y in [0, 1]:
            mask = labels == y
            sub = pd.to_numeric(df.loc[mask, "prob_positive"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            out["by_label"][str(y)] = _series_stats(sub)

    return out


def main() -> None:
    ap = argparse.ArgumentParser("Analyze prob_positive distributions for before/translated eval CSVs")
    ap.add_argument("--before_csv", type=Path, required=True)
    ap.add_argument("--translated_csv", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    res = {
        "before": _analyze_csv(args.before_csv, bins=int(args.bins)),
        "translated": _analyze_csv(args.translated_csv, bins=int(args.bins)),
    }

    # quick ratio diagnostics
    b_mean = res["before"]["overall"]["mean"]
    t_mean = res["translated"]["overall"]["mean"]
    b_med = res["before"]["overall"]["median"]
    t_med = res["translated"]["overall"]["median"]
    res["ratios"] = {
        "translated_over_before_mean_ratio": float((t_mean + 1e-8) / (b_mean + 1e-8)) if np.isfinite(b_mean) and np.isfinite(t_mean) else float("nan"),
        "translated_over_before_median_ratio": float((t_med + 1e-8) / (b_med + 1e-8)) if np.isfinite(b_med) and np.isfinite(t_med) else float("nan"),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"Saved probability diagnostics: {args.out_json}")


if __name__ == "__main__":
    main()
