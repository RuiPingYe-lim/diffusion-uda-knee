#!/usr/bin/env python3
"""Accept a diagnostic teacher on the held-out BUSI source-test split.

This script is reporting-only. It must not be used for teacher checkpoint
selection, calibration fitting, threshold tuning, or early stopping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from v9_calibration import Gate, scores

VIEW_DIRECTORIES = {
    "source": "real",
    "U1": "fake_1",
    "U5": "fake_5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_weights", type=Path, required=True)
    parser.add_argument("--new_weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images_root", type=Path, required=True)
    parser.add_argument("--split", default="src_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--legacy_tolerance", type=float, default=0.10)
    return parser.parse_args()


def load_heldout(
    manifest: Path,
    images_root: Path,
    split: str,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    frame = pd.read_csv(manifest)
    required = {"key", "split", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{manifest} is missing required columns {missing}")
    frame = frame[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"split {split!r} is empty in {manifest}")
    if split.lower() in {"src_train", "train", "src_valid", "validation", "val"}:
        raise ValueError("v10 requires a held-out source-test split")
    if frame["key"].astype(str).duplicated().any():
        raise ValueError("held-out manifest contains duplicate case keys")
    labels = frame["label"].to_numpy(dtype=int)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("v10 accepts binary source labels only")
    paths: dict[str, list[str]] = {}
    for view, directory_name in VIEW_DIRECTORIES.items():
        directory = images_root / directory_name
        view_paths = [directory / f"{key}.png" for key in frame["key"].astype(str)]
        missing_paths = [str(path) for path in view_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} {view} images are missing; "
                f"first={missing_paths[0]}"
            )
        paths[view] = [str(path.resolve()) for path in view_paths]
    return labels, paths


def report_teacher(
    tag: str,
    checkpoint: Path,
    labels: np.ndarray,
    view_paths: dict[str, list[str]],
    device: torch.device,
    tolerance: float,
) -> None:
    network = Gate(checkpoint).to(device).eval()
    diagnostic_scores = {
        view: scores(network, paths, device) for view, paths in view_paths.items()
    }
    print(f"\n############## {tag} ##############")
    print("  rendering |  AUC   |  acc   | bal-acc | corr with source (P/S)")
    for view, score in diagnostic_scores.items():
        prediction = (score > 0).astype(int)
        correlation = "--"
        if view != "source":
            correlation = "%.3f / %.3f" % (
                pearsonr(diagnostic_scores["source"], score)[0],
                spearmanr(diagnostic_scores["source"], score).statistic,
            )
        print(
            "  %-9s | %.4f | %.4f | %.4f  | %s"
            % (
                view,
                roc_auc_score(labels, score),
                accuracy_score(labels, prediction),
                balanced_accuracy_score(labels, prediction),
                correlation,
            )
        )

    sign = 2 * labels - 1
    print("  -- legacy margin change and actual class-wise safety charge --")
    for view in ("U1", "U5"):
        margin_change = sign * (diagnostic_scores[view] - diagnostic_scores["source"])
        charge = np.maximum(-margin_change - tolerance, 0.0)
        print(
            f"  {view}: margin change benign "
            f"{margin_change[labels == 0].mean():+.3f}, malignant "
            f"{margin_change[labels == 1].mean():+.3f}; charge benign "
            f"{charge[labels == 0].mean():.3f}, malignant "
            f"{charge[labels == 1].mean():.3f}"
        )


def main() -> None:
    args = parse_args()
    labels, view_paths = load_heldout(
        args.manifest,
        args.images_root,
        args.split,
    )
    device = torch.device(
        args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    )
    print(
        f"held-out BUSI source test: split={args.split}, n={len(labels)}, "
        f"classes={dict(pd.Series(labels).value_counts())}"
    )
    report_teacher(
        "OLD teacher (raw-only)",
        args.old_weights,
        labels,
        view_paths,
        device,
        args.legacy_tolerance,
    )
    report_teacher(
        "NEW teacher (render-robust)",
        args.new_weights,
        labels,
        view_paths,
        device,
        args.legacy_tolerance,
    )
    print(
        "\nThe held-out table is an acceptance report only. "
        "It must not feed checkpoint or hyperparameter selection."
    )


if __name__ == "__main__":
    main()
