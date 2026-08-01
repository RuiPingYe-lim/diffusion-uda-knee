#!/usr/bin/env python3
"""Measure how much translated-teacher degradation calibration can recover.

Fit calibration only on a source validation split and evaluate it on a
case-disjoint held-out source test split. The test split must not be used to
train the source or render-robust teacher, fit calibration, select a checkpoint,
or tune any threshold.

The direct manifest schema contains ``case_id``, ``split``, ``label``, and one
path column per rendering (by default ``raw,U1,U5``). For the held-out UNSB
result schema, pass ``--test_images_root`` with a manifest containing ``key``
and ``label``; paths are then resolved as ``real/``, ``fake_1/``, and
``fake_5/`` under that root.

Only source labels are read. No target-domain label is accepted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torchvision import models
from torchvision import transforms as T

DEFAULT_RENDERINGS = ("raw", "U1", "U5")
RESULT_DIRECTORIES = {
    "raw": "real",
    "U1": "fake_1",
    "U5": "fake_5",
}

CLF_TF = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Lambda(
            lambda tensor: tensor.repeat(3, 1, 1) if tensor.shape[0] == 1 else tensor
        ),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


class SpatialAttention(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dimensions, dimensions, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention = self.conv(features)
        batch, channels, height, width = attention.shape
        attention = self.softmax(attention.view(batch, channels, -1))
        maximum = attention.amax(2, keepdim=True).clamp_min(1e-6)
        return features * (attention / maximum).view(batch, channels, height, width)


class Gate(nn.Module):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.space_attn = SpatialAttention(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 2)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["state_dict"]
        state = {key.replace("module.", ""): value for key, value in state.items()}
        modules = {
            "stem.": self.stem,
            "space_attn.": self.space_attn,
            "classifier.": self.classifier,
        }
        for prefix, module in modules.items():
            module.load_state_dict(
                {
                    key[len(prefix) :]: value
                    for key, value in state.items()
                    if key.startswith(prefix)
                },
                strict=False,
            )

    @torch.no_grad()
    def score(self, images: torch.Tensor) -> np.ndarray:
        features = self.stem(images)
        features = self.space_attn(features)
        logits = self.classifier(self.avgpool(features).flatten(1))
        return (logits[:, 1] - logits[:, 0]).cpu().numpy()


def parse_renderings(value: str) -> tuple[str, ...]:
    renderings = tuple(item.strip() for item in value.split(",") if item.strip())
    if "raw" not in renderings or len(renderings) < 2:
        raise ValueError("renderings must contain raw and at least one translation")
    return renderings


def _resolve_manifest_path(value: object, manifest: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rendering image not found: {path}")
    return str(path)


def validate_source_frame(frame: pd.DataFrame, name: str) -> None:
    labels = set(frame["label"].astype(int).unique())
    if labels - {0, 1}:
        raise ValueError(f"{name} contains non-binary source labels: {sorted(labels)}")
    duplicated = frame["case_id"].astype(str).duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, "case_id"].astype(str).head(5).tolist()
        raise ValueError(f"{name} contains duplicate case_id values: {examples}")


def load_direct_manifest(
    manifest: Path,
    split: str,
    renderings: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    required = {"case_id", "split", "label", *renderings}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{manifest} is missing required columns {missing}")
    frame = frame[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"split {split!r} is empty in {manifest}")
    frame["case_id"] = frame["case_id"].astype(str)
    for rendering in renderings:
        frame[rendering] = [
            _resolve_manifest_path(value, manifest) for value in frame[rendering]
        ]
    validate_source_frame(frame, f"{manifest}:{split}")
    return frame.reset_index(drop=True)


def load_result_manifest(
    manifest: Path,
    split: str,
    images_root: Path,
    renderings: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    required = {"key", "split", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{manifest} is missing required columns {missing}")
    frame = frame[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"split {split!r} is empty in {manifest}")
    frame["case_id"] = frame["key"].astype(str)
    for rendering in renderings:
        if rendering not in RESULT_DIRECTORIES:
            raise ValueError(
                f"no UNSB result-directory mapping is defined for {rendering!r}"
            )
        directory = images_root / RESULT_DIRECTORIES[rendering]
        paths = [directory / f"{key}.png" for key in frame["key"]]
        missing_paths = [str(path) for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} {rendering} images are missing; "
                f"first={missing_paths[0]}"
            )
        frame[rendering] = [str(path.resolve()) for path in paths]
    validate_source_frame(frame, f"{manifest}:{split}")
    return frame.reset_index(drop=True)


def assert_heldout_protocol(
    fit: pd.DataFrame,
    test: pd.DataFrame,
    test_split: str,
    allow_nonheldout_test: bool,
) -> None:
    overlap = sorted(set(fit["case_id"]) & set(test["case_id"]))
    if overlap:
        raise ValueError(
            f"calibration-fit/test case overlap ({len(overlap)}): {overlap[:5]}"
        )
    if not allow_nonheldout_test and test_split.lower() in {
        "src_train",
        "train",
        "src_valid",
        "validation",
        "val",
    }:
        raise ValueError(
            "calibration evaluation must use held-out source test cases; "
            "pass --allow_nonheldout_test only for an explicitly non-final diagnostic"
        )


def scores(
    network: Gate,
    paths: list[str],
    device: torch.device,
) -> np.ndarray:
    output = []
    for start in range(0, len(paths), 32):
        images = torch.stack(
            [
                CLF_TF(Image.open(path).convert("L"))
                for path in paths[start : start + 32]
            ]
        ).to(device)
        output.append(network.score(images))
    return np.concatenate(output)


def expected_calibration_error(
    probability: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (probability > edges[index]) & (probability <= edges[index + 1])
        if selected.sum():
            error += selected.mean() * abs(
                labels[selected].mean() - probability[selected].mean()
            )
    return float(error)


def metrics(score: np.ndarray, labels: np.ndarray, tag: str) -> dict[str, float | str]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -60.0, 60.0)))
    prediction = (score > 0).astype(int)
    unique_scores = np.unique(score)
    if unique_scores.size == 1:
        thresholds = np.array(
            [
                np.nextafter(unique_scores[0], -np.inf),
                np.nextafter(unique_scores[0], np.inf),
            ]
        )
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        thresholds = np.concatenate(
            [
                [np.nextafter(unique_scores[0], -np.inf)],
                midpoints,
                [np.nextafter(unique_scores[-1], np.inf)],
            ]
        )
    threshold_results = [
        (accuracy_score(labels, score > threshold), threshold)
        for threshold in thresholds
    ]
    best_accuracy, best_threshold = max(threshold_results)
    return {
        "tag": tag,
        "auc": float(roc_auc_score(labels, score)),
        "acc": float(accuracy_score(labels, prediction)),
        "bacc": float(balanced_accuracy_score(labels, prediction)),
        "oracle_acc": float(best_accuracy),
        "opt_thr": float(best_threshold),
        "ece": expected_calibration_error(probability, labels),
        "brier": float(brier_score_loss(labels, probability)),
        "nll": float(
            log_loss(
                labels,
                np.clip(probability, 1e-6, 1.0 - 1e-6),
                labels=[0, 1],
            )
        ),
    }


def show(result: dict[str, float | str]) -> None:
    print(
        "  %-34s AUC %.4f | acc %.4f | bal-acc %.4f | "
        "oracle-thr acc %.4f (thr %+.2f) | ECE %.3f Brier %.3f NLL %.3f"
        % (
            result["tag"],
            result["auc"],
            result["acc"],
            result["bacc"],
            result["oracle_acc"],
            result["opt_thr"],
            result["ece"],
            result["brier"],
            result["nll"],
        )
    )


def fit_to_source(
    translated_score: np.ndarray,
    source_score: np.ndarray,
    mode: str,
) -> tuple[float, float]:
    """Fit ``source ~= a * translated + b`` in the requested family."""

    if mode == "bias":
        return 1.0, float(np.mean(source_score - translated_score))
    if mode == "temperature":

        def objective(scale: float) -> float:
            return float(np.mean((scale * translated_score - source_score) ** 2))

        result = minimize_scalar(
            objective,
            bounds=(1e-3, 100.0),
            method="bounded",
        )
        return float(result.x), 0.0
    design = np.stack(
        [translated_score, np.ones_like(translated_score)],
        axis=1,
    )
    solution, *_ = np.linalg.lstsq(design, source_score, rcond=None)
    scale = float(max(solution[0], 1e-3))
    bias = float(np.mean(source_score - scale * translated_score))
    return scale, bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fit_manifest", type=Path, required=True)
    parser.add_argument("--fit_split", default="src_valid")
    parser.add_argument("--test_manifest", type=Path, required=True)
    parser.add_argument("--test_split", default="src_test")
    parser.add_argument(
        "--test_images_root",
        type=Path,
        default=None,
        help="UNSB result images root for a key-based held-out manifest",
    )
    parser.add_argument("--renderings", default="raw,U1,U5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_nonheldout_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderings = parse_renderings(args.renderings)
    fit = load_direct_manifest(
        args.fit_manifest,
        args.fit_split,
        renderings,
    )
    if args.test_images_root is None:
        test = load_direct_manifest(
            args.test_manifest,
            args.test_split,
            renderings,
        )
    else:
        test = load_result_manifest(
            args.test_manifest,
            args.test_split,
            args.test_images_root,
            renderings,
        )
    assert_heldout_protocol(
        fit,
        test,
        args.test_split,
        args.allow_nonheldout_test,
    )
    device = torch.device(
        args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    )
    network = Gate(args.weights).to(device).eval()
    print(
        f"fit calibration on {args.fit_split} (n={len(fit)}); "
        f"evaluate once on {args.test_split} (n={len(test)})"
    )

    score_sets: dict[str, dict[str, np.ndarray]] = {}
    for part_name, frame in (("fit", fit), ("test", test)):
        score_sets[part_name] = {
            "label": frame["label"].to_numpy(dtype=int),
        }
        for rendering in renderings:
            score_sets[part_name][rendering] = scores(
                network,
                frame[rendering].tolist(),
                device,
            )
        print(f"scored {part_name}", flush=True)

    test_labels = score_sets["test"]["label"]
    print("\n############## RAW REFERENCE ##############")
    show(metrics(score_sets["test"]["raw"], test_labels, "raw"))

    for rendering in renderings:
        if rendering == "raw":
            continue
        print(f"\n############## {rendering} ##############")
        show(
            metrics(
                score_sets["test"][rendering],
                test_labels,
                f"{rendering} uncalibrated",
            )
        )
        modes = (
            ("bias", "bias-only d+b"),
            ("temperature", "temperature a*d"),
            ("affine", "affine a*d+b"),
        )
        for mode, name in modes:
            scale, bias = fit_to_source(
                score_sets["fit"][rendering],
                score_sets["fit"]["raw"],
                mode,
            )
            calibrated = scale * score_sets["test"][rendering] + bias
            show(
                metrics(
                    calibrated,
                    test_labels,
                    f"{name} [to-source] (a={scale:.2f} b={bias:+.2f})",
                )
            )
        platt = LogisticRegression(max_iter=2000).fit(
            score_sets["fit"][rendering].reshape(-1, 1),
            score_sets["fit"]["label"],
        )
        scale = float(platt.coef_[0][0])
        bias = float(platt.intercept_[0])
        show(
            metrics(
                scale * score_sets["test"][rendering] + bias,
                test_labels,
                f"Platt [to-labels] (a={scale:.2f} b={bias:+.2f})",
            )
        )
        pearson = pearsonr(
            score_sets["test"]["raw"],
            score_sets["test"][rendering],
        )[0]
        spearman = spearmanr(
            score_sets["test"]["raw"],
            score_sets["test"][rendering],
        ).statistic
        print(
            "  per-case agreement with raw: "
            f"Pearson {pearson:.3f} | Spearman {spearman:.3f}"
        )

    print("\n############## LEGACY MARGIN ASYMMETRY ##############")
    sign = 2 * test_labels - 1
    for rendering in renderings:
        if rendering == "raw":
            continue
        margin_change = sign * (
            score_sets["test"][rendering] - score_sets["test"]["raw"]
        )
        offset = np.mean(score_sets["test"][rendering] - score_sets["test"]["raw"])
        print(
            f"  {rendering}: benign {margin_change[test_labels == 0].mean():+.3f} | "
            f"malignant {margin_change[test_labels == 1].mean():+.3f} | "
            f"offset {offset:+.3f}"
        )


if __name__ == "__main__":
    main()
