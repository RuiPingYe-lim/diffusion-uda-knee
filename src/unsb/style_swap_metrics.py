"""Statistics for the DOSC target-reference style-swap audit.

The audit holds a source case fixed, changes only the unlabeled target
reference, and uses common bridge noise across references. A separate
fixed-reference arm changes only bridge noise. This module contains the pure
validation and summary functions so the causal protocol can be unit-tested
without an upstream UNSB checkout.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

SWAP_COLUMNS = {
    "case_id",
    "label",
    "reference_id",
    "reference_probe_score",
    "source_score",
    "output_score",
}
NOISE_COLUMNS = {
    "case_id",
    "label",
    "replicate_id",
    "source_score",
    "output_score",
}


def make_path_noises(
    source_image: torch.Tensor,
    num_timesteps: int,
    batch_size: int,
    seed: int,
    common_across_batch: bool,
) -> tuple[torch.Tensor, ...]:
    """Create reproducible bridge-transition noise tensors.

    Common noise has batch size one and is broadcast by
    ``DoscSBModel.translate_with_condition``. Independent noise has one draw
    per style/reference replicate.
    """

    if source_image.ndim != 4 or source_image.shape[0] != 1:
        raise ValueError("source_image must have shape [1,C,H,W]")
    if int(num_timesteps) < 1:
        raise ValueError("num_timesteps must be positive")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device=source_image.device)
    generator.manual_seed(int(seed))
    leading = 1 if common_across_batch else int(batch_size)
    shape = (leading, *source_image.shape[1:])
    return tuple(
        torch.randn(
            shape,
            dtype=source_image.dtype,
            device=source_image.device,
            generator=generator,
        )
        for _ in range(int(num_timesteps) - 1)
    )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns {missing}")
    if frame.empty:
        raise ValueError(f"{name} is empty")


def _validate_case_values(frame: pd.DataFrame, name: str) -> None:
    for column in ("label", "source_score"):
        counts = frame.groupby("case_id")[column].nunique(dropna=False)
        invalid = counts[counts != 1]
        if not invalid.empty:
            raise ValueError(
                f"{name} has inconsistent {column} values for cases "
                f"{invalid.index.astype(str).tolist()[:5]}"
            )
    labels = set(frame["label"].astype(int).unique())
    if labels - {0, 1}:
        raise ValueError(f"{name} contains non-binary labels {sorted(labels)}")


def validate_complete_swap_grid(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Require exactly one row for every source-case/reference pair."""

    _require_columns(frame, SWAP_COLUMNS, "style-swap frame")
    duplicate = frame.duplicated(["case_id", "reference_id"])
    if duplicate.any():
        examples = (
            frame.loc[duplicate, ["case_id", "reference_id"]]
            .astype(str)
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"style-swap frame contains duplicate pairs: {examples}")
    _validate_case_values(frame, "style-swap frame")
    case_ids = sorted(frame["case_id"].astype(str).unique())
    reference_ids = sorted(frame["reference_id"].astype(str).unique())
    expected = len(case_ids) * len(reference_ids)
    if len(frame) != expected:
        counts = frame.groupby("case_id")["reference_id"].nunique()
        raise ValueError(
            "style-swap frame is not a complete case/reference grid; "
            f"expected {expected} rows, got {len(frame)}, "
            f"reference counts={counts.value_counts().to_dict()}"
        )
    per_reference_score = frame.groupby("reference_id")[
        "reference_probe_score"
    ].nunique(dropna=False)
    if (per_reference_score != 1).any():
        raise ValueError("reference_probe_score must be fixed per reference_id")
    return case_ids, reference_ids


def validate_noise_grid(frame: pd.DataFrame, expected_cases: Iterable[str]) -> None:
    """Require a balanced fixed-reference path-noise control."""

    _require_columns(frame, NOISE_COLUMNS, "path-noise frame")
    duplicate = frame.duplicated(["case_id", "replicate_id"])
    if duplicate.any():
        raise ValueError("path-noise frame contains duplicate case/replicate pairs")
    _validate_case_values(frame, "path-noise frame")
    actual_cases = set(frame["case_id"].astype(str))
    expected = set(str(value) for value in expected_cases)
    if actual_cases != expected:
        raise ValueError(
            "style-swap and path-noise frames use different cases: "
            f"missing={sorted(expected - actual_cases)[:5]}, "
            f"extra={sorted(actual_cases - expected)[:5]}"
        )
    counts = frame.groupby("case_id")["replicate_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError(
            "path-noise frame has unequal replicate counts: "
            f"{counts.value_counts().to_dict()}"
        )


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(rankdata(x), rankdata(y))


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or int(draws) <= 0:
        value = float(values.mean())
        return value, value
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    samples = values[indices].mean(axis=1)
    return (
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    )


def _reference_permutation_pvalue(
    reference_scores: np.ndarray,
    reference_effects: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    draws: int,
) -> float:
    if not np.isfinite(observed) or int(draws) <= 0:
        return float("nan")
    extreme = 0
    draws = int(draws)
    for _ in range(draws):
        permuted = rng.permutation(reference_scores)
        candidate = _pearson(permuted, reference_effects)
        if np.isfinite(candidate) and abs(candidate) >= abs(observed) - 1e-12:
            extreme += 1
    return float((extreme + 1) / (draws + 1))


def _per_case_variability(
    frame: pd.DataFrame,
    replicate_column: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for case_id, group in frame.groupby("case_id", sort=True):
        group = group.sort_values(replicate_column)
        output = group["output_score"].to_numpy(dtype=float)
        source_score = float(group["source_score"].iloc[0])
        prediction = output > float(threshold)
        source_prediction = source_score > float(threshold)
        rows.append(
            {
                "case_id": str(case_id),
                "label": int(group["label"].iloc[0]),
                "score_std": float(np.std(output, ddof=0)),
                "score_range": float(np.ptp(output)),
                "internal_prediction_flip": float(np.unique(prediction).size > 1),
                "any_source_prediction_flip": float(
                    np.any(prediction != source_prediction)
                ),
                "mean_output_score": float(output.mean()),
            }
        )
    return pd.DataFrame(rows)


def _case_slopes(frame: pd.DataFrame) -> np.ndarray:
    slopes = []
    for _, group in frame.groupby("case_id", sort=True):
        x = group["reference_probe_score"].to_numpy(dtype=float)
        y = group["output_score"].to_numpy(dtype=float) - group[
            "source_score"
        ].to_numpy(dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denominator = float(np.dot(x, x))
        slopes.append(
            float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")
        )
    return np.asarray(slopes, dtype=float)


def summarize_style_swap(
    swap: pd.DataFrame,
    path_noise: pd.DataFrame,
    threshold: float = 0.0,
    bootstrap_draws: int = 2000,
    permutation_draws: int = 2000,
    seed: int = 7,
) -> dict[str, object]:
    """Summarize one bridge depth of the causal style-swap experiment."""

    case_ids, reference_ids = validate_complete_swap_grid(swap)
    validate_noise_grid(path_noise, case_ids)
    rng = np.random.default_rng(int(seed))

    swap_case = _per_case_variability(swap, "reference_id", threshold)
    noise_case = _per_case_variability(path_noise, "replicate_id", threshold)
    paired = swap_case.merge(
        noise_case,
        on=["case_id", "label"],
        suffixes=("_style", "_noise"),
        validate="one_to_one",
    )
    slopes = _case_slopes(swap)
    slope_ci = _bootstrap_mean_ci(slopes, rng, bootstrap_draws)
    std_difference = (
        paired["score_std_style"].to_numpy() - paired["score_std_noise"].to_numpy()
    )
    std_difference_ci = _bootstrap_mean_ci(std_difference, rng, bootstrap_draws)

    reference_table = (
        swap.assign(
            output_delta=swap["output_score"].astype(float)
            - swap["source_score"].astype(float)
        )
        .groupby("reference_id", as_index=False)
        .agg(
            reference_probe_score=("reference_probe_score", "first"),
            mean_output_delta=("output_delta", "mean"),
        )
        .sort_values("reference_id")
    )
    reference_scores = reference_table["reference_probe_score"].to_numpy(dtype=float)
    reference_effects = reference_table["mean_output_delta"].to_numpy(dtype=float)
    reference_pearson = _pearson(reference_scores, reference_effects)
    reference_spearman = _spearman(reference_scores, reference_effects)
    permutation_p = _reference_permutation_pvalue(
        reference_scores,
        reference_effects,
        reference_pearson,
        rng,
        permutation_draws,
    )

    source_cases = (
        swap[["case_id", "label", "source_score"]]
        .drop_duplicates("case_id")
        .sort_values("case_id")
    )
    mean_outputs = (
        swap.groupby("case_id", as_index=False)["output_score"]
        .mean()
        .rename(columns={"output_score": "mean_output_score"})
    )
    aggregate = source_cases.merge(
        mean_outputs,
        on="case_id",
        validate="one_to_one",
    )
    per_reference_auc = {}
    for reference_id, group in swap.groupby("reference_id", sort=True):
        per_reference_auc[str(reference_id)] = _safe_auc(
            group["label"].to_numpy(dtype=int),
            group["output_score"].to_numpy(dtype=float),
        )
    finite_reference_auc = np.asarray(
        [value for value in per_reference_auc.values() if np.isfinite(value)],
        dtype=float,
    )

    style_std = swap_case["score_std"].to_numpy(dtype=float)
    noise_std = noise_case["score_std"].to_numpy(dtype=float)
    finite_slopes = slopes[np.isfinite(slopes)]
    summary: dict[str, object] = {
        "n_cases": len(case_ids),
        "n_references": len(reference_ids),
        "n_noise_replicates": int(
            path_noise.groupby("case_id")["replicate_id"].nunique().iloc[0]
        ),
        "reference_probe_score_std": float(np.std(reference_scores, ddof=0)),
        "source_auc": _safe_auc(
            aggregate["label"].to_numpy(dtype=int),
            aggregate["source_score"].to_numpy(dtype=float),
        ),
        "mean_over_references_auc": _safe_auc(
            aggregate["label"].to_numpy(dtype=int),
            aggregate["mean_output_score"].to_numpy(dtype=float),
        ),
        "per_reference_auc": per_reference_auc,
        "per_reference_auc_mean": (
            float(finite_reference_auc.mean())
            if finite_reference_auc.size
            else float("nan")
        ),
        "per_reference_auc_min": (
            float(finite_reference_auc.min())
            if finite_reference_auc.size
            else float("nan")
        ),
        "style_score_std_mean": float(style_std.mean()),
        "path_noise_score_std_mean": float(noise_std.mean()),
        "style_to_noise_std_ratio": float(
            style_std.mean() / max(noise_std.mean(), 1e-12)
        ),
        "style_minus_noise_std_mean": float(std_difference.mean()),
        "style_minus_noise_std_ci95": list(std_difference_ci),
        "style_score_range_mean": float(swap_case["score_range"].mean()),
        "path_noise_score_range_mean": float(noise_case["score_range"].mean()),
        "style_internal_prediction_flip_rate": float(
            swap_case["internal_prediction_flip"].mean()
        ),
        "path_noise_internal_prediction_flip_rate": float(
            noise_case["internal_prediction_flip"].mean()
        ),
        "style_any_source_prediction_flip_rate": float(
            swap_case["any_source_prediction_flip"].mean()
        ),
        "path_noise_any_source_prediction_flip_rate": float(
            noise_case["any_source_prediction_flip"].mean()
        ),
        "case_slope_mean": (
            float(finite_slopes.mean()) if finite_slopes.size else float("nan")
        ),
        "case_slope_ci95": list(slope_ci),
        "reference_effect_pearson": reference_pearson,
        "reference_effect_spearman": reference_spearman,
        "reference_permutation_pvalue": permutation_p,
    }

    if "output_reference_cosine" in swap.columns:
        summary["output_reference_cosine_mean"] = float(
            swap["output_reference_cosine"].mean()
        )
    if "output_style_distance_from_source" in swap.columns:
        summary["output_style_distance_from_source_mean"] = float(
            swap["output_style_distance_from_source"].mean()
        )

    class_summary = {}
    for label in (0, 1):
        style_class = swap_case[swap_case["label"] == label]
        noise_class = noise_case[noise_case["label"] == label]
        class_summary[str(label)] = {
            "n": int(len(style_class)),
            "style_score_std_mean": float(style_class["score_std"].mean()),
            "path_noise_score_std_mean": float(noise_class["score_std"].mean()),
            "style_score_range_mean": float(style_class["score_range"].mean()),
            "style_internal_prediction_flip_rate": float(
                style_class["internal_prediction_flip"].mean()
            ),
            "style_any_source_prediction_flip_rate": float(
                style_class["any_source_prediction_flip"].mean()
            ),
        }
    summary["by_class"] = class_summary
    summary["reference_effects"] = reference_table.to_dict("records")
    return summary
