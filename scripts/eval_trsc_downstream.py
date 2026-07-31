#!/usr/bin/env python3
"""Unlock and compare a complete TRSC downstream run matrix.

Training runs select their epoch only from source validation and store target
probabilities without labels. This reporting script refuses partial matrices,
joins locked target labels once, and reports seed-paired, case-paired AUC
contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in parse_csv(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def parse_contrast(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("contrast must have the form LEFT:RIGHT")
    left, right = (item.strip() for item in value.split(":", 1))
    if not left or not right or left == right:
        raise argparse.ArgumentTypeError("contrast arms must be distinct and non-empty")
    return left, right


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--sealed_dir", type=Path, required=True)
    parser.add_argument("--target_labels", type=Path, required=True)
    parser.add_argument("--target_case_col", default="case_id")
    parser.add_argument("--target_label_col", default="label")
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--seeds", default="7,16,42")
    parser.add_argument("--baseline", default="raw")
    parser.add_argument(
        "--contrast",
        action="append",
        type=parse_contrast,
        default=[],
        metavar="LEFT:RIGHT",
    )
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def load_labels(path: Path, case_col: str, label_col: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    header = pd.read_csv(path, nrows=0)
    missing = sorted({case_col, label_col} - set(header.columns))
    if missing:
        raise ValueError(f"target label manifest is missing columns: {missing}")
    labels = pd.read_csv(path, usecols=[case_col, label_col]).rename(
        columns={case_col: "case_id", label_col: "label"}
    )
    labels["case_id"] = labels["case_id"].astype(str)
    labels["label"] = labels["label"].astype(int)
    if labels["case_id"].duplicated().any():
        raise ValueError("target label manifest contains duplicate case identifiers")
    if set(labels["label"].unique()) != {0, 1}:
        raise ValueError("target label manifest must contain both binary classes")
    return labels.sort_values("case_id").reset_index(drop=True)


def load_run(
    runs_root: Path,
    sealed_dir: Path,
    condition: str,
    seed: int,
    labels: pd.DataFrame,
) -> tuple[dict[str, object], np.ndarray]:
    name = f"{condition}_s{seed}"
    run_dir = runs_root / name
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.csv"
    sealed_path = sealed_dir / f"{name}_target_percase.csv"
    missing = [
        str(path)
        for path in (config_path, history_path, sealed_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"incomplete run {name}; missing={missing}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(config.get("target_labels_used", True)):
        raise ValueError(f"run {name} does not attest target_labels_used=false")
    if str(config.get("arm")) != condition or int(config.get("seed")) != seed:
        raise ValueError(f"run identity mismatch in {config_path}")
    selected_epoch = int(config["selected_epoch"])
    history = pd.read_csv(history_path)
    if "target_auc_SEALED" in history.columns:
        raise ValueError(
            f"run {name} contains target AUC during training and violates the locked protocol"
        )
    if selected_epoch not in set(history["epoch"].astype(int)):
        raise ValueError(f"selected epoch {selected_epoch} is absent from {history_path}")

    predictions = pd.read_csv(sealed_path)
    if "label" in predictions.columns:
        raise ValueError(f"sealed predictions for {name} must not contain target labels")
    required = {"epoch", "case_id", "prob"}
    if required - set(predictions.columns):
        missing_columns = required - set(predictions.columns)
        raise ValueError(
            f"sealed predictions for {name} are missing {missing_columns}"
        )
    predictions = predictions[predictions["epoch"].astype(int) == selected_epoch].copy()
    predictions["case_id"] = predictions["case_id"].astype(str)
    if predictions["case_id"].duplicated().any():
        raise ValueError(f"selected target predictions contain duplicate cases for {name}")
    merged = labels.merge(
        predictions[["case_id", "prob"]],
        on="case_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(merged["_merge"]) != {"both"}:
        missing_labels = (
            merged.loc[merged["_merge"] != "both", "case_id"].head(5).tolist()
        )
        raise ValueError(f"target case mismatch for {name}: {missing_labels}")
    merged = merged.sort_values("case_id").reset_index(drop=True)
    probability = merged["prob"].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError(f"non-finite target probabilities in {name}")
    auc = float(roc_auc_score(merged["label"], probability))
    record = {
        "condition": condition,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "target_auc": auc,
        "source_select_score": float(config["selected_score"]),
        "config_sha256": sha256_file(config_path),
        "history_sha256": sha256_file(history_path),
        "sealed_sha256": sha256_file(sealed_path),
    }
    return record, probability


def paired_bootstrap(
    labels: np.ndarray,
    left: dict[int, np.ndarray],
    right: dict[int, np.ndarray],
    seeds: tuple[int, ...],
    draws: int,
    seed: int,
) -> dict[str, float | list[float]]:
    point_by_seed = [
        float(roc_auc_score(labels, left[item]) - roc_auc_score(labels, right[item]))
        for item in seeds
    ]
    point = float(np.mean(point_by_seed))
    if draws == 0:
        return {
            "delta_auc": point,
            "ci95": [float("nan"), float("nan")],
            "probability_delta_gt_zero": float("nan"),
        }

    rng = np.random.default_rng(seed)
    bootstrapped = []
    case_count = len(labels)
    for _ in range(draws):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        case_indices = rng.integers(0, case_count, size=case_count)
        sampled_labels = labels[case_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        deltas = [
            roc_auc_score(sampled_labels, left[int(item)][case_indices])
            - roc_auc_score(sampled_labels, right[int(item)][case_indices])
            for item in sampled_seeds
        ]
        bootstrapped.append(float(np.mean(deltas)))
    if not bootstrapped:
        raise RuntimeError("paired bootstrap produced no valid binary resamples")
    values = np.asarray(bootstrapped)
    return {
        "delta_auc": point,
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def auc_pair_contribution(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    positive = probability[labels == 1][:, None]
    negative = probability[labels == 0][None, :]
    return (positive > negative).astype(float) + 0.5 * (positive == negative)


def pairwise_diagnostics(
    labels: np.ndarray,
    left: dict[int, np.ndarray],
    right: dict[int, np.ndarray],
    seeds: tuple[int, ...],
) -> dict[str, float | int]:
    improved = degraded = unchanged = 0
    class_shifts = {0: [], 1: []}
    for seed in seeds:
        left_pairs = auc_pair_contribution(labels, left[seed])
        right_pairs = auc_pair_contribution(labels, right[seed])
        difference = left_pairs - right_pairs
        improved += int((difference > 0).sum())
        degraded += int((difference < 0).sum())
        unchanged += int((difference == 0).sum())
        for label in (0, 1):
            mask = labels == label
            class_shifts[label].append(
                float(np.mean(left[seed][mask] - right[seed][mask]))
            )
    return {
        "left_improved_pairs": improved,
        "left_degraded_pairs": degraded,
        "unchanged_pairs": unchanged,
        "left_minus_right_mean_score_class0": float(
            np.mean(class_shifts[0])
        ),
        "left_minus_right_mean_score_class1": float(
            np.mean(class_shifts[1])
        ),
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    conditions = parse_csv(args.conditions)
    seeds = parse_seeds(args.seeds)
    invalid_conditions = [
        name for name in conditions if not VALID_NAME.fullmatch(name)
    ]
    if invalid_conditions:
        raise ValueError(f"invalid condition names: {invalid_conditions}")
    if args.baseline not in conditions:
        raise ValueError("baseline must be included in conditions")
    if args.bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be non-negative")

    labels_path = args.target_labels.expanduser().resolve()
    labels = load_labels(
        labels_path,
        args.target_case_col,
        args.target_label_col,
    )
    label_array = labels["label"].to_numpy(dtype=int)
    runs_root = args.runs_root.expanduser().resolve()
    sealed_dir = args.sealed_dir.expanduser().resolve()
    records = []
    per_case_rows = []
    probabilities: dict[str, dict[int, np.ndarray]] = {
        condition: {} for condition in conditions
    }
    for condition in conditions:
        for seed in seeds:
            record, probability = load_run(
                runs_root,
                sealed_dir,
                condition,
                seed,
                labels,
            )
            records.append(record)
            probabilities[condition][seed] = probability
            for case_id, label, item in zip(
                labels["case_id"],
                label_array,
                probability,
            ):
                per_case_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "case_id": case_id,
                        "label": int(label),
                        "prob": float(item),
                    }
                )

    contrasts = list(args.contrast)
    if not contrasts:
        contrasts = [
            (condition, args.baseline)
            for condition in conditions
            if condition != args.baseline
        ]
    contrast_rows = []
    for index, (left_name, right_name) in enumerate(contrasts):
        if left_name not in probabilities or right_name not in probabilities:
            raise ValueError(f"unknown contrast {left_name}:{right_name}")
        result = paired_bootstrap(
            label_array,
            probabilities[left_name],
            probabilities[right_name],
            seeds,
            args.bootstrap_draws,
            args.bootstrap_seed + index,
        )
        diagnostics = pairwise_diagnostics(
            label_array,
            probabilities[left_name],
            probabilities[right_name],
            seeds,
        )
        contrast_rows.append(
            {
                "left": left_name,
                "right": right_name,
                "delta_auc": result["delta_auc"],
                "ci95_low": result["ci95"][0],
                "ci95_high": result["ci95"][1],
                "probability_delta_gt_zero": result[
                    "probability_delta_gt_zero"
                ],
                **diagnostics,
            }
        )

    per_run = pd.DataFrame(records)
    per_case = pd.DataFrame(per_case_rows)
    summary_rows = []
    for condition, group in per_run.groupby("condition", sort=False):
        summary_rows.append(
            {
                "condition": condition,
                "mean_target_auc": float(group["target_auc"].mean()),
                "std_target_auc": float(group["target_auc"].std(ddof=0)),
                "per_seed_auc": {
                    str(int(row.seed)): float(row.target_auc)
                    for row in group.itertuples(index=False)
                },
            }
        )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    contrast_frame = pd.DataFrame(contrast_rows)
    per_run.to_csv(out_dir / "downstream_per_run.csv", index=False)
    per_case.to_csv(out_dir / "downstream_per_case.csv", index=False)
    contrast_frame.to_csv(out_dir / "downstream_contrasts.csv", index=False)
    report = {
        "protocol": {
            "complete_matrix_required": True,
            "conditions": list(conditions),
            "seeds": list(seeds),
            "target_labels_used_during_training": False,
            "target_labels_used_for_final_reporting": True,
            "target_cases": len(labels),
            "bootstrap_draws": args.bootstrap_draws,
        },
        "hashes": {
            "target_labels": sha256_file(labels_path),
        },
        "summary": summary_rows,
        "contrasts": contrast_rows,
    }
    (out_dir / "downstream_summary.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(per_run[["condition", "seed", "selected_epoch", "target_auc"]].to_string(index=False))
    print("\nPaired target-AUC contrasts")
    print(contrast_frame.to_string(index=False))
    print(f"\nSaved report to {out_dir}")


if __name__ == "__main__":
    main()
