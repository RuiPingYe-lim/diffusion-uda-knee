#!/usr/bin/env python3
"""Build a strict BUSI(source A) -> BrEaST(target B) DOSC-UNSB dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

IMAGE_COLUMN_CANDIDATES = ("image_path", "path", "img_path", "filepath", "file")
LABEL_COLUMN_CANDIDATES = ("label", "target", "y", "class", "cls")
CASE_COLUMN_CANDIDATES = ("case_id", "patient_id", "id", "case", "study_id")


def detect_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Iterable[str],
    required: bool,
) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Column {requested!r} not found; available={list(frame.columns)}")
        return requested
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if required:
        raise ValueError(f"Could not detect required column; available={list(frame.columns)}")
    return None


def resolve_image_path(value: str, csv_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (csv_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def safe_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token[:80] or fallback


def load_source_records(
    csv_path: Path,
    split: str,
    image_column: str | None,
    label_column: str | None,
    case_column: str | None,
) -> list[dict[str, object]]:
    frame = pd.read_csv(csv_path)
    image_col = detect_column(frame, image_column, IMAGE_COLUMN_CANDIDATES, required=True)
    label_col = detect_column(frame, label_column, LABEL_COLUMN_CANDIDATES, required=True)
    case_col = detect_column(frame, case_column, CASE_COLUMN_CANDIDATES, required=False)
    records = []
    for index, row in frame.iterrows():
        label = int(row[label_col])
        if label not in (0, 1):
            raise ValueError(f"Expected binary source label at {csv_path}:{index + 2}, got {label}")
        case_id = str(row[case_col]) if case_col else f"{split}_{index:05d}"
        records.append(
            {
                "source_path": resolve_image_path(row[image_col], csv_path),
                "label": label,
                "case_id": case_id,
                "split": split,
            }
        )
    if not records:
        raise ValueError(f"Source CSV is empty: {csv_path}")
    return records


def load_target_records(
    csv_path: Path,
    split: str,
    image_column: str | None,
    case_column: str | None,
) -> list[dict[str, object]]:
    frame = pd.read_csv(csv_path)
    image_col = detect_column(frame, image_column, IMAGE_COLUMN_CANDIDATES, required=True)
    case_col = detect_column(frame, case_column, CASE_COLUMN_CANDIDATES, required=False)
    records = []
    for index, row in frame.iterrows():
        case_id = str(row[case_col]) if case_col else f"{split}_{index:05d}"
        records.append(
            {
                "source_path": resolve_image_path(row[image_col], csv_path),
                "case_id": case_id,
                "split": split,
            }
        )
    if not records:
        raise ValueError(f"Target CSV is empty: {csv_path}")
    return records


def assert_disjoint(records_a, records_b, name_a: str, name_b: str) -> None:
    cases_a = {str(record["case_id"]) for record in records_a}
    cases_b = {str(record["case_id"]) for record in records_b}
    overlap = sorted(cases_a & cases_b)
    if overlap:
        raise ValueError(
            f"{name_a}/{name_b} case overlap ({len(overlap)}): {overlap[:5]}"
        )


def place_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported placement mode: {mode}")


def prepare_output(root: Path, overwrite: bool) -> None:
    managed_names = (
        "trainA",
        "trainB",
        "testA",
        "testB",
        "trainA_manifest.csv",
        "testA_manifest.csv",
        "dataset_manifest.json",
    )
    existing = [root / name for name in managed_names if (root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Managed DOSC dataset files already exist under {root}; pass --overwrite"
        )
    if overwrite:
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)


def materialize_domain(
    records: list[dict[str, object]],
    root: Path,
    directory_name: str,
    prefix: str,
    mode: str,
    include_labels: bool,
) -> list[dict[str, object]]:
    manifest_rows = []
    destination_dir = root / directory_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        source_path = Path(record["source_path"])
        case_token = safe_token(str(record["case_id"]), fallback=f"{index:05d}")
        suffix = source_path.suffix.lower() or ".png"
        filename = f"{prefix}_{index:05d}_{case_token}{suffix}"
        destination = destination_dir / filename
        place_image(source_path, destination, mode)
        row = {
            "relative_path": destination.relative_to(root).as_posix(),
            "case_id": str(record["case_id"]),
            "source_split": str(record["split"]),
        }
        if include_labels:
            row["label"] = int(record["label"])
        manifest_rows.append(row)
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_train_csv", type=Path, required=True)
    parser.add_argument("--source_val_csv", type=Path, required=True)
    parser.add_argument("--target_train_csv", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--source_image_col")
    parser.add_argument("--source_label_col")
    parser.add_argument("--source_case_col")
    parser.add_argument("--target_image_col")
    parser.add_argument("--target_case_col")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_train = load_source_records(
        args.source_train_csv,
        "src_train",
        args.source_image_col,
        args.source_label_col,
        args.source_case_col,
    )
    source_val = load_source_records(
        args.source_val_csv,
        "src_valid",
        args.source_image_col,
        args.source_label_col,
        args.source_case_col,
    )
    target_train = load_target_records(
        args.target_train_csv,
        "tgt_train",
        args.target_image_col,
        args.target_case_col,
    )
    assert_disjoint(source_train, source_val, "source train", "source validation")

    output_root = args.out_root.resolve()
    prepare_output(output_root, overwrite=args.overwrite)
    train_a_manifest = materialize_domain(
        source_train,
        output_root,
        "trainA",
        "source",
        args.mode,
        include_labels=True,
    )
    materialize_domain(
        target_train,
        output_root,
        "trainB",
        "target",
        args.mode,
        include_labels=False,
    )
    test_a_manifest = materialize_domain(
        source_train + source_val,
        output_root,
        "testA",
        "source_eval",
        args.mode,
        include_labels=True,
    )
    materialize_domain(
        target_train,
        output_root,
        "testB",
        "target_ref",
        args.mode,
        include_labels=False,
    )

    pd.DataFrame(train_a_manifest).to_csv(
        output_root / "trainA_manifest.csv",
        index=False,
    )
    pd.DataFrame(test_a_manifest).to_csv(
        output_root / "testA_manifest.csv",
        index=False,
    )
    summary = {
        "direction": "AtoB",
        "source_domain": "BUSI",
        "target_domain": "BrEaST",
        "target_labels_used": False,
        "placement_mode": args.mode,
        "counts": {
            "trainA_source": len(source_train),
            "trainB_target": len(target_train),
            "testA_source_train_and_val": len(source_train) + len(source_val),
            "testB_target_train_references": len(target_train),
        },
        "inputs": {
            "source_train_csv": str(args.source_train_csv.resolve()),
            "source_val_csv": str(args.source_val_csv.resolve()),
            "target_train_csv": str(args.target_train_csv.resolve()),
        },
    }
    with (output_root / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
