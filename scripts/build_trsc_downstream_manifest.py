#!/usr/bin/env python3
"""Build a matched classifier manifest from target-reference UNSB variants.

The source cases and labels come only from ``testA_manifest.csv``. Each
``--variant NAME=PATH`` points to one UNSB ``test_latest/images`` directory.
All variants must contain the same rendered source image and the requested
``fake_k`` outputs for every case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REQUIRED_COLUMNS = {"relative_path", "case_id", "label", "source_split"}
VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(
        sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    )
    if not steps or any(step < 1 for step in steps):
        raise ValueError("steps must contain positive one-based integers")
    return steps


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must have the form NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not VALID_NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"invalid variant name {name!r}; use letters, digits, and underscores"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"variant image directory not found: {path}")
    return name, path


def load_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def require_image(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} image not found: {path}")
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        metavar="NAME=PATH",
        help="Repeat for every translator arm; PATH is test_latest/images",
    )
    parser.add_argument("--steps", default="1,5")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = parse_steps(args.steps)
    source_manifest = args.source_manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest not found: {source_manifest}")

    variants = dict(args.variant)
    if len(variants) != len(args.variant):
        raise ValueError("variant names must be unique")

    source = pd.read_csv(source_manifest)
    missing_columns = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing_columns:
        raise ValueError(f"source manifest is missing columns: {missing_columns}")
    source = source[
        ["relative_path", "case_id", "label", "source_split"]
    ].copy()
    source["case_id"] = source["case_id"].astype(str)
    source["label"] = source["label"].astype(int)
    if set(source["label"].unique()) - {0, 1}:
        raise ValueError("source manifest must contain binary labels 0/1")
    if source["case_id"].duplicated().any():
        duplicates = source.loc[
            source["case_id"].duplicated(), "case_id"
        ].head(5)
        raise ValueError(f"duplicate source cases: {duplicates.tolist()}")
    if set(source["source_split"].astype(str)) - {"src_train", "src_valid"}:
        raise ValueError("source_split must contain only src_train/src_valid")

    first_name = next(iter(variants))
    first_root = variants[first_name]
    rows: list[dict[str, object]] = []
    for record in source.itertuples(index=False):
        filename = Path(str(record.relative_path)).name
        canonical_raw = require_image(
            first_root / "real" / filename,
            f"{first_name} real",
        )
        canonical_pixels = load_pixels(canonical_raw)
        row: dict[str, object] = {
            "case_id": str(record.case_id),
            "label": int(record.label),
            "split": str(record.source_split),
            "raw": str(canonical_raw),
        }
        for variant_name, variant_root in variants.items():
            variant_raw = require_image(
                variant_root / "real" / filename,
                f"{variant_name} real",
            )
            if not np.array_equal(canonical_pixels, load_pixels(variant_raw)):
                raise ValueError(
                    "variant raw render mismatch for "
                    f"case={record.case_id}: {first_name} vs {variant_name}"
                )
            for step in steps:
                translated = require_image(
                    variant_root / f"fake_{step}" / filename,
                    f"{variant_name} U{step}",
                )
                row[f"{variant_name}_U{step}"] = str(translated)
        rows.append(row)

    output = pd.DataFrame(rows)
    train_cases = set(output.loc[output["split"] == "src_train", "case_id"])
    valid_cases = set(output.loc[output["split"] == "src_valid", "case_id"])
    overlap = sorted(train_cases & valid_cases)
    if overlap:
        raise ValueError(f"source train/valid case overlap: {overlap[:5]}")

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out, index=False)
    metadata = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "output_manifest": str(out),
        "output_manifest_sha256": sha256_file(out),
        "target_labels_used": False,
        "raw_render_variant": first_name,
        "variants": {name: str(path) for name, path in variants.items()},
        "steps": list(steps),
        "counts": {
            "src_train": len(train_cases),
            "src_valid": len(valid_cases),
        },
        "condition_columns": [
            column
            for column in output.columns
            if column not in {"case_id", "label", "split", "raw"}
        ],
    }
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
