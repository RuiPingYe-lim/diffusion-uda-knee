#!/usr/bin/env python3
"""Install the TRSC model and legacy-compatible aliases into upstream UNSB."""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = REPO_ROOT / "src" / "unsb"
OVERLAY_FILES = {
    OVERLAY_ROOT / "dosc_modules.py": Path("models/dosc_modules.py"),
    OVERLAY_ROOT / "dosc_sb_model.py": Path("models/dosc_sb_model.py"),
    OVERLAY_ROOT / "dosc_unaligned_dataset.py": Path("data/dosc_unaligned_dataset.py"),
    OVERLAY_ROOT / "trsc_sb_model.py": Path("models/trsc_sb_model.py"),
    OVERLAY_ROOT / "trsc_joint_modules.py": Path("models/trsc_joint_modules.py"),
    OVERLAY_ROOT / "trsc_joint_sb_model.py": Path("models/trsc_joint_sb_model.py"),
    OVERLAY_ROOT / "dabrf_modules.py": Path("models/dabrf_modules.py"),
    OVERLAY_ROOT / "trsc_dabrf_joint_sb_model.py": Path(
        "models/trsc_dabrf_joint_sb_model.py"
    ),
    OVERLAY_ROOT / "safebridge_modules.py": Path(
        "models/safebridge_modules.py"
    ),
    OVERLAY_ROOT / "safebridge_joint_sb_model.py": Path(
        "models/safebridge_joint_sb_model.py"
    ),
    OVERLAY_ROOT / "trsc_unaligned_dataset.py": Path("data/trsc_unaligned_dataset.py"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unsb_root(root: Path) -> None:
    required = (
        root / "models" / "sb_model.py",
        root / "models" / "__init__.py",
        root / "data" / "base_dataset.py",
        root / "data" / "__init__.py",
        root / "train.py",
        root / "test.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The target is not a compatible UNSB checkout; missing: " + ", ".join(missing)
        )
    sb_text = (root / "models" / "sb_model.py").read_text(encoding="utf-8")
    required_tokens = ("class SBModel", "def compute_G_loss", "def calculate_NCE_loss")
    absent_tokens = [token for token in required_tokens if token not in sb_text]
    if absent_tokens:
        raise RuntimeError(
            "Unsupported UNSB SBModel API; missing tokens: " + ", ".join(absent_tokens)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsb_root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unsb_root = args.unsb_root.resolve()
    validate_unsb_root(unsb_root)
    actions = []
    for source, relative_destination in OVERLAY_FILES.items():
        destination = unsb_root / relative_destination
        if not source.is_file():
            raise FileNotFoundError(f"Overlay source not found: {source}")
        if destination.exists():
            if sha256(source) == sha256(destination):
                action = "unchanged"
            elif not args.force:
                raise FileExistsError(
                    f"Refusing to replace modified overlay file {destination}; pass --force"
                )
            else:
                action = "replace"
        else:
            action = "create"
        actions.append(
            {
                "action": action,
                "source": str(source),
                "destination": str(destination),
                "sha256": sha256(source),
            }
        )

    if not args.dry_run:
        for item in actions:
            if item["action"] == "unchanged":
                continue
            destination = Path(item["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source"], destination)
            py_compile.compile(str(destination), doraise=True)

    print(json.dumps({"dry_run": args.dry_run, "actions": actions}, indent=2))


if __name__ == "__main__":
    main()
