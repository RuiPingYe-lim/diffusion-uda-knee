#!/usr/bin/env python3
"""Train a render-robust diagnostic teacher for the DOSC CIDP constraint.

WHY THIS EXISTS
---------------
The legacy DOSC loss compared the true-class margin of the source image with
that of the translated image. That measurement is useful only when the teacher
reads both renderings consistently.

The v9 development diagnostic found that the 128px source PNG and its 256px
source rendering score identically, ruling out resolution as the cause. It
also found that threshold recalibration recovers only part of the U1 accuracy
loss, indicating calibration drift plus local ranking loss. Those v9 values
were measured on ``src_train`` and must not be quoted as held-out acceptance.

The authoritative 130-case BUSI source-test result, which neither teacher
trained on, is:

    rendering          old AUC/acc       render-robust AUC/acc
    source             0.9386/0.9000          0.9367/0.9231
    U1                 0.8747/0.7385          0.9242/0.8846
    U5                 0.8017/0.7154          0.9207/0.8385

That matters because the margin m_y = (2y-1)d is not calibration-invariant.
The measured offset is b ~ -3.94, and the per-class margin change is +2.10 for
benign but -7.75 for malignant. Since the legacy safety term only penalised
drops, it read as "do not translate malignant cases". The new CIDP loss removes
positive affine drift before measuring local margin and rank degradation. A
render-robust teacher remains useful, but it is not a substitute for CIDP.

WHAT THIS DOES
--------------
Fine-tunes the same architecture on a mixture of renderings of the same source
cases: raw PNG, the 256px generator-input rendering, and UNSB translations.
Every view carries its case's source label. Target-domain labels are never
read. Target-train appearance is present only through the already-generated
UNSB translations, which is allowed by the UDA protocol.

ACCEPTANCE
----------
Checkpoint selection uses only the disjoint source validation split and
maximises its worst-rendering AUC. The held-out BUSI test split is evaluated
later by ``scripts/v10_heldout.py`` and must never be supplied as either the
training or validation split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_existing_classifier_on_csv import build_model  # noqa: E402

# da_manifest.csv columns, verified by pixel statistics rather than by name:
#   src_path 128px raw (mean 64.50/std 55.81) | raw 256px source render (64.52/55.70)
#   U1, U5   UNSB translations (51.87 / 55.87)
# The cache/fusion_*_busi.csv manifests are NOT usable here: their `before_png` is itself a
# translation (results_u2b_rev/.../fake_5/) and their fake_* columns point at a deleted
# cycle-back directory.
DEFAULT_VIEWS = "src_path,raw,U1,U5"


def build_transform(resize: int) -> T.Compose:
    """The validated source-classifier contract (reproduces val AUC 0.9913 exactly)."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((resize, resize), antialias=True),
            T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_records(
    manifest: Path,
    split: str,
    views: list[str],
) -> list[dict[str, object]]:
    frame = pd.read_csv(manifest)
    required = {"split", "case_id", "label", *views}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{manifest} is missing {missing_columns}; expected da_manifest.csv schema"
        )
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"split {split!r} is empty in {manifest}")
    records = []
    seen_cases: dict[str, int] = {}
    for row_number, row in frame.iterrows():
        if pd.isna(row["case_id"]):
            raise ValueError(f"empty case_id in {manifest}:{row_number + 2}")
        case_id = str(row["case_id"]).strip()
        if not case_id:
            raise ValueError(f"empty case_id in {manifest}:{row_number + 2}")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(
                f"expected binary source label in {manifest}:{row_number + 2}, got {label}"
            )
        if case_id in seen_cases:
            raise ValueError(
                f"duplicate case_id {case_id!r} in split {split!r}; "
                "the teacher expects one manifest row per case"
            )
        seen_cases[case_id] = label
        available = {}
        for view in views:
            value = row[view]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"missing {view} path for case {case_id!r} "
                    f"in {manifest}:{row_number + 2}"
                )
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"{view} image for case {case_id!r} does not exist: {path}"
                )
            available[view] = str(path.resolve())
        records.append(
            {
                "case_id": case_id,
                "label": label,
                "views": available,
            }
        )
    counts = {v: sum(v in r["views"] for r in records) for v in views}
    print(f"[{split}] {len(records)} cases; views present: {counts}")
    return records


def assert_disjoint_cases(
    train_records: list[dict[str, object]],
    validation_records: list[dict[str, object]],
) -> None:
    train_cases = {str(record["case_id"]) for record in train_records}
    validation_cases = {str(record["case_id"]) for record in validation_records}
    overlap = sorted(train_cases & validation_cases)
    if overlap:
        raise ValueError(
            f"source train/validation case overlap ({len(overlap)}): {overlap[:5]}"
        )


class MixedRenderDataset(Dataset):
    """One item per case; the rendering is RESAMPLED every epoch.

    Sampling one view per case, rather than emitting every view, keeps the epoch size -- and
    therefore the effective learning-rate schedule -- identical to the original source-only
    training, so the two runs stay comparable.
    """

    def __init__(
        self, records, resize: int, train: bool, seed: int = 0, view: str | None = None
    ):
        self.records = records
        self.tf = build_transform(resize)
        self.train = train
        self.view = view  # None -> resample; otherwise force one rendering
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        views = record["views"]
        name = self.view if (self.view in views) else self.rng.choice(sorted(views))
        image = Image.open(views[name]).convert("L")
        if self.train:
            if self.rng.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.rotate(self.rng.uniform(-10, 10), resample=Image.BILINEAR)
        return self.tf(image), int(record["label"])


@torch.no_grad()
def evaluate(model, records, resize, device, view, batch_size=32):
    subset = [r for r in records if view in r["views"]]
    if not subset:
        return None
    loader = DataLoader(
        MixedRenderDataset(subset, resize, train=False, view=view),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        probs.append(F.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    prob, label = np.concatenate(probs), np.concatenate(labels)
    auc = roc_auc_score(label, prob) if len(np.unique(label)) == 2 else float("nan")
    return {
        "auc": float(auc),
        "acc": float(((prob >= 0.5).astype(int) == label).mean()),
        "n": int(len(label)),
    }


def report(model, records, resize, device, views, title):
    print(f"\n--- {title} ---")
    rows = {}
    for view in views:
        m = evaluate(model, records, resize, device, view)
        if m is None:
            continue
        rows[view] = m
        print("  %-10s AUC %.4f   acc %.4f   n=%d" % (view, m["auc"], m["acc"], m["n"]))
    if rows:
        worst = min(rows, key=lambda k: rows[k]["auc"])
        best = max(rows, key=lambda k: rows[k]["auc"])
        print(
            "  best %s %.4f | worst %s %.4f | spread %.4f"
            % (
                best,
                rows[best]["auc"],
                worst,
                rows[worst]["auc"],
                rows[best]["auc"] - rows[worst]["auc"],
            )
        )
    return rows


def load_into(model, weights: Path, device, tag: str):
    blob = torch.load(weights, map_location=device, weights_only=False)
    state = (
        blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    )
    result = model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items()}, strict=False
    )
    missing = [k for k in result.missing_keys if "num_batches" not in k]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"{tag} checkpoint/model mismatch: missing={missing[:4]}, "
            f"unexpected={list(result.unexpected_keys)[:4]}"
        )
    print(f"[{tag}] loaded {weights}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="da_manifest.csv with explicit source train/validation splits",
    )
    p.add_argument("--train_split", default="src_train")
    p.add_argument("--val_split", default="src_valid")
    p.add_argument(
        "--views", default=DEFAULT_VIEWS, help="comma-separated manifest columns to mix"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--init_weights",
        type=Path,
        default=None,
        help="warm-start from the validated raw-only teacher (recommended)",
    )
    p.add_argument(
        "--baseline_weights",
        type=Path,
        default=None,
        help="old teacher, scored on the same renderings for a before/after table",
    )
    p.add_argument("--backbone", default="custom_resnet50_space")
    p.add_argument("--pretrained", default="imagenet")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_split == args.val_split:
        raise ValueError("train_split and val_split must be different")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    )
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    train_records = build_records(args.manifest, args.train_split, views)
    val_records = build_records(args.manifest, args.val_split, views)
    assert_disjoint_cases(train_records, val_records)
    present = [v for v in views if any(v in r["views"] for r in train_records)]
    print("renderings mixed during training: %s" % present)

    model = build_model(
        args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        device=device,
    )
    if args.init_weights is not None:
        load_into(model, args.init_weights, device, "init")

    if args.baseline_weights is not None:
        baseline = build_model(
            args.backbone,
            num_classes=args.num_classes,
            pretrained="none",
            device=device,
        )
        load_into(baseline, args.baseline_weights, device, "baseline")
        report(
            baseline,
            val_records,
            args.image_size,
            device,
            present,
            "BASELINE teacher (raw-only training)",
        )
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loader = DataLoader(
        MixedRenderDataset(
            train_records,
            args.image_size,
            train=True,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_worst, best_state, best_epoch, best_rows = -1.0, None, -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimiser.step()
            total += float(loss) * images.shape[0]
            seen += images.shape[0]
        rows = {
            view: metrics
            for view in present
            if (
                metrics := evaluate(
                    model,
                    val_records,
                    args.image_size,
                    device,
                    view,
                )
            )
        }
        worst = min(m["auc"] for m in rows.values())
        mean = float(np.mean([m["auc"] for m in rows.values()]))
        print(
            "epoch %02d  loss %.4f  val AUC mean %.4f  worst %.4f"
            % (epoch, total / max(seen, 1), mean, worst),
            flush=True,
        )
        if worst > best_worst:
            best_worst, best_epoch, best_rows = worst, epoch, rows
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no epoch produced a usable validation score")
    model.load_state_dict(best_state)
    rows = report(
        model,
        val_records,
        args.image_size,
        device,
        present,
        f"RENDER-ROBUST teacher (best epoch {best_epoch}, selected on worst rendering)",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "epoch": best_epoch,
            "best_metric_name": "worst_rendering_auc",
            "best_metric_value": best_worst,
            "val_metrics": rows,
            "renderings": present,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "data_protocol": {
                "manifest": str(args.manifest.resolve()),
                "train_split": args.train_split,
                "validation_split": args.val_split,
                "train_cases": len(train_records),
                "validation_cases": len(val_records),
                "case_overlap": 0,
                "heldout_test_used_for_selection": False,
                "target_labels_used": False,
            },
        },
        args.out,
    )
    summary = args.out.with_suffix(args.out.suffix + ".json")
    summary.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "worst_rendering_auc": best_worst,
                "per_rendering": best_rows,
                "renderings": present,
                "train_split": args.train_split,
                "validation_split": args.val_split,
                "heldout_test_used_for_selection": False,
                "target_labels_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved teacher: {args.out}\nSaved summary: {summary}")


if __name__ == "__main__":
    main()
