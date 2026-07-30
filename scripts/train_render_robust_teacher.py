#!/usr/bin/env python3
"""Train a RENDER-ROBUST diagnostic teacher for the DOSC margin constraint.

WHY THIS EXISTS
---------------
``dosc_modules.diagnostic_non_degradation_loss`` compares the true-class margin of the
source image against that of the TRANSLATED image and penalises the drop. That is only a
diagnostic-safety signal if the teacher can read both renderings equally well.

The teacher currently exported (``gate_busi2breast_cache/best_checkpoint.pt``) was trained
on raw BUSI PNGs only. Measured on this repo's data:

    raw BUSI (image_path, 128x128)              acc 0.984
    UNSB input rendering (before_png, 256x256)  acc 0.578   <-- 27/32 malignant wrong

and, on translated images, a frozen source classifier loses 0.149 AUC while a probe
retrained on that rendering loses only 0.023. The lesion survives translation; the frozen
teacher simply cannot read it. Used as-is, the margin term therefore fires on APPEARANCE
CHANGE rather than on diagnostic damage, i.e. it becomes a leash toward the identity
mapping -- the exact degeneration DOSC is meant to avoid, and doubly harmful here because
the translator already under-translates (~3% of the class-conditional domain gap).

WHAT THIS DOES
--------------
Fine-tunes the same architecture on a MIXTURE of renderings of the SAME source cases:
raw PNG, the 256px generator-input rendering, and the precomputed translations. Every view
of a case carries that case's SOURCE label, so no target label is ever touched and the UDA
boundary is untouched -- target images are not read at all by this script.

ACCEPTANCE
----------
The run prints AUC/accuracy PER RENDERING on the source validation split, next to the old
checkpoint's numbers when ``--baseline_weights`` is given. The teacher is fit for purpose
when its worst rendering is close to its best; a teacher that is excellent on raw and poor
on translations is the failure this script exists to remove. Checkpoint selection uses the
WORST-rendering AUC for that reason.
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

RAW_VIEW = "raw"
SOURCE_RENDER_CANDIDATES = ("before_png", "source_png", "src_png")
IMAGE_COL_CANDIDATES = ("image_path", "path", "img_path", "filepath")


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


def detect(frame: pd.DataFrame, candidates, required: bool = True):
    lower = {c.lower(): c for c in frame.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise ValueError(f"none of {candidates} in columns {list(frame.columns)}")
    return None


def stem_of(path_value: str) -> str:
    return Path(str(path_value)).stem


def key_stem(key_value: str) -> str:
    """'src_train__00000' -> '00000'; falls back to the whole key."""
    text = str(key_value)
    return text.rsplit("__", 1)[-1] if "__" in text else text


def build_records(manifest_csv: Path, raw_csv: Path | None, max_fakes: int):
    """One record per case: {label, views: {name -> path}}.

    Views come from the manifest (256px source rendering + translations). The raw PNG is
    joined in by filename stem when a raw CSV is given; a failed join degrades to
    "manifest views only" with a warning rather than silently mismatching labels.
    """
    man = pd.read_csv(manifest_csv)
    src_col = detect(man, SOURCE_RENDER_CANDIDATES)
    label_col = detect(man, ("label", "target", "y"))
    fake_cols = [c for c in man.columns if c.lower().startswith("fake")]
    fake_cols = sorted(fake_cols)[:max_fakes] if max_fakes >= 0 else sorted(fake_cols)
    key_col = detect(man, ("key", "case_id"), required=False)

    raw_by_stem = {}
    if raw_csv is not None:
        raw = pd.read_csv(raw_csv)
        raw_img = detect(raw, IMAGE_COL_CANDIDATES)
        raw_lab = detect(raw, ("label", "target", "y"))
        for _, row in raw.iterrows():
            raw_by_stem[stem_of(row[raw_img])] = (str(row[raw_img]), int(row[raw_lab]))

    records, joined, mismatched = [], 0, 0
    for _, row in man.iterrows():
        label = int(row[label_col])
        views = {"src_render": str(row[src_col])}
        for c in fake_cols:
            if isinstance(row[c], str) and row[c]:
                views[c] = str(row[c])
        if raw_by_stem:
            stem = key_stem(row[key_col]) if key_col else stem_of(row[src_col])
            hit = raw_by_stem.get(stem)
            if hit is not None:
                if hit[1] != label:
                    mismatched += 1
                else:
                    views[RAW_VIEW] = hit[0]
                    joined += 1
        records.append({"label": label, "views": views})

    if raw_by_stem:
        if mismatched:
            raise ValueError(
                f"{mismatched} cases had conflicting labels between {raw_csv} and "
                f"{manifest_csv}; refusing to train on an ambiguous join"
            )
        if joined == 0:
            print(f"[warn] no raw view could be joined from {raw_csv}; continuing without it")
        else:
            print(f"[join] raw view attached to {joined}/{len(records)} cases")
    return records, [RAW_VIEW, "src_render"] + fake_cols


class MixedRenderDataset(Dataset):
    """One item per case; the rendering is RESAMPLED every epoch.

    Sampling one view per case (rather than emitting every view) keeps the epoch size and
    therefore the effective learning-rate schedule identical to the original source-only
    training, so the two runs stay comparable.
    """

    def __init__(self, records, resize: int, train: bool, seed: int = 0, view: str | None = None):
        self.records = records
        self.tf = build_transform(resize)
        self.train = train
        self.view = view                     # None -> sample; otherwise force one rendering
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        views = record["views"]
        if self.view is not None:
            name = self.view
            if name not in views:
                name = "src_render" if "src_render" in views else next(iter(views))
        else:
            name = self.rng.choice(sorted(views.keys()))
        image = Image.open(views[name]).convert("L")
        if self.train:
            if self.rng.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            angle = self.rng.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.BILINEAR)
        return self.tf(image), int(record["label"])


@torch.no_grad()
def evaluate(model, records, resize, device, view, batch_size=32):
    subset = [r for r in records if view in r["views"]]
    if not subset:
        return None
    loader = DataLoader(
        MixedRenderDataset(subset, resize, train=False, view=view),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    prob = np.concatenate(probs)
    label = np.concatenate(labels)
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(label, prob) if len(np.unique(label)) == 2 else float("nan")
    return {"auc": float(auc), "acc": float((pred == label).mean()), "n": int(len(label))}


def report(model, records, resize, device, views, title):
    print(f"\n--- {title} (source validation, per rendering) ---")
    rows = {}
    for view in views:
        metrics = evaluate(model, records, resize, device, view)
        if metrics is None:
            continue
        rows[view] = metrics
        print("  %-12s AUC %.4f   acc %.4f   n=%d" % (view, metrics["auc"], metrics["acc"], metrics["n"]))
    if rows:
        worst = min(rows, key=lambda k: rows[k]["auc"])
        print("  worst rendering: %s (AUC %.4f)" % (worst, rows[worst]["auc"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_manifest", type=Path, required=True,
                        help="source TRAIN manifest with before_png + fake_* columns")
    parser.add_argument("--val_manifest", type=Path, required=True,
                        help="source VALIDATION manifest, same schema")
    parser.add_argument("--raw_train_csv", type=Path, default=None, help="optional raw-PNG source train CSV")
    parser.add_argument("--raw_val_csv", type=Path, default=None, help="optional raw-PNG source val CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline_weights", type=Path, default=None,
                        help="old teacher checkpoint, scored on the same renderings for comparison")
    parser.add_argument("--init_weights", type=Path, default=None,
                        help="warm-start from an existing source classifier (recommended: the "
                             "validated raw-only teacher). Keeps the new teacher anchored to the "
                             "model whose source-validation numbers are already known.")
    parser.add_argument("--backbone", default="custom_resnet50_space")
    parser.add_argument("--pretrained", default="imagenet")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_fakes", type=int, default=5, help="-1 for all fake_* columns")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")

    train_records, view_names = build_records(args.train_manifest, args.raw_train_csv, args.max_fakes)
    val_records, _ = build_records(args.val_manifest, args.raw_val_csv, args.max_fakes)
    present = [v for v in view_names if any(v in r["views"] for r in train_records)]
    print("renderings in training mixture: %s" % present)
    print("train cases %d | val cases %d" % (len(train_records), len(val_records)))

    model = build_model(args.backbone, num_classes=args.num_classes, pretrained=args.pretrained, device=device)
    if args.init_weights is not None:
        blob = torch.load(args.init_weights, map_location=device, weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        result = model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        missing = [k for k in result.missing_keys if "num_batches" not in k]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"warm-start checkpoint does not match the model: missing={missing[:4]}, "
                f"unexpected={list(result.unexpected_keys)[:4]}"
            )
        print(f"[init] warm-started from {args.init_weights}")

    if args.baseline_weights is not None:
        baseline = build_model(args.backbone, num_classes=args.num_classes, pretrained="none", device=device)
        blob = torch.load(args.baseline_weights, map_location=device, weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        baseline.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        report(baseline, val_records, args.image_size, device, present, "BASELINE teacher (raw-only training)")
        del baseline
        torch.cuda.empty_cache() if device.type == "cuda" else None

    loader = DataLoader(
        MixedRenderDataset(train_records, args.image_size, train=True, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        rows = {}
        for view in present:
            metrics = evaluate(model, val_records, args.image_size, device, view)
            if metrics is not None:
                rows[view] = metrics
        worst = min(m["auc"] for m in rows.values()) if rows else float("nan")
        mean = float(np.mean([m["auc"] for m in rows.values()])) if rows else float("nan")
        print("epoch %02d  loss %.4f  val AUC mean %.4f  worst %.4f" % (epoch, total / max(seen, 1), mean, worst),
              flush=True)
        if worst > best_worst:
            best_worst, best_epoch = worst, epoch
            best_rows = rows
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no epoch produced a usable validation score")
    model.load_state_dict(best_state)
    rows = report(model, val_records, args.image_size, device, present,
                  f"RENDER-ROBUST teacher (best epoch {best_epoch}, selected on worst rendering)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "epoch": best_epoch,
            "best_metric_name": "worst_rendering_auc",
            "best_metric_value": best_worst,
            "val_metrics": rows,
            "args": vars(args) | {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
            "renderings": present,
            "note": "source labels only; no target image or label is read by this script",
        },
        args.out,
    )
    summary = args.out.with_suffix(args.out.suffix + ".json")
    summary.write_text(json.dumps({"best_epoch": best_epoch, "worst_rendering_auc": best_worst,
                                   "per_rendering": best_rows, "renderings": present}, indent=2), encoding="utf-8")
    print(f"\nSaved teacher: {args.out}\nSaved summary: {summary}")


if __name__ == "__main__":
    main()
