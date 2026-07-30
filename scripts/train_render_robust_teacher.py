#!/usr/bin/env python3
"""Train a RENDER-ROBUST diagnostic teacher for the DOSC safety constraint.

WHY THIS EXISTS
---------------
``dosc_modules.diagnostic_non_degradation_loss`` compares the true-class margin of the source
image with that of the TRANSLATED image and penalises the drop. That is only a diagnostic
signal if the teacher reads both renderings equally well. Measured on this data with the
exported teacher (``gate_busi2breast_cache``), scoring d = z1 - z0:

    rendering            AUC      acc     best acc reachable by ANY threshold
    raw PNG (128px)     0.9999   0.9912            0.9934
    raw render (256px)  0.9996   0.9844                --
    U1 (translated)     0.9516   0.7522            0.8850
    U5 (translated)     0.8900   0.7102            0.8208

Two consequences. First, resolution is not the issue at all -- 128px and 256px score
identically; translation is. Second, only about 56% of the accuracy loss on U1 is a
threshold effect (0.7522 -> 0.8850 by recalibration); the remaining 44% is a genuine loss of
separability, and AUC falls 0.048. So the teacher is BOTH miscalibrated and mildly degraded
on translated images.

That matters because the margin m_y = (2y-1)d is not calibration-invariant. The measured
offset is b ~ -3.94, and the per-class margin change is +2.10 for benign but -7.75 for
malignant. Since the safety term only penalises drops, it fires on essentially every
malignant case and never on benign ones: as exported, the constraint reads as "do not
translate malignant cases". A teacher that reads translated images correctly removes the
bulk of that spurious, class-asymmetric gradient.

WHAT THIS DOES
--------------
Fine-tunes the same architecture on a MIXTURE of renderings of the SAME source cases -- raw
PNG, the 256px generator-input rendering, and the UNSB translations -- every view carrying
its case's SOURCE label. No target image and no target label is read anywhere in this file,
so the UDA boundary is untouched.

ACCEPTANCE
----------
Per-rendering AUC/accuracy on the source validation split, printed next to the old
checkpoint's numbers when ``--baseline_weights`` is given. The teacher is fit for purpose
when its worst rendering is close to its best; excellent on raw and poor on translations is
the failure this script exists to remove, so checkpoint selection uses the WORST-rendering
AUC rather than the mean.
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


def build_records(manifest: Path, split: str, views: list[str]):
    frame = pd.read_csv(manifest)
    if "split" not in frame.columns:
        raise ValueError(f"{manifest} has no 'split' column; expected da_manifest.csv schema")
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"split {split!r} is empty in {manifest}")
    missing = [v for v in views if v not in frame.columns]
    if missing:
        raise ValueError(f"views {missing} not in {list(frame.columns)}")
    records = []
    for _, row in frame.iterrows():
        available = {v: str(row[v]) for v in views if isinstance(row[v], str) and row[v]}
        if not available:
            continue
        records.append({"label": int(row["label"]), "views": available})
    counts = {v: sum(v in r["views"] for r in records) for v in views}
    print(f"[{split}] {len(records)} cases; views present: {counts}")
    return records


class MixedRenderDataset(Dataset):
    """One item per case; the rendering is RESAMPLED every epoch.

    Sampling one view per case, rather than emitting every view, keeps the epoch size -- and
    therefore the effective learning-rate schedule -- identical to the original source-only
    training, so the two runs stay comparable.
    """

    def __init__(self, records, resize: int, train: bool, seed: int = 0, view: str | None = None):
        self.records = records
        self.tf = build_transform(resize)
        self.train = train
        self.view = view                      # None -> resample; otherwise force one rendering
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
    loader = DataLoader(MixedRenderDataset(subset, resize, train=False, view=view),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        probs.append(F.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy())
        labels.append(targets.numpy())
    prob, label = np.concatenate(probs), np.concatenate(labels)
    auc = roc_auc_score(label, prob) if len(np.unique(label)) == 2 else float("nan")
    return {"auc": float(auc), "acc": float(((prob >= 0.5).astype(int) == label).mean()), "n": int(len(label))}


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
        print("  best %s %.4f | worst %s %.4f | spread %.4f"
              % (best, rows[best]["auc"], worst, rows[worst]["auc"], rows[best]["auc"] - rows[worst]["auc"]))
    return rows


def load_into(model, weights: Path, device, tag: str):
    blob = torch.load(weights, map_location=device, weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    result = model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    missing = [k for k in result.missing_keys if "num_batches" not in k]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"{tag} checkpoint/model mismatch: missing={missing[:4]}, "
                           f"unexpected={list(result.unexpected_keys)[:4]}")
    print(f"[{tag}] loaded {weights}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("/root/autodl-tmp/breast/da_route/da_manifest.csv"))
    p.add_argument("--train_split", default="src_train")
    p.add_argument("--val_split", default="src_valid")
    p.add_argument("--views", default=DEFAULT_VIEWS, help="comma-separated manifest columns to mix")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init_weights", type=Path, default=None,
                   help="warm-start from the validated raw-only teacher (recommended)")
    p.add_argument("--baseline_weights", type=Path, default=None,
                   help="old teacher, scored on the same renderings for a before/after table")
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
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    train_records = build_records(args.manifest, args.train_split, views)
    val_records = build_records(args.manifest, args.val_split, views)
    present = [v for v in views if any(v in r["views"] for r in train_records)]
    print("renderings mixed during training: %s" % present)

    model = build_model(args.backbone, num_classes=args.num_classes, pretrained=args.pretrained, device=device)
    if args.init_weights is not None:
        load_into(model, args.init_weights, device, "init")

    if args.baseline_weights is not None:
        baseline = build_model(args.backbone, num_classes=args.num_classes, pretrained="none", device=device)
        load_into(baseline, args.baseline_weights, device, "baseline")
        report(baseline, val_records, args.image_size, device, present, "BASELINE teacher (raw-only training)")
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    loader = DataLoader(MixedRenderDataset(train_records, args.image_size, train=True, seed=args.seed),
                        batch_size=args.batch_size, shuffle=True, num_workers=0)
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
            total += float(loss) * images.shape[0]; seen += images.shape[0]
        rows = {v: m for v in present if (m := evaluate(model, val_records, args.image_size, device, v))}
        worst = min(m["auc"] for m in rows.values())
        mean = float(np.mean([m["auc"] for m in rows.values()]))
        print("epoch %02d  loss %.4f  val AUC mean %.4f  worst %.4f" % (epoch, total / max(seen, 1), mean, worst),
              flush=True)
        if worst > best_worst:
            best_worst, best_epoch, best_rows = worst, epoch, rows
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no epoch produced a usable validation score")
    model.load_state_dict(best_state)
    rows = report(model, val_records, args.image_size, device, present,
                  f"RENDER-ROBUST teacher (best epoch {best_epoch}, selected on worst rendering)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "epoch": best_epoch,
                "best_metric_name": "worst_rendering_auc", "best_metric_value": best_worst,
                "val_metrics": rows, "renderings": present,
                "args": {k: str(v) for k, v in vars(args).items()},
                "note": "source labels only; no target image or label is read by this script"},
               args.out)
    summary = args.out.with_suffix(args.out.suffix + ".json")
    summary.write_text(json.dumps({"best_epoch": best_epoch, "worst_rendering_auc": best_worst,
                                   "per_rendering": best_rows, "renderings": present}, indent=2), encoding="utf-8")
    print(f"\nSaved teacher: {args.out}\nSaved summary: {summary}")


if __name__ == "__main__":
    main()
