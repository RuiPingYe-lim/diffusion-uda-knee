#!/usr/bin/env python
"""Evaluate UNSB (Unpaired Neural Schrodinger Bridge) translation with the FROZEN
source classifier — the same probe used for BBDM, so the two are comparable.

UNSB k2m = KneeMRI(A, target) -> MRNet(B, source), i.e. target->source, exactly
the direction a frozen source-only classifier needs.

Arms (all scored by the same frozen MRNet source-only classifier):
  direct   : the original mean-projection target PNG from the test CSV
             (validity gate: must reproduce the known direct AUC, e.g. 0.7281)
  real     : UNSB's own copy of the input (its preprocessing, 256px)
  fake_1..N: the successive Schrodinger-bridge steps; fake_N is the final
             translation. Scoring every step shows whether the discriminative
             signal survives translation or is progressively destroyed.

Reports each arm's case-level AUC plus a paired case bootstrap CI against a
reference arm (default: real), on shared resample indices.

Motivation: UNSB images look visually BETTER than BBDM's, but image quality and
preservation of class-discriminative content are different things — BBDM's
translated images collapsed a frozen classifier to chance. This measures which.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

_THIS = Path(__file__).resolve()
for _p in (_THIS.parent, _THIS.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eval_existing_classifier_on_csv import build_model  # noqa: E402

_TT = T.ToTensor()


def load_clf_input(path: Path, image_size: int) -> torch.Tensor:
    """Any PNG (L or RGB) -> [3,S,S] in [-1,1], the classifier's eval transform."""
    img = Image.open(path).convert("L")
    g = T.Resize((image_size, image_size), antialias=True)(_TT(img)).clamp(0, 1)
    return g.repeat(3, 1, 1) * 2.0 - 1.0


@torch.no_grad()
def score_paths(model, paths: List[Path], image_size: int, device, batch: int = 64) -> np.ndarray:
    probs: List[float] = []
    buf: List[torch.Tensor] = []

    def flush():
        if buf:
            x = torch.stack(buf).to(device)
            probs.extend(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy().tolist())
            buf.clear()

    for p in paths:
        buf.append(load_clf_input(p, image_size))
        if len(buf) >= batch:
            flush()
    flush()
    return np.asarray(probs)


def paired_ci(y, p_arm, p_ref, n_boot, seed) -> Dict[str, float]:
    rng = np.random.RandomState(seed); n = len(y); d = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n); yl = y[idx]
        if yl.min() == yl.max():
            continue
        d.append(roc_auc_score(yl, p_arm[idx]) - roc_auc_score(yl, p_ref[idx]))
    a = np.asarray(d)
    return {"boot_mean": round(float(a.mean()), 4),
            "lo": round(float(np.percentile(a, 2.5)), 4),
            "hi": round(float(np.percentile(a, 97.5)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser("evaluate UNSB translation with a frozen source classifier")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--test_csv", type=Path, required=True, help="target test CSV: image_path,label,case_id")
    ap.add_argument("--unsb_images_dir", type=Path, required=True, help="UNSB .../test_latest/images (has real/, fake_N/)")
    ap.add_argument("--arms", type=str, default="real,fake_1,fake_2,fake_3,fake_4,fake_5")
    ap.add_argument("--ref_arm", type=str, default="real")
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--expect_direct_auc", type=float, default=None, help="validity gate on the 'direct' arm")
    ap.add_argument("--direct_tol", type=float, default=0.005)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" not in args.device or torch.cuda.is_available()) else "cpu")
    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    ck = torch.load(args.classifier, map_location=device, weights_only=False)
    state = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    r = model.load_state_dict(state, strict=False)
    bad = [k for k in list(r.missing_keys) + list(r.unexpected_keys) if "rsa" not in k.lower()]
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:6]}")
    model.eval()

    df = pd.read_csv(args.test_csv)
    for c in ["image_path", "label", "case_id"]:
        if c not in df.columns:
            raise ValueError(f"{args.test_csv} must have {c}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # keep only cases whose UNSB images exist for EVERY arm (so all arms are paired)
    keep, missing = [], 0
    for _, row in df.iterrows():
        cid = str(row["case_id"])
        if all((args.unsb_images_dir / a / f"{cid}.png").exists() for a in arms):
            keep.append(row)
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing} cases lack UNSB images for some arm -> dropped (paired arms only)")
    sub = pd.DataFrame(keep).reset_index(drop=True)
    if sub.empty:
        raise RuntimeError("no cases with UNSB images for all arms")
    y = sub["label"].to_numpy(int)
    cids = [str(c) for c in sub["case_id"].tolist()]

    probs: Dict[str, np.ndarray] = {}
    # 'direct' = the original mean-projection PNG named in the CSV (validity anchor)
    probs["direct"] = score_paths(model, [Path(p) for p in sub["image_path"].tolist()], args.image_size, device)
    for a in arms:
        probs[a] = score_paths(model, [args.unsb_images_dir / a / f"{c}.png" for c in cids], args.image_size, device)

    aucs = {k: float(roc_auc_score(y, v)) for k, v in probs.items()}
    direct_ok = None
    if args.expect_direct_auc is not None:
        direct_ok = abs(aucs["direct"] - args.expect_direct_auc) <= args.direct_tol

    ref = args.ref_arm
    cis = {a: paired_ci(y, probs[a], probs[ref], args.n_boot, args.seed) for a in arms if a != ref}
    cis[f"{ref}_minus_direct"] = paired_ci(y, probs[ref], probs["direct"], args.n_boot, args.seed)

    out = {
        "n_cases": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "auc": {k: round(v, 4) for k, v in aucs.items()},
        "expect_direct_auc": args.expect_direct_auc,
        "direct_reproduced": direct_ok,
        "ref_arm": ref,
        "paired_CI_vs_ref": cis,
    }

    print("\n============ UNSB translation eval (frozen source classifier) ============")
    print(f"  n={out['n_cases']} (pos {out['n_pos']}/neg {out['n_neg']})")
    if direct_ok is not None:
        print(f"  [validity] direct AUC={aucs['direct']:.4f} vs expected {args.expect_direct_auc} -> "
              f"{'OK' if direct_ok else '!!! MISMATCH (probe VOID)'}")
    print("  --- AUC by arm ---")
    for k in ["direct"] + arms:
        print(f"    {k:10s} {aucs[k]:.4f}")
    print(f"  --- paired CI vs '{ref}' ---")
    for k, c in cis.items():
        print(f"    {k:22s} boot_mean={c['boot_mean']:+.4f}  CI[{c['lo']:+.4f}, {c['hi']:+.4f}]")
    print("==========================================================================\n")

    if args.out_csv:
        d = {"case_id": cids, "label": y}
        d.update({f"prob_{k}": v for k, v in probs.items()})
        pd.DataFrame(d).to_csv(args.out_csv, index=False); print(f"[write] {args.out_csv}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8"); print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
