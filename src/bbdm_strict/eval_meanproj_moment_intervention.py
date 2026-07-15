#!/usr/bin/env python
"""Mean-projection FIXED MOMENT-INTERVENTION probe (renamed from "Oracle audit").

NOTE (review): this is NOT an oracle / upper bound for C2 -- it measures
AUC(f_frozen(M(x_target))) for a fixed first/second-moment transform M, which is
a different classifier/training/transform-direction than C2. A null here bounds
only THIS fixed intervention, not arbitrary appearance-translation methods.


Uses an EXISTING source-only ('none') classifier — NO retraining — to compare,
on the SAME mean-projection target test set that produced the reported number:

    direct_meanproj : target image fed to the classifier as-is
    moment_meanproj : target image moment-matched (per-image affine) to the
                      SOURCE global mean/std, then fed to the same classifier

The point of the audit: the mean-projection PNGs were exported AFTER a per-case
1/99-percentile + min-max intensity normalisation, which may already have erased
the scanner-appearance gap.  If so, moment matching on this pipeline is a near
identity transform and cannot help — regardless of any downstream path training.

Outputs
-------
* direct/moment case-level AUC (direct MUST reproduce the training-time number);
* paired case-level bootstrap Delta AUC and 95% CI (shared resample indices);
* source vs target per-image mean/std distributions (is the style gap gone?);
* mean pixel MAE introduced by the moment transform (distance from identity);
* fraction of pixels clipped to {0,1} by the transform;
* closure of the intensity mean/std gap toward the source.

VERDICT (stop rule)
-------------------
* Delta < 0.01  OR  bootstrap CI crosses 0  -> STOP pixel-path tuning; end knee C2.
* Delta >= 0.03 AND CI strictly above 0     -> appearance lever survives; Round 2 OK.
* otherwise                                 -> weak / ambiguous.

The classifier is loaded read-only and the target *test* labels are used only to
score AUC after the (fixed) checkpoint — nothing here is tuned on them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

# --- resolve imports against the live code root (parent of this file's dir) ----
_THIS = Path(__file__).resolve()
for _p in (_THIS.parent, _THIS.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_existing_classifier_on_csv import build_model  # noqa: E402


# ----------------------------------------------------------------------------- #
# Preprocessing that EXACTLY mirrors CSVBinaryDataset's eval transform:
#   ToTensor -> Resize(antialias) -> repeat(3) -> Normalize(0.5,0.5) == 2x-1
# We split it so the moment transform can be injected in the [0,1], 1-channel,
# post-resize space (identical to where a translated view would live).
# ----------------------------------------------------------------------------- #
_TO_TENSOR = T.ToTensor()


def load_gray01_resized(path: Path, image_size: int) -> torch.Tensor:
    """Return a [1,S,S] float tensor in [0,1] using the eval resize (antialias)."""
    img = Image.open(path).convert("L")
    resize = T.Resize((int(image_size), int(image_size)), antialias=True)
    return resize(_TO_TENSOR(img)).clamp(0.0, 1.0)  # [1,S,S]


def to_classifier_input(gray01: torch.Tensor) -> torch.Tensor:
    """[1,S,S] in [0,1] -> [3,S,S] in [-1,1] (Normalize(0.5,0.5))."""
    return gray01.repeat(3, 1, 1) * 2.0 - 1.0


def moment_match01(
    gray01: torch.Tensor, s_mean: float, s_std: float, eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-image affine to source (mean,std); population std (ddof=0) to match
    numpy .std() used everywhere else.  Returns (clipped01, stats) where stats
    holds the transform MAE, clip fraction, and the ACTUAL post-clip mean/std
    (post-clip so closure reflects the real, clipping-distorted result).
    """
    m = gray01.mean()
    s = gray01.std(unbiased=False).clamp_min(eps)
    raw = (gray01 - m) / s * float(s_std) + float(s_mean)
    clipped = raw.clamp(0.0, 1.0)
    stats = {
        "mm_mae": float((clipped - gray01).abs().mean().item()),
        "mm_clip_frac": float(((raw < 0.0) | (raw > 1.0)).float().mean().item()),
        "mm_mean": float(clipped.mean().item()),
        "mm_std": float(clipped.std(unbiased=False).item()),
    }
    return clipped, stats


# ----------------------------------------------------------------------------- #
# Source global statistics computed with the SAME resize the classifier sees.
# ----------------------------------------------------------------------------- #
def compute_source_stats(
    csv_path: Path, image_col: str, image_size: int, max_n: int, seed: int
) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    parent = Path(csv_path).parent
    idx = list(range(len(df)))
    import random as _random

    _random.Random(int(seed)).shuffle(idx)
    idx = idx[: min(int(max_n), len(idx))]
    s = ss = cnt = 0.0
    per_mean: List[float] = []
    per_std: List[float] = []
    for i in idx:
        raw = str(df.iloc[i][image_col]).strip()
        p = Path(raw)
        if not p.exists():
            p = parent / raw
        g = load_gray01_resized(p, image_size)  # [1,S,S]
        s += float(g.sum().item())
        ss += float((g * g).sum().item())
        cnt += float(g.numel())
        per_mean.append(float(g.mean().item()))
        per_std.append(float(g.std(unbiased=False).item()))
    mean = s / cnt
    std = float(np.sqrt(max(ss / cnt - mean * mean, 1e-8)))
    return {
        "global_mean": float(mean),
        "global_std": float(std),
        "per_image_mean": per_mean,
        "per_image_std": per_std,
        "n": len(idx),
    }


def _detect(df: pd.DataFrame, prefer, candidates, required=True):
    if prefer and prefer in df.columns:
        return prefer
    low = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise ValueError(f"None of {candidates} in {list(df.columns)}")
    return None


@torch.no_grad()
def score_target(
    model: torch.nn.Module,
    csv_path: Path,
    image_col: str,
    label_col: str,
    case_col,
    image_size: int,
    s_mean: float,
    s_std: float,
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parent = Path(csv_path).parent
    rows: List[Dict[str, object]] = []
    buf_direct: List[torch.Tensor] = []
    buf_moment: List[torch.Tensor] = []
    meta: List[Dict[str, object]] = []

    def flush():
        if not buf_direct:
            return
        xd = torch.stack(buf_direct).to(device)
        xm = torch.stack(buf_moment).to(device)
        pd_ = torch.softmax(model(xd), dim=1)[:, 1].cpu().numpy()
        pm_ = torch.softmax(model(xm), dim=1)[:, 1].cpu().numpy()
        for k, mrow in enumerate(meta):
            rows.append({**mrow, "prob_direct": float(pd_[k]), "prob_moment": float(pm_[k])})
        buf_direct.clear()
        buf_moment.clear()
        meta.clear()

    for i in range(len(df)):
        raw = str(df.iloc[i][image_col]).strip()
        p = Path(raw)
        if not p.exists():
            p = parent / raw
        g = load_gray01_resized(p, image_size)  # [1,S,S] in [0,1]
        gm, mm = moment_match01(g, s_mean, s_std)
        buf_direct.append(to_classifier_input(g))
        buf_moment.append(to_classifier_input(gm))
        lab = df.iloc[i][label_col]
        case_id = str(df.iloc[i][case_col]) if case_col else str(i)
        meta.append(
            {
                "case_id": case_id,
                "label": int(lab),
                "tgt_mean": float(g.mean().item()),
                "tgt_std": float(g.std(unbiased=False).item()),
                **mm,
            }
        )
        if len(buf_direct) >= batch_size:
            flush()
    flush()
    return pd.DataFrame(rows)


def paired_bootstrap_delta(
    case_label: np.ndarray,
    prob_direct: np.ndarray,
    prob_moment: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(int(seed))
    n = len(case_label)
    deltas: List[float] = []
    kept = 0
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, n)  # shared resample for both arms
        yl = case_label[idx]
        if yl.min() == yl.max():
            continue
        ad = roc_auc_score(yl, prob_direct[idx])
        am = roc_auc_score(yl, prob_moment[idx])
        deltas.append(am - ad)
        kept += 1
    arr = np.asarray(deltas, dtype=float)
    return {
        "delta_mean_boot": float(arr.mean()) if kept else float("nan"),
        "delta_ci_lo": float(np.percentile(arr, 2.5)) if kept else float("nan"),
        "delta_ci_hi": float(np.percentile(arr, 97.5)) if kept else float("nan"),
        "boot_kept": int(kept),
    }


def main() -> None:
    ap = argparse.ArgumentParser("mean-projection Oracle audit (knee C2 gate)")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--source_train_csv", type=Path, required=True)
    ap.add_argument("--target_test_csv", type=Path, required=True)
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--image_col", type=str, default="image_path")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--case_col", type=str, default="case_id")
    ap.add_argument("--source_stat_n", type=int, default=100000, help="cap on source imgs for stats")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expect_direct_auc", type=float, default=0.7281,
                    help="direct-arm reproduction target; a mismatch beyond --direct_auc_tol VOIDS the audit")
    ap.add_argument("--direct_auc_tol", type=float, default=0.005)
    ap.add_argument("--mcid", type=float, default=0.03, help="minimum clinically important delta-AUC (futility = CI upper < mcid)")
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if ("cuda" not in args.device or torch.cuda.is_available()) else "cpu")

    # --- model (read-only) ---------------------------------------------------
    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    ck = torch.load(args.classifier, map_location=device, weights_only=False)
    state = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    res = model.load_state_dict(state, strict=False)
    bad_m = [k for k in res.missing_keys if "rsa" not in k.lower()]
    bad_u = [k for k in res.unexpected_keys if "rsa" not in k.lower()]
    if bad_m or bad_u:
        raise RuntimeError(f"checkpoint mismatch: missing={bad_m[:6]} unexpected={bad_u[:6]}")
    model.eval()
    print(f"[load] {args.classifier} (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}; rsa ok)")

    # --- source global stats (same resize the classifier sees) ---------------
    ss = compute_source_stats(args.source_train_csv, args.image_col, args.image_size, args.source_stat_n, args.seed)
    s_mean, s_std = ss["global_mean"], ss["global_std"]
    print(f"[source] n={ss['n']} global_mean={s_mean:.4f} global_std={s_std:.4f}")

    # --- score target: direct + moment ---------------------------------------
    df = score_target(
        model, args.target_test_csv, args.image_col, args.label_col, args.case_col,
        args.image_size, s_mean, s_std, device,
    )
    if df.empty:
        raise RuntimeError("no target rows scored")

    # per-case aggregation (mean-proj is 1 img/case; general-safe)
    case = (
        df.groupby("case_id", sort=False)
        .agg(label=("label", "first"),
             prob_direct=("prob_direct", "mean"),
             prob_moment=("prob_moment", "mean"))
        .reset_index()
    )
    y = case["label"].to_numpy(int)
    pdirect = case["prob_direct"].to_numpy(float)
    pmoment = case["prob_moment"].to_numpy(float)
    auc_direct = float(roc_auc_score(y, pdirect))
    auc_moment = float(roc_auc_score(y, pmoment))
    delta = auc_moment - auc_direct
    boot = paired_bootstrap_delta(y, pdirect, pmoment, args.n_boot, args.seed)

    # --- appearance-gap diagnostics ------------------------------------------
    src_m = np.asarray(ss["per_image_mean"]); src_s = np.asarray(ss["per_image_std"])
    tgt_m = df["tgt_mean"].to_numpy(float); tgt_s = df["tgt_std"].to_numpy(float)
    gap_mean = float(abs(src_m.mean() - tgt_m.mean()))
    gap_std = float(abs(src_s.mean() - tgt_s.mean()))
    # FIX (review): standardize the cross-domain MEAN gap by the pooled BETWEEN-CASE
    # SD of per-image means -- NOT the within-image pooled pixel std (s_std), which
    # is a different quantity and understated the gap in the original writeup.
    xcase_sd = float(np.sqrt((src_m.std() ** 2 + tgt_m.std() ** 2) / 2.0))
    gap_mean_in_srcstd = float(gap_mean / (xcase_sd + 1e-9))
    mm_mae = float(df["mm_mae"].mean())
    clip_frac = float(df["mm_clip_frac"].mean())
    # closure of the intensity gap toward the source, using ACTUAL post-clip
    # moment means/stds (clipping keeps them from landing exactly on source).
    mm_m = df["mm_mean"].to_numpy(float); mm_s = df["mm_std"].to_numpy(float)
    gap_mean_before = float(np.mean(np.abs(tgt_m - s_mean)))
    gap_mean_after = float(np.mean(np.abs(mm_m - s_mean)))
    closure_mean = float(1.0 - gap_mean_after / (gap_mean_before + 1e-9))
    gap_std_before = float(np.mean(np.abs(tgt_s - s_std)))
    gap_std_after = float(np.mean(np.abs(mm_s - s_std)))
    closure_std = float(1.0 - gap_std_after / (gap_std_before + 1e-9))

    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())

    # --- direct-arm reproduction gate: the whole audit is VOID if this fails --
    direct_diff = None
    direct_ok = True
    if args.expect_direct_auc is not None:
        direct_diff = float(abs(auc_direct - float(args.expect_direct_auc)))
        direct_ok = direct_diff <= float(args.direct_auc_tol)

    # --- verdict (review: futility = CI UPPER bound below the MCID, NOT "CI crosses 0") -----
    # MCID = minimum clinically important difference (preset effect size, e.g. 0.03).
    # PROCEED  : ci_lo > 0 AND delta >= MCID   (effect is real and >= MCID)
    # STOP     : ci_hi < MCID                  (can rule OUT a meaningful effect -> futile)
    # INCONCL. : otherwise                     (CI spans the MCID -> undecided; get more data)
    ci_lo, ci_hi = boot["delta_ci_lo"], boot["delta_ci_hi"]
    MCID = float(args.mcid)
    if not direct_ok:
        verdict = (f"VOID: direct arm did NOT reproduce expected {args.expect_direct_auc} "
                   f"(got {auc_direct:.4f}, |d|={direct_diff:.4f} > tol {args.direct_auc_tol}); "
                   "preprocessing/checkpoint diverges from training -> audit invalid, ignore delta.")
    elif ci_lo > 0 and delta >= MCID:
        verdict = f"PROCEED: effect real and >= MCID={MCID} (ci_lo={ci_lo:.3f}>0)."
    elif ci_hi < MCID:
        verdict = f"STOP (futility): CI upper {ci_hi:.3f} < MCID={MCID} -> can rule out a meaningful effect."
    else:
        verdict = f"INCONCLUSIVE: CI [{ci_lo:.3f},{ci_hi:.3f}] spans MCID={MCID} -> undecided, need more cases/folds."

    summary = {
        "auc_direct": round(auc_direct, 4),
        "auc_moment": round(auc_moment, 4),
        "delta_auc_point": round(delta, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in boot.items()},
        "n_cases": int(len(case)), "n_pos": n_pos, "n_neg": n_neg,
        "source_global_mean": round(s_mean, 4), "source_global_std": round(s_std, 4),
        "source_perimg_mean_mean": round(float(src_m.mean()), 4),
        "source_perimg_mean_sd": round(float(src_m.std()), 4),
        "source_perimg_std_mean": round(float(src_s.mean()), 4),
        "target_perimg_mean_mean": round(float(tgt_m.mean()), 4),
        "target_perimg_mean_sd": round(float(tgt_m.std()), 4),
        "target_perimg_std_mean": round(float(tgt_s.mean()), 4),
        "gap_mean_abs": round(gap_mean, 4),
        "gap_std_abs": round(gap_std, 4),
        "gap_mean_in_source_std_units": round(gap_mean_in_srcstd, 4),
        "moment_transform_mean_MAE": round(mm_mae, 5),
        "moment_transform_clip_fraction": round(clip_frac, 5),
        "closure_mean_after_clip": round(closure_mean, 4),
        "closure_std_after_clip": round(closure_std, 4),
        "expect_direct_auc": args.expect_direct_auc,
        "direct_auc_diff": (round(direct_diff, 4) if direct_diff is not None else None),
        "direct_reproduced": bool(direct_ok),
        "verdict": verdict,
    }

    print("\n================ MEAN-PROJ ORACLE AUDIT ================")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=======================================================")
    if args.expect_direct_auc is not None:
        tag = "OK" if direct_ok else "!!! PIPELINE MISMATCH -> AUDIT VOID"
        print(f"[sanity] direct_auc={auc_direct:.4f} vs expected={args.expect_direct_auc:.4f} "
              f"(|d|={direct_diff:.4f}, tol={args.direct_auc_tol}) {tag}")
    print(f"\nVERDICT: {verdict}\n")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        case.to_csv(str(args.out_csv).replace(".csv", "_percase.csv"), index=False)
        print(f"[write] per-image -> {args.out_csv}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[write] summary -> {args.out_json}")


if __name__ == "__main__":
    main()
