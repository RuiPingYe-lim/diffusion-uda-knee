#!/usr/bin/env python
"""b' — fixed first/second-moment intervention probe (NOT a C2 upper bound).

Honest framing (methodological correction 2026-07-14): this measures
    AUC( f_frozen( M(x_target) ) )
for a FROZEN source-only classifier and a fixed moment transform M. C2 instead
measures AUC( f_path-trained(x_target) ) — a different classifier, training and
transform direction; there is NO upper-bound relation. A null here only says
THIS intervention does not help the frozen classifier.

Arms (frozen classifier, DIAGNOSTIC target set; the locked final test is never
read unless --i_understand_touch_locked_test is passed):
  direct        : target image as-is
  moment_global : each target img affine-matched to ONE pooled source-train
                  global (mean,std)
  moment_bank   : each target img matched to every source-train per-case
                  (mean,std) (bank_all = all K cases), classifier probs
                  AVERAGED over the bank -> deterministic prediction

Reported: three paired case-level bootstrap CIs on SHARED resample indices:
  bank-direct, bank-global, global-direct  (so "bank vs global" has its own CI).

Validity: --self_test checks the moment-bank and bootstrap branches (which the
source-val AUC gate does NOT cover); the source-val gate checks direct/load/
transform. Both must pass before the probe is trusted.

Decision rule (on bank_all - direct):
  Delta >= 0.01  -> worth ONE real breast C2 screen
  Delta <= 0     -> stop moment-path (low ROI, with knee)
  0 < Delta<0.01 -> very weak; run C2 only if its cost is minimal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

_TO_TENSOR = T.ToTensor()
_MIN_EFFECT = 0.01  # decision threshold on bank_all - direct


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gray01_resized(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    resize = T.Resize((int(image_size), int(image_size)), antialias=True)
    return resize(_TO_TENSOR(img)).clamp(0.0, 1.0)


def to_clf(gray01: torch.Tensor) -> torch.Tensor:
    return gray01.repeat(3, 1, 1) * 2.0 - 1.0


def moment_match01(gray01: torch.Tensor, s_mean: float, s_std: float, eps: float = 1e-6) -> torch.Tensor:
    m = gray01.mean()
    s = gray01.std(unbiased=False).clamp_min(eps)
    return ((gray01 - m) / s * float(s_std) + float(s_mean)).clamp(0.0, 1.0)


def resolve(raw: str, parent: Path) -> Path:
    p = Path(str(raw).strip())
    return p if p.exists() else (parent / p)


def source_stats(csv: Path, image_col: str, image_size: int, seed: int, max_pool: int) -> Dict:
    df = pd.read_csv(csv)
    parent = Path(csv).parent
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    idx = idx[: min(max_pool, len(idx))]
    s = ss = cnt = 0.0
    per_mean: List[float] = []
    per_std: List[float] = []
    for i in idx:
        g = load_gray01_resized(resolve(str(df.iloc[i][image_col]), parent), image_size)
        s += float(g.sum()); ss += float((g * g).sum()); cnt += float(g.numel())
        per_mean.append(float(g.mean())); per_std.append(float(g.std(unbiased=False)))
    g_mean = s / cnt
    g_std = float(np.sqrt(max(ss / cnt - g_mean * g_mean, 1e-8)))
    return {"global_mean": float(g_mean), "global_std": float(g_std),
            "perimg_mean": per_mean, "perimg_std": per_std, "n_pool": len(idx),
            "xcase_sd_of_means": float(np.std(per_mean))}


@torch.no_grad()
def score_arms(model, csv: Path, image_col: str, label_col: str, case_col,
               image_size: int, glob: Tuple[float, float], bank: List[Tuple[float, float]],
               device, var_batch: int = 256) -> pd.DataFrame:
    df = pd.read_csv(csv)
    parent = Path(csv).parent
    rows: List[Dict] = []
    for i in range(len(df)):
        g = load_gray01_resized(resolve(str(df.iloc[i][image_col]), parent), image_size)
        variants = [to_clf(g), to_clf(moment_match01(g, *glob))]
        for (m, sd) in bank:
            variants.append(to_clf(moment_match01(g, m, sd)))
        probs: List[float] = []
        for j in range(0, len(variants), var_batch):
            x = torch.stack(variants[j:j + var_batch]).to(device)
            probs.extend(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy().tolist())
        probs = np.asarray(probs)
        case_id = str(df.iloc[i][case_col]) if (case_col and case_col in df.columns) else str(i)
        rows.append({"case_id": case_id, "label": int(df.iloc[i][label_col]),
                     "prob_direct": float(probs[0]), "prob_moment_global": float(probs[1]),
                     "prob_moment_bank": float(np.mean(probs[2:])) if len(probs) > 2 else float("nan"),
                     "tgt_mean": float(g.mean()), "tgt_std": float(g.std(unbiased=False))})
    return pd.DataFrame(rows)


def case_reduce(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    agg = {c: (c, "mean") for c in cols}
    g = df.groupby("case_id", sort=False).agg(label=("label", "first"), **agg).reset_index()
    y = g["label"].to_numpy(int)
    return y, {c: g[c].to_numpy(float) for c in cols}


def three_contrast_bootstrap(y, p_dir, p_glob, p_bank, n_boot, seed) -> Dict:
    """Shared-index paired bootstrap for bank-direct, bank-global, global-direct."""
    rng = np.random.RandomState(seed); n = len(y)
    d_bd, d_bg, d_gd = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n); yl = y[idx]
        if yl.min() == yl.max():
            continue
        a_dir = roc_auc_score(yl, p_dir[idx])
        a_glob = roc_auc_score(yl, p_glob[idx])
        a_bank = roc_auc_score(yl, p_bank[idx])
        d_bd.append(a_bank - a_dir); d_bg.append(a_bank - a_glob); d_gd.append(a_glob - a_dir)

    def ci(a):
        a = np.asarray(a)
        return {"lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)),
                "boot_mean": float(a.mean()), "kept": int(len(a))}
    return {"bank_minus_direct": ci(d_bd), "bank_minus_global": ci(d_bg), "global_minus_direct": ci(d_gd)}


def decide_bank(delta: float) -> str:
    if delta >= _MIN_EFFECT:
        return f"bank_all Delta={delta:+.4f} >= 0.01 -> worth ONE real breast C2 screen"
    if delta <= 0:
        return f"bank_all Delta={delta:+.4f} <= 0 -> stop moment-path (low ROI, with knee)"
    return f"bank_all Delta={delta:+.4f} in (0,0.01) -> very weak; C2 only if cost minimal"


def run_self_test(model, csv, image_col, label_col, image_size, glob, device) -> None:
    """Validate the moment-bank and bootstrap branches (src-val gate does not)."""
    df = pd.read_csv(csv); parent = Path(csv).parent
    g = load_gray01_resized(resolve(str(df.iloc[0][image_col]), parent), image_size)
    # (1) matching an image to its OWN moments is (near) identity
    self_mm = moment_match01(g, float(g.mean()), float(g.std(unbiased=False)))
    id_err = float((self_mm - g).abs().max())
    assert id_err < 1e-4, f"self-moment-match not identity: {id_err}"
    # (2) a 1-style bank == global arm, exactly (indexing/averaging correctness)
    d1 = score_arms(model, csv, image_col, label_col, None, image_size, glob, [glob], device)
    bank_eq = float((d1["prob_moment_bank"] - d1["prob_moment_global"]).abs().max())
    assert bank_eq < 1e-6, f"1-style bank != global arm: {bank_eq}"
    # (3) bootstrap of identical arms -> exactly zero deltas
    y, cols = case_reduce(d1, ["prob_direct"])
    bt = three_contrast_bootstrap(y, cols["prob_direct"], cols["prob_direct"], cols["prob_direct"], 200, 0)
    zero = max(abs(bt["bank_minus_direct"]["lo"]), abs(bt["bank_minus_direct"]["hi"]))
    assert zero == 0.0, f"self-bootstrap not zero: {zero}"
    print(f"[self_test] OK  identity_err={id_err:.2e}  bank==global_err={bank_eq:.2e}  self_boot={zero}")


def main() -> None:
    ap = argparse.ArgumentParser("b': fixed moment intervention probe with bank_all + 3 paired CIs")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--source_train_csv", type=Path, required=True)
    ap.add_argument("--source_val_csv", type=Path, required=True)
    ap.add_argument("--diag_csv", type=Path, required=True, help="target diagnostic (train+valid); never the locked test")
    ap.add_argument("--expect_srcval_auc", type=float, required=True)
    ap.add_argument("--locked_test_csv", type=Path, default=None)
    ap.add_argument("--i_understand_touch_locked_test", action="store_true")
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="imagenet")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--image_col", type=str, default="image_path")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--case_col", type=str, default="case_id")
    ap.add_argument("--max_pool", type=int, default=100000, help="cap source styles for bank_all")
    ap.add_argument("--var_batch", type=int, default=256)
    ap.add_argument("--srcval_tol", type=float, default=0.01)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.locked_test_csv is not None and not args.i_understand_touch_locked_test:
        print("[guard] locked test given without explicit flag -> NOT read.")
        args.locked_test_csv = None

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
    ck_hash = sha256_file(args.classifier)
    diag_hash = sha256_file(args.diag_csv)
    print(f"[load] {args.classifier}  sha256={ck_hash[:12]}  diag_sha256={diag_hash[:12]}")

    st = source_stats(args.source_train_csv, args.image_col, args.image_size, args.seed, args.max_pool)
    glob = (st["global_mean"], st["global_std"])
    bank = list(zip(st["perimg_mean"], st["perimg_std"]))  # bank_all = every source case
    print(f"[source] pool_n={st['n_pool']} global=({glob[0]:.4f},{glob[1]:.4f}) bank_all_K={len(bank)}")

    if args.self_test:
        run_self_test(model, args.diag_csv, args.image_col, args.label_col, args.image_size, glob, device)

    # validity gate (source val; no target test touched)
    sv = score_arms(model, args.source_val_csv, args.image_col, args.label_col, args.case_col,
                    args.image_size, glob, [], device, args.var_batch)
    ysv, csv_cols = case_reduce(sv, ["prob_direct"])
    srcval_direct = float(roc_auc_score(ysv, csv_cols["prob_direct"]))
    srcval_ok = abs(srcval_direct - args.expect_srcval_auc) <= args.srcval_tol
    print(f"[validity] src_val direct AUC={srcval_direct:.4f} vs {args.expect_srcval_auc} "
          f"-> {'OK' if srcval_ok else '!!! VOID'}")

    # diagnostic arms
    df = score_arms(model, args.diag_csv, args.image_col, args.label_col, args.case_col,
                    args.image_size, glob, bank, device, args.var_batch)
    cols = ["prob_direct", "prob_moment_global", "prob_moment_bank"]
    y, cd = case_reduce(df, cols)
    auc = {c: float(roc_auc_score(y, cd[c])) for c in cols}
    bt = three_contrast_bootstrap(y, cd["prob_direct"], cd["prob_moment_global"], cd["prob_moment_bank"],
                                  args.n_boot, args.seed)
    delta_bank = auc["prob_moment_bank"] - auc["prob_direct"]

    # correct standardized appearance gap (pooled cross-case SD of per-image means)
    tgt_mean = df["tgt_mean"].to_numpy(float)
    xsd = float(np.sqrt((st["xcase_sd_of_means"] ** 2 + float(np.std(tgt_mean)) ** 2) / 2.0))
    gap_mean = abs(float(np.mean(st["perimg_mean"])) - float(tgt_mean.mean()))

    out = {
        "checkpoint_sha256": ck_hash, "diag_manifest_sha256": diag_hash,
        "source_train_csv": str(args.source_train_csv), "diag_csv": str(args.diag_csv),
        "n_cases": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "bank_all_K": len(bank),
        "gap_mean_in_xcase_SD": round(gap_mean / (xsd + 1e-9), 3),
        "srcval_direct_auc": round(srcval_direct, 4), "srcval_reproduced": bool(srcval_ok),
        "auc_direct": round(auc["prob_direct"], 4),
        "auc_moment_global": round(auc["prob_moment_global"], 4),
        "auc_moment_bank_all": round(auc["prob_moment_bank"], 4),
        "CI_bank_minus_direct": {k: round(v, 4) for k, v in bt["bank_minus_direct"].items()},
        "CI_bank_minus_global": {k: round(v, 4) for k, v in bt["bank_minus_global"].items()},
        "CI_global_minus_direct": {k: round(v, 4) for k, v in bt["global_minus_direct"].items()},
        "decision": ("N/A (VOID src-val)" if not srcval_ok else decide_bank(delta_bank)),
    }

    print("\n=========== b' moment_bank_all + 3 paired CIs ===========")
    print(f"  validity src_val direct={srcval_direct:.4f} -> {'valid' if srcval_ok else 'VOID'}")
    print(f"  diag n={out['n_cases']} (pos {out['n_pos']}/neg {out['n_neg']})  gap={out['gap_mean_in_xcase_SD']} xcaseSD  bank_all_K={len(bank)}")
    print(f"  AUC  direct={auc['prob_direct']:.4f}  global={auc['prob_moment_global']:.4f}  bank_all={auc['prob_moment_bank']:.4f}")
    for name, key in [("bank - direct", "bank_minus_direct"), ("bank - global", "bank_minus_global"), ("global - direct", "global_minus_direct")]:
        c = bt[key]
        print(f"  {name:16s} boot_mean={c['boot_mean']:+.4f}  CI[{c['lo']:+.4f}, {c['hi']:+.4f}]")
    print(f"  DECISION: {out['decision']}")
    print("=========================================================\n")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False); print(f"[write] {args.out_csv}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8"); print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
