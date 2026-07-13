import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eval_existing_classifier_on_csv import (
    CSVImageDataset,
    LABEL_COL_CANDIDATES,
    build_model,
    detect_column,
    load_weights,
    run_inference,
    set_seed,
)


PAIR_LABEL_CANDIDATES = ["target_label", "label", "target", "y", "class", "cls"]


def _auc_roc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    ys = y_true[order]
    tp_cum = np.cumsum(ys == 1)
    fp_cum = np.cumsum(ys == 0)
    tpr = np.concatenate([[0.0], tp_cum / max(pos, 1)])
    fpr = np.concatenate([[0.0], fp_cum / max(neg, 1)])
    return float(np.trapz(tpr, fpr))


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= float(thr)).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(len(y_true), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 0.0 if (prec + sens) == 0 else (2.0 * prec * sens) / (prec + sens)
    balacc = 0.5 * (sens + spec)
    auc = _auc_roc_binary(y_true, y_prob)

    return {
        "AUC": float(auc),
        "ACC": float(acc),
        "SPEC": float(spec),
        "SENS": float(sens),
        "PREC": float(prec),
        "F1": float(f1),
        "BALACC": float(balacc),
        "N": int(len(y_true)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def _candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    uniq = np.unique(np.clip(y_prob.astype(np.float64), 0.0, 1.0))
    base = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    cands = np.unique(np.concatenate([uniq, base]))
    return cands


def select_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    cands = _candidate_thresholds(y_prob)
    rows: List[Dict[str, float]] = []

    for t in cands:
        m = metrics_at_threshold(y_true, y_prob, float(t))
        m["threshold"] = float(t)
        m["YoudenJ"] = float(m["SENS"] + m["SPEC"] - 1.0)
        rows.append(m)

    sweep = pd.DataFrame(rows)

    idx_j = int(sweep["YoudenJ"].idxmax())
    idx_f1 = int(sweep["F1"].idxmax())
    idx_ba = int(sweep["BALACC"].idxmax())

    picks = [
        ("YoudenJ", sweep.iloc[idx_j]),
        ("BestF1", sweep.iloc[idx_f1]),
        ("BestBalAcc", sweep.iloc[idx_ba]),
    ]

    fixed = metrics_at_threshold(y_true, y_prob, 0.5)
    fixed_row = {"op": "Fixed0.5", "threshold": 0.5, **fixed, "YoudenJ": fixed["SENS"] + fixed["SPEC"] - 1.0}

    out_rows: List[Dict[str, float]] = []
    for name, r in picks:
        out_rows.append(
            {
                "op": name,
                "threshold": float(r["threshold"]),
                "AUC": float(r["AUC"]),
                "ACC": float(r["ACC"]),
                "SPEC": float(r["SPEC"]),
                "SENS": float(r["SENS"]),
                "PREC": float(r["PREC"]),
                "F1": float(r["F1"]),
                "BALACC": float(r["BALACC"]),
                "YoudenJ": float(r["YoudenJ"]),
                "N": int(r["N"]),
            }
        )
    out_rows.append(fixed_row)

    return pd.DataFrame(out_rows)


def make_loader(
    csv_path: Path,
    image_col: Optional[str],
    label_col: Optional[str],
    volume_col: Optional[str],
    prefer_volume_col: bool,
    resize: int,
    root_dir: Optional[Path],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = CSVImageDataset(
        csv_path=csv_path,
        image_col=image_col,
        label_col=label_col,
        volume_col=volume_col,
        prefer_volume_col=prefer_volume_col,
        resize=resize,
        root_dir=root_dir,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def collect_probs_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    probs, _, _, labels, has_label = run_inference(
        model=model,
        loader=loader,
        device=device,
        temperature=temperature,
        threshold=0.5,
    )
    valid = has_label & np.isin(labels, [0, 1])
    if not np.any(valid):
        raise ValueError("No valid binary labels {0,1} found for this dataset.")
    return probs[valid], labels[valid].astype(int)


def eval_with_threshold_table(
    split_name: str,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    th_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    auc_all = _auc_roc_binary(y_true, y_prob)

    for _, r in th_df.iterrows():
        op = str(r["op"])
        thr = float(r["threshold"])
        m = metrics_at_threshold(y_true, y_prob, thr)
        m["AUC"] = float(auc_all)
        rows.append(
            {
                "split": split_name,
                "op": op,
                "threshold": thr,
                "AUC": float(m["AUC"]),
                "ACC": float(m["ACC"]),
                "SPEC": float(m["SPEC"]),
                "SENS": float(m["SENS"]),
                "PREC": float(m["PREC"]),
                "F1": float(m["F1"]),
                "BALACC": float(m["BALACC"]),
                "N": int(m["N"]),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Calibrate thresholds on source cache val, then evaluate before/translated")

    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--backbone", type=str, default="custom_resnet50_space")
    ap.add_argument("--pretrained", type=str, default="none")

    ap.add_argument("--source_val_csv", type=Path, required=True)
    ap.add_argument("--source_val_root", type=Path, default=None)
    ap.add_argument("--source_image_col", type=str, default=None)
    ap.add_argument("--source_label_col", type=str, default=None)
    ap.add_argument("--source_volume_col", type=str, default=None)
    ap.add_argument("--source_prefer_volume_col", type=int, default=0)

    ap.add_argument("--pairs_csv", type=Path, required=True)
    ap.add_argument("--pairs_root", type=Path, default=None)
    ap.add_argument("--before_col", type=str, default="before_png")
    ap.add_argument("--translated_col", type=str, default="translated_png")
    ap.add_argument("--pairs_label_col", type=str, default=None)

    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device.lower()) else "cpu")

    model = build_model(args.backbone, num_classes=2, pretrained=args.pretrained, device=device)
    load_weights(model, args.weights, device=device)
    model.to(device)
    model.eval()

    source_loader = make_loader(
        csv_path=args.source_val_csv,
        image_col=args.source_image_col,
        label_col=args.source_label_col,
        volume_col=args.source_volume_col,
        prefer_volume_col=bool(int(args.source_prefer_volume_col)),
        resize=int(args.resize),
        root_dir=args.source_val_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    src_prob, src_true = collect_probs_labels(model, source_loader, device, float(args.temperature))
    th_table = select_thresholds(src_true, src_prob)

    pairs_df = pd.read_csv(args.pairs_csv)
    pairs_label_col = detect_column(pairs_df, args.pairs_label_col, PAIR_LABEL_CANDIDATES, required=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_before_csv = args.pairs_csv.parent / "_tmp_before_eval.csv"
    temp_trans_csv = args.pairs_csv.parent / "_tmp_translated_eval.csv"
    pairs_root = args.pairs_root if args.pairs_root is not None else args.pairs_csv.parent

    pairs_df[[args.before_col, pairs_label_col]].rename(
        columns={args.before_col: "image_path", pairs_label_col: "label"}
    ).to_csv(temp_before_csv, index=False)
    pairs_df[[args.translated_col, pairs_label_col]].rename(
        columns={args.translated_col: "image_path", pairs_label_col: "label"}
    ).to_csv(temp_trans_csv, index=False)

    before_loader = make_loader(
        csv_path=temp_before_csv,
        image_col="image_path",
        label_col="label",
        volume_col=None,
        prefer_volume_col=False,
        resize=int(args.resize),
        root_dir=pairs_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    trans_loader = make_loader(
        csv_path=temp_trans_csv,
        image_col="image_path",
        label_col="label",
        volume_col=None,
        prefer_volume_col=False,
        resize=int(args.resize),
        root_dir=pairs_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    before_prob, before_true = collect_probs_labels(model, before_loader, device, float(args.temperature))
    trans_prob, trans_true = collect_probs_labels(model, trans_loader, device, float(args.temperature))

    before_metrics = eval_with_threshold_table("before_png", before_prob, before_true, th_table)
    trans_metrics = eval_with_threshold_table("translated_png", trans_prob, trans_true, th_table)
    all_metrics = pd.concat([before_metrics, trans_metrics], axis=0, ignore_index=True)

    th_path = args.out_dir / "ops_table_val.csv"
    metrics_path = args.out_dir / "target_before_translated_by_threshold.csv"
    before_prob_path = args.out_dir / "before_probs.csv"
    trans_prob_path = args.out_dir / "translated_probs.csv"

    th_table.to_csv(th_path, index=False)
    all_metrics.to_csv(metrics_path, index=False)

    pd.DataFrame({"prob_positive": before_prob, "label": before_true}).to_csv(before_prob_path, index=False)
    pd.DataFrame({"prob_positive": trans_prob, "label": trans_true}).to_csv(trans_prob_path, index=False)

    try:
        temp_before_csv.unlink(missing_ok=True)
        temp_trans_csv.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"[Done] Threshold table saved: {th_path}")
    print(f"[Done] Target eval table saved: {metrics_path}")
    print("[Source val thresholds]")
    print(th_table[["op", "threshold", "AUC", "ACC", "SPEC", "SENS", "PREC", "F1", "BALACC"]].to_string(index=False))
    print("[Target metrics by threshold]")
    print(all_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
