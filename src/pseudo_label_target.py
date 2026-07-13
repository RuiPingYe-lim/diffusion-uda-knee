"""
用源域分类器(best_checkpoint)对目标域图打伪标签。
输出 CSV: 把 label 列替换成 预测类别(伪标签)，保留 image_path/case_id/slice_idx，
额外写 true_label_bak(真标签，仅诊断) 和 pseudo_prob。
这样后续 BBDM label_random 配对读到的 "label" 就是伪标签，全程不碰真标签。
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from eval_existing_classifier_on_csv import (
    build_model, detect_column, resolve_path, array_to_pil, IMAGE_COL_CANDIDATES,
)


def eval_tf(sz):
    return T.Compose([
        T.ToTensor(), T.Resize((sz, sz), antialias=True),
        T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


class DS(Dataset):
    def __init__(self, df, image_col, csv_parent, sz):
        self.df, self.ic, self.cp, self.tf = df, image_col, csv_parent, eval_tf(sz)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = resolve_path(str(r[self.ic]), None, self.cp)
        img = array_to_pil(np.load(p)) if str(p).lower().endswith(".npy") else Image.open(p).convert("L")
        return self.tf(img), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.weights, map_location=dev, weights_only=False)
    sz = int(ck.get("args", {}).get("image_size", 224)) if isinstance(ck, dict) else 224
    model = build_model(a.backbone, 2, "none", dev)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    print("loaded  missing=%d unexpected=%d image_size=%d" % (len(res.missing_keys), len(res.unexpected_keys), sz))
    model.eval()

    df = pd.read_csv(a.in_csv)
    ic = detect_column(df, None, IMAGE_COL_CANDIDATES, required=True)
    dl = DataLoader(DS(df, ic, Path(a.in_csv).parent, sz), batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    preds = np.zeros(len(df), dtype=int)
    probs = np.zeros(len(df), dtype=float)
    with torch.no_grad():
        for xb, idx in dl:
            p = torch.softmax(model(xb.to(dev)), 1)[:, 1].cpu().numpy()
            idx = idx.numpy()
            probs[idx] = p
            preds[idx] = (p >= 0.5).astype(int)

    out = df.copy()
    if "label" in out.columns:
        true = pd.to_numeric(out["label"], errors="coerce")
        m = true.isin([0, 1])
        if m.any():
            acc = float((preds[m.values] == true[m].astype(int).values).mean())
            print("[DIAGNOSTIC ONLY] pseudo vs TRUE label acc = %.4f (NOT used for pairing)" % acc)
        out["true_label_bak"] = out["label"]
    out["label"] = preds
    out["pseudo_prob"] = probs
    out.to_csv(a.out_csv, index=False)
    u, c = np.unique(preds, return_counts=True)
    print("pseudo-label dist:", dict(zip(u.tolist(), c.tolist())))
    print("saved:", a.out_csv)


if __name__ == "__main__":
    main()
