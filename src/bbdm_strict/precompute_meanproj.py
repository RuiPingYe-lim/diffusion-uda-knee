"""
Mean-projection preprocessing (paper-1 aligned pipeline).

Convert each case's npy volume -> mean projection over slices (proj="mean") ->
per-image percentile intensity normalisation (intensity_norm01) -> one grayscale
PNG per case, and write a CSV (image_path,label,case_id). Mirrors the first paper's
data.py so both papers share the same mean-projection representation.

Usage:
  python precompute_meanproj.py --npy_root <.../knees_npy> --plane sagittal \
     --csv <case_id,label csv> --out_dir <png dir> --out_csv <out csv>
"""
import os, glob, argparse
import numpy as np, pandas as pd
from PIL import Image


def list_stem_lengths(dirpath):
    stems = []
    for f in glob.glob(os.path.join(dirpath, "*.npy")):
        stem = os.path.splitext(os.path.basename(f))[0]
        stems.append(len(stem.split("-")[0]))
    return sorted(set(stems))


def find_file(npy_root, plane, cid, zero_pad=0):
    base = os.path.join(npy_root, plane) if plane else npy_root
    cid = str(cid)
    pads = [4, 6, 8, 10]
    if zero_pad and int(zero_pad) not in pads:
        pads = [int(zero_pad)] + pads
    cands = [cid] + ([cid.zfill(z) for z in pads] if cid.isdigit() else [])
    cands = list(dict.fromkeys(cands))
    for c in cands:
        f = os.path.join(base, f"{c}.npy")
        if os.path.isfile(f):
            return f
    if cid.isdigit():
        for z in list_stem_lengths(base):
            f = os.path.join(base, f"{cid.zfill(z)}.npy")
            if os.path.isfile(f):
                return f
    return None


def intensity_norm01(arr):
    vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
    if vmax > vmin:
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr.astype(np.float32)


def volume_to_2d(arr, proj="mean"):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        s, h, w = arr.shape[0], arr.shape[-2], arr.shape[-1]
        looks_shw = s >= 4 and h >= 32 and w >= 32
        looks_hwc = arr.shape[-1] in (1, 3) and arr.shape[0] == h and arr.shape[1] == w
        if looks_shw and not looks_hwc:
            img = arr.mean(axis=0) if proj == "mean" else arr.max(axis=0)
        else:
            c = arr.shape[-1]
            img = arr[..., 0] if c == 1 else arr.mean(axis=-1)
    else:
        img = arr[arr.shape[0] // 2]
    return intensity_norm01(img.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_root", required=True)
    ap.add_argument("--plane", default="sagittal")
    ap.add_argument("--csv", required=True, help="case_id,label")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv)
    rows, miss = [], 0
    for _, r in df.iterrows():
        cid = str(r["case_id"]); lab = int(r["label"])
        f = find_file(a.npy_root, a.plane, cid)
        if f is None:
            miss += 1; continue
        img = volume_to_2d(np.load(f, allow_pickle=False), "mean")
        p = os.path.join(a.out_dir, f"{cid}.png")
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "L").save(p)
        rows.append({"image_path": p, "label": lab, "case_id": cid})
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    print(f"done {a.csv}: {len(rows)} written, {miss} missing -> {a.out_csv}")


if __name__ == "__main__":
    main()
