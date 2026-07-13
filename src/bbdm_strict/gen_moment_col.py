"""
给已有的融合配对 csv 增加一列 moment_png：对 before 图做矩匹配(对齐到源域均值/对比度)。
用于把 0.795 的矩匹配图并入融合分类器。
"""
import argparse, os
import numpy as np, pandas as pd
from PIL import Image


def src_stats(paths, size=128):
    s = ss = n = 0.0
    for p in paths:
        g = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float32) / 255.0
        s += g.sum(); ss += (g ** 2).sum(); n += g.size
    m = s / n; return float(m), float(np.sqrt(max(ss / n - m ** 2, 1e-8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_ref", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    srcdf = pd.read_csv(a.source_csv)
    rng = np.random.RandomState(0)
    sp = srcdf.iloc[rng.choice(len(srcdf), min(a.n_ref, len(srcdf)), replace=False)]["image_path"].tolist()
    SM, SS = src_stats(sp)

    df = pd.read_csv(a.pairs_csv)
    outpaths = []
    for i, r in df.iterrows():
        g = np.asarray(Image.open(r["before_png"]).convert("L"), dtype=np.float32) / 255.0
        m, s = g.mean(), g.std() + 1e-6
        gm = np.clip((g - m) / s * SS + SM, 0, 1)
        op = os.path.join(a.out_dir, "%06d.png" % i)
        Image.fromarray((gm * 255).astype(np.uint8), "L").save(op)
        outpaths.append(op)
    df["moment_png"] = outpaths
    df.to_csv(a.pairs_csv, index=False)  # 原地加列
    print("added moment_png to", a.pairs_csv, "| source mean=%.3f std=%.3f" % (SM, SS))


if __name__ == "__main__":
    main()
