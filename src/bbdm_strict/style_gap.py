"""
量化 源域(MRNet) 与 目标域(KneeMRI) 的"风格差异在哪里"。
纯 CPU：强度直方图 + 径向平均功率谱(FFT)。
输出：数值 + 一张对比图 style_gap.png。
不占 GPU，可与其它任务并行。
"""
import argparse
import numpy as np, pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_gray(path, size):
    im = Image.open(path).convert("L").resize((size, size))
    return np.asarray(im, dtype=np.float32) / 255.0


def radial_power(img):
    # 2D FFT -> 幅度谱 -> 径向平均
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    mag = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), mag.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def sample_paths(csv, n, seed=0):
    df = pd.read_csv(csv)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx]["image_path"].tolist()


def summarize(paths, size):
    hist = np.zeros(256)
    means, stds, gradmag = [], [], []
    ps = None; nps = 0
    for p in paths:
        g = load_gray(p, size)
        u8 = (g * 255).astype(int)
        hist += np.bincount(u8.ravel(), minlength=256)
        means.append(g.mean()); stds.append(g.std())
        gy, gx = np.gradient(g)
        gradmag.append(np.sqrt(gx ** 2 + gy ** 2).mean())
        rp = radial_power(g)
        if ps is None: ps = np.zeros_like(rp)
        L = min(len(ps), len(rp)); ps[:L] += rp[:L]; nps += 1
    hist /= hist.sum()
    return {
        "mean": float(np.mean(means)), "std_intra": float(np.mean(stds)),
        "bright_std_across": float(np.std(means)),
        "gradmag": float(np.mean(gradmag)),
        "hist": hist, "ps": ps / nps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="/root/autodl-tmp/knee_allslices/style_gap.png")
    a = ap.parse_args()

    S = summarize(sample_paths(a.source_csv, a.n), a.size)
    Tg = summarize(sample_paths(a.target_csv, a.n), a.size)

    print("== 强度/锐度 ==")
    print("           source(MRNet)   target(KneeMRI)")
    print("亮度均值    %.3f            %.3f" % (S["mean"], Tg["mean"]))
    print("单图对比度  %.3f            %.3f" % (S["std_intra"], Tg["std_intra"]))
    print("梯度幅值    %.4f           %.4f   (越大越锐/高频越多)" % (S["gradmag"], Tg["gradmag"]))

    # 频段能量占比：低/中/高
    def band(ps):
        n = len(ps); lo = ps[1:n//6].sum(); mi = ps[n//6:n//2].sum(); hi = ps[n//2:].sum()
        tot = lo + mi + hi
        return lo/tot, mi/tot, hi/tot
    sl, sm, sh = band(S["ps"]); tl, tm, th = band(Tg["ps"])
    print("\n== 频段能量占比(低/中/高) ==")
    print("source  低=%.3f 中=%.3f 高=%.3f" % (sl, sm, sh))
    print("target  低=%.3f 中=%.3f 高=%.3f" % (tl, tm, th))

    # 画图
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(S["hist"], label="source MRNet", color="tab:blue")
    ax[0].plot(Tg["hist"], label="target KneeMRI", color="tab:red")
    ax[0].set_title("Intensity histogram"); ax[0].set_xlabel("pixel value"); ax[0].legend()

    fs = np.arange(1, len(S["ps"]))
    ax[1].loglog(fs, S["ps"][1:], label="source MRNet", color="tab:blue")
    ax[1].loglog(fs, Tg["ps"][1:], label="target KneeMRI", color="tab:red")
    ax[1].set_title("Radial power spectrum"); ax[1].set_xlabel("spatial freq (radius)"); ax[1].set_ylabel("power"); ax[1].legend()

    L = min(len(S["ps"]), len(Tg["ps"]))
    ratio = np.log10((Tg["ps"][1:L] + 1e-12) / (S["ps"][1:L] + 1e-12))
    ax[2].plot(np.arange(1, L), ratio, color="tab:purple")
    ax[2].axhline(0, color="gray", ls="--", lw=0.8)
    ax[2].set_title("log10(target/source) power  per freq")
    ax[2].set_xlabel("spatial freq (radius)"); ax[2].set_ylabel("log ratio (>0: target more)")
    plt.tight_layout(); plt.savefig(a.out, dpi=110)
    print("\nsaved figure:", a.out)


if __name__ == "__main__":
    main()
