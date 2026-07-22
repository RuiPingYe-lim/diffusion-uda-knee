"""Analyze the matched three-arm gate: primary = true_fakes - repeat_before.

Reads gate3arm/<control>_s<seed>/percase.csv, aggregates prob by case_id, and:
  * verifies the case set is IDENTICAL across all controls/seeds (else abort);
  * per (control, seed): case-level AUC;
  * PRIMARY contrast true_fakes - repeat_before with a TWO-LEVEL (seed x case)
    paired bootstrap 95% CI (resample seeds w/ replacement, then cases w/ shared
    indices for both arms);
  * secondary contrasts repeat_before - direct and true_fakes - direct.

Usage: python analyze_gate.py <gate3arm_dir> [n_boot]
"""
import sys, glob, os, re
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/breast/exp/gate3arm"
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
CONTROLS = ["direct", "repeat_before", "true_fakes"]


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


def load():
    # data[control][seed] = DataFrame(case_id,label,prob) aggregated by case
    data = {c: {} for c in CONTROLS}
    for f in sorted(glob.glob(os.path.join(ROOT, "*_s*", "percase.csv"))):
        m = re.search(r"/(direct|repeat_before|true_fakes)_s(\d+)/percase.csv$", f.replace("\\", "/"))
        if not m:
            continue
        ctrl, seed = m.group(1), int(m.group(2))
        df = pd.read_csv(f)
        cg = df.groupby("case_id", sort=True).agg(label=("label", "first"),
                                                  prob=("prob_positive", "mean")).reset_index()
        data[ctrl][seed] = cg
    return data


def main():
    data = load()
    seeds = sorted(set.intersection(*[set(data[c].keys()) for c in CONTROLS]) or set())
    if not seeds:
        print("!! no complete (control x seed) set found under", ROOT); return
    # verify identical case set + label alignment across everything
    ref = data[CONTROLS[0]][seeds[0]][["case_id", "label"]].reset_index(drop=True)
    for c in CONTROLS:
        for s in seeds:
            cur = data[c][s][["case_id", "label"]].reset_index(drop=True)
            if not cur.equals(ref):
                raise SystemExit(f"!! case_id/label mismatch in {c} seed {s} -- arms not comparable")
    y = ref["label"].to_numpy(int)
    n = len(y)
    # per (control, seed) AUC
    print(f"cases={n} pos={(y==1).sum()} neg={(y==0).sum()} seeds={seeds}")
    P = {c: {s: data[c][s]["prob"].to_numpy(float) for s in seeds} for c in CONTROLS}
    print("\n per-arm case-AUC (mean over seeds):")
    for c in CONTROLS:
        a = [auc(y, P[c][s]) for s in seeds]
        print(f"   {c:14s} {np.mean(a):.4f} ± {np.std(a):.4f}   per-seed={[round(x,4) for x in a]}")

    def two_level(cA, cB):
        rng = np.random.RandomState(0); deltas = []
        for _ in range(N_BOOT):
            ss = rng.choice(seeds, len(seeds), replace=True)
            idx = rng.randint(0, n, n)
            yl = y[idx]
            if yl.min() == yl.max():
                continue
            d = np.mean([auc(yl, P[cA][s][idx]) - auc(yl, P[cB][s][idx]) for s in ss])
            deltas.append(d)
        a = np.asarray(deltas)
        pt = np.mean([auc(y, P[cA][s]) - auc(y, P[cB][s]) for s in seeds])
        return pt, float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    print("\n paired contrasts (two-level seed×case bootstrap 95% CI):")
    for cA, cB in [("true_fakes", "repeat_before"), ("repeat_before", "direct"), ("true_fakes", "direct")]:
        pt, lo, hi = two_level(cA, cB)
        star = "  <-- PRIMARY" if (cA, cB) == ("true_fakes", "repeat_before") else ""
        verdict = "CI>0" if lo > 0 else ("CI<0" if hi < 0 else "CI spans 0")
        print(f"   {cA} - {cB:14s} Δ={pt:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  {verdict}{star}")


if __name__ == "__main__":
    main()
