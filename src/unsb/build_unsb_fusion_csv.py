"""
从 UNSB 翻译输出构造交叉注意力融合分类器的配对 CSV。

融合思路(见 ../bbdm_strict/fusion_classifier.py):原图(before)作为 query,
交叉注意力融合多张 UNSB 翻译视图(fake_1/3/5)。分类不只依赖翻译图,而是"原图 + 翻译"。

对源域(有标签,用于训融合分类器)和目标域(用于评估)各构造一份 CSV:
  列 = before_png, f1, f3, f5, label, case_id

用法:
  python build_unsb_fusion_csv.py \
    --label_csv <某域的 case_id,label csv> \
    --before_dir <该域原图目录> \
    --unsb_dir  <该域 UNSB 翻译 results .../images> \
    --out_csv   <输出 csv>
注:源域原图过 UNSB 时,把源域图当作输入 A 跑一遍 UNSB(见 scripts/run_unsb_fusion.sh)。
"""
import argparse, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_csv", required=True, help="含 case_id,label")
    ap.add_argument("--before_dir", required=True, help="原图目录(每 case 一张 <case_id>.png)")
    ap.add_argument("--unsb_dir", required=True, help="UNSB results .../images(含 fake_1/3/5)")
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.label_csv)
    rows, miss = [], 0
    for _, r in df.iterrows():
        c = str(r["case_id"])
        bp = os.path.join(a.before_dir, f"{c}.png")
        f1 = os.path.join(a.unsb_dir, "fake_1", f"{c}.png")
        f3 = os.path.join(a.unsb_dir, "fake_3", f"{c}.png")
        f5 = os.path.join(a.unsb_dir, "fake_5", f"{c}.png")
        if not all(os.path.isfile(p) for p in [bp, f1, f3, f5]):
            miss += 1
            continue
        rows.append({"before_png": bp, "f1": f1, "f3": f3, "f5": f5,
                     "label": int(r["label"]), "case_id": c})
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    print(f"{a.out_csv}: 写出 {len(rows)}, 缺失 {miss}")


if __name__ == "__main__":
    main()
