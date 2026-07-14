"""
评估 UNSB 翻译图的目标域 AUC(翻译-only)。

UNSB(外部仓库 cyclomon/UNSB)把目标域图翻译成源域风格,输出多档 NFE 步数的结果
(fake_1 / fake_3 / fake_5,步数越多风格越强)。本脚本用**源域分类器**对这些翻译图打分,
与"直接迁移(原图,不翻译)"对比,验证翻译是否提升判别性能。

用法:
  python eval_unsb_translation.py \
    --clf_ckpt <source_classifier.pt> \
    --label_csv <target_test.csv>            # 含 case_id,label \
    --before_dir <目标原图目录>              # 每个 case 一张 <case_id>.png(直接迁移用) \
    --unsb_dir <UNSB results .../images>     # 含 real/ fake_1/ fake_3/ fake_5/ \
    --fakes fake_1,fake_3,fake_5
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
import torchvision.transforms as T
from PIL import Image

# 复用仓库里的模型构建器
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_existing_classifier_on_csv import build_model


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    o = np.argsort(-p); ys = y[o]
    return float(np.trapz(np.r_[0, np.cumsum(ys == 1) / pos], np.r_[0, np.cumsum(ys == 0) / neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True)
    ap.add_argument("--backbone", default="custom_resnet50_space")
    ap.add_argument("--label_csv", required=True, help="含 case_id,label")
    ap.add_argument("--before_dir", required=True, help="目标原图目录(直接迁移对照)")
    ap.add_argument("--unsb_dir", required=True, help="UNSB results .../images(含 real/ fake_*/)")
    ap.add_argument("--fakes", default="fake_1,fake_3,fake_5")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = T.Compose([
        T.ToTensor(), T.Resize((224, 224), antialias=True),
        T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    clf = build_model(a.backbone, 2, "none", dev)
    ck = torch.load(a.clf_ckpt, map_location=dev, weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck))
    clf.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    clf.to(dev).eval()

    lab = pd.read_csv(a.label_csv)

    @torch.no_grad()
    def ev(get_path):
        P, Y = [], []
        for _, r in lab.iterrows():
            p = get_path(str(r["case_id"]))
            x = tf(Image.open(p).convert("L")).unsqueeze(0).to(dev)
            P.append(float(torch.softmax(clf(x), 1)[0, 1]))
            Y.append(int(r["label"]))
        return auc(Y, P)

    print("直接迁移(原图)           AUC=%.3f" % ev(lambda c: os.path.join(a.before_dir, f"{c}.png")))
    for k in a.fakes.split(","):
        print("UNSB %-8s 翻译图         AUC=%.3f" % (k, ev(lambda c, k=k: os.path.join(a.unsb_dir, k, f"{c}.png"))))


if __name__ == "__main__":
    main()
