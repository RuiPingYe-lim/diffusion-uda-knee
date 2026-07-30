"""Provenance check: is the frozen source classifier valid on the UNSB rendering?

Result: it is NOT. The gate classifier scores raw BUSI (`busi_train.csv:image_path`,
128x128) at acc 0.984, but the UNSB input rendering (`fusion_train_busi.csv:before_png`,
256x256) at acc 0.578 -- malignant 27/32 wrong, though AUC stays 0.916. Since U1 =
G(before_png) lives in that same rendering, the frozen classifier cannot validly probe
translated images: margins measured that way are the known "erosion = style sensitivity"
artefact. Use a probe retrained on the translated rendering instead (v6_retrained_probe.py).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, os
from PIL import Image
from torchvision import transforms as T, models
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CK = "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt"
CLF_TF = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True),
                    T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
                    T.Normalize([0.5] * 3, [0.5] * 3)])


class SA(nn.Module):
    def __init__(s, d):
        super().__init__(); s.conv = nn.Conv2d(d, d, 1); s.soft = nn.Softmax(dim=2)

    def forward(s, x):
        a = s.conv(x); b, c, h, w = a.shape; a = s.soft(a.view(b, c, -1))
        m = a.amax(2, keepdim=True).clamp_min(1e-6); return x * (a / m).view(b, c, h, w)


class Net(nn.Module):
    def __init__(s, ck):
        super().__init__(); m = models.resnet50(weights=None)
        s.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        s.space_attn = SA(2048); s.avgpool = nn.AdaptiveAvgPool2d(1); s.classifier = nn.Linear(2048, 2)
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        for p, mod in {"stem.": s.stem, "space_attn.": s.space_attn, "classifier.": s.classifier}.items():
            mod.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)}, strict=False)

    @torch.no_grad()
    def forward(s, x):
        return s.classifier(s.avgpool(s.space_attn(s.stem(x))).flatten(1))


def evalcsv(net, df, col):
    P, Y = [], []
    for b0 in range(0, len(df), 16):
        r = df.iloc[b0:b0 + 16]
        xb = torch.stack([CLF_TF(Image.open(p).convert("L")) for p in r[col]]).to(DEV)
        P.append(F.softmax(net(xb), 1)[:, 1].cpu().numpy()); Y.append(r["label"].astype(int).values)
    p = np.concatenate(P); y = np.concatenate(Y); pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return roc_auc_score(y, p), accuracy_score(y, pred), (tp, tn, fp, fn), len(y)


net = Net(CK).to(DEV).eval()
bt = pd.read_csv("/root/autodl-tmp/breast/cache/busi_train.csv")
ft = pd.read_csv("/root/autodl-tmp/breast/cache/fusion_train_busi.csv")
print("busi_train cols", list(bt.columns), "n", len(bt))
print("fusion_train cols", list(ft.columns), "n", len(ft))

# stratified 64
rng = np.random.RandomState(0)
sub = pd.concat([bt[bt.label == c].iloc[rng.permutation((bt.label == c).sum())[:32]] for c in [0, 1]])
a, ac, cm, n = evalcsv(net, sub, "image_path")
print("\n[gate clf on busi_train image_path, 64]  AUC=%.4f acc=%.4f (tp,tn,fp,fn)=%s n=%d" % (a, ac, cm, n))

subf = pd.concat([ft[ft.label == c].iloc[rng.permutation((ft.label == c).sum())[:32]] for c in [0, 1]])
a, ac, cm, n = evalcsv(net, subf, "before_png")
print("[gate clf on fusion before_png, 64]      AUC=%.4f acc=%.4f (tp,tn,fp,fn)=%s n=%d" % (a, ac, cm, n))

# compare one example image_path vs before_png
def stat(p):
    im = np.array(Image.open(p).convert("L")).astype(np.float32)
    return im.shape, float(im.min()), float(im.max()), float(im.mean())


print("\nexample image_path[0]:", bt.image_path.iloc[0])
print("  ", stat(bt.image_path.iloc[0]))
print("example before_png[0]:", ft.before_png.iloc[0])
print("  ", stat(ft.before_png.iloc[0]))
print("same basename?", os.path.basename(str(bt.image_path.iloc[0])), "vs", os.path.basename(str(ft.before_png.iloc[0])))
