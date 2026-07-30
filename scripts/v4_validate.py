#!/usr/bin/env python
"""STEP 1: faithfully reproduce the task classifier's ORIGINAL validation.

Exact pipeline copied from eval_existing_classifier_on_csv.py:
  read  : Image.open(path).convert('L')
  transform: ToTensor -> Resize((224,224), antialias=True) -> repeat(3) -> Normalize(0.5)
  model : HybridResNet(custom_resnet50_space) = stem -> SpaceAttention -> avgpool -> classifier
Acceptance: reproduced AUC/acc/confusion must match the checkpoint's recorded val_metrics
  (AUC 0.9913, acc 0.9692, tp21 tn42 fp2 fn0, n=65).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T, models
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASK_CKPT = "/root/autodl-tmp/breast/exp/gate_busi2breast_cache/best_checkpoint.pt"
VAL_CSV = "/root/autodl-tmp/breast/cache/busi_valid.csv"


def build_transform(resize=224):
    return T.Compose([
        T.ToTensor(),
        T.Resize((resize, resize), antialias=True),
        T.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class SpaceAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv2d(d, d, 1)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        att = self.conv(x); b, c, h, w = att.shape
        att = self.soft(att.view(b, c, -1))
        m = att.amax(dim=2, keepdim=True).clamp_min(1e-6)
        att = (att / m).view(b, c, h, w)
        return x * att


class HybridResNet(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        m = models.resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.space_attn = SpaceAttention(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, 2)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        pref = {"stem.": self.stem, "space_attn.": self.space_attn, "classifier.": self.classifier}
        report = {}
        for p, mod in pref.items():
            sub = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            miss, unexp = mod.load_state_dict(sub, strict=False)
            report[p] = (len(sub), len([k for k in miss if "num_batches" not in k]), len(unexp))
        print("[load]", {k: v for k, v in report.items()})

    @torch.no_grad()
    def forward(self, x):
        f = self.stem(x)
        z = self.avgpool(self.space_attn(f)).flatten(1)
        return self.classifier(z)


def main():
    df = pd.read_csv(VAL_CSV)
    tf = build_transform(224)
    model = HybridResNet(TASK_CKPT).to(DEV).eval()
    probs, ys = [], []
    for b0 in range(0, len(df), 16):
        rows = df.iloc[b0:b0 + 16]
        xb = torch.stack([tf(Image.open(p).convert("L")) for p in rows["image_path"]]).to(DEV)
        with torch.no_grad():
            p1 = F.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
        probs.append(p1); ys.append(rows["label"].astype(int).values)
    p = np.concatenate(probs); y = np.concatenate(ys); pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print("\nREPRO  AUC=%.4f  acc=%.4f  tp=%d tn=%d fp=%d fn=%d  n=%d" %
          (roc_auc_score(y, p), accuracy_score(y, pred), tp, tn, fp, fn, len(y)))
    print("RECORD AUC=0.9913  acc=0.9692  tp=21 tn=42 fp=2 fn=0  n=65")
    ok = abs(roc_auc_score(y, p) - 0.9913) < 0.005 and tp == 21 and tn == 42 and fp == 2 and fn == 0
    print("\nPIPELINE MATCH:" , "YES -- preprocessing confirmed" if ok else "NO -- still mismatched")


if __name__ == "__main__":
    main()
