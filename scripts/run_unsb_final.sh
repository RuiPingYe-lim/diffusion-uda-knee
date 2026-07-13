#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
LOG=/root/autodl-tmp/unsb_final.log
: > $LOG
exec > >(tee -a $LOG) 2>&1
echo "###### UNSB epoch-30 权威重跑 start ######"

echo "== 1. 用 epoch30 重新翻译 KneeMRI(184) 和 MRNet(1130) =="
cd /root/autodl-tmp/UNSB
python test.py --dataroot ./datasets/knee2mrnet --name k2m_SB --checkpoints_dir ./checkpoints \
  --mode sb --eval --phase test --num_test 300 --epoch latest --gpu_ids 0 --results_dir ./results 2>&1 | tail -1
python test.py --dataroot ./datasets/mrnet_src --name k2m_SB --checkpoints_dir ./checkpoints \
  --mode sb --eval --phase test --num_test 2000 --epoch latest --gpu_ids 0 --results_dir ./results_mrnet 2>&1 | tail -1

echo "== 2. 只用翻译图 的 AUC(前向 M->K) =="
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
python - <<'PY'
import sys,numpy as np,pandas as pd,torch
import torchvision.transforms as T
from PIL import Image
sys.path.insert(0,"/root/autodl-tmp/knee/code2")
from idea2_diffusion_baseline.eval_existing_classifier_on_csv import build_model
dev=torch.device("cuda")
def auc(y,p):
    y=np.asarray(y);p=np.asarray(p);pos,neg=(y==1).sum(),(y==0).sum()
    o=np.argsort(-p);ys=y[o];return float(np.trapz(np.r_[0,np.cumsum(ys==1)/pos],np.r_[0,np.cumsum(ys==0)/neg]))
tf=T.Compose([T.ToTensor(),T.Resize((224,224),antialias=True),T.Lambda(lambda t:t.repeat(3,1,1) if t.shape[0]==1 else t),T.Normalize([0.5]*3,[0.5]*3)])
clf=build_model("custom_resnet50_space",2,"none",dev)
clf.load_state_dict(torch.load("/root/autodl-tmp/meanproj_stage/cls_mrnet_mp.pt",map_location=dev,weights_only=False)["model"]);clf.to(dev).eval()
lab=pd.read_csv("/root/autodl-tmp/meanproj_stage/kneemri_test.csv")
U="/root/autodl-tmp/UNSB/results/k2m_SB/test_latest/images"; MP="/root/autodl-tmp/meanproj_stage/kneemri_test"
@torch.no_grad()
def ev(g):
    P,Y=[],[]
    for _,r in lab.iterrows():
        x=tf(Image.open(g(str(r["case_id"]))).convert("L")).unsqueeze(0).to(dev)
        P.append(float(torch.softmax(clf(x),1)[0,1]));Y.append(int(r["label"]))
    return auc(Y,P)
print("  直接迁移           AUC=%.3f"%ev(lambda c:f"{MP}/{c}.png"))
for k in ["fake_1","fake_3","fake_5"]:
    print("  UNSB %-6s        AUC=%.3f"%(k,ev(lambda c,k=k:f"{U}/{k}/{c}.png")))
PY

echo "== 3. 交叉注意力融合(原图+UNSB fake_1/3/5) =="
bash /root/autodl-tmp/run_unsb_fusion.sh 2>&1 | grep -e "volume_AUC" | tail -1
echo "UNSB_FINAL_DONE"
