#!/bin/bash
# ============================================================================
# C1 根因诊断 (PURE):同图矩匹配端点 BBDM,只保留桥损失 (无自保持/重建冲突)。
#
#   桥两端 = 同一张目标图 (x_B=原图, x_A=该图矩匹配到源域风格),内容一致、只差风格。
#   纯配置 (bbdm_knee_moment_self_pure.json) 关闭了 self_l1/ssim/edge/latent_recon/
#   source_recon/supcon,避免"自保持损失把输出拉回目标风格"污染诊断。
#   moment_self 训练不使用任何目标标签 (dataset 返回 label=-1)。
#
# 判读用 Oracle 四参照点 (不是历史 0.795):
#   direct_128 | moment_oracle | moment_vae | moment_bbdm
#     moment_bbdm ≈ moment_vae            -> 随机配对/损失冲突是主因(线复活)
#     moment_bbdm << moment_vae           -> 桥实现问题(latent scale / t=T / 采样)
#     moment_vae  << moment_oracle        -> 损失来自 VAE / 128 分辨率
#     moment_bbdm 追平 moment_vae 但≈direct -> 翻译不加判别信息,转源->目标增强
#
# 需要带 GPU 的 autodl 实例。用法: bash run_moment_self_diag.sh
# ============================================================================
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
CODE=/root/autodl-tmp/knee/code2/idea2_diffusion_baseline
AS=/root/autodl-tmp/knee_allslices
CFG=$CODE/bbdm_strict/configs/bbdm_knee_moment_self_pure.json
CLF=$AS/exp/cls_allslices/best_checkpoint.pt
BCK=$AS/exp/bbdm_runs/bbdm_moment_self_pure/checkpoints/latest.pt
LOG=$AS/moment_self_diag.log
cd $CODE/bbdm_strict
exec > >(tee -a $LOG) 2>&1
echo "############ C1 PURE 诊断: moment_self BBDM  start $(date) ############"

echo "===== [1] 训练 PURE BBDM (moment_self, 仅桥损失) ====="
python train_strict_bbdm.py --config $CFG || { echo TRAIN_FAIL; exit 1; }

echo "===== [2] Oracle 四参照点评估 (direct/moment/moment+VAE/BBDM) ====="
echo "@@@ RESULT_ORACLE @@@"
python eval_moment_self_oracle.py \
  --clf_ckpt $CLF \
  --slice_csv $AS/kneemri_test/allslices.csv \
  --source_csv $AS/mrnet_train/allslices.csv \
  --bbdm_config $CFG --bbdm_ckpt $BCK \
  --num_inference_steps 50 --image_size 128 --n_ref 1000 --seed 42 \
  --n_boot 2000 --equiv_margin 0.02 --out_csv $AS/moment_self_oracle_percase.csv

echo "MOMENT_SELF_DIAG_DONE $(date)"
