#!/bin/bash
# ============================================================================
# C1 诊断实验:同图矩匹配端点 BBDM
#   桥的两端 = 同一张目标图 (x_B=原图, x_A=该图矩匹配到源域风格),
#   结构完全一致、只差风格 -> 检验"随机配对"是不是 BBDM 学错任务的根因。
#
# 结论判读 (目标域 KneeMRI test 体级 AUC):
#   ~0.795  -> 主因确实是随机端点配对, 修好配对翻译就能追平矩匹配;
#   仍<0.72 -> 还有 latent scale / 时间步 / 反向采样等实现问题;
#   ≈0.795 但不超 -> 扩散翻译本身不额外提供判别信息(与 UNSB 结论一致)。
#
# 对照:  label_random BBDM ~0.60 | VAE 纯重建 0.726 | 直接迁移 0.742 | 矩匹配 0.795
#
# 需要带 GPU 的 autodl 实例。用法: bash run_moment_self_diag.sh
# ============================================================================
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
CODE=/root/autodl-tmp/knee/code2/idea2_diffusion_baseline
AS=/root/autodl-tmp/knee_allslices
CFG=$CODE/bbdm_strict/configs/bbdm_knee_moment_self.json
CLF=$AS/exp/cls_allslices/best_checkpoint.pt
BCK=$AS/exp/bbdm_runs/bbdm_moment_self/checkpoints/latest.pt
LOG=$AS/moment_self_diag.log
cd $CODE/bbdm_strict
exec > >(tee -a $LOG) 2>&1
echo "############ C1 诊断:moment_self BBDM  start $(date) ############"

echo "===== [1] 训练 BBDM (pair_mode=moment_self, 同图矩匹配端点) ====="
python train_strict_bbdm.py --config $CFG || { echo TRAIN_FAIL; exit 1; }

echo "===== [2] 翻译评估 (目标 KneeMRI test, 体级 AUC) ====="
echo "@@@ RESULT_MOMENT_SELF @@@"
python eval_volume.py --clf_ckpt $CLF --slice_csv $AS/kneemri_test/allslices.csv \
  --mode translate --bbdm_config $CFG --bbdm_ckpt $BCK --num_inference_steps 50 --image_size 128

echo "MOMENT_SELF_DIAG_DONE $(date)"
